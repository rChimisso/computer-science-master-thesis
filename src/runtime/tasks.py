import concurrent.futures
import copy
import json
import multiprocessing
import os
import pickle
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from src.runtime.records import checksum, digest, read, stop, write
from src.runtime.resources import admission, memory, request

REGISTRY = {}
"""Active coordinators indexed by their absolute run directory."""

def current(config: dict):
  """Find the coordinator only in its owning process.

  :param config: Bound task configuration.
  :return: Active scheduler or None for a direct scientific call.
  """
  return None if config.get('_local') else REGISTRY.get(str(Path(config['paths'].get('run', '.')).resolve()))

class Scheduler:
  """Persist and supervise resource-bounded spawned computation tasks.

  :ivar root: Run directory.
  :ivar directory: Durable scheduler state outside sealed evidence.
  :ivar limits: Resource admission policy.
  :ivar config: Cancellation settings.
  :ivar records: Persisted task graph.
  :ivar futures: Process-local result futures.
  :ivar payloads: Trusted descriptor payloads.
  :ivar running: Owned process handles and progress offsets.
  :ivar mutex: Coordinator synchronization lock.
  :ivar closing: Event requesting scheduler shutdown.
  :ivar thread: Resource admission and monitoring thread.
  :ivar failure: First authoritative failure.
  :ivar blocks: Unresolved conditional work descriptions.
  :ivar started: Active session start time.
  :ivar previous_seconds: Previously accumulated active duration.
  :ivar forecast: Original timing evidence and projected future work.
  """

  def __init__(self, root: Path, limits: dict, config: dict):
    """Restore the durable task graph, never trusting stale running status.

    :param root: Stable run namespace.
    :param limits: Worker and thread ceilings.
    :param config: Bound cancellation configuration.
    """
    self.root, self.limits, self.config = Path(root), limits, config
    self.directory = self.root / 'execution'
    self.directory.mkdir(parents=True, exist_ok=True)
    self.records = read(self.directory / 'tasks.json') if (self.directory / 'tasks.json').exists() else {}
    for row in self.records.values():
      if row['status'] != 'completed':
        row['status'] = 'restoring'
        row['waiting'] = 'awaiting workflow registration'
    self.futures, self.payloads, self.running = {}, {}, {}
    self.mutex, self.closing = threading.RLock(), threading.Event()
    self.failure = None
    self.blocks = {}
    self.started = time.monotonic()
    previous = read(self.directory / 'status.json') if (self.directory / 'status.json').exists() else {}
    self.previous_seconds = previous.get('active_seconds', 0)
    from src.runtime.forecast import Forecast
    self.forecast = Forecast(self.root, config, config.get('_forecast_scope'))
    self.thread = threading.Thread(target=self.loop, name='task-coordinator', daemon=True)

  def __enter__(self):
    """Register the resource coordinator and begin supervision.

    :return: This scheduler.
    """
    REGISTRY[str(self.root.resolve())] = self
    self.forecast.recover(self.records)
    self.thread.start()
    return self

  def __exit__(self, kind, error, traceback):
    """Drain or cancel workers before removing the coordinator registration.

    :param kind: Exception type.
    :param error: Exception instance.
    :param traceback: Exception traceback.
    """
    if error and self.failure is None:
      self.failure = repr(error)
      (self.root / 'STOP').touch()
    self.closing.set()
    self.thread.join()
    REGISTRY.pop(str(self.root.resolve()), None)

  def conditional(self, key: str, description: str | None) -> None:
    """Track work whose exact tasks depend on future selections.

    :param key: Stable study identifier.
    :param description: Explanation, or None to resolve the placeholder.
    """
    with self.mutex:
      if description is None:
        self.blocks.pop(key, None)
      else:
        self.blocks[key] = description

  @contextmanager
  def coordinator_task(self, operation: str, identity: dict, dataset: str = ''):
    """Record a single-writer decision or publication performed by the coordinator.

    :param operation: Selection, freeze or report publication.
    :param identity: Deterministic inputs to the coordinator-owned operation.
    :param dataset: Optional dataset context.
    :yield: Owned task record until publication completes.
    """
    key = digest({'coordinator': operation, 'identity': identity})
    started = time.monotonic()
    with self.mutex:
      row = self.records.setdefault(key, {'attempts': 0, 'seconds': 0.0})
      row.update({
        'operation': operation,
        'forecast_action': self.forecast.scope.get('action'),
        'dataset': dataset,
        'studies': [operation],
        'status': 'running',
        'attempts': row['attempts'] + 1,
        'dependencies': [],
        'pid': os.getpid(),
        'resources': dict.fromkeys(('cpu_workers', 'cpu_threads', 'gpu_workers', 'host_bytes', 'gpu_bytes', 'exclusive'), 0)
      })
    try:
      yield row
    except BaseException as error:
      with self.mutex:
        row.update({'status': 'failed', 'error': repr(error)})
        self.failure = self.failure or repr(error)
      raise
    else:
      with self.mutex:
        row['status'] = 'completed'
    finally:
      with self.mutex:
        row['seconds'] += time.monotonic() - started

  def submit(self, operation: str, payload: dict, identity: dict, dependencies: tuple = ()):
    """Register one uniquely owned task and its already-declared prerequisites.

    :param operation: Worker operation.
    :param payload: Numerical arguments and data descriptors.
    :param identity: Stable numerical/task identity.
    :param dependencies: Futures returned by this scheduler.
    :return: Future yielding the compact task value.
    """
    if operation in ('neural', 'final-fit') and payload['spec']['model'] in ('lstm', 'transformer'):
      from src.runtime.records import sources
      config = payload['config']
      settings = {
        'data': payload['data'].fingerprint,
        'model': payload['spec']['model'],
        'device': config['execution']['device'],
        'training': config['training'],
        'neural': config['neural'][payload['spec']['model']]
      }
      directory = Path(config['paths']['results']) / 'metrics/neural-calibration' / digest(settings)
      local = config | {'paths': config['paths'] | {'evidence': str(directory)}}
      payload = payload | {'config': local}
      benchmark = self.submit('benchmark', payload, settings | {'source': sources(numerical=True)})
      dependencies = (*dependencies, benchmark)
    key = digest({'operation': operation, 'identity': identity})
    with self.mutex:
      if key in self.futures:
        self.records[key]['studies'] = sorted(set(self.records[key].get('studies', [])) | set(payload.get('config', {}).get('_study', [])))
        return self.futures[key]
      future = concurrent.futures.Future()
      future.task_id = key
      self.futures[key] = future
      self.payloads[key] = payload
      resources = request(operation, payload.get('spec', {}), payload['config'])
      from src.runtime.estimates import family
      prototype = {
        'operation': operation,
        'dataset': getattr(payload.get('data'), 'name', payload.get('name', '')),
        'representation': getattr(payload.get('data'), 'fingerprint', None),
        'spec': payload.get('spec', {})
      }
      for measured in self.records.values():
        if family(measured) == family(prototype):
          peaks = measured.get('measurements', {})
          host = peaks.get('parent_peak_rss_bytes', 0) + peaks.get('child_peak_rss_bytes', 0)
          resources['host_bytes'] = max(resources['host_bytes'], int(host * 1.25))
          if measured.get('isolated_gpu_measurement'):
            resources['gpu_bytes'] = max(resources['gpu_bytes'], int(measured.get('gpu_increment_bytes', 0) * 1.25))
      if resources['cpu_threads'] > self.limits['cpu_threads']:
        raise ValueError(f"{operation} needs at least {resources['cpu_threads']} CPU threads; increase --cpu-threads")
      data = payload.get('data')
      row = self.records.setdefault(key, {
        'operation': operation,
        'dataset': getattr(data, 'name', payload.get('name', '')),
        'representation': getattr(data, 'fingerprint', None),
        'spec': payload.get('spec', {}),
        'seed': payload.get('seed'),
        'fold': payload.get('fold', {}).get('fold', payload.get('phase')),
        'phase': payload.get('phase', 'development'),
        'studies': payload['config'].get('_study', []),
        'status': 'pending',
        'attempts': 0,
        'seconds': 0.0
      })
      from src.runtime.forecast import descriptor
      workload = descriptor(operation, payload)
      row.update({
        'workload': workload,
        'forecast_action': self.forecast.scope.get('action'),
        'dependencies': [item.task_id for item in dependencies],
        'resources': resources,
        'units': workload['samples']
      })
      result = self.directory / 'results' / f'{key}.json'
      if row['status'] == 'completed' and result.exists() and operation != 'data':
        if row.get('result_sha256') != checksum(result):
          raise ValueError('Completed task result envelope changed: ' + key)
        saved = read(result)
        for path, expected in saved.get('artifacts', {}).items():
          if not Path(path).exists() or checksum(path) != expected:
            raise ValueError('Completed task artifact changed: ' + path)
        future.set_result(self.value(saved['value']))
        row['reused'] = True
        return future
      row['status'] = 'pending'
      row['waiting'] = 'dependencies' if any(not item.done() for item in dependencies) else 'awaiting admission'
      row['reused'] = False
      if self.failure:
        row['status'] = 'blocked'
        future.set_exception(RuntimeError(self.failure))
      return future

  def value(self, value):
    """Resolve compact task references without sending large arrays over IPC.

    :param value: Published compact value.
    :return: Authoritative record or data descriptor.
    """
    if isinstance(value, dict) and 'descriptor' in value:
      if checksum(value['descriptor']) != value['sha256']:
        raise ValueError('Dataset descriptor changed')
      with Path(value['descriptor']).open('rb') as stream:
        return pickle.load(stream)
    if isinstance(value, dict) and 'record' in value:
      record = read(value['record'])
      from src.runtime.storage import intentionally_pruned
      pairs = [(row.get('state_path'), row.get('state_sha256')) for row in record.get('rows', [])]
      pairs.append((record.get('checkpoint'), record.get('checkpoint_sha256')))
      for path, expected in pairs:
        if path and not intentionally_pruned(path, self.config) and checksum(path) != expected:
          raise ValueError('Completed fitting artifact changed: ' + path)
      return record
    if isinstance(value, dict) and 'pipeline' in value:
      from src.runtime.pipelines import verify
      verify(value['pipeline'])
    if isinstance(value, dict) and 'artifacts' in value and 'directory' in value:
      from src.runtime.pipelines import verify
      verify(value)
    return value

  def experiment(self, data, fold: dict, spec: dict, seed: int, config: dict, penalties=None, phase='development'):
    """Declare extraction/readout dependencies, or a complete neural fitting task.

    :param data: Prepared dataset descriptor.
    :param fold: Isolated training/held memberships.
    :param spec: Numerical candidate.
    :param seed: Initialization seed.
    :param config: Bound scientific settings.
    :param penalties: Fixed penalties or the declared grid.
    :param phase: Evaluation population.
    :return: Future resolving the fitting record.
    """
    config = config | {'_study': config.get('_study_labels', {}).get(digest(spec), config.get('_study', []))}
    payload = {
      'data': data,
      'fold': fold,
      'spec': spec,
      'seed': seed,
      'config': config,
      'penalties': penalties,
      'phase': phase
    }
    identity = {
      'source': config['_identity'],
      'dataset': data.name,
      'data': data.fingerprint,
      'fold': fold,
      'seed': seed,
      'spec': spec,
      'penalties': penalties,
      'phase': phase
    }
    path = Path(config['paths']['results']) / 'metrics/jobs' / digest(identity) / 'result.json'
    if path.exists() and read(path).get('status') == 'completed':
      from src.studies.experiments import experiment
      result = experiment(data, fold, spec, seed, config | {'_local': True}, penalties, phase)
      future = concurrent.futures.Future()
      future.task_id = digest({'operation': 'neural' if spec['model'] in ('lstm', 'transformer') else 'readout', 'identity': identity})
      future.set_result(result)
      with self.mutex:
        self.futures[future.task_id] = future
        self.records[future.task_id] = {
          'operation': 'fit',
          'dataset': data.name,
          'studies': config.get('_study', []),
          'spec': spec,
          'seed': seed,
          'fold': fold['fold'],
          'phase': phase,
          'status': 'completed',
          'reused': True,
          'seconds': result.get('seconds', 0),
          'dependencies': []
        }
      return future
    if spec['model'] in ('lstm', 'transformer'):
      return self.submit('neural', payload, identity)
    training = fold['train']
    indices = training + fold['validation']
    adapter_identity = {
      'data': data.fingerprint,
      'train': training,
      'mode': spec.get('adapter', config['datasets'][data.name]['adapter']),
      'inputs': spec['inputs'],
      'seed': spec.get('projection_seed', 42)
    }
    adapter = self.submit('adapter', payload | {'train': training}, adapter_identity)
    feature_spec = {name: value for name, value in spec.items() if name not in ('view', 'summary', 'solver')}
    feature_payload = payload | {
      'train': training,
      'indices': indices,
      'adapter_result': str(self.directory / 'results' / f'{adapter.task_id}.json')
    }
    feature = self.submit('feature', feature_payload, {
      'adapter': adapter.task_id,
      'indices': indices,
      'spec': feature_spec,
      'seed': 42 if spec['model'] == 'projection' else seed
    }, (adapter,))
    return self.submit('readout', payload, identity, (feature,))

  def launch(self, key: str, available: dict) -> None:
    """Start a non-daemon worker with a private process group and progress stream.

    :param key: Ready task identifier.
    :param available: Latest host and native GPU measurement.
    """
    from src.runtime.task_worker import worker
    directory = self.root / 'temporary/scheduler'
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / f'{key}.pickle'
    with payload.open('wb') as stream:
      pickle.dump(self.payloads[key], stream)
    events = self.directory / 'logs' / f'{key}.jsonl'
    events.parent.mkdir(parents=True, exist_ok=True)
    result = self.directory / 'results' / f'{key}.json'
    result.parent.mkdir(parents=True, exist_ok=True)
    result.unlink(missing_ok=True)
    arguments = (
      str(payload),
      str(result),
      str(events),
      self.records[key]['operation'],
      os.getpid(),
      self.records[key]['resources']['cpu_threads']
    )
    process = multiprocessing.get_context('spawn').Process(target=worker, args=arguments)
    process.start()
    row = self.records[key]
    row.pop('progress_initial', None)
    row.pop('progress', None)
    row.update({
      'status': 'running',
      'attempts': row['attempts'] + 1,
      'started_utc': time.time(),
      'pid': process.pid,
      'waiting': None
    })
    row['concurrency'] = {
      'active_tasks': sorted(self.running),
      'policy': self.limits,
      'gpu_measurement_domain': 'Whole-device memory, including unrelated applications'
    }
    gpu = row['resources']['gpu_workers']
    if gpu and available['gpu_total'] is not None:
      row['initial_global_gpu_bytes'] = available['gpu_total'] - available['gpu_free']
      row['isolated_gpu_measurement'] = not any(self.records[item]['resources']['gpu_workers'] for item in self.running)
      for item in self.running:
        if self.records[item]['resources']['gpu_workers']:
          self.records[item]['isolated_gpu_measurement'] = False
    self.running[key] = {
      'process': process,
      'started': time.monotonic(),
      'offset': events.stat().st_size if events.exists() else 0,
      'events': events,
      'result': result
    }

  def inspect(self, key: str) -> None:
    """Collect progress and atomically published completion from one owned worker.

    :param key: Active task identifier.
    """
    active, row = self.running[key], self.records[key]
    if active['events'].exists():
      with active['events'].open() as stream:
        stream.seek(active['offset'])
        for line in stream:
          try:
            event = json.loads(line)
          except json.JSONDecodeError:
            break
          if event['unit'] in ('samples', 'epochs', 'batches'):
            initial = row.setdefault('progress_initial', {}).get(event['unit'])
            if initial is None or initial['id'] != event['id']:
              row['progress_initial'][event['unit']] = event
            row.setdefault('progress', {})[event['unit']] = event
          active['offset'] += len(line.encode())
    process = active['process']
    if process.is_alive():
      return
    process.join()
    row['finished_utc'] = time.time()
    row['seconds'] += time.monotonic() - active['started']
    saved = read(active['result']) if active['result'].exists() else {'status': 'failed', 'error': f'Worker exited {process.exitcode} without a committed result'}
    row['status'] = saved['status']
    if saved['status'] == 'completed' and process.exitcode == 0:
      row['measurements'] = saved.get('measurements', {})
      row['result_sha256'] = checksum(active['result'])
      row['reused'] = row.get('progress', {}).get('samples', {}).get('status') == 'cached'
      if not row['reused']:
        row['measured_seconds'] = time.monotonic() - active['started']
      try:
        self.futures[key].set_result(self.value(saved['value']))
      except BaseException as error:
        saved = {'status': 'failed', 'error': repr(error)}
    if saved['status'] != 'completed' or process.exitcode:
      row.update({'status': saved['status'] if saved['status'] != 'completed' else 'failed', 'error': saved.get('error', str(process.exitcode))})
      self.failure = self.failure or row['error']
      if not self.futures[key].done():
        self.futures[key].set_exception(RuntimeError(row['error']))
    del self.running[key]

  def publish(self) -> None:
    """Persist a consistent task graph and lightweight monitor snapshot."""
    from src.runtime.estimates import summarize
    rows = copy.deepcopy(self.records)
    projected = self.forecast.pending(rows)
    snapshot = summarize(rows, self.limits, self.blocks, self.forecast.history, projected)
    snapshot.update({
      'updated_utc': time.time(),
      'pid': os.getpid(),
      'active_seconds': self.previous_seconds + time.monotonic() - self.started,
      'error': self.failure
    })
    snapshot['state'] = 'pausing' if self.failure else 'running'
    write(self.directory / 'tasks.json', rows)
    write(self.directory / 'status.json', snapshot)

  def loop(self) -> None:
    """Supervise task admission, cancellations and durable progress."""
    stopped_at = None
    last_memory, available = 0.0, None
    try:
      while True:
        with self.mutex:
          try:
            stop(self.config)
          except TimeoutError as error:
            self.failure = self.failure or str(error)
          for key in list(self.running):
            self.inspect(key)
          if self.failure or self.closing.is_set():
            pending = [key for key, future in self.futures.items() if not future.done() and key not in self.running]
            if self.failure:
              (self.root / 'STOP').touch()
              stopped_at = stopped_at or time.monotonic()
              for key in pending:
                self.records[key]['status'] = 'blocked'
                self.futures[key].set_exception(RuntimeError(self.failure))
              if time.monotonic() - stopped_at > 30:
                for active in self.running.values():
                  try:
                    os.killpg(active['process'].pid, signal.SIGKILL)
                  except ProcessLookupError:
                    pass
            if not self.running and not pending:
              break
          if not self.failure:
            if time.monotonic() - last_memory > 3:
              available, last_memory = memory(), time.monotonic()
            for active_key in self.running:
              active_row = self.records[active_key]
              if active_row['resources']['gpu_workers'] and available['gpu_total'] is not None:
                used = available['gpu_total'] - available['gpu_free']
                active_row['peak_global_gpu_bytes'] = max(active_row.get('peak_global_gpu_bytes', 0), used)
                active_row['gpu_increment_bytes'] = max(active_row.get('gpu_increment_bytes', 0), used - active_row.get('initial_global_gpu_bytes', used))
            ready = [key for key in self.futures if not self.futures[key].done() and key not in self.running]
            active_datasets = [self.records[key]['dataset'] for key in self.running]
            active_gpu = sum(self.records[key]['resources']['gpu_workers'] for key in self.running)
            active_studies = [study for key in self.running for study in self.records[key].get('studies', [])]
            ready.sort(key=lambda key: (
              not self.records[key]['resources'].get('exclusive'),
              not self.records[key]['resources'].get('gpu_exclusive'),
              not (self.records[key]['resources']['gpu_workers'] and not active_gpu),
              active_datasets.count(self.records[key]['dataset']),
              sum(active_studies.count(study) for study in self.records[key].get('studies', [])),
              self.records[key].get('attempts', 0)
            ))
            draining = None
            gpu_waiting = False
            for key in ready:
              row = self.records[key]
              if any(not self.futures[dep].done() for dep in row['dependencies']):
                row['waiting'] = 'dependencies'
                continue
              requirements = row['resources']
              active = [self.records[item]['resources'] for item in self.running]
              reason = admission(requirements, active, self.limits, available)
              barrier = draining or ('exclusive GPU draining' if gpu_waiting and requirements['gpu_workers'] else None)
              if not reason and not barrier:
                self.launch(key, available)
              else:
                row['waiting'] = reason or barrier
                if requirements['exclusive']:
                  draining = 'exclusive task draining'
                elif requirements.get('gpu_exclusive'):
                  gpu_waiting = True
                if requirements['gpu_workers'] and not any(item['gpu_workers'] for item in active) and reason == 'cpu threads budget' and not draining:
                  draining = 'GPU host threads draining'
                if not self.running and reason in ('host memory', 'GPU memory', 'GPU memory unavailable'):
                  self.failure = f'Task cannot fit available memory: {key}: {reason}'
          self.publish()
        time.sleep(1)
    except BaseException as error:
      self.failure = repr(error)
      (self.root / 'STOP').touch()
      for active in self.running.values():
        try:
          os.killpg(active['process'].pid, signal.SIGKILL)
        except ProcessLookupError:
          pass
        active['process'].join()
      for future in self.futures.values():
        if not future.done():
          future.set_exception(RuntimeError(self.failure))
    finally:
      with self.mutex:
        self.publish()
