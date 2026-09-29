import os
import pickle
import traceback
from pathlib import Path

def perform(operation: str, payload: dict):
  """Execute one owned operation using the unchanged scientific implementations.

  :param operation: Task kind.
  :param payload: Data descriptors and numerical arguments.
  :return: Compact task result or a path to its authoritative record.
  """
  import numpy as np
  from src.runtime.records import read
  data, config = payload.get('data'), payload['config'] | {'_local': True}
  spec, seed = payload.get('spec', {}), payload.get('seed', 42)
  if operation == 'data':
    from src.data.datasets import load
    from src.runtime.records import digest
    from src.runtime.scheduling import lock
    key = digest([
      payload['name'],
      payload.get('split', 'train'),
      config['data'][payload['name']]
    ])
    with lock(Path(config['paths']['cache']) / 'locks' / f'dataset-{key}.lock', config):
      return load(payload['name'], config, payload.get('split', 'train'), payload.get('official', False))
  if operation == 'adapter':
    from src.features.adapters import fit_adapter
    mode = spec.get('adapter', config['datasets'][data.name]['adapter']) if spec.get('inputs') != 0 else 'identity'
    return fit_adapter(data, np.asarray(payload['train']), mode, spec['inputs'], spec.get('projection_seed', 42), config)
  if operation == 'feature':
    from src.features.extraction import extract
    adapter = read(payload['adapter_result'])['value']
    _, record, _ = extract(data, np.asarray(payload['train']), np.asarray(payload['indices']), spec, seed, config, adapter)
    return {'metadata': str(Path(record['directory']) / 'metadata.json')}
  if operation == 'benchmark':
    from src.models.training import benchmark_neural
    return benchmark_neural(data, spec['model'], config)
  if operation in ('readout', 'neural'):
    from src.studies.experiments import experiment
    result = experiment(data, payload['fold'], spec, seed, config, payload.get('penalties'), payload.get('phase', 'development'))
    return {'record': result['path']}
  if operation == 'export':
    from src.runtime.pipelines import export
    return export(data, np.asarray(payload['train']), spec, seed, payload['phase'], read(payload['job']), config)
  if operation == 'final-fit':
    from src.studies.final import fit
    return fit(data, payload['entry'], seed, config)
  if operation == 'final-evaluate':
    from src.studies.final import evaluate
    return {'record': evaluate(data, payload['held'], payload['entry'], payload['fitted'], seed, config)['path']}
  if operation == 'resources':
    from src.studies.resources import measure
    return measure(payload['datasets'], payload['frozen'], config)
  if operation == 'fixture':
    import time
    for index in range(payload.get('steps', 1)):
      from src.runtime.records import stop
      stop(config)
      time.sleep(payload.get('delay', 0.01))
    if payload.get('fail'):
      raise RuntimeError('Requested synthetic failure')
    return payload.get('value', 1)
  raise ValueError('Unknown task operation: ' + operation)

def configure_threads(threads: int) -> None:
  """Limit libraries imported later in this worker or its child processes.

  :param threads: Reserved native thread ceiling.
  """
  if not isinstance(threads, int) or threads < 1:
    raise ValueError('Worker thread reservation must be a positive integer')
  for name in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'BLIS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS'
  ):
    os.environ[name] = str(threads)
  os.environ['OMP_DYNAMIC'] = 'FALSE'
  os.environ['MKL_DYNAMIC'] = 'FALSE'

def worker(payload_path: str, result_path: str, events: str, operation: str, parent_pid: int, threads: int) -> None:
  """Run in an independent process group, publishing owned results atomically.

  :param payload_path: Trusted run-local pickle of descriptors and arguments.
  :param result_path: Owned compact result destination.
  :param events: Owned append-only progress stream.
  :param operation: Explicit computation kind.
  :param parent_pid: Coordinator identity before spawning.
  :param threads: CPU threads reserved by the scheduler.
  """
  configure_threads(threads)
  os.setsid()
  import ctypes
  import signal
  def parent_died(signum, frame) -> None:
    """Terminate the entire owned process group after coordinator death.

    :param signum: Parent-death signal.
    :param frame: Interrupted execution frame.
    """
    os.killpg(os.getpgrp(), signal.SIGKILL)
  signal.signal(signal.SIGTERM, parent_died)
  ctypes.CDLL(None).prctl(1, signal.SIGTERM)
  if os.getppid() != parent_pid:
    parent_died(signal.SIGTERM, None)
  os.environ['THESIS_PROGRESS_EVENTS'] = events
  os.environ['THESIS_PROGRESS'] = 'off'
  from src.runtime.records import checksum, write
  with Path(payload_path).open('rb') as stream:
    payload = pickle.load(stream)
  try:
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=threads):
      value = perform(operation, payload)
    if operation == 'data':
      target = Path(result_path).with_suffix('.pickle')
      with target.open('wb') as stream:
        pickle.dump(value, stream)
      value = {'descriptor': str(target), 'sha256': checksum(target)}
    artifacts = {}
    if operation == 'benchmark':
      path = Path(payload['config']['paths']['evidence']) / payload['data'].name / f"benchmark-{payload['spec']['model']}.json"
      artifacts[str(path)] = checksum(path)
    if isinstance(value, dict):
      for key in ('record', 'metadata'):
        if key in value:
          artifacts[value[key]] = checksum(value[key])
    import resource
    measurements = {
      'parent_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
      'child_peak_rss_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
      'host_boundary': 'Separate process peaks; their sum is a conservative reservation, not measured simultaneous RSS'
    }
    write(result_path, {
      'status': 'completed',
      'value': value,
      'artifacts': artifacts,
      'measurements': measurements
    })
  except BaseException as error:
    write(result_path, {
      'status': 'interrupted' if isinstance(error, (TimeoutError, KeyboardInterrupt)) else 'failed',
      'error': repr(error),
      'traceback': traceback.format_exc()
    })
    raise
