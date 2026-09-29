from collections import Counter, defaultdict
from statistics import median

from src.runtime.forecast import extraction_key

def family(row: dict) -> tuple:
  """Group comparable tasks for resource measurements.

  :param row: Task record.
  :return: Timing-family identity.
  """
  spec = row.get('spec', {})
  return (
    row['operation'],
    row['dataset'],
    row.get('representation'),
    spec.get('model'),
    spec.get('qubits', spec.get('size')),
    spec.get('inputs'),
    spec.get('depth'),
    spec.get('adapter'),
    spec.get('reset', 'partial')
  )

def timing_family(row: dict) -> tuple:
  """Index compatible timing records without scanning unrelated completed work.

  :param row: Recorded or projected workload.
  :return: Dataset, numerical workload family and comparable settings.
  """
  spec = row.get('spec', {})
  operation = row['operation']
  if operation == 'final-fit':
    operation = 'neural' if spec.get('model') in ('lstm', 'transformer') else 'feature'
  elif operation == 'final-evaluate' and spec.get('model') not in ('lstm', 'transformer'):
    operation = 'feature'
  return (operation, row.get('dataset'), *(spec.get(key, 'partial' if key == 'reset' else None) for key in ('model', 'qubits', 'size', 'inputs', 'depth', 'reset', 'recipe', 'adapter')))

def estimate(row: dict, history: list) -> tuple:
  """Estimate every task, preferring current throughput over history and broad defaults.

  :param row: Active, queued or projected task.
  :param history: Original completed-task and compatible-cache timings.
  :return: Interval, evidence label, optional rate and progress unit.
  """
  operation, spec = row['operation'], row.get('spec', {})
  neural = spec.get('model') in ('lstm', 'transformer') and operation in ('neural', 'final-fit')
  unit = 'epochs' if neural else 'samples'
  progress = row.get('progress', {}).get(unit)
  if row.get('forecast_seconds'):
    return row['forecast_seconds'], 'conservative stage allowance', None, unit
  composite = operation in ('final-fit', 'final-evaluate') and not neural
  progress_is_extraction = not composite or not progress or not progress.get('task') or '/features/' in progress['task']
  if row.get('status') == 'running' and progress and progress.get('total') and progress_is_extraction:
    initial = row.get('progress_initial', {}).get(unit, {'completed': 0, 'elapsed_seconds': 0})
    count = progress['completed'] - initial['completed']
    elapsed = progress.get('elapsed_seconds', 0) - initial.get('elapsed_seconds', 0)
    if count > 0 and elapsed > 0:
      remaining = max(0, progress['total'] - progress['completed']) * elapsed / count
      overhead = 120 if operation in ('final-fit', 'final-evaluate') else 15
      return [remaining * 1.1 + overhead, remaining * 1.6 + overhead * 2], 'live epoch ceiling' if neural else 'live throughput', count / elapsed, unit
  samples = row.get('workload', {}).get('samples', row.get('units', 0)) or 1
  epochs = row.get('workload', {}).get('epochs', spec.get('fixed_epochs', 120)) if neural else 1
  target = 'neural' if neural else 'feature' if operation in ('final-fit', 'final-evaluate') else operation
  measured = []
  cached = []
  for previous in history:
    if previous.get('status') != 'completed' or previous.get('seconds', 0) <= 0 or previous.get('dataset') != row.get('dataset'):
      continue
    other = previous.get('spec', {})
    if timing_family(previous) != timing_family(row):
      continue
    if previous.get('progress', {}).get('samples', {}).get('status') == 'cached' and not previous.get('timing_origin'):
      continue
    count = previous.get('workload', {}).get('samples', previous.get('units', 0))
    if target in ('feature', 'neural') and not count:
      continue
    seconds = previous['seconds']
    if target in ('feature', 'neural'):
      seconds *= samples / count
    if neural:
      previous_epochs = previous.get('progress', {}).get('epochs', {}).get('completed') or previous.get('workload', {}).get('epochs', other.get('fixed_epochs', epochs))
      seconds *= epochs / max(1, previous_epochs)
    (cached if previous.get('timing_origin') else measured).append(seconds)
  values = measured or cached
  if values:
    overhead = 120 if operation == 'final-fit' else 30 if operation == 'final-evaluate' else 0
    return [median(values) * 1.1 + overhead, max(median(values) * 1.65, max(values) * 1.15) + overhead * 2], 'task measurements' if measured else 'saved neural measurements' if neural else 'original cache measurements', None, unit
  if target == 'feature':
    if spec.get('model') == 'qrc':
      per_sample = 0.8 * 4 ** max(0, spec.get('qubits', 6) - 6) * max(1, spec.get('depth', 1) / 2)
      seconds = samples * per_sample
    else:
      seconds = max(30, samples * 0.05)
  elif neural:
    seconds = max(300, samples * epochs * 0.01)
  else:
    seconds = {'readout': 120, 'adapter': 300, 'export': 120, 'data': 300, 'benchmark': 600, 'resources': 1800, 'report': 900}.get(operation, 300)
  return [seconds, seconds * 3], 'conservative fallback', None, unit

def schedule_duration(tasks: dict, durations: dict, limits: dict) -> float:
  """Simulate dependencies and resource ceilings, counting all estimated work.

  :param tasks: Actual and projected graph records.
  :param durations: Remaining seconds for every unfinished task.
  :param limits: Admission ceilings.
  :return: Projected elapsed duration, conservatively serial if graph admission fails.
  """
  pending = dict(sorted(durations.items(), key=lambda item: tasks[item[0]].get('status') != 'running'))
  completed = {key for key, row in tasks.items() if row['status'] == 'completed'}
  running, clock = {}, 0.0
  while pending or running:
    for key in list(pending):
      row = tasks[key]
      if any(dep not in completed and dep in durations for dep in row.get('dependencies', [])):
        continue
      requirements = row.get('resources', {})
      active = [tasks[item].get('resources', {}) for item in running]
      if active and (requirements.get('exclusive') or any(value.get('exclusive') for value in active)):
        continue
      if requirements.get('gpu_workers') and any(value.get('gpu_workers') and (requirements.get('gpu_exclusive') or value.get('gpu_exclusive')) for value in active):
        continue
      if any(sum(value.get(name, 0) for value in active) + requirements.get(name, 0) > ceiling for name, ceiling in limits.items()):
        continue
      running[key] = clock + max(0.001, pending.pop(key))
    if not running:
      return clock + sum(pending.values())
    clock = min(running.values())
    for key in [key for key, finish in running.items() if finish <= clock]:
      del running[key]
      completed.add(key)
  return clock

def summarize(tasks: dict, limits: dict, conditional: dict, history: list | None = None, projected: dict | None = None) -> dict:
  """Describe progress and a conservative full forecast without omitting untimed work.

  :param tasks: Durable actual task graph.
  :param limits: Admission ceilings.
  :param conditional: Descriptions of unresolved work, covered by fallback allowances.
  :param history: Compatible original cache timings.
  :param projected: Explicit forecast-only future tasks and stage allowances.
  :return: Serializable progress, per-dataset studies and full ETA with evidence mix.
  """
  evidence = defaultdict(list)
  for row in list(tasks.values()) + (history or []):
    if row.get('status') == 'completed':
      evidence[timing_family(row)].append(row)
  counts = Counter(row['status'] for row in tasks.values())
  for key in ('completed', 'running', 'ready', 'waiting', 'restoring', 'failed', 'blocked'):
    counts.setdefault(key, 0)
  for row in tasks.values():
    if row['status'] in ('pending', 'interrupted'):
      counts['ready' if all(tasks.get(dep, {}).get('status') == 'completed' for dep in row.get('dependencies', [])) else 'waiting'] += 1
  datasets, studies = {}, {}
  for row in tasks.values():
    name = row.get('dataset') or 'run-wide'
    state = 'completed' if row['status'] == 'completed' else 'pending'
    groups = [datasets.setdefault(name, {'completed': 0, 'pending': 0})]
    for study in row.get('studies') or [row.get('phase') or row['operation']]:
      groups.append(studies.setdefault(study, {}).setdefault(name, {'completed': 0, 'pending': 0}))
    for group in groups:
      group[state] += 1
  graph = dict(tasks) | (projected or {})
  if not projected:
    for key in conditional:
      graph['allowance:' + key] = {'operation': 'conditional', 'dataset': '', 'status': 'pending', 'forecast_seconds': [3600, 86400], 'forecast_only': True}
  fits = [key for key, row in graph.items() if row['operation'] == 'final-fit' and row['status'] != 'completed']
  if fits:
    graph['forecast:training-barrier'] = {
      'operation': 'barrier',
      'dataset': '',
      'status': 'pending',
      'dependencies': fits,
      'forecast_seconds': [0, 0]
    }
    for key, row in list(graph.items()):
      if row['operation'] == 'final-evaluate':
        graph[key] = row | {'dependencies': list(row.get('dependencies', [])) + ['forecast:training-barrier']}
  active, low, high, bases = [], {}, {}, Counter()
  covered = {extraction_key(row) for row in graph.values() if row['status'] == 'completed'} - {None}
  for key, row in sorted(graph.items(), key=lambda item: item[1]['status'] != 'running'):
    if row['status'] == 'completed':
      continue
    interval, basis, rate, unit = estimate(row, evidence[timing_family(row)])
    bank = extraction_key(row)
    if bank and bank in covered and row['status'] != 'running':
      interval, basis = [30, 180], 'shared extraction reuse allowance'
    covered.add(bank)
    low[key], high[key] = interval
    bases[basis] += 1
    if row['status'] == 'running':
      active.append({
        'id': key,
        **{name: row.get(name) for name in ('operation', 'dataset', 'spec', 'seed', 'fold', 'phase', 'pid', 'resources', 'studies')},
        'device': row.get('workload', {}).get('device', 'GPU' if row.get('resources', {}).get('gpu_workers') else 'CPU'),
        'progress': row.get('progress', {}),
        'rate_per_second': rate,
        'rate_unit': unit,
        'eta_seconds': interval,
        'eta_basis': basis
      })
  total = [schedule_duration(graph, values, limits) for values in (low, high)]
  return {
    'counts': dict(counts),
    'reused_tasks': sum(bool(row.get('reused')) for row in tasks.values() if row['status'] == 'completed'),
    'datasets': datasets,
    'studies': studies,
    'limits': limits,
    'active': active,
    'conditional': conditional,
    'eta': {
      'total_seconds': total,
      'evidence': dict(bases),
      'projected_tasks': len(projected or {}),
      'fallback_tasks': sum(value for key, value in bases.items() if 'fallback' in key or 'allowance' in key),
      'interpretation': 'Conservative planning range, not a confidence interval; fallback and future-stage allowances included'
    }
  }
