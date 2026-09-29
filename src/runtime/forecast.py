import json
from pathlib import Path

from src.runtime.records import digest, read
from src.runtime.resources import device, request as reservation

def descriptor(operation: str, payload: dict) -> dict:
  """Describe work for forecasting without opening held data or changing computation.

  :param operation: Worker operation.
  :param payload: Submitted data descriptors and numerical settings.
  :return: Serializable sample and epoch counts and execution device.
  """
  data = payload.get('held') if operation == 'final-evaluate' else payload.get('data')
  training = payload.get('train', payload.get('fold', {}).get('train', []))
  indices = payload.get('indices', training)
  samples = len(indices)
  if operation in ('final-fit', 'final-evaluate', 'data') and data is not None:
    samples = len(data.labels)
  spec, config = payload.get('spec', {}), payload['config']
  neural = spec.get('model') in ('lstm', 'transformer')
  epochs = spec.get('fixed_epochs', config.get('training', {}).get('max_epochs', 120)) if neural else 1
  return {'samples': max(1, samples), 'epochs': epochs, 'device': device(operation, spec, config)}

def signature(row: dict) -> str:
  """Identify a planned phase task independently of paths and display settings.

  :param row: Actual or projected task.
  :return: Stable planning identity, not a numerical cache identity.
  """
  return digest([row['operation'], row.get('dataset', ''), row.get('seed'), row.get('spec', {}), row.get('phase')])

def extraction_key(row: dict) -> str | None:
  """Identify extraction shared by observation/readout variants within a phase.

  :param row: Actual or projected task with sample population.
  :return: Shared extraction identity, or None for non-reservoir work.
  """
  if row['operation'] not in ('feature', 'final-fit', 'final-evaluate') or row.get('spec', {}).get('model') in ('lstm', 'transformer', None):
    return None
  return digest([
    row['operation'],
    row.get('dataset'),
    row.get('phase'),
    row.get('fold'),
    42 if row['spec'].get('model') == 'projection' else row.get('seed'),
    row.get('workload', {}).get('samples', row.get('units')),
    {key: value for key, value in row['spec'].items() if key not in ('view', 'summary', 'solver')}
  ])

class Forecast:
  """Recover numerical timing evidence and bound not-yet-registered workflow stages.

  :ivar root: Run directory.
  :ivar config: Resolved settings used only to describe expected work.
  :ivar scope: Requested action, datasets and optional tasks.
  :ivar history: Compatible original extraction measurements from shared caches.
  :ivar frozen: Most recently read frozen manifest.
  :ivar stamp: Frozen manifest modification time used to avoid repeated parsing.
  """

  def __init__(self, root: Path, config: dict, scope: dict | None = None):
    """Prepare planning state without reading datasets or initializing a GPU.

    :param root: Run directory.
    :param config: Numerical execution configuration.
    :param scope: Action and resolved request, absent for small scheduler fixtures.
    """
    self.root, self.config, self.scope = root, config, scope or {}
    self.history, self.frozen, self.stamp = [], None, None

  def recover(self, tasks: dict) -> None:
    """Recover original cache timings once, requiring matching numerical identities.

    :param tasks: Saved tasks mapping representation fingerprints to datasets.
    """
    from src.runtime.numerical import identity
    identities = {}
    datasets = {row.get('representation'): row['dataset'] for row in tasks.values() if row.get('representation')}
    for path in sorted((Path(self.config['paths']['cache']) / 'features').glob('*/metadata.json')):
      try:
        value = json.loads(path.read_text())
        source = value['identity']
        spec = source['spec']
        model = spec['model']
        if model not in identities:
          identities[model] = identity(model, self.config)
        if value.get('status') != 'completed' or source['source'] != identities[model] or not value.get('completed_samples') or not value.get('seconds'):
          continue
        name = datasets.get(source['data'])
        if source.get('data'):

          self.history.append({
            'operation': 'feature',
            'dataset': name or '',
            'representation': source['data'],
            'spec': spec,
            'units': value['completed_samples'],
            'seconds': value['seconds'],
            'status': 'completed',
            'timing_origin': str(path)
          })
      except (OSError, ValueError, KeyError, TypeError):
        continue

    confirmation = self.root / 'evidence/metrics/confirmation.json'
    if confirmation.exists():
      for job in read(confirmation).get('jobs', []):
        if job['role'] not in ('practical:lstm', 'practical:transformer'):
          continue
        try:
          value = json.loads(Path(job['path']).read_text())
          value = value.get('record', value)
          settings = value['identity']
          epochs = len(value['neural']['history'])
          self.history.append({
            'operation': 'neural',
            'dataset': job['dataset'],
            'spec': settings['spec'],
            'units': len(settings['fold']['train']),
            'workload': {'samples': len(settings['fold']['train']), 'epochs': epochs},
            'seconds': value['seconds'],
            'status': 'completed',
            'timing_origin': job['path']
          })
        except (OSError, KeyError, ValueError, TypeError):
          continue

  def pending(self, tasks: dict) -> dict:
    """Materialize forecast-only future work, replacing it as real tasks appear.

    :param tasks: Current persistent execution graph.
    :return: Projected tasks that are never dispatched as computation.
    """
    datasets = {row.get('representation'): row['dataset'] for row in tasks.values() if row.get('representation')}
    for record in self.history:
      if record.get('representation') in datasets:
        record['dataset'] = datasets[record['representation']]
    if not self.scope:
      return {}
    path = self.root / 'frozen.json'
    if path.exists() and path.stat().st_mtime_ns != self.stamp:
      self.frozen = read(path)['manifest']
      self.stamp = path.stat().st_mtime_ns
    action = self.scope.get('action', 'run')
    requested = self.scope.get('request', {})
    official = action == 'evaluate' or (action == 'run' and requested.get('official_test', True) and not requested.get('diagnostic_only'))
    virtual = {}
    observed = {signature(row) for row in tasks.values()}
    progress = read(self.root / 'progress.json') if (self.root / 'progress.json').exists() else {}
    if self.frozen:
      confirmation = self.root / 'evidence/metrics/confirmation.json'
      confirmed = confirmation.exists() and read(confirmation).get('status') == 'completed'
      phases = ([] if action == 'evaluate' or confirmed else ['confirmation']) + (['official'] if official else [])
      for phase in phases:
        for name, entries in self.frozen['entries'].items():
          for entry in entries.values():
            for seed in self.frozen['config']['seeds']:
              spec = entry['spec']
              neural = spec['model'] in ('lstm', 'transformer')
              operations = ['final-fit', 'final-evaluate'] if phase == 'official' else ['neural' if neural else 'feature', 'export']
              for operation in operations:
                samples = (2264 if name == 'shd' else 264) if operation == 'final-evaluate' else (8156 if name == 'shd' else 1077)
                row = {
                  'operation': operation,
                  'dataset': name,
                  'spec': spec,
                  'seed': seed,
                  'phase': phase,
                  'fold': phase,
                  'status': 'pending',
                  'dependencies': [],
                  'resources': reservation(operation, spec, self.config),
                  'units': samples,
                  'workload': {'samples': samples, 'epochs': spec.get('fixed_epochs', 120) if neural else 1},
                  'forecast_only': True
                }
                key = signature(row)
                if key not in observed:
                  virtual['forecast:' + key] = row
    else:
      from src.studies.design import groups
      from src.runtime.estimates import estimate
      for name in requested.get('datasets', []):
        if name in progress.get('completed', []):
          continue
        declarations = groups(name, requested.get('studies', ['comparison']), requested.get('models', []), self.config)
        candidates = {digest(spec): spec for group in declarations for spec in group['candidates']}
        low, high = 0.0, 0.0
        for spec in candidates.values():
          sample_count = 6340 if name == 'shd' else 880
          row = {'operation': 'neural' if spec['model'] in ('lstm', 'transformer') else 'feature', 'dataset': name, 'spec': spec, 'units': sample_count}
          interval, _, _, _ = estimate(row, list(tasks.values()) + self.history)
          low += interval[0] * 4
          high += interval[1] * 4
        replication = max(1, len(requested.get('seeds', self.config['seeds'])))
        supporting = 2 if set(requested.get('studies', [])) & {'mapping', 'learning-curves', 'preprocessing'} else 1
        spent = sum(row.get('seconds', 0) for row in tasks.values() if row.get('dataset') == name and row.get('status') == 'completed' and not row.get('reused'))
        remaining = [max(600, low * replication * supporting - spent), max(3600, high * replication * supporting - spent)]
        virtual['development-envelope:' + name] = self.allowance('development-envelope', remaining, name)
      if official:
        virtual['future-final-refits'] = self.allowance('future-final-refits', [24 * 3600, 96 * 3600])
    for operation, interval in (('report', [300, 1800]), ('resources', [300, 3600])):
      if operation == 'resources' and (action != 'run' or not requested.get('resources')):
        continue
      existing = [row for row in tasks.values() if row['operation'] == operation and row.get('forecast_action') == action]
      if not existing:
        virtual['finalization:' + operation] = self.allowance(operation, interval)
    return virtual

  def allowance(self, operation: str, interval: list, dataset: str = '') -> dict:
    """Describe an explicit conservative allowance without claiming measured evidence.

    :param operation: Future stage name.
    :param interval: Lower and upper planning seconds.
    :param dataset: Optional dataset context.
    :return: Undispatched exclusive planning task.
    """
    return {
      'operation': operation,
      'dataset': dataset,
      'status': 'pending',
      'spec': {},
      'dependencies': [],
      'resources': {'cpu_workers': 1, 'cpu_threads': self.config['execution']['cpu_threads'], 'gpu_workers': 0, 'exclusive': True},
      'forecast_seconds': interval,
      'forecast_only': True
    }
