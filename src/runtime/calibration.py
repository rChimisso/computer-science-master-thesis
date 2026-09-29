import statistics
import time
from pathlib import Path

from src.runtime.configuration import bind, load
from src.runtime.records import digest, read, sources, write
from src.runtime.resources import policy
from src.runtime.tasks import Scheduler

def workloads(root: Path, config: dict) -> list:
  """Prepare complete development sequences and fixed train-fitted adapters.

  :param root: Independent calibration namespace.
  :param config: Unchanged numerical policy.
  :return: Repeated extraction payloads spanning both datasets and CPU/GPU families.
  """
  import numpy as np
  from src.data.datasets import load as load_data
  from src.features.adapters import fit_adapter
  rows = []
  for name in ('shd', 'dvs'):
    data = load_data(name, config)
    training = data.development
    ordered = sorted(training.tolist(), key=lambda index: len(data.counts(index)))
    adapter = fit_adapter(data, training, config['datasets'][name]['adapter'], 4, 42, config)
    path = root / 'adapters' / f'{name}.json'
    write(path, {'value': adapter})
    for family, size, count in (('qrc', 6, 128), ('qrc', 8, 24), ('crc', 24, 1024)):
      indices = [ordered[index] for index in np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)]
      spec = {
        'model': family,
        'inputs': 4,
        'view': 'all'
      }
      spec.update({'qubits': size, 'depth': config['datasets'][name]['depth']} if family == 'qrc' else {'size': size} | config['anchors']['crc'])
      rows.append({
        'data': data,
        'train': training.tolist(),
        'indices': indices,
        'spec': spec,
        'seed': 42,
        'adapter_result': str(path),
        'config': config
      })
  return rows

def trial(root: Path, payloads: list, limits: dict, serial: bool, deadline: float) -> dict:
  """Measure complete extraction jobs, excluding adapter preparation and cache hits.

  :param root: Unique trial namespace.
  :param payloads: Repeated complete-sequence workloads.
  :param limits: Scheduling ceilings being tested.
  :param serial: Await every workload before submitting another.
  :param deadline: Absolute shared calibration limit.
  :return: Measured total throughput, owned task timings and resource peaks.
  """
  config = bind(payloads[0]['config'], root, deadline)
  config['paths']['cache'] = str(root / 'temporary')
  config['_study'] = ['calibration']
  started = time.perf_counter()
  with Scheduler(root, limits, config) as scheduler:
    futures = []
    for payload in payloads:
      local = payload | {'config': config}
      future = scheduler.submit('feature', local, {
        'data': payload['data'].fingerprint,
        'spec': payload['spec'],
        'indices': payload['indices']
      })
      futures.append(future)
      if serial:
        future.result()
    for future in futures:
      future.result()
  seconds = time.perf_counter() - started
  records = [read(future.result()['metadata']) for future in futures]
  return {
    'status': 'completed',
    'limits': limits,
    'serial': serial,
    'seconds': seconds,
    'samples': sum(len(payload['indices']) for payload in payloads),
    'samples_per_second': sum(len(payload['indices']) for payload in payloads) / seconds,
    'tasks': scheduler.records,
    'extractions': [{key: row[key] for key in ('completed_samples', 'seconds', 'checksums', 'timings') if key in row} for row in records]
  }

def calibrate(root: Path, minutes: float = 30) -> dict:
  """Compare conservative scheduling policies without selecting scientific parameters.

  :param root: Independent evidence destination.
  :param minutes: Wall-clock calibration allowance.
  :return: Completed repetitions and a conservative measured recommendation.
  """
  deadline = time.time() + minutes * 60
  config = bind(load(), root, deadline)
  root.mkdir(parents=True, exist_ok=True)
  config['_identity'] = digest(sources())
  payloads = workloads(root, config)
  designs = [
    ('serial', policy(1, 8, 1), True),
    ('cpu1-gpu1', policy(1, 8, 1), False),
    ('cpu2-gpu1', policy(2, 8, 1), False),
    ('cpu4-gpu1', policy(4, 8, 1), False),
    ('cpu2-gpu2', policy(2, 8, 2), False)
  ]
  result = {
    'source': sources(),
    'workloads': [{
      'dataset': payload['data'].name,
      'spec': payload['spec'],
      'membership': payload['data'].membership(payload['indices']),
      'steps': [len(payload['data'].counts(index)) for index in payload['indices']]
    } for payload in payloads],
    'boundaries': 'Complete development sequences, one upload, unchanged native thread counts. Includes spawned-worker setup and result persistence; no fitting or test evaluation. Native GPU measurements cover the whole device.',
    'trials': []
  }
  import platform
  import subprocess
  result['hardware'] = {
    'platform': platform.platform(),
    'cpu': next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
    'gpu': subprocess.run([
      'nvidia-smi',
      '--query-gpu=name,driver_version,memory.total,temperature.gpu,pstate',
      '--format=csv,noheader'
    ], check=True, capture_output=True, text=True).stdout.strip()
  }
  warm = [payload | {'indices': payload['indices'][:1]} for payload in payloads]
  trial(root / 'warmup', warm, policy(1, 8, 1), True, deadline)
  for repeat in range(3):
    for name, limits, serial in designs:
      if time.time() >= deadline:
        break
      try:
        measured = trial(root / f'{name}-{repeat}', payloads, limits, serial, deadline)
      except Exception as error:
        measured = {'status': 'incomplete', 'error': repr(error)}
      result['trials'].append({'name': name, 'repetition': repeat} | measured)
      write(root / 'calibration.json', result)
      if measured['status'] != 'completed':
        break
  complete = {}
  for name, limits, serial in designs:
    rows = [row for row in result['trials'] if row['name'] == name and row['status'] == 'completed']
    if len(rows) == 3:
      complete[name] = {
        'limits': limits,
        'median_seconds': statistics.median(row['seconds'] for row in rows),
        'serial': serial
      }
  if complete:
    best = min(row['median_seconds'] for row in complete.values())
    eligible = {name: row for name, row in complete.items() if row['median_seconds'] <= best / 0.95}
    choice = min(eligible, key=lambda name: (eligible[name]['limits']['cpu_workers'] + eligible[name]['limits']['gpu_workers'], eligible[name]['limits']['gpu_workers'], name))
    result['recommendation'] = {
      'name': choice,
      **eligible[choice],
      'rule': 'Completed three-repetition median; fewer workers within five percent throughput'
    }
  result['completed_policies'] = complete
  result['status'] = 'completed' if len(complete) == len(designs) else 'partial'
  write(root / 'calibration.json', result)
  return result

def main() -> None:
  """Run an explicitly requested local calibration without changing the study."""
  import argparse
  parser = argparse.ArgumentParser(description='Bounded development-only scheduling calibration')
  parser.add_argument('--output', required=True)
  parser.add_argument('--minutes', type=float, default=30)
  arguments = parser.parse_args()
  calibrate(Path(arguments.output), arguments.minutes)

if __name__ == '__main__':
  main()
