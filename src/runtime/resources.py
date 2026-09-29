import os
import subprocess
from pathlib import Path

def policy(cpu_workers: int | None = None, cpu_threads: int | None = None, gpu_workers: int | None = None) -> dict:
  """Resolve independent scheduling ceilings without changing numerical settings.

  :param cpu_workers: Maximum CPU computation tasks.
  :param cpu_threads: Total native thread reservations.
  :param gpu_workers: Maximum simultaneous Aer GPU tasks.
  :return: Validated resource ceilings.
  """
  available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
  result = {
    'cpu_workers': cpu_workers if cpu_workers is not None else 2,
    'cpu_threads': cpu_threads if cpu_threads is not None else max(1, min(8, available - 2)),
    'gpu_workers': gpu_workers if gpu_workers is not None else 1
  }
  if any(not isinstance(value, int) or value < 1 for value in result.values()):
    raise ValueError('Worker and thread limits must be positive integers')
  return result

def memory() -> dict:
  """Inspect available host and native GPU memory without importing GPU frameworks.

  :return: Available/total bytes, with unknown GPU memory reported explicitly.
  """
  values = {}
  for line in Path('/proc/meminfo').read_text().splitlines():
    key, value = line.split(':', 1)
    values[key] = int(value.split()[0]) * 1024
  result = {
    'host_available': values['MemAvailable'],
    'host_total': values['MemTotal'],
    'gpu_free': None,
    'gpu_total': None
  }
  try:
    completed = subprocess.run([
      'nvidia-smi',
      '--query-gpu=memory.free,memory.total',
      '--format=csv,noheader,nounits'
    ], capture_output=True, text=True, timeout=3, check=True)
    free, total = map(int, completed.stdout.splitlines()[0].split(','))
    result.update({'gpu_free': free * 1024 ** 2, 'gpu_total': total * 1024 ** 2})
  except (OSError, subprocess.SubprocessError, ValueError, IndexError):
    pass
  return result

def device(operation: str, spec: dict, config: dict) -> str:
  """Resolve the computation device consistently for admission and monitoring.

  :param operation: Worker operation.
  :param spec: Numerical model specification.
  :param config: Resolved execution settings.
  :return: CPU, CUDA device, or explicit Aer device label.
  """
  settings = config['execution']
  if operation in ('adapter', 'readout', 'data'):
    return 'CPU'
  if spec.get('model') == 'qrc':
    return 'Aer ' + settings['qrc_large' if spec.get('qubits') == 8 else 'qrc_small']
  if spec.get('model') in ('lstm', 'transformer') or operation == 'resources':
    return settings['device']
  if spec.get('model') == 'crc' and operation in ('feature', 'export', 'final-fit', 'final-evaluate'):
    return settings['crc_device']
  return 'CPU'

def request(operation: str, spec: dict, config: dict) -> dict:
  """Reserve native threads and conservative memory for a task family.

  :param operation: Worker operation.
  :param spec: Numerical model specification.
  :param config: Immutable per-task numerical settings.
  :return: Admission requirements; measured peaks can increase these floors.
  """
  execution = config['execution']
  neural = spec.get('model') in ('lstm', 'transformer')
  quantum = spec.get('model') == 'qrc'
  selected = device(operation, spec, config)
  gpu = selected.startswith('cuda') or selected == 'Aer GPU'
  gpu |= operation == 'resources' and execution['device'].startswith('cuda')
  if operation in ('adapter', 'readout', 'data'):
    gpu = False
  exclusive = operation in ('benchmark', 'resources')
  threads = config['training']['cpu_threads'] if neural else execution['cpu_threads']
  if operation in ('feature', 'export', 'final-fit', 'final-evaluate') and quantum:
    threads = max(threads, execution['gpu_threads'] + 1)
  host = 2 * 1024 ** 3 if operation in ('readout', 'data') or neural else 1024 ** 3
  gpu_bytes = (5 * 1024 ** 3 if neural or operation == 'resources' else 768 * 1024 ** 2) if gpu else 0
  return {
    'cpu_threads': threads,
    'cpu_workers': int(not gpu),
    'gpu_workers': int(gpu),
    'exclusive': exclusive,
    'gpu_exclusive': bool(neural and gpu),
    'host_bytes': host,
    'gpu_bytes': gpu_bytes
  }

def admission(wanted: dict, active: list, limits: dict, available: dict) -> str | None:
  """Explain why a task cannot start under the current reservations.

  :param wanted: New task requirements.
  :param active: Running task reservations.
  :param limits: Worker/thread ceilings.
  :param available: Latest free-memory sample; active reservations are additionally withheld to cover allocations after sampling.
  :return: Waiting reason, or None when admission is safe.
  """
  if active and (wanted['exclusive'] or any(row['exclusive'] for row in active)):
    return 'exclusive task'
  if wanted['gpu_workers'] and any(row['gpu_workers'] and (wanted.get('gpu_exclusive') or row.get('gpu_exclusive')) for row in active):
    return 'exclusive GPU'
  for key in ('cpu_workers', 'cpu_threads', 'gpu_workers'):
    if sum(row[key] for row in active) + wanted[key] > limits[key]:
      return key.replace('_', ' ') + ' budget'
  host_reserved = sum(row['host_bytes'] for row in active)
  if wanted['host_bytes'] > min(available['host_available'], available['host_total']) * 0.75 - host_reserved:
    return 'host memory'
  if wanted['gpu_workers']:
    if available['gpu_free'] is None:
      return 'GPU memory unavailable'
    gpu_reserved = sum(row['gpu_bytes'] for row in active)
    if wanted['gpu_bytes'] > min(available['gpu_free'], available['gpu_total'] * 0.75) - gpu_reserved:
      return 'GPU memory'
  return None
