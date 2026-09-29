import datetime
import sys
import time
from pathlib import Path

from src.runtime.records import read

def process_identity(pid: int) -> str | None:
  """Read process start ticks, avoiding accidental reuse of an unrelated PID.

  :param pid: Operating-system process identifier.
  :return: Linux start ticks, or None if the process has exited.
  """
  try:
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return None if fields[0] == 'Z' else fields[19]
  except (OSError, IndexError):
    return None

def status(root: Path) -> dict:
  """Read saved status without datasets or numerical framework imports.

  :param root: Run directory.
  :return: Snapshot enriched with verified coordinator liveness.
  """
  launch = read(root / 'execution/coordinator.json') if (root / 'execution/coordinator.json').exists() else {}
  snapshot = read(root / 'execution/status.json') if (root / 'execution/status.json').exists() else {
    'counts': {},
    'active': [],
    'eta': {},
    'datasets': {}
  }
  alive = bool(launch.get('pid') and process_identity(launch['pid']) == launch.get('start_ticks'))
  if alive and snapshot.get('pid') != launch['pid']:
    snapshot = {
      'counts': {},
      'active': [],
      'eta': {},
      'datasets': {},
      'limits': launch.get('limits', {}),
      'updated_utc': launch['updated_utc'],
      'active_seconds': snapshot.get('active_seconds', 0),
      'state': 'starting; restoring saved task state'
    }
  state = launch.get('state', 'not started')
  if state == 'running':
    state = snapshot.get('state', state)
  snapshot.update({
    'run': root.name,
    'alive': alive,
    'state': state
  })
  if not alive and snapshot['state'] in ('starting', 'running', 'pausing'):
    snapshot['state'] = 'stopped unexpectedly; resume required'
  snapshot['stale'] = alive and time.time() - snapshot.get('updated_utc', launch.get('updated_utc', 0)) > 15
  return snapshot

def duration(seconds: float) -> str:
  """Format a duration without suggesting subsecond forecasting precision.

  :param seconds: Nonnegative duration.
  :return: Compact human-readable duration.
  """
  return f'{seconds / 3600:.1f} h' if seconds >= 3600 else f'{seconds / 60:.0f} min'

def render(record: dict) -> str:
  """Render one concise summary and active-task rows.

  :param record: Read-only coordinator snapshot.
  :return: Plain terminal text.
  """
  lines = [f"{record['run']} | {record['state']}" + (' | STALE HEARTBEAT' if record.get('stale') else '')]
  reservations = {key: sum(row['resources'][key] for row in record['active']) for key in ('cpu_workers', 'cpu_threads', 'gpu_workers')}
  lines.append(f"Active coordinator time: {duration(record.get('active_seconds', 0))} | reserved: {reservations} | limits: {record.get('limits', {})}")
  if record.get('updated_utc'):
    lines.append('Updated: ' + datetime.datetime.fromtimestamp(record['updated_utc']).astimezone().strftime('%b %d %H:%M:%S'))
  lines.append('Tasks: ' + ', '.join(f'{key}={value}' for key, value in record['counts'].items() if key != 'pending'))
  lines.append(f"Verified reused tasks: {record.get('reused_tasks', 0)}")
  for name, counts in record['datasets'].items():
    label = 'Run-wide' if name == 'run-wide' else name.upper()
    lines.append(f"{label}: completed={counts.get('completed', 0)}, pending={counts.get('pending', 0)}")
  for name, datasets in record.get('studies', {}).items():
    parts = [f"{dataset.upper()}: completed={counts.get('completed', 0)}, pending={counts.get('pending', 0)}" for dataset, counts in sorted(datasets.items())]
    lines.append(f"{name} | " + ' | '.join(parts))
  eta = record.get('eta', {})
  interval = eta.get('total_seconds')
  if interval and record['state'] != 'completed':
    reference = record.get('updated_utc', time.time()) if record['state'] in ('running', 'starting') and not record.get('stale') else time.time()
    updated = datetime.datetime.fromtimestamp(reference).astimezone()
    dates = [(updated + datetime.timedelta(seconds=value)).strftime('%b %d %H:%M') for value in interval]
    lines.append(f'Total remaining: {duration(interval[0])} - {duration(interval[1])} ({dates[0]} - {dates[1]})')
    lines.append(f"Conservative planning range | evidence: {eta.get('evidence', {})} | projected tasks: {eta.get('projected_tasks', 0)}")
    if record['state'] not in ('running', 'starting') or record.get('stale'):
      lines.append('Completion dates assume immediate resumption; processing duration excludes paused time.')
  for row in record['active']:
    spec = row.get('spec') or {}
    labels = ' '.join(f'{key}={spec[key]}' for key in ('model', 'qubits', 'inputs', 'depth', 'size') if key in spec)
    units = row.get('progress', {})
    detail = ' | '.join(f"{key} {value['completed']}/{value.get('total', '?')}" for key, value in units.items())
    remaining = row.get('eta_seconds')
    estimate = f" | ETA {duration(remaining[0])}-{duration(remaining[1])}" if remaining else ' | estimating startup work'
    if row.get('rate_per_second'):
      estimate = f" | {row['rate_per_second']:.2g} {row['rate_unit']}/s" + estimate
    resources = row['resources']
    reserved = ', '.join(f'{key}={resources.get(key, 0)}' for key in ('cpu_workers', 'cpu_threads', 'gpu_workers'))
    memory = f"RAM={resources.get('host_bytes', 0) / 1024 ** 3:.2f} GiB, VRAM={resources.get('gpu_bytes', 0) / 1024 ** 3:.2f} GiB"
    lines.append(f"{row['dataset'] or 'run-wide'} {row['operation']} {labels} seed={row.get('seed')} fold={row.get('fold')} | {detail}{estimate}")
    lines.append(f"  Device: {row.get('device', 'coordinator')} | reserved: {reserved}, {memory}")
  if record.get('error'):
    lines.append('Error: ' + record['error'])
  if record.get('alive', record['state'] == 'running'):
    lines.append('Ctrl+C disconnects this view; computation continues. Use pause --run to stop safely.')
  return '\n'.join(lines)

def watch(root: Path) -> dict:
  """Attach a read-only terminal view; disconnect never signals computation.

  :param root: Run namespace.
  :return: Last observed status.
  """
  previous, last = '', {}
  try:
    while True:
      last = status(root)
      text = render(last)
      if sys.stderr.isatty():
        sys.stderr.write('\x1b[2J\x1b[H' + text + '\n')
        sys.stderr.flush()
      elif text != previous:
        print(text, file=sys.stderr, flush=True)
      previous = text
      if not last['alive']:
        return last
      time.sleep(1 if sys.stderr.isatty() else 30)
  except (KeyboardInterrupt, BrokenPipeError):
    print('Disconnected; the workflow continues in the background.', file=sys.stderr)
    return last

def pause(root: Path) -> dict:
  """Request cooperative cancellation and confirm the coordinator has exited.

  :param root: Run namespace.
  :return: Verified stopped status.
  """
  record = status(root)
  if not record['alive']:
    return record
  (root / 'STOP').touch()
  started = time.monotonic()
  while status(root)['alive']:
    if time.monotonic() - started > 90:
      raise RuntimeError('Pause has not completed; inspect coordinator status before any forced termination')
    time.sleep(0.5)
  return status(root)
