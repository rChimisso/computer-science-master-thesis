import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.runtime.monitor import process_identity, status, watch
from src.runtime.records import read, write

def launch(action: str, request: dict, root: Path, limits: dict, deadline: float, resume: bool = False, detach: bool = False) -> dict:
  """Start a session-independent coordinator and optionally attach its monitor.

  :param action: Run, frozen reproduction, or explicit evaluation.
  :param request: Resolved scientific request or frozen manifest path.
  :param root: Output namespace.
  :param limits: Independent scheduling limits.
  :param deadline: Absolute stop timestamp.
  :param resume: Permit compatible existing work.
  :param detach: Return without opening a live monitor.
  :return: Initial or last observed execution status.
  """
  lock_path = Path('output/locks/workflow.lock')
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  lease = lock_path.open('a')
  try:
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError:
    lease.close()
    raise RuntimeError('A managed workflow is already running in this workspace')
  if (root / 'manifest.json').exists() and action == 'run' and not resume:
    lease.close()
    raise ValueError('Existing run requires --resume')
  directory = root / 'execution'
  directory.mkdir(parents=True, exist_ok=True)
  payload = {
    'action': action,
    'request': request,
    'root': str(root),
    'limits': limits,
    'deadline': None if deadline == float('inf') else deadline,
    'resume': resume
  }
  path = directory / 'request.json'
  write(path, payload, sort_keys=False)
  (root / 'STOP').unlink(missing_ok=True)
  with (directory / 'console.log').open('ab', buffering=0) as stream:
    process = subprocess.Popen([
      sys.executable,
      '-u',
      '-m',
      'src.runtime.execution',
      str(path)
    ], stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True, pass_fds=(lease.fileno(),))
  lease.close()
  for _ in range(100):
    record = read(directory / 'coordinator.json') if (directory / 'coordinator.json').exists() else {}
    if record.get('pid') == process.pid:
      return status(root) if detach else watch(root)
    if process.poll() is not None:
      raise RuntimeError(f"Coordinator startup failed; see {directory / 'console.log'}")
    time.sleep(0.1)
  raise RuntimeError('Coordinator did not publish its startup record')

def publish_terminal(root: Path, record: dict) -> None:
  """Persist the coordinator outcome after workers and report publication finish.

  :param root: Run namespace.
  :param record: Terminal coordinator state, including its completion timestamp.
  """
  if record['state'] not in ('completed', 'failed', 'paused'):
    raise ValueError('Terminal publication requires a stopped coordinator state')
  destination = root / 'execution/status.json'
  snapshot = read(destination) if destination.exists() else {}
  snapshot.update({'state': record['state'], 'pid': record['pid'], 'updated_utc': record['updated_utc'], 'error': record.get('error'), 'active': []})
  if record['state'] == 'completed':
    snapshot['error'] = None
    snapshot['eta'] = {}
    progress_path = root / 'progress.json'
    if progress_path.exists():
      progress = read(progress_path)
      progress.pop('error', None)
      write(progress_path, progress)
  write(destination, snapshot)

def coordinate(path: Path) -> None:
  """Own scheduling, persistence and failure handling independently of any terminal.

  :param path: Explicit local execution request.
  """
  payload = read(path)
  root, limits = Path(payload['root']), payload['limits']
  destination = root / 'execution/coordinator.json'
  record = {
    'pid': os.getpid(),
    'start_ticks': process_identity(os.getpid()),
    'state': 'starting',
    'updated_utc': time.time(),
    'action': payload['action'],
    'limits': limits
  }
  write(destination, record)

  def interrupt(signum, frame) -> None:
    """Translate process signals into cooperative run cancellation.

    :param signum: Received signal.
    :param frame: Interrupted frame.
    """
    (root / 'STOP').touch()

  signal.signal(signal.SIGTERM, interrupt)
  signal.signal(signal.SIGINT, interrupt)
  signal.signal(signal.SIGHUP, signal.SIG_IGN)
  deadline = payload['deadline'] if payload['deadline'] is not None else float('inf')
  os.environ['THESIS_PROGRESS_EVENTS'] = str(root / 'execution/coordinator-events.jsonl')
  os.environ['THESIS_PROGRESS'] = 'off'
  from src.runtime.configuration import bind
  from src.runtime.tasks import Scheduler
  from src.workflow import run, evaluate_run
  try:
    config = bind(payload['request']['config'] if payload['action'] == 'run' else read(payload['request']['manifest'])['manifest']['config'], root, deadline)
    record['state'] = 'running'
    write(destination, record)
    config['_forecast_scope'] = {'action': payload['action'], 'request': payload['request']}
    with Scheduler(root, limits, config) as scheduler:
      if payload['action'] == 'run':
        result = run(payload['request'], root.name, payload['resume'], deadline)
      else:
        result = evaluate_run(payload['request']['manifest'], payload['action'] == 'evaluate', deadline)
      if scheduler.failure:
        raise RuntimeError(scheduler.failure)
    record.update({'state': 'completed', 'result': result})
  except BaseException as error:
    record.update({'state': 'paused' if (root / 'STOP').exists() else 'failed', 'error': repr(error)})
    from src.runtime.records import write as save
    progress = read(root / 'progress.json') if (root / 'progress.json').exists() else {}
    progress.update({'status': 'interrupted', 'error': repr(error)})
    save(root / 'progress.json', progress)
    raise
  finally:
    record['updated_utc'] = time.time()
    write(destination, record)
    publish_terminal(root, record)

if __name__ == '__main__':
  coordinate(Path(sys.argv[1]))
