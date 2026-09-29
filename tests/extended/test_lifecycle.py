import os

from src.runtime.records import read

from tests.support import SyntheticCase

class LifecycleTests(SyntheticCase):
  """Exercise detached execution, cooperative pause and abrupt process death."""

  def test_detached_lifecycle_and_reconnect(self):
    """The launcher and monitor may exit while computation survives, then pauses cleanly."""
    import json
    import signal
    import subprocess
    import sys
    import time
    from src.runtime.monitor import pause, status
    from src.runtime.resources import policy
    config = {key: value for key, value in self.config.items() if not key.startswith('_')} | {'paths': self.config['paths'] | {'runs': str(self.root)}}
    root = self.root / 'detached'
    payload = {'request': {'config': config}, 'root': str(root), 'limits': policy(1, 4, 1)}
    path = self.root / 'launch.json'
    path.write_text(json.dumps(payload))
    subprocess.run([sys.executable, '-m', 'tests.extended.lifecycle_fixture', 'launch', str(path)], check=True, stdout=subprocess.DEVNULL, timeout=20)
    try:
      self.assertTrue(status(root)['alive'])
      from src.runtime.execution import launch
      with self.assertRaisesRegex(RuntimeError, 'already running'):
        launch('run', payload['request'], root, payload['limits'], float('inf'), True, True)
      monitor = subprocess.Popen([sys.executable, '-m', 'src', 'status', '--run', str(root), '--watch'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      time.sleep(1)
      monitor.send_signal(signal.SIGINT)
      monitor.wait(timeout=5)
      self.assertTrue(status(root)['alive'])
    finally:
      stopped = pause(root)
    self.assertFalse(stopped['alive'])
    self.assertEqual(stopped['state'], 'paused')

  def test_coordinator_crash_stops_owned_workers(self):
    """Abrupt coordinator death kills owned process groups and leaves resumable records."""
    import json
    import signal
    import subprocess
    import sys
    import time
    from src.runtime.monitor import process_identity, status
    from src.runtime.resources import policy
    config = {key: value for key, value in self.config.items() if not key.startswith('_')} | {'paths': self.config['paths'] | {'runs': str(self.root)}}
    root = self.root / 'crashed'
    path = self.root / 'launch.json'
    path.write_text(json.dumps({'request': {'config': config}, 'root': str(root), 'limits': policy(1, 4, 1)}))
    subprocess.run([sys.executable, '-m', 'tests.extended.lifecycle_fixture', 'launch', str(path)], check=True, stdout=subprocess.DEVNULL, timeout=20)
    active = []
    for _ in range(100):
      active = status(root)['active']
      if active:
        break
      time.sleep(0.1)
    self.assertTrue(active)
    coordinator = read(root / 'execution/coordinator.json')
    os.kill(coordinator['pid'], signal.SIGKILL)
    for _ in range(100):
      if all(process_identity(row['pid']) is None for row in active):
        break
      time.sleep(0.1)
    self.assertTrue(all(process_identity(row['pid']) is None for row in active))
    self.assertFalse(status(root)['alive'])
    self.assertIn('stopped unexpectedly', status(root)['state'])
