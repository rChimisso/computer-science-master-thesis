import json
import sys
from pathlib import Path
from unittest.mock import patch

from src.runtime.configuration import bind
from src.runtime.execution import coordinate, launch
from src.runtime.records import read
from src.runtime.tasks import current

def synthetic_run(request, run_id, resume, deadline):
  """Supply a long, cancellable synthetic task to the real coordinator.

  :param request: Fixture settings.
  :param run_id: Stable run name.
  :param resume: Requested resumption flag.
  :param deadline: Stop timestamp.
  :return: Synthetic completion status.
  """
  root = Path(request['config']['paths']['runs']) / run_id
  config = bind(request['config'], root, deadline)
  scheduler = current(config)
  scheduler.submit('fixture', {
    'config': config,
    'steps': 1000,
    'delay': 0.1
  }, {'fixture': True}).result()
  return {'status': 'completed'}

def main() -> None:
  """Exercise production detachment with a substituted scientific fixture only."""
  if sys.argv[1] == 'launch':
    import subprocess
    original = subprocess.Popen

    def child(command, **kwargs):
      """Select this fixture module while preserving production process options.

      :param command: Original coordinator command.
      :param kwargs: Production process isolation arguments.
      :return: Detached coordinator process.
      """
      command = list(command)
      command[command.index('src.runtime.execution')] = 'tests.extended.lifecycle_fixture'
      return original(command, **kwargs)

    payload = read(sys.argv[2])
    with patch('src.runtime.execution.subprocess.Popen', side_effect=child):
      print(json.dumps(launch('run', payload['request'], Path(payload['root']), payload['limits'], float('inf'), True, True)))
  else:
    with patch('src.workflow.run', side_effect=synthetic_run):
      coordinate(Path(sys.argv[1]))

if __name__ == '__main__':
  main()
