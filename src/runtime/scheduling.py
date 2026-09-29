import fcntl
import time
from contextlib import contextmanager
from pathlib import Path

from src.runtime.records import stop

@contextmanager
def lock(path: Path, config: dict):
  """Serialize cache writers across threads and processes with cancellation.

  :param path: Persistent lock-file identity.
  :param config: Deadline and cancellation policy.
  :yield: Exclusive access until the context exits.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a") as stream:
    while True:
      stop(config)
      try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
      except BlockingIOError:
        time.sleep(0.05)
    try:
      yield
    finally:
      fcntl.flock(stream, fcntl.LOCK_UN)
