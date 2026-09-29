import functools
import hashlib
import inspect
import itertools
import json
import os
import sys
import threading
import time

from tqdm import tqdm

class Progress:
  """Display task progress without changing scientific state or persistence.

  :ivar label: Human-readable task identity.
  :ivar total: Declared work units, or unknown for an open-ended stage.
  :ivar completed: Successfully finished work, including restored progress.
  :ivar unit: Work-unit name.
  :ivar bar: Terminal renderer, absent in plain or disabled mode.
  :ivar position: Reserved terminal row.
  :ivar started: Monotonic start time for this invocation.
  :ivar last_log: Last plain progress-event timestamp.
  :ivar status: Terminal task state.
  :ivar mode: Effective display mode for this task.
  :ivar stream: Diagnostic output destination.
  :ivar identifier: Unique invocation identifier for matching log events.
  :ivar parent: Enclosing task identifier in the current worker thread.
  """

  lock = threading.RLock()
  """Shared lock protecting rendering and row allocation across worker threads."""
  positions = set()
  """Terminal rows currently owned by active tasks."""
  setting = "auto"
  """Default display policy, overridden by the command-line option."""
  sequence = itertools.count()
  """Monotonic invocation numbers, allocated under the shared rendering lock."""
  context = threading.local()
  """Per-thread task ancestry, independent of worker scientific state."""

  def __init__(self, label: str, total: int | None = None, initial: int = 0, unit: str = "items"):
    """Describe a task; output begins when entering its context.

    :param label: Task identity, including dataset or model context where available.
    :param total: Maximum known units, or unknown for a stage.
    :param initial: Previously completed units restored from authoritative records.
    :param unit: Name of each unit of work.
    """
    if initial < 0 or total is not None and (total < 0 or initial > total):
      raise ValueError("Invalid progress counts")
    self.label, self.total, self.completed, self.unit = label, total, initial, unit
    self.bar, self.position = None, None
    self.started = self.last_log = time.monotonic()
    self.status = "running"
    with self.lock:
      self.identifier = f"{os.getpid()}:{next(self.sequence)}"
    self.parent = None
    self.stream = sys.stderr
    self.mode = os.environ.get("THESIS_PROGRESS", self.setting)
    if self.mode not in ("auto", "plain", "off"):
      raise ValueError("Progress mode must be auto, plain or off")

  def __enter__(self):
    """Allocate a terminal row or emit a plain start event.

    :return: Active progress task.
    """
    with self.lock:
      stack = getattr(self.context, "stack", [])
      self.parent = stack[-1] if stack else None
      self.context.stack = [*stack, self.identifier]
      if self.mode == "auto" and self.stream.isatty():
        self.position = next(value for value in range(len(self.positions) + 1) if value not in self.positions)
        self.positions.add(self.position)
        format_string = None if self.unit in ("samples", "batches", "epochs", "files", "bytes", "predictions") else "{desc}: {n_fmt}/{total_fmt} [{elapsed}]"
        self.bar = tqdm(total=self.total, initial=self.completed, desc=self.label, unit=self.unit, position=self.position, leave=False, dynamic_ncols=True, mininterval=0.5, file=self.stream, bar_format=format_string)
      self.event("started")
    return self

  def event(self, status: str, **details) -> None:
    """Write a structured diagnostic event without interfering with terminal bars.

    :param status: Event or terminal status.
    :param details: Additional diagnostic quantities.
    """
    destination = os.environ.get("THESIS_PROGRESS_EVENTS")
    if self.mode == "off" and not destination:
      return
    record = {
      "task": self.label,
      "id": self.identifier,
      "parent": self.parent,
      "status": status,
      "completed": self.completed,
      "total": self.total,
      "unit": self.unit,
      "elapsed_seconds": round(time.monotonic() - self.started, 3)
    }
    with self.lock:
      if destination:
        with open(destination, "a") as stream:
          stream.write(json.dumps(record | details, sort_keys=True) + "\n")
      elif self.mode != "off":
        tqdm.write(json.dumps(record | details, sort_keys=True), file=self.stream)

  def update(self, amount: int = 1) -> None:
    """Advance only after the corresponding work has succeeded.

    :param amount: Newly completed work units.
    """
    with self.lock:
      if amount < 0 or self.total is not None and self.completed + amount > self.total:
        raise ValueError("Progress exceeds declared work")
      self.completed += amount
      if self.bar is not None:
        self.bar.update(amount)
      elif time.monotonic() - self.last_log >= (1 if os.environ.get("THESIS_PROGRESS_EVENTS") else 30):
        self.event("running")
        self.last_log = time.monotonic()

  def __exit__(self, kind, error, traceback):
    """Release the row and distinguish completed, stopped and failed work.

    :param kind: Raised exception type, if any.
    :param error: Raised exception instance.
    :param traceback: Original exception traceback.
    :return: False so exceptions always propagate.
    """
    if self.status == "running":
      self.status = "completed" if self.total is None or self.completed == self.total else "stopped"
    if kind is not None:
      self.status = "interrupted" if issubclass(kind, (KeyboardInterrupt, TimeoutError, GeneratorExit)) else "failed"
    with self.lock:
      if self.bar is not None:
        self.bar.close()
        self.positions.remove(self.position)
      self.event(self.status)
      self.context.stack = [item for item in getattr(self.context, "stack", []) if item != self.identifier]
    return False

def configure(mode: str) -> None:
  """Choose progress rendering without adding display options to frozen settings.

  :param mode: Auto terminal detection, plain logs or disabled output.
  """
  if mode not in ("auto", "plain", "off"):
    raise ValueError("Unknown progress mode")
  Progress.setting = mode
  os.environ["THESIS_PROGRESS"] = mode

def track(values, label: str, total: int | None = None, initial: int = 0, unit: str = "items"):
  """Iterate work, counting an item only when its caller finishes processing it.

  :param values: Remaining work items.
  :param label: Task identity.
  :param total: Full size, including previously completed work.
  :param initial: Restored completion count.
  :param unit: Name of work items.
  :yield: Next work item, with unchanged order and values.
  """
  if total is None and hasattr(values, "__len__"):
    total = initial + len(values)
  with Progress(label, total, initial, unit) as task:
    for value in values:
      yield value
      task.update()

def stage(label: str):
  """Identify a callable as a stage with contextual start and completion events.

  :param label: Stable operation name.
  :return: Decorator preserving the original callable's interface.
  """
  def decorate(function):
    """Wrap one callable without changing its return or failure semantics.

    :param function: Scientific operation.
    :return: Instrumented callable.
    """
    signature = inspect.signature(function)
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
      """Run the operation with a contextual progress identity.

      :param args: Original positional arguments.
      :param kwargs: Original keyword arguments.
      :return: Unchanged operation result.
      """
      arguments = signature.bind(*args, **kwargs).arguments
      fields = []
      for key in ("data", "name", "split", "model", "recipe", "seed", "phase", "official", "root"):
        if key in arguments:
          value = arguments[key]
          fields.append(f"{key}={getattr(value, 'name', value)}")
      spec = arguments.get("spec", arguments.get("entry", {}).get("spec", {}))
      for key in ("model", "qubits", "inputs", "size", "depth"):
        if key in spec:
          fields.append(f"{key}={spec[key]}")
      if "fold" in arguments:
        fields.append(f"fold={arguments['fold'].get('fold')}")
      for key in ("spec", "fold", "entry"):
        if key in arguments:
          value = json.dumps(arguments[key], sort_keys=True, default=str)
          fields.append(f"{key}={hashlib.sha256(value.encode()).hexdigest()[:10]}")
      with Progress(" / ".join([label, *fields])):
        return function(*args, **kwargs)
    return wrapped
  return decorate
