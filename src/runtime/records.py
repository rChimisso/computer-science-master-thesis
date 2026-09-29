import hashlib
import json
import os
import time
from pathlib import Path

def digest(value: object) -> str:
  """Hash a canonical JSON value.

  :param value: Serializable identity.
  :return: SHA-256 identity.
  """
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def checksum(path: str | Path) -> str:
  """Hash a file with bounded memory.

  :param path: Existing file.
  :return: SHA-256 checksum.
  """
  result = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      result.update(block)
  return result.hexdigest()

def read(path: str | Path) -> dict:
  """Read a JSON record.

  :param path: Record location.
  :return: Decoded record.
  """
  value = json.loads(Path(path).read_text())
  if isinstance(value, dict) and value.get("evidence_format") == 1:
    from src.runtime.storage import unpack
    return unpack(value["record"], Path(path))
  return value

def write(path: str | Path, value: object, sort_keys: bool = True) -> None:
  """Atomically replace a record after flushing its contents.

  :param path: Destination.
  :param value: Serializable contents.
  :param sort_keys: Canonicalize mappings unless configuration transport requires their declared iteration order.
  """
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  with temporary.open("w") as stream:
    json.dump(value, stream, indent=2, sort_keys=sort_keys, allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
  temporary.replace(path)

def sources(numerical: bool = False) -> dict:
  """Fingerprint active implementations using portable names.

  :param numerical: Exclude orchestration and display from model checkpoint identities.
  :return: Relative source checksums.
  """
  return {str(path): checksum(path) for path in sorted(Path("src").rglob("*.py")) if not numerical or path.parts[1] in ('data', 'models', 'features', 'evaluation')}

def stop(config: dict) -> None:
  """Check a deadline or cooperative cancellation file.

  :param config: Resolved runtime settings.
  :raises TimeoutError: When execution must checkpoint and return.
  """
  if time.time() >= config.get("_deadline", float("inf")) or (Path(config["paths"].get("run", config["paths"]["results"])) / "STOP").exists():
    raise TimeoutError("Execution stopped; saved work can be resumed")

def frozen_record(path: str | Path) -> dict:
  """Validate a frozen configuration before any evaluation data access.

  :param path: Explicit frozen-manifest location.
  :return: Checked record.
  """
  record = read(path)
  if record.get("status") != "frozen" or record.get("fingerprint") != digest(record.get("manifest")):
    raise ValueError("An intact frozen manifest is required")
  from src.runtime.provenance import validate_source
  validate_source(record["manifest"]["source"], Path(path).parent)
  from src.methodology.parameters import validate_register
  body = record["manifest"]
  validate_register(body["config"], body["parameter_register"])
  for dataset, entries in body["entries"].items():
    decision = body["decisions"][dataset]
    if decision["status"] != "completed" or decision["seeds"] != body["config"]["seeds"]:
      raise ValueError("Frozen selection coverage is incomplete")
    for role, entry in entries.items():
      if entry != decision["groups"][role]["selected"]:
        raise ValueError("Frozen settings differ from the development decision")
  return record
