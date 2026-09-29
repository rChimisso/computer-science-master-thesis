import copy
import os
import shutil
from pathlib import Path

import numpy as np

from src.runtime.records import checksum, digest, read, write

def evidence_root(run: Path) -> Path:
  """Resolve the durable evidence directory for an authored workflow run.

  :param run: Execution root.
  :return: Directory containing metric and fitting records.
  """
  return run / "evidence"

def work_root(config: dict) -> Path:
  """Resolve disposable execution storage.

  :param config: Bound settings.
  :return: Run-local working directory.
  """
  return Path(config["paths"]["temporary"])

def pack(value: object, path: Path, root: Path, key: str = "") -> object:
  """Store shared memberships and compact predictions using relative verified references.

  :param value: Serializable job content.
  :param path: Owning record path.
  :param root: Durable evidence root.
  :param key: Current field name.
  :return: Encoded record content.
  """
  if isinstance(value, dict) and set(value) == {"identities", "labels", "groups"}:
    target = root / "memberships" / f"{digest(value)}.json"
    if not target.exists():
      write(target, value)
    return {"$evidence": "membership", "path": os.path.relpath(target, path.parent), "sha256": checksum(target)}
  if isinstance(value, list) and key in ("predictions", "train_predictions"):
    array = np.asarray(value, dtype=np.uint16)
    target = root / "predictions" / f"{digest(value)}.npy"
    if not target.exists():
      target.parent.mkdir(parents=True, exist_ok=True)
      temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
      with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
      temporary.replace(target)
    return {"$evidence": "predictions", "path": os.path.relpath(target, path.parent), "sha256": checksum(target)}
  if isinstance(value, dict):
    return {name: pack(item, path, root, name) for name, item in value.items()}
  if isinstance(value, list):
    return [pack(item, path, root) for item in value]
  return value

def unpack(value: object, path: Path) -> object:
  """Read compact references without accessing models, caches or datasets.

  :param value: Encoded object.
  :param path: Owning record path.
  :return: Hydrated job content.
  """
  if isinstance(value, dict) and "$evidence" in value:
    target = path.parent / value["path"]
    boundary = next((parent for parent in path.parents if parent.name == "evidence"), None)
    if boundary is None or not target.resolve().is_relative_to(boundary.resolve()):
      raise ValueError("Compact evidence reference escaped its run")
    if checksum(target) != value["sha256"]:
      raise ValueError("Durable evidence checksum mismatch")
    return np.load(target, allow_pickle=False).tolist() if value["$evidence"] == "predictions" else read(target)
  if isinstance(value, dict):
    return {name: unpack(item, path) for name, item in value.items()}
  if isinstance(value, list):
    return [unpack(item, path) for item in value]
  return value

def save_job(path: Path, record: dict, config: dict) -> None:
  """Persist a job in the authored evidence format.

  :param path: Destination record.
  :param record: Current execution result.
  :param config: Bound storage settings.
  """
  write(path, {"evidence_format": 1, "record": pack(record, path, Path(config["paths"]["results"]))})

def snapshot_dataset(data, config: dict, population: str = "train") -> dict:
  """Retain small dataset statistics independently of disposable count arrays.

  :param data: Loaded indexed source.
  :param config: Bound settings.
  :param population: Official source split label.
  :return: Persisted dataset description.
  """
  rows = []
  confirmation = set(data.confirmation)
  for index, identity in enumerate(data.identities):
    values = data.counts(index)
    rows.append({
      "identity": identity,
      "label": int(data.labels[index]),
      "group": int(data.groups[index]),
      "lighting": data.lighting[index],
      "events": int(values.sum(dtype=np.uint64)),
      "steps": len(values),
      "partition": "official" if population == "test" else "confirmation" if index in confirmation else "development"
    })
  record = {
    "dataset": data.name,
    "population": population,
    "classes": data.classes,
    "channels": data.counts(0).shape[1],
    "bin_us": data.bin_us,
    "data_fingerprint": data.fingerprint,
    "representation": config["data"][data.name],
    "duration_definition": "Number of valid bins multiplied by bin width, not exact event span",
    "rows": rows
  }
  write(Path(config["paths"]["results"]) / "datasets" / f"{data.name}-{population}.json", record)
  return record

def scalar_history(history: list) -> list:
  """Remove sample predictions from epoch histories without dropping scalar metrics.

  :param history: Executed epochs.
  :return: Compact epoch records.
  """
  result = copy.deepcopy(history)
  for epoch in result:
    for population in ("train", "validation"):
      if epoch.get(population):
        epoch[population] = {key: value for key, value in epoch[population].items() if key not in ("predictions", "per_class", "confusion_matrix")}
  return result

def cleanup(run: Path, report_record: dict) -> dict:
  """Prune only verified, completed run-local work, keeping shared artifacts intact.

  :param run: Run root.
  :param report_record: Successfully published report status.
  :return: Cleanup ledger or explicit deferral.
  """
  root = evidence_root(run)
  if report_record.get("development_status") != "completed":
    return {"status": "deferred", "reason": "Run evidence is incomplete"}
  progress = read(run / "progress.json") if (run / "progress.json").exists() else {}
  if progress.get("status") not in ("completed", "diagnostic_completed"):
    return {"status": "deferred", "reason": "Run is not successfully complete"}
  from src.runtime.pipelines import verify
  packages = read(root / "pipelines.json") if (root / "pipelines.json").exists() else {}
  for package in packages.values():
    verify(package)
    if not package.get("verified_predictions"):
      raise ValueError("Cannot prune before pipeline prediction verification")
  for phase in ("confirmation", "official"):
    path = root / "metrics" / f"{phase}.json"
    if path.exists():
      phase_record = read(path)
      if phase_record["status"] != "completed":
        return {"status": "deferred", "reason": f"{phase} is incomplete"}
      for entry in phase_record["jobs"]:
        key = f"{phase}:{entry['dataset']}:{entry['role']}:{entry['seed']}"
        if key not in packages:
          raise ValueError("Final model package missing before cleanup")
  temporary = run / "temporary"
  files = {str(path.relative_to(run)): checksum(path) for path in temporary.rglob("*") if path.is_file()}
  previous_path = root / "cleanup.json"
  previous = read(previous_path) if previous_path.exists() else {"removed": {}}
  ledger = {"status": "prepared", "removed": previous["removed"] | files}
  write(previous_path, ledger)
  if temporary.exists():
    shutil.rmtree(temporary)
  ledger["status"] = "completed"
  write(previous_path, ledger)
  return ledger

def intentionally_pruned(path: str, config: dict) -> bool:
  """Distinguish ledgered removal from corrupt or missing fitted artifacts.

  :param path: Previously recorded artifact.
  :param config: Bound settings.
  :return: Whether successful cleanup explicitly removed this artifact.
  """
  root = Path(config["paths"]["results"])
  ledger = root / "cleanup.json"
  run = Path(config["paths"].get("run", root))
  try:
    relative = str(Path(path).relative_to(run))
  except ValueError:
    return False
  return not Path(path).exists() and ledger.exists() and relative in read(ledger).get("removed", {})

def seal(run: Path) -> dict:
  """Record durable source hashes after successful execution, excluding maintenance ledgers.

  :param run: Execution namespace.
  :return: Persisted evidence integrity inventory.
  """
  root = evidence_root(run)
  artifacts = {str(path.relative_to(root)): checksum(path) for path in root.rglob("*") if path.is_file() and path.name not in ("integrity.json", "cleanup.json") and not path.name.startswith(".")}
  record = {"version": 1, "artifacts": artifacts}
  write(root / "integrity.json", record)
  return record

def compact(run: Path, config: dict) -> dict:
  """Retain sample predictions only where declared reporting consumes them.

  :param run: Successfully executed run namespace.
  :param config: Bound evidence storage.
  :return: Compaction counts, preserving every candidate's numerical scores.
  """
  import json
  root = evidence_root(run)
  keep = {}

  def selections(value: object) -> None:
    """Find selected fitting references, including supporting selections.

    :param value: Nested selection evidence.
    """
    if isinstance(value, dict):
      if "selected" in value and isinstance(value["selected"], dict):
        selected = value["selected"]
        for path in selected.get("jobs", []):
          keep.setdefault(str(Path(path)), set()).add(selected["selection"]["lambda"])
      for item in value.values():
        selections(item)
    elif isinstance(value, list):
      for item in value:
        selections(item)

  path = root / "metrics/selections.json"
  if path.exists():
    selections(read(path))
  for name in ("confirmation", "official", "preprocessing", "learning-curves"):
    path = root / "metrics" / f"{name}.json"
    if path.exists():
      record = read(path)
      for entry in record.get("jobs", record.get("rows", [])):
        path = Path(entry.get("path", entry.get("job", "")))
        if path.is_file():
          keep.setdefault(str(path), set()).add(read(path)["rows"][0]["lambda"])

  def stripped(value: object) -> object:
    """Remove duplicated predictions while preserving scalar and confusion evidence.

    :param value: Job or nested metric object.
    :return: Prediction-free copy.
    """
    if isinstance(value, dict):
      return {key: stripped(item) for key, item in value.items() if key not in ("predictions", "train_predictions")}
    if isinstance(value, list):
      return [stripped(item) for item in value]
    return value

  for path in sorted((root / "metrics/jobs").glob("*/result.json")):
    job = read(path)
    if job["status"] != "completed":
      continue
    saved = stripped(job)
    for index, row in enumerate(job["rows"]):
      if row["lambda"] in keep.get(str(path), set()):
        for key in ("predictions", "train_predictions"):
          if key not in row:
            raise ValueError("A reported fit lost required predictions before compaction")
          saved["rows"][index][key] = row[key]
    save_job(path, saved, config)
  referenced = set()

  def references(value: object, path: Path) -> None:
    """Collect live compact array references without loading their data.

    :param value: Raw JSON structure.
    :param path: Owning JSON record.
    """
    if isinstance(value, dict):
      if value.get("$evidence") == "predictions":
        referenced.add((path.parent / value["path"]).resolve())
      for item in value.values():
        references(item, path)
    elif isinstance(value, list):
      for item in value:
        references(item, path)

  for path in root.rglob("*.json"):
    references(json.loads(path.read_text()), path)
  removed = 0
  for path in (root / "predictions").glob("*.npy"):
    if path.resolve() not in referenced:
      path.unlink()
      removed += 1
  record = {"reported_fits": len(keep), "unreferenced_prediction_arrays_removed": removed}
  write(root / "compaction.json", record)
  return record
