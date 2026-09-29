import shutil
from pathlib import Path

from src.runtime.records import checksum, frozen_record, read, write

def prepare(path: str | Path, run_id: str | None = None) -> Path:
  """Create a frozen replay namespace without configuration selection.

  :param path: Authored workflow frozen manifest.
  :param run_id: Optional independent output name.
  :return: Verified frozen manifest for fixed fitting and evaluation.
  """
  path = Path(path)
  frozen = frozen_record(path)
  if run_id is None:
    return path
  if run_id in (".", "..") or Path(run_id).name != run_id:
    raise ValueError("Run identifier must be a single directory name")
  body = frozen["manifest"]
  root = Path(body["config"]["paths"]["runs"]) / run_id
  target = root / "frozen.json"
  if target.exists():
    if read(target) != frozen:
      raise ValueError("Existing replay differs from requested frozen protocol")
  else:
    if root.exists() and any(root.iterdir()):
      raise ValueError("Replay requires a new run directory or its matching frozen manifest")
    producer = path.parent / 'evidence/provenance'
    if (producer / 'implementation.tar.gz').exists():
      inventory = read(producer / 'implementation.json')
      if inventory['source'] != body['source'] or checksum(producer / 'implementation.tar.gz') != inventory['sha256']:
        raise ValueError('Frozen producer source snapshot changed')
      destination = root / 'evidence/provenance'
      destination.mkdir(parents=True, exist_ok=True)
      for name in ('implementation.tar.gz', 'implementation.json'):
        shutil.copyfile(producer / name, destination / name)
    write(root / "manifest.json", {
      "evidence_version": 1,
      "config": body["config"],
      "parameter_register": body["parameter_register"],
      "source": body["source"],
      "environment": body["environment"],
      "datasets": list(body["entries"]),
      "studies": [],
      "models": [],
      "resources": False,
      "declarations": {name: [] for name in body["entries"]},
      "reproduction": {"source_manifest": str(path), "sha256": checksum(path), "selection_reopened": False}
    })
    write(target, frozen)
  frozen_record(target)
  return target
