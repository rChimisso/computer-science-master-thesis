from pathlib import Path

import h5py
import numpy as np

from src.runtime.progress import stage, track
from src.data.acquisition import acquire
from src.data.gesture import bin_dvs_clip
from src.data.indexed import StudyData
from src.data.spikes import PreprocessingSpec, bin_shd_events
from src.runtime.records import checksum, digest, read, stop, write

@stage("data.datasets.load")
def load(name: str, config: dict, split: str = "train", official: bool = False) -> StudyData:
  """Prepare independent ragged counts from raw sources with explicit access gating.

  :param name: Dataset identifier.
  :param config: Preprocessing and storage settings.
  :param split: Official train or test split.
  :param official: Authorization from the frozen evaluation action.
  :return: Dataset with stable original identities and no padded bins.
  """
  if split == "test" and not official:
    raise ValueError("Test data cannot be opened by development actions")
  source = acquire(name, split, config, official)
  settings = config["data"][name]
  identity = {
    "dataset": name,
    "split": split,
    "settings": settings,
    "loader": checksum(__file__),
    "binning": checksum("src/data/spikes.py" if name == "shd" else "src/data/gesture.py")
  }
  if name == "shd":
    identity["raw"] = checksum(source)
  else:
    paths = sorted(source.glob("user*_*/*.npy"))
    if not paths:
      raise ValueError("No DVS clips in the expected schema")
    identity["raw"] = {str(path.relative_to(source)): checksum(path) for path in track(paths, f"{name}/{split}/hash-sources", unit="files")}
  root = Path(config["paths"]["cache"]) / "datasets" / digest(identity)
  destination = root / "manifest.json"
  record = read(destination) if destination.exists() else {
    "identity": identity,
    "status": "running",
    "rows": []
  }
  if record["identity"] != identity:
    raise ValueError("Dataset identity changed")
  root.mkdir(parents=True, exist_ok=True)
  for row in track(record["rows"], f"{name}/{split}/verify-cache", unit="samples"):
    if checksum(root / row["file"]) != row["sha256"]:
      raise ValueError("Count cache checksum mismatch")
  if record["status"] != "completed":
    if name == "shd":
      with h5py.File(source, "r") as incoming:
        labels = np.asarray(incoming["labels"], dtype=np.int64)
        groups = np.asarray(incoming["extra/speaker"], dtype=np.int64)
        spec = PreprocessingSpec(settings["bin_us"], settings["channels"], "count", settings["duration_us"])
        for index in track(range(len(record["rows"]), len(labels)), f"{name}/{split}/bin", total=len(labels), initial=len(record["rows"]), unit="samples"):
          stop(config)
          times = (np.asarray(incoming["spikes/times"][index]) * 1000000).astype(np.int64)
          units = np.asarray(incoming["spikes/units"][index], dtype=np.int64)
          if len(times) != len(units) or np.any(times < 0) or np.any(times >= settings["duration_us"]) or np.any(units < 0) or np.any(units >= 700):
            raise ValueError("Invalid SHD events or forbidden truncation")
          events = np.empty(len(times), dtype=[("t", np.int64), ("x", np.int64)])
          events["t"], events["x"] = times, units
          counts, length = bin_shd_events(events, spec)
          save_sample(root, record, counts[:length], int(labels[index]), int(groups[index]), f"shd-{split}:{index}", "", len(times))
    else:
      for path in track(paths[len(record["rows"]):], f"{name}/{split}/bin", total=len(paths), initial=len(record["rows"]), unit="samples"):
        stop(config)
        subject, lighting = path.parent.name.split("_", 1)
        events = np.load(path, mmap_mode="r")
        counts = bin_dvs_clip(events, settings["bin_us"], settings["grid"])
        save_sample(root, record, counts, int(path.stem), int(subject[4:]), f"dvs-{split}:{path.parent.name}/{path.name}", lighting, len(events))
    record["status"] = "completed"
    write(destination, record)
  rows = record["rows"]
  groups = np.asarray([row["group"] for row in rows], dtype=np.int64)
  labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
  expected = (8156 if split == "train" else 2264) if name == "shd" else (1077 if split == "train" else 264)
  if len(rows) != expected:
    raise ValueError(f"Incomplete {name}/{split} dataset: expected {expected}, received {len(rows)}")
  classes = 20 if name == "shd" else 11
  if set(labels) != set(range(classes)):
    raise ValueError("Missing or unexpected dataset classes")
  confirmation = np.isin(groups, settings["confirmation_groups"]) if split == "train" else np.ones(len(rows), dtype=bool)
  return StudyData(
    name,
    [root / row["file"] for row in rows],
    labels,
    groups,
    [row["identity"] for row in rows],
    np.flatnonzero(~confirmation),
    np.flatnonzero(confirmation),
    classes,
    settings["bin_us"],
    digest(record),
    [row["lighting"] for row in rows],
    raw_fingerprint=digest(identity["raw"])
  )

def save_sample(root: Path, record: dict, counts: np.ndarray, label: int, group: int, identity: str, lighting: str, events: int) -> None:
  """Commit a conserved complete sample before updating progress.

  :param root: Cache directory.
  :param record: Mutable progress record.
  :param counts: Unpadded counts.
  :param label: Class.
  :param group: Speaker or subject.
  :param identity: Original sample identity.
  :param lighting: DVS condition.
  :param events: Original event count.
  """
  if int(counts.sum(dtype=np.uint64)) != events:
    raise ValueError("Event-count conservation failed")
  filename = f"{len(record['rows'])}.npy"
  temporary = root / "sample.tmp.npy"
  np.save(temporary, counts)
  temporary.replace(root / filename)
  record["rows"].append({
    "file": filename,
    "sha256": checksum(root / filename),
    "label": label,
    "group": group,
    "identity": identity,
    "lighting": lighting,
    "events": events,
    "steps": len(counts)
  })
  if len(record["rows"]) % 64 == 0:
    write(root / "manifest.json", record)
