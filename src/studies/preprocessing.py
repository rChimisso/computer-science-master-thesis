import copy
from pathlib import Path

import numpy as np

from src.runtime.progress import stage, track
from src.data.datasets import load
from src.runtime.records import write
from src.studies.experiments import experiment

class RawCounts:
  """Expose count values while preserving dataset membership and batching.

  :ivar data: Underlying audited count source.
  """

  def __init__(self, data):
    """Wrap the source without changing its counts.

    :param data: Underlying dataset.
    """
    self.data = data

  def __getattr__(self, name):
    """Delegate metadata and count access.

    :param name: Requested attribute.
    :return: Underlying value.
    """
    return getattr(object.__getattribute__(self, 'data'), name)

  def values(self, index: int) -> np.ndarray:
    """Return raw counts for the paper-informed input condition.

    :param index: Sample row.
    :return: Float32 unpadded counts.
    """
    return np.asarray(self.data.counts(index), dtype=np.float32)

@stage("studies.preprocessing.run_preprocessing")
def run_preprocessing(data, seeds: list, config: dict) -> dict:
  """Repeat the declared SHD representation diagnostic on its original partition.

  :param data: Main source, used to select the dataset only.
  :param seeds: Requested comparison seeds.
  :param config: Standalone numerical configuration.
  :return: Separately labelled diagnostic outcomes.
  """
  if data.name != "shd":
    return {"status": "not_applicable"}
  rows = []
  for seed in seeds:
    for label, representation in track(config["supporting"]["preprocessing"]["representations"].items(), f"preprocessing/seed={seed}", unit="representations"):
      local = copy.deepcopy(config)
      local["data"]["shd"].update({key: representation[key] for key in ("channels", "bin_us")})
      local["training"].update({key: config["supporting"]["preprocessing"][key] for key in ("max_epochs", "patience")})
      from src.runtime.tasks import current
      scheduler = current(local)
      if scheduler:
        source = copy.deepcopy(scheduler.submit('data', {'name': 'shd', 'config': local}, {'name': 'shd', 'settings': local['data']['shd'], 'source': local['_identity']}).result())
      else:
        source = load("shd", local)
      if representation["values"] == "count":
        source = RawCounts(source)
      source.fingerprint = f"{source.fingerprint}:{label}"
      fold = {
        "fold": "representation-diagnostic",
        "train": source.development.tolist(),
        "validation": source.confirmation.tolist()
      }
      job = experiment(source, fold, {"model": "lstm", "recipe": "baseline"}, seed, local, phase="preprocessing-diagnostic")
      rows.append({
        "representation": label,
        "seed": seed,
        "job": job["path"]
      })
  result = {
    "status": "completed" if seeds == config["seeds"] else "partial",
    "population": "Previously used speakers 0/1; diagnostic, not independent confirmation",
    "rows": rows
  }
  write(Path(config["paths"]["results"]) / "metrics/preprocessing.json", result)
  return result
