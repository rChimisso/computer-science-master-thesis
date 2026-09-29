import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src.runtime.progress import stage, track
from src.evaluation.readout import direct_readout, save_state
from src.features.extraction import extract
from src.runtime.records import checksum, digest, read, stop
from src.runtime.storage import intentionally_pruned, save_job, work_root

@stage("studies.experiments.experiment")
def experiment(data, fold: dict, spec: dict, seed: int, config: dict, penalties: list | None = None, phase: str = "development") -> dict:
  """Execute or resume a single declared fitting/evaluation cell.

  :param data: Source data.
  :param fold: Explicit training and held indices.
  :param spec: Frozen numeric candidate.
  :param seed: Initialization seed.
  :param config: Bound runtime configuration.
  :param penalties: Optional frozen penalty for confirmation or learning curves.
  :param phase: Population and interpretation boundary.
  :return: Complete metrics, predictions and fitted-state references.
  """
  from src.runtime.tasks import current
  scheduler = current(config)
  if scheduler is not None:
    return scheduler.experiment(data, fold, spec, seed, config, penalties, phase).result()
  train = np.asarray(fold["train"], dtype=np.int64)
  held = np.asarray(fold["validation"], dtype=np.int64)
  if len(set(train)) != len(train) or len(set(held)) != len(held) or set(train) & set(held):
    raise ValueError("Invalid sample membership")
  if any("-train:" not in data.identities[int(i)] for i in train):
    raise ValueError("Training cannot use test examples")
  if phase == "development" and not set(train).union(held) <= set(data.development):
    raise ValueError("Development cannot inspect confirmation or official test")
  if phase in ("development", "confirmation") and set(data.groups[train]) & set(data.groups[held]):
    raise ValueError("Held-group isolation failed")
  identity = {
    "source": config["_identity"],
    "dataset": data.name,
    "data": data.fingerprint,
    "fold": fold,
    "seed": seed,
    "spec": spec,
    "penalties": penalties,
    "phase": phase
  }
  root = Path(config["paths"]["results"]) / "metrics/jobs" / digest(identity)
  destination = root / "result.json"
  if destination.exists():
    saved = read(destination)
    if saved["status"] == "completed":
      for row in saved["rows"]:
        if row.get("state_path") and not intentionally_pruned(row["state_path"], config) and checksum(row["state_path"]) != row["state_sha256"]:
          raise ValueError("Fitted state changed")
      if saved.get("checkpoint") and not intentionally_pruned(saved["checkpoint"], config) and checksum(saved["checkpoint"]) != saved["checkpoint_sha256"]:
        raise ValueError("Neural checkpoint changed")
      return saved
  root.mkdir(parents=True, exist_ok=True)
  record = {
    "identity": identity,
    "classes": data.classes,
    "status": "running",
    "path": str(destination),
    "membership": {"train": data.membership(train), "held": data.membership(held)},
    "rows": []
  }
  save_job(destination, record, config)
  started = time.perf_counter()
  try:
    stop(config)
    if spec["model"] in ("lstm", "transformer"):
      from src.models.training import train_neural
      result = train_neural(data, train, held, spec["model"], spec["recipe"], seed, config, spec.get("fixed_epochs"))
      row = {
        "lambda": None,
        "train": result["best"]["train"],
        "validation": result["best"]["validation"],
        "predictions": result["best"]["validation"]["predictions"],
        "train_predictions": result["best"]["train"]["predictions"]
      }
      record.update({
        "rows": [row],
        "neural_status": result["status"],
        "best_epoch": result["best"]["epoch"],
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "neural": result
      })
    else:
      indices = np.concatenate((train, held))
      values, features, adapter = extract(data, train, indices, spec, seed, config)
      record.update({"features": features, "adapter": adapter})
      grid = penalties if penalties is not None else config["readout"]["lambdas"]
      solver = spec.get("solver", "cholesky")
      with threadpool_limits(limits=config["execution"]["cpu_threads"]):
        for ceiling in (1000, 5000):
          rows = []
          for penalty in track(grid, f"{data.name}/{spec['model']}/seed={seed}/ridge", unit="penalties"):
            stop(config)
            if solver == "cholesky":
              row, state, _ = direct_readout(values[:len(train)], values[len(train):], data.labels[train], data.labels[held], penalty, data.classes)
            else:
              from src.evaluation.readout import fit_readout
              local = config | {"readout": config["readout"] | {"tolerance": 0.001, "max_iterations": ceiling}}
              row, state = fit_readout(values[:len(train)], values[len(train):], data.labels[train], data.labels[held], penalty, data.classes, local, ceiling)
            states = work_root(config) / "readouts" / digest(identity)
            states.mkdir(parents=True, exist_ok=True)
            row.update(save_state(states, f"lambda-{penalty}", state))
            rows.append(row)
          record["rows"] = rows
          save_job(destination, record, config)
          if all(row["status"] == "completed" for row in rows):
            break
        if any(row["status"] != "completed" for row in record["rows"]):
          raise ValueError("Unresolved solver limit cannot enter selection")
    record.update({"status": "completed", "seconds": time.perf_counter() - started})
    save_job(destination, record, config)
    return record
  except BaseException as error:
    record.update({"status": "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed", "error": repr(error)})
    save_job(destination, record, config)
    raise
