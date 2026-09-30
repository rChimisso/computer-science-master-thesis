import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src.models.protocol import FittingProtocol
from src.runtime.progress import stage
from src.evaluation.metrics import classification_metrics
from src.evaluation.readout import direct_readout, fit_readout, predict_readout, save_state
from src.features.extraction import extract
from src.runtime.records import digest, read, stop
from src.runtime.storage import save_job, work_root

@stage("studies.final.fit")
def fit(data, entry: dict, seed: int, config: dict) -> dict:
  """Complete a fixed full-training refit before any official-test loader is called.

  :param data: Official training source only.
  :param entry: Frozen configuration and penalty.
  :param seed: Frozen seed.
  :param config: Bound immutable protocol.
  :return: Checksummed inference pipeline and training predictions.
  """
  indices = np.arange(len(data.labels))
  spec = entry["spec"]
  penalty = entry["selection"]["lambda"]
  identity = {
    "source": config["_identity"],
    "data": data.fingerprint,
    "spec": spec,
    "penalty": penalty,
    "seed": seed,
    "membership": data.membership(indices)
  }
  root = Path(config["paths"]["results"]) / "models/final" / digest(identity)
  path = root / "fit.json"
  if path.exists():
    record = read(path)
    if record["identity"] != identity:
      raise ValueError("Final fitting membership or configuration changed")
    if record["status"] == "completed":
      from src.runtime.pipelines import verify
      verify(record["pipeline"])
      return record
  root.mkdir(parents=True, exist_ok=True)
  stop(config)
  started = time.perf_counter()
  if spec["model"] in ("lstm", "transformer"):
    from src.models.training import train_neural
    result = train_neural(data, indices, np.asarray([], dtype=int), spec["model"], spec["recipe"], seed, config, spec["fixed_epochs"])
    record = {
      "artifact": result["checkpoint"],
      "sha256": result["checkpoint_sha256"],
      "train": result["best"]["train"],
      "train_predictions": result["best"]["train"]["predictions"],
      "adapter": None,
      "neural": result
    }
  else:
    values, features, adapter = extract(data, indices, indices, spec, seed, config)
    with threadpool_limits(limits=config["execution"]["cpu_threads"]):
      if FittingProtocol.from_spec(spec).solver == "cholesky":
        row, state, _ = direct_readout(values, values[:1], data.labels, data.labels[:1], penalty, data.classes)
      else:
        for limit in (1000, 5000):
          local = config | {"readout": config["readout"] | {"tolerance": 0.001, "max_iterations": limit}}
          row, state = fit_readout(values, values[:1], data.labels, data.labels[:1], penalty, data.classes, local, limit)
          if row["status"] == "completed":
            break
        if row["status"] != "completed":
          raise ValueError("Frozen readout hit its solver ceiling")
    working = work_root(config) / "final" / digest(identity)
    working.mkdir(parents=True, exist_ok=True)
    artifact = save_state(working, "readout", state)
    record = {
      "artifact": artifact["state_path"],
      "sha256": artifact["state_sha256"],
      "train": row["train"],
      "train_predictions": row["train_predictions"],
      "adapter": adapter,
      "features": features
    }
  record.update({
    "status": "completed",
    "identity": identity,
    "seconds": time.perf_counter() - started
  })
  from src.runtime.pipelines import export
  record["pipeline"] = export(data, indices, spec, seed, "official", record, config)
  save_job(path, record, config)
  return record

@stage("studies.final.evaluate")
def evaluate(data, held, entry: dict, fitted: dict, seed: int, config: dict) -> dict:
  """Apply a completed frozen pipeline without refitting on test data.

  :param data: Original full training source.
  :param held: Explicitly authorized official test source.
  :param entry: Frozen settings.
  :param fitted: Completed fitting record.
  :param seed: Frozen seed.
  :param config: Immutable runtime settings.
  :return: Standard saved evaluation job with membership-bound predictions.
  """
  spec = entry["spec"]
  indices = np.arange(len(held.labels))
  identity = {
    "source": config["_identity"],
    "fit": digest(fitted),
    "held": held.fingerprint,
    "spec": spec,
    "seed": seed,
    "phase": "official",
    "fold": {"fold": "official"}
  }
  root = Path(config["paths"]["results"]) / "metrics/jobs" / digest(identity)
  path = root / "result.json"
  from src.runtime.pipelines import predict, verify
  verify(fitted["pipeline"])
  if path.exists():
    record = read(path)
    if record["status"] == "completed":
      return record
  if spec["model"] in ("lstm", "transformer"):
    predictions = predict(fitted["pipeline"], held, indices, config).tolist()
  else:
    values, _, _ = extract(held, np.asarray([], dtype=int), indices, spec, seed, config, fitted["adapter"])
    with np.load(Path(fitted["pipeline"]["directory"]) / "weights.npz", allow_pickle=False) as state:
      predictions = predict_readout(values, state).tolist()
  metric = classification_metrics(held.labels, np.asarray(predictions), held.classes)
  row = {
    "solver": FittingProtocol.from_spec(spec).solver,
    "lambda": entry["selection"]["lambda"],
    "train": fitted["train"],
    "validation": metric,
    "train_predictions": fitted["train_predictions"],
    "predictions": predictions
  }
  record = {
    "identity": identity,
    "classes": data.classes,
    "status": "completed",
    "path": str(path),
    "rows": [row],
    "adapter": fitted["adapter"],
    "membership": {"train": data.membership(np.arange(len(data.labels))), "held": held.membership(indices)},
    "fit_artifact": fitted["artifact"],
    "fit_sha256": fitted["sha256"]
  }
  save_job(path, record, config)
  return record
