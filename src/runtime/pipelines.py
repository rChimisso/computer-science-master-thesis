import hashlib
import os
from pathlib import Path

import numpy as np

from src.runtime.records import checksum, digest, read, write

def verify(package: dict) -> dict:
  """Check a retained inference package without importing its execution backend.

  :param package: Run-local reference to shared artifacts.
  :return: Validated pipeline description.
  """
  root = Path(package["directory"])
  for name, expected in package["artifacts"].items():
    if checksum(root / name) != expected:
      raise ValueError("Final inference package changed")
  return read(root / "pipeline.json")

def export(data, indices: np.ndarray, spec: dict, seed: int, phase: str, job: dict, config: dict) -> dict:
  """Retain only inference parameters and verify prediction replay before pruning.

  :param data: Training source used by this fit.
  :param indices: Ordered fitting membership.
  :param spec: Frozen numerical model definition.
  :param seed: Initialization seed.
  :param phase: Confirmation or official fitting boundary.
  :param job: Completed fit or experiment result.
  :param config: Bound runtime settings.
  :return: Checksummed shared package reference.
  """
  arrays = {}
  if spec["model"] in ("lstm", "transformer"):
    import torch
    checkpoint = job.get("checkpoint", job.get("artifact"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arrays = {key: value.detach().numpy() for key, value in state["model"].items()}
  else:
    path = job["rows"][0]["state_path"] if "rows" in job else job["artifact"]
    with np.load(path, allow_pickle=False) as state:
      arrays = {key: state[key].copy() for key in state.files}
    if spec["model"] == "crc":
      from src.features.observations import classical_reference
      reservoir = classical_reference(data, spec, seed)
      arrays.update({"input_weights": reservoir.input_weights, "recurrent_weights": reservoir.recurrent_weights, "bias": reservoir.bias})
  description = {
    "version": 1,
    "dataset": data.name,
    "phase": phase,
    "spec": spec,
    "seed": seed,
    "membership": digest(data.membership(indices)),
    "classes": data.classes,
    "channels": data.counts(int(indices[0])).shape[1],
    "bin_us": data.bin_us,
    "preprocessing": config["data"][data.name],
    "adapter": job.get("adapter"),
    "segments": spec.get("segments", config["datasets"][data.name]["segments"]),
    "method": config["method"],
    "array_identities": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for key, value in arrays.items()}
  }
  if spec["model"] in ("lstm", "transformer"):
    description.update({"neural": config["neural"], "neural_recipes": config["neural_recipes"], "batch_size": state["identity"]["batch_size"]})
  if spec["model"] == "qrc":
    rng = np.random.default_rng(seed)
    description["gate_angles"] = rng.uniform(-np.pi, np.pi, (spec["depth"], spec["qubits"], 2)).tolist()
  root = Path(config["paths"]["cache"]) / "models" / digest(description)
  root.mkdir(parents=True, exist_ok=True)
  from src.runtime.scheduling import lock
  with lock(Path(config["paths"]["cache"]) / "locks" / f"pipeline-{root.name}.lock", config):
    target = root / "weights.npz"
    if not (root / "pipeline.json").exists():
      temporary = root / f"weights-{os.getpid()}.tmp"
      with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
      temporary.replace(target)
      write(root / "pipeline.json", description)
    elif read(root / "pipeline.json") != description:
      raise ValueError("Pipeline identity collision")
    with np.load(target, allow_pickle=False) as saved:
      for key, value in arrays.items():
        np.testing.assert_array_equal(saved[key], value)
  package = {
    "id": root.name,
    "directory": str(root),
    "artifacts": {name: checksum(root / name) for name in ("weights.npz", "pipeline.json")},
    "inference_bytes": sum(path.stat().st_size for path in root.iterdir() if path.is_file())
  }
  expected = job["rows"][0]["train_predictions"] if "rows" in job else job["train_predictions"]
  checked = indices[:min(3, len(indices))]
  actual = predict(package, data, checked, config)
  np.testing.assert_array_equal(actual, np.asarray(expected)[:len(checked)])
  package["verified_predictions"] = {"membership": data.membership(checked), "predictions": actual.tolist(), "check": "exact class agreement"}
  return package

def predict(package: dict, data, indices: np.ndarray, config: dict) -> np.ndarray:
  """Apply a retained inference pipeline without optimization or fitted-statistic updates.

  :param package: Verified model reference.
  :param data: Input dataset with compatible preprocessing.
  :param indices: Requested ordered examples.
  :param config: Device and cancellation policy.
  :return: Predicted classes in source order.
  """
  description = verify(package)
  spec = description["spec"]
  if data.classes != description["classes"] or data.bin_us != description["bin_us"] or data.counts(int(indices[0])).shape[1] != description["channels"]:
    raise ValueError("Inference input schema differs from retained pipeline")
  with np.load(Path(package["directory"]) / "weights.npz", allow_pickle=False) as stored:
    arrays = {key: stored[key] for key in stored.files}
  if spec["model"] in ("lstm", "transformer"):
    import torch
    from src.models.training import neural_settings, neural_evaluate
    from src.models.temporal_reference import build_network
    device = torch.device(config["execution"]["device"])
    local = config | {"neural": description["neural"], "neural_recipes": description["neural_recipes"]}
    network = build_network(description["channels"], data.classes, spec["model"], neural_settings(spec["model"], spec["recipe"], local), device)
    network.load_state_dict({key: torch.from_numpy(value.copy()).to(device) for key, value in arrays.items()})
    batch_size = min(description["batch_size"], config["training"]["batch_size"])
    if batch_size < 1:
      raise ValueError("Inference batch size must be positive")
    return np.asarray(neural_evaluate(network, data, indices, batch_size, device, config)["predictions"])
  from src.features.adapters import apply_adapter
  from src.features.extraction import summarize
  from src.features.observations import quantum_views
  from src.runtime.records import stop
  from src.evaluation.readout import predict_readout
  quantum = None
  if spec["model"] == "qrc":
    from src.models.quantum_execution import AerReservoir
    settings = {key: spec[key] for key in ("qubits", "inputs", "depth")}
    settings.update({
      "seed": description["seed"],
      "gate_angles": description["gate_angles"],
      "reset_mode": spec.get("reset", "partial"),
      "observables": ["x", "y", "z", "zz_ring"] if spec["qubits"] == 6 else ["x", "y", "z"]
    })
    quantum = AerReservoir(settings, config["execution"]["qrc_large"] if spec["qubits"] == 8 else config["execution"]["qrc_small"], config["execution"]["gpu_threads"])
  features = []
  try:
    for index in indices:
      stop(config)
      values = apply_adapter(data.values(int(index)), description["adapter"])
      if quantum:
        trajectory = quantum.transform([values], cancellation=lambda: stop(config))["values"][0]
        values = quantum_views(trajectory, spec)[spec.get("view", "all")]
      elif spec["model"] == "crc":
        state = np.zeros(spec["size"])
        trajectory = []
        leak = -np.expm1(-data.bin_us / (1000 * spec["tau"]))
        for row in values:
          if spec.get("reset") == "all":
            state.fill(0)
          state = (1 - leak) * state + leak * np.tanh(arrays["input_weights"] @ row + arrays["recurrent_weights"] @ state + arrays["bias"])
          trajectory.append(state.copy())
        values = np.asarray(trajectory)
        if spec.get("view") == "products":
          values = np.concatenate((values, values[:, -6:] * np.roll(values[:, -6:], -1, axis=1)), axis=1)
      summary = summarize(values, description["segments"])
      if spec.get("summary") == "segment_mean_final":
        width = values.shape[1]
        summary = np.concatenate((summary[:description["segments"] * width], summary[-width:]))
      features.append(summary)
  finally:
    if quantum:
      quantum.close()
  return predict_readout(np.asarray(features), arrays)
