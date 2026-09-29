import hashlib
import copy
import time
from pathlib import Path

import numpy as np

from src.runtime.progress import Progress, stage
from src.features.summary import summarize_sequence_features
from src.features.adapters import apply_adapter, fit_adapter
from src.features.observations import classical_reference, quantum_views
from src.runtime.records import checksum, digest, read, stop, write
from src.runtime.scheduling import lock
from src.runtime.numerical import identity as numerical_identity, merge, moments

def summarize(values: np.ndarray, segments: int) -> np.ndarray:
  """Summarize only valid bins in the established ordering.

  :param values: Complete unpadded trajectory.
  :param segments: Normalized segment count.
  :return: Means, population deviations and final state.
  """
  return summarize_sequence_features(values[None], np.asarray([len(values)]), segments, "segment_mean_std_final")[0]

@stage("features.extraction.extract")
def extract(data, training: np.ndarray, indices: np.ndarray, spec: dict, seed: int, config: dict, adapter: dict | None = None) -> tuple:
  """Serialize equivalent feature writers while permitting independent backends.

  :param data: Audited dataset.
  :param training: Adapter fitting indices.
  :param indices: Extraction membership.
  :param spec: Numerical reservoir settings.
  :param seed: Initialization seed.
  :param config: Runtime configuration.
  :param adapter: Optional previously fitted transformation.
  :return: Complete summary bank and transformation.
  """
  numerical = numerical_identity(spec["model"], config)
  identity = {
    "data": data.fingerprint,
    "training": training.tolist(),
    "indices": indices.tolist(),
    "spec": {key: value for key, value in spec.items() if key not in ("view", "summary", "solver")},
    "seed": 42 if spec["model"] == "projection" else seed,
    "source": numerical
  }
  with lock(Path(config["paths"]["cache"]) / "locks" / f"{digest(identity)}.lock", config):
    return extract_locked(data, training, indices, spec, seed, config, numerical, adapter)

def extract_locked(data, training: np.ndarray, indices: np.ndarray, spec: dict, seed: int, config: dict, numerical: dict, adapter: dict | None = None) -> tuple:
  """Extract all compatible views once into an atomic resumable cache.

  :param data: Dataset with explicit original identities.
  :param training: Permitted adapter-fitting rows.
  :param indices: Ordered extraction membership.
  :param spec: Reservoir and summary settings.
  :param seed: Dynamics initialization seed.
  :param config: Numerical identity, execution policy and paths.
  :param numerical: Numerical source/build identity computed once by the public extractor.
  :param adapter: Previously fitted adapter for held-set extraction.
  :return: Selected summary array, complete descriptor and fitted adapter.
  """
  mode = spec.get("adapter", config["datasets"][data.name]["adapter"])
  if spec.get("inputs") == 0:
    mode = "identity"
  if adapter is None:
    if any("-train:" not in data.identities[int(i)] for i in training):
      raise ValueError("Adapter fitting cannot inspect official test samples")
    key = digest([
      data.fingerprint,
      training.tolist(),
      mode,
      spec["inputs"],
      spec.get("projection_seed", 42)
    ])
    with lock(Path(config["paths"]["cache"]) / "locks" / f"adapter-{key}.lock", config):
      adapter = fit_adapter(data, training, mode, spec["inputs"], spec.get("projection_seed", 42), config)
  if adapter.get("fingerprint") != digest({key: value for key, value in adapter.items() if key != "fingerprint"}):
    raise ValueError("Fitted input adapter checksum changed")
  segments = spec.get("segments", config["datasets"][data.name]["segments"])
  settings = {key: value for key, value in spec.items() if key not in ("view", "summary", "solver")}
  identity = {
    "source": numerical,
    "data": data.fingerprint,
    "membership": data.membership(indices),
    "adapter": adapter,
    "spec": settings,
    "segments": segments,
    "seed": 42 if spec["model"] == "projection" else seed,
    "execution": numerical["execution"]
  }
  root = Path(config["paths"]["cache"]) / "features" / digest(identity)
  path = root / "metadata.json"
  record = read(path) if path.exists() else {
    "identity": identity,
    "status": "running",
    "completed_samples": 0,
    "seconds": 0.0,
    "checksums": {}
  }
  if record["identity"] != identity:
    raise ValueError("Feature cache identity changed")
  if record["status"] == "completed":
    for filename, expected in record["checksums"].items():
      if checksum(root / filename) != expected:
        raise ValueError("Completed feature cache changed")
    with Progress(f"{data.name}/{spec['model']}/seed={seed}/features/{root.name[:10]}", len(indices), len(indices), "samples") as task:
      task.status = "cached"
    return selected_view(root, spec), record | {"cache_hit": True}, adapter
  root.mkdir(parents=True, exist_ok=True)
  quantum = None
  classical = None
  arrays = {}
  prefixes = {}
  execution = config["execution"]
  training_set = set(training.tolist())
  batch = execution["quantum_batch"] if spec["model"] == "qrc" else execution["crc_batch"]
  try:
    if spec["model"] == "qrc":
      from src.models.quantum_execution import AerReservoir
      settings = {key: spec[key] for key in ("qubits", "inputs", "depth")}
      settings.update({
        "seed": seed,
        "reset_mode": spec.get("reset", "partial"),
        "observables": [
          "x",
          "y",
          "z",
          "zz_ring"
        ] if spec["qubits"] == 6 else [
          "x",
          "y",
          "z"
        ]
      })
      quantum = AerReservoir(settings, execution["qrc_large"] if spec["qubits"] == 8 else execution["qrc_small"], execution["gpu_threads"], lambda: stop(config))
    elif spec["model"] == "crc":
      import torch
      from src.models.streaming_crc import StreamingCRC
      torch.set_num_threads(execution["cpu_threads"])
      classical = StreamingCRC(classical_reference(data, spec, seed), execution["crc_device"], execution["crc_graphs"])
    with Progress(f"{data.name}/{spec['model']}/seed={seed}/features/{root.name[:10]}", len(indices), record["completed_samples"], "samples") as task:
      for start in range(record["completed_samples"], len(indices), batch):
        stop(config)
        committed = copy.deepcopy(record)
        started = time.perf_counter()
        rows = indices[start:start + batch]
        sequences = [apply_adapter(data.values(int(index)), adapter) for index in rows]
        for index, sequence in zip(rows, sequences):
          population = "train" if int(index) in training_set else "held"
          diagnostics = record.setdefault("diagnostics", {}).setdefault(population, {})
          diagnostics["input"] = merge(diagnostics.get("input", {}), moments(sequence))
        timings = {}
        if quantum is not None:
          local = config | {"_deadline": min(config["_deadline"], time.time() + execution["job_seconds"])}
          transformed = quantum.transform(sequences, cancellation=lambda: stop(local))
          trajectories = transformed["values"]
          timings = {key: value for key, value in transformed.items() if key.endswith("_seconds")}
          output = {}
          for index, trajectory in zip(rows, trajectories):
            population = "train" if int(index) in training_set else "held"
            diagnostics = record["diagnostics"][population]
            diagnostics["observations"] = merge(diagnostics.get("observations", {}), moments(trajectory))
            for view, values in quantum_views(trajectory, spec).items():
              output.setdefault(view, []).append(summarize(values, segments))
        elif classical is not None and spec.get("reset", "partial") == "partial":
          views = ("all", "products") if spec["size"] == 18 else ("all",)
          output, timings = classical.extract(sequences, segments, views, lambda: stop(config))
          for index, statistics in zip(rows, timings.pop("state_statistics")):
            population = "train" if int(index) in training_set else "held"
            diagnostics = record["diagnostics"][population]
            diagnostics["states"] = merge(diagnostics.get("states", {}), statistics)
        elif classical is not None:
          reference = classical_reference(data, spec, seed)
          output = {"all": []}
          for index, sequence in zip(rows, sequences):
            population = "train" if int(index) in training_set else "held"
            diagnostics = record["diagnostics"][population]
            states = reference.parameters.leak_rate * np.tanh(sequence @ reference.input_weights.T + reference.bias)
            output["all"].append(summarize(states, segments))
            diagnostics["states"] = merge(diagnostics.get("states", {}), moments(states))
        else:
          output = {"all": [summarize(sequence, segments) for sequence in sequences]}
        for view, values in output.items():
          values = np.asarray(values, dtype=np.float64)
          if not np.isfinite(values).all():
            raise ValueError("Nonfinite features")
          filename = f"{view}.npy"
          if view not in arrays:
            shape = (len(indices), values.shape[1])
            if record["completed_samples"] and not (root / filename).exists():
              raise ValueError("Interrupted feature array missing")
            arrays[view] = np.lib.format.open_memmap(root / filename, mode="r+" if (root / filename).exists() else "w+", dtype=np.float64, shape=shape)
            if arrays[view].shape != shape or arrays[view].dtype != np.float64:
              raise ValueError("Interrupted feature shape changed")
            prefixes[view] = hashlib.sha256(np.ascontiguousarray(arrays[view][:start]).tobytes())
            if start and record["prefixes"][view] != prefixes[view].hexdigest():
              raise ValueError("Interrupted feature prefix changed")
          arrays[view][start:start + len(rows)] = values
          arrays[view].flush()
          prefixes[view].update(values.tobytes())
        record.update({
          "completed_samples": start + len(rows),
          "seconds": record["seconds"] + time.perf_counter() - started,
          "prefixes": {view: value.hexdigest() for view, value in prefixes.items()}
        })
        for key, value in timings.items():
          record.setdefault("timings", {})[key] = record.get("timings", {}).get(key, 0) + value
        write(path, record)
        committed = copy.deepcopy(record)
        task.update(len(rows))
    record.update({
      "status": "completed",
      "checksums": {file.name: checksum(file) for file in root.glob("*.npy")},
      "directory": str(root)
    })
    write(path, record)
  except BaseException as error:
    if "committed" in locals():
      record = committed
    record.update({"status": "interrupted" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "failed", "error": repr(error)})
    write(path, record)
    raise
  finally:
    if quantum is not None:
      quantum.close()
  return selected_view(root, spec), record | {"cache_hit": False}, adapter

def selected_view(root: Path, spec: dict) -> np.ndarray:
  """Read a requested observation and summary view from a shared extraction.

  :param root: Completed feature bank.
  :param spec: View and optional simpler summary.
  :return: Ordered feature rows.
  """
  values = np.load(root / f"{spec.get('view', 'all')}.npy", mmap_mode="r")
  if spec.get("summary", "segment_mean_std_final") == "segment_mean_final":
    segments = spec["segments"]
    width = values.shape[1] // (2 * segments + 1)
    return np.concatenate((values[:, :segments * width], values[:, -width:]), axis=1)
  return values
