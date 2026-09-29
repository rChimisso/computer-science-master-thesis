import multiprocessing as mp
import resource
import time
from pathlib import Path

import numpy as np

from src.runtime.progress import stage, track
from src.features.adapters import apply_adapter
from src.features.observations import classical_reference
from src.runtime.records import read, stop, write

def resource_worker(connection, data, entry: dict, seed: int, config: dict) -> None:
  """Measure one practical pipeline in an isolated process after fitting has completed.

  :param connection: Parent communication endpoint.
  :param data: Training-only benchmark source.
  :param entry: Frozen practical pipeline.
  :param seed: Initialization seed.
  :param config: Runtime paths and device settings.
  """
  try:
    from src.runtime.progress import configure
    configure("off")
    import torch
    from src.data.indexed import padded_batch
    from src.models.training import neural_settings
    from src.models.temporal_reference import build_network
    from src.models.training_support import configure_device
    from src.models.streaming_crc import StreamingCRC
    from src.evaluation.readout import predict_readout
    from src.runtime.pipelines import verify
    spec = entry["spec"]
    job = read(entry["resource_job"])
    if job["status"] != "completed":
      raise ValueError("Resource measurements require a completed fit")
    package = entry["pipeline"]
    description = verify(package)
    membership = np.asarray(sorted(data.confirmation, key=lambda index: (len(data.counts(index)), int(index))))
    positions = np.linspace(0, len(membership) - 1, min(64, len(membership)), dtype=int)
    indices = membership[positions]
    device = configure_device(seed, config["execution"]["device"] if spec["model"] != "crc" else config["execution"]["crc_device"], config["execution"]["cpu_threads"])
    batch_size = config["execution"]["crc_batch"] if spec["model"] == "crc" else min(description["batch_size"], config["training"]["batch_size"])
    if batch_size < 1:
      raise ValueError("Resource batch size must be positive")
    if device.type == "cuda":
      torch.cuda.set_per_process_memory_fraction(config["execution"]["memory_fraction"], device)
    started = time.perf_counter()
    if spec["model"] == "crc":
      state = np.load(Path(package["directory"]) / "weights.npz")
      reference = classical_reference(data, spec, seed)
      extractor = StreamingCRC(reference, str(device), config["execution"]["crc_graphs"])
      adapter = job["adapter"]
      parameters = sum(value.size for value in (reference.input_weights, reference.recurrent_weights, reference.bias))
      learned = state["coefficients"].size + state["intercept"].size
      normalization_parameters = state["means"].size + state["scales"].size
      artifact_bytes = package["inference_bytes"]
      def infer(batch: np.ndarray) -> np.ndarray:
        """Measure full CRC input conversion, extraction and ridge prediction.

        :param batch: Sample membership.
        :return: Predicted classes.
        """
        segments = spec.get("segments", config["datasets"][data.name]["segments"])
        chunks = []
        for start in range(0, len(batch), batch_size):
          stop(config)
          sequences = [apply_adapter(data.values(int(index)), adapter) for index in batch[start:start + batch_size]]
          chunks.append(extractor.extract(sequences, segments, ("all",))[0]["all"])
        features = np.concatenate(chunks)
        if spec.get("summary") == "segment_mean_final":
          width = spec["size"]
          features = np.concatenate((features[:, :segments * width], features[:, -width:]), axis=1)
        return predict_readout(features, state)
    else:
      network = build_network(data.counts(0).shape[1], data.classes, spec["model"], neural_settings(spec["model"], spec["recipe"], config), device)
      with np.load(Path(package["directory"]) / "weights.npz") as checkpoint:
        network.load_state_dict({key: torch.from_numpy(checkpoint[key].copy()).to(device) for key in checkpoint.files})
      network.eval()
      parameters = 0
      normalization_parameters = 0
      learned = sum(value.numel() for value in network.parameters())
      artifact_bytes = package["inference_bytes"]
      def infer(batch: np.ndarray) -> np.ndarray:
        """Measure preprocessing, transfer, neural inference and prediction recovery.

        :param batch: Sample membership.
        :return: Predicted classes.
        """
        predictions = []
        with torch.inference_mode():
          for start in range(0, len(batch), batch_size):
            stop(config)
            values, lengths, _ = padded_batch(data, batch[start:start + batch_size])
            predictions.append(network(torch.from_numpy(values).to(device), torch.from_numpy(lengths).to(device)).argmax(dim=1).cpu().numpy())
        return np.concatenate(predictions)
    if device.type == "cuda":
      torch.cuda.synchronize(device)
    setup = time.perf_counter() - started
    infer(indices)
    measurements = []
    for repeat in range(3):
      for label, batch in (("single", indices[:1]), ("batch", indices)):
        if device.type == "cuda":
          torch.cuda.synchronize(device)
        started = time.perf_counter()
        infer(batch)
        if device.type == "cuda":
          torch.cuda.synchronize(device)
        measurements.append({
          "repeat": repeat,
          "workload": label,
          "samples": len(batch),
          "seconds": time.perf_counter() - started
        })
    result = {
      "dataset": data.name,
      "model": spec["model"],
      "role": entry["role"],
      "pipeline_id": package["id"],
      "configuration": job["identity"]["spec"],
      "device": str(device),
      "threads": config["execution"]["cpu_threads"],
      "batch_size": batch_size,
      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
      "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
      "tf32_cudnn": torch.backends.cudnn.allow_tf32,
      "seed": seed,
      "membership": data.membership(indices),
      "setup_seconds": setup,
      "measurements": measurements,
      "fixed_parameters": parameters,
      "learned_parameters": learned,
      "fitted_normalization_parameters": normalization_parameters,
      "inference_artifact_bytes": artifact_bytes,
      "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
      "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
      "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
      "training_job": job["path"],
      "complete_fitting_seconds": job["seconds"],
      "readout_fitting_seconds": job["rows"][0].get("fitting_seconds"),
      "extraction_seconds": job.get("features", {}).get("seconds"),
      "features_reused": job.get("features", {}).get("cache_hit"),
      "optimization_seconds": sum(row["learning_seconds"] for row in job.get("neural", {}).get("history", [])) if job.get("neural") else None,
      "checkpoint_bytes": Path(job["checkpoint"]).stat().st_size if job.get("checkpoint") and Path(job["checkpoint"]).exists() else None,
      "optimizer_checkpoint_bytes": Path(job["checkpoint"]).with_name("last.pt").stat().st_size if job.get("checkpoint") and Path(job["checkpoint"]).with_name("last.pt").exists() else None,
      "feature_cache_bytes": sum(path.stat().st_size for path in Path(job["features"]["directory"]).glob("*.npy")) if job.get("features") else None,
      "timing_population": "Length-stratified development-confirmation subset; not official-test inference",
      "fitting_boundary": "Recorded complete fit wall time, including extraction or cache loading and evaluation; feature reuse is explicit",
      "energy": None,
      "host_memory_boundary": "Peak isolated process RSS, including imported runtime, dataset metadata and fitted pipeline; not incremental model-only RAM"
    }
    connection.send(result)
  except BaseException as error:
    connection.send({"error": repr(error)})
  finally:
    connection.close()

@stage("studies.resources.measure")
def measure(datasets: dict, frozen: dict, config: dict) -> dict:
  """Measure enabled practical models sequentially in isolated processes.

  :param datasets: Loaded development sources.
  :param frozen: Frozen settings.
  :param config: Bound runtime configuration.
  :return: Resource records with explicit physical interpretation boundaries.
  """
  records = []
  evidence = Path(config["paths"]["results"])
  confirmation = read(evidence / "metrics/confirmation.json")
  pipelines = read(evidence / "pipelines.json")
  context = mp.get_context("spawn")
  for name, entries in frozen["manifest"]["entries"].items():
    for role, entry in track(entries.items(), f"{name}/resources", unit="models"):
      spec = entry["spec"]
      if role.startswith("practical:"):
        for seed in config["seeds"]:
          fitting = next(row for row in confirmation["jobs"] if row["dataset"] == name and row["role"] == role and row["seed"] == seed)
          measured = entry | {"role": role, "resource_job": fitting["path"], "pipeline": pipelines[f"confirmation:{name}:{role}:{seed}"]}
          parent, child = context.Pipe()
          worker = context.Process(target=resource_worker, args=(child, datasets[name], measured, seed, config))
          worker.start()
          child.close()
          try:
            while not parent.poll(0.2):
              stop(config)
              if not worker.is_alive():
                raise RuntimeError("Resource worker exited without results")
            record = parent.recv()
            if "error" in record:
              raise RuntimeError(record["error"])
            records.append(record)
          finally:
            if worker.is_alive():
              worker.terminate()
            worker.join()
            parent.close()
  result = {
    "status": "completed",
    "practical": records,
    "quantum_simulator_costs_are_hardware_costs": False
  }
  write(Path(config["paths"]["results"]) / "metrics/resources.json", result)
  return result
