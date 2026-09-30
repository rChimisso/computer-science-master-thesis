import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.runtime.progress import Progress, stage, track
from src.data.indexed import StudyData, padded_batch
from src.runtime.records import checksum, digest, read, sources, stop, write
from src.evaluation.metrics import classification_metrics
from src.models.training_support import atomic_checkpoint, configure_device, restore_rng, rng_state, synchronize
from src.runtime.storage import scalar_history, work_root
from src.models.temporal_reference import build_network
from src.models.protocol import FittingProtocol

def neural_settings(model: str, recipe: str, config: dict) -> dict:
  """Resolve one predeclared neural regularization recipe.

  :param model: LSTM or Transformer.
  :param recipe: Baseline, dropout, or regularization.
  :param config: Resolved supplementary settings and declared recipes.
  :return: Explicit model and optimizer settings.
  """
  settings = copy.deepcopy(config["neural"][model])
  if recipe in ("dropout", "regularization"):
    settings.update(config["neural_recipes"][model][recipe])
  elif recipe != "baseline":
    raise ValueError("Unknown predeclared neural recipe")
  return settings

def make_neural(data: StudyData, model: str, recipe: str, config: dict, device: torch.device) -> tuple:
  """Construct a dataset-aware model and unchanged optimizer family.

  :param data: Dataset dimensions and classes.
  :param model: Model family.
  :param recipe: Registered regularization recipe.
  :param config: Reference and training settings.
  :param device: Explicit execution device.
  :return: Model and optimizer.
  """
  settings = neural_settings(model, recipe, config)
  network = build_network(data.counts(0).shape[1], data.classes, model, settings, device)
  if FittingProtocol.from_spec({"model": model}).solver == "adamax":
    optimizer = torch.optim.Adamax(network.parameters(), lr=settings["learning_rate"], betas=tuple(settings["betas"]), eps=settings["epsilon"], weight_decay=settings["weight_decay"])
  else:
    optimizer = torch.optim.AdamW(network.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
  return network, optimizer

def neural_batches(data: StudyData, indices: np.ndarray, batch_size: int, shuffle: bool) -> list[np.ndarray]:
  """Build length-aware batches with reproducible random membership.

  :param data: Sequence source.
  :param indices: Eligible source rows.
  :param batch_size: Common feasible family batch size.
  :param shuffle: Shuffle within length-sorted random windows during training.
  :return: Ordered minibatch index arrays.
  """
  order = np.random.permutation(indices) if shuffle else indices.copy()
  blocks = []
  window = batch_size * 10
  for start in range(0, len(order), window):
    block = sorted(order[start:start + window], key=lambda index: len(data.counts(int(index))))
    blocks.extend(np.asarray(block[offset:offset + batch_size]) for offset in range(0, len(block), batch_size))
  if shuffle:
    blocks = [blocks[int(i)] for i in np.random.permutation(len(blocks))]
  return blocks

def neural_evaluate(network: nn.Module, data: StudyData, indices: np.ndarray, batch_size: int, device: torch.device, config: dict) -> dict:
  """Evaluate fixed parameters without selecting on the held partition.

  :param network: Trained temporal model.
  :param data: Dataset source.
  :param indices: Evaluation membership.
  :param batch_size: Evaluation batch size.
  :param device: Explicit device.
  :param config: Cancellation settings.
  :return: Metrics, ordered predictions, loss, and inference duration.
  """
  network.eval()
  predictions = {}
  loss_sum = 0.0
  synchronize(device)
  started = time.perf_counter()
  with torch.no_grad():
    for batch in track(neural_batches(data, indices, batch_size, False), f"{data.name}/inference", unit="batches"):
      stop(config)
      values, lengths, labels = padded_batch(data, batch)
      logits = network(torch.from_numpy(values).to(device), torch.from_numpy(lengths).to(device))
      loss = nn.functional.cross_entropy(logits, torch.from_numpy(labels).to(device), reduction="sum")
      if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite evaluation loss")
      loss_sum += loss.item()
      predictions.update(zip(batch.tolist(), logits.argmax(dim=1).cpu().tolist()))
  synchronize(device)
  predicted = np.asarray([predictions[int(i)] for i in indices])
  result = classification_metrics(data.labels[indices], predicted, data.classes)
  result.update({
    "loss": loss_sum / len(indices),
    "predictions": predicted.tolist(),
    "seconds": time.perf_counter() - started
  })
  return result

def benchmark_batch(data: StudyData, indices: np.ndarray, model: str, config: dict, device: torch.device) -> dict:
  """Measure one calibration attempt in a scope that releases failed allocations.

  :param data: Development source.
  :param indices: Longest complete clips in the candidate batch.
  :param model: Neural family.
  :param config: Optimizer and architecture settings.
  :param device: Explicit training device.
  :return: Successful timing and memory measurements.
  """
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
  network, optimizer = make_neural(data, model, "baseline", config, device)
  started = time.perf_counter()
  values, lengths, labels = padded_batch(data, indices)
  for _ in range(2):
    optimizer.zero_grad(set_to_none=True)
    logits = network(torch.from_numpy(values).to(device), torch.from_numpy(lengths).to(device))
    loss = nn.functional.cross_entropy(logits, torch.from_numpy(labels).to(device))
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
  synchronize(device)
  return {
    "two_updates_seconds": time.perf_counter() - started,
    "maximum_steps": int(lengths.max()),
    "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
  }

@stage("models.training.benchmark_neural")
def benchmark_neural(data: StudyData, model: str, config: dict) -> int:
  """Find and persist a common memory-safe batch size on longest clips.

  :param data: Development source only.
  :param model: Registered model family.
  :param config: Device and output settings.
  :return: Largest feasible predeclared batch size.
  """
  path = Path(config["paths"]["evidence"]) / data.name / f"benchmark-{model}.json"
  fingerprint = digest({
    "data": data.fingerprint,
    "source": sources(numerical=True),
    "device": config["execution"]["device"],
    "training": config["training"],
    "neural": config["neural"][model]
  })
  if path.exists():
    record = read(path)
    if record["fingerprint"] == fingerprint:
      return record["batch_size"]
  device = configure_device(42, config["execution"]["device"], config["training"]["cpu_threads"])
  longest = np.asarray(sorted(data.development, key=lambda i: len(data.counts(int(i))), reverse=True))
  failures = []
  if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(config["execution"]["memory_fraction"], device)
  for size in [value for value in (128, 64, 32, 16, 8, 4, 2, 1) if value <= config["training"]["batch_size"]]:
    stop(config)
    try:
      measured = benchmark_batch(data, longest[:size], model, config, device)
    except torch.cuda.OutOfMemoryError:
      failures.append(size)
    else:
      write(path, measured | {"fingerprint": fingerprint, "batch_size": size, "failures": failures})
      return size
    finally:
      if device.type == "cuda":
        torch.cuda.empty_cache()
  raise RuntimeError("Even one complete clip does not fit the device")

@stage("models.training.train_neural")
def train_neural(data: StudyData, training: np.ndarray, validation: np.ndarray, model: str, recipe: str, seed: int, config: dict, fixed_epochs: int | None = None) -> dict:
  """Run or resume one fully fingerprinted inner-fold or fixed-duration fit.

  :param data: Indexed dataset source.
  :param training: Fitting membership.
  :param validation: Held membership, empty for final fitting without evaluation.
  :param model: Neural family.
  :param recipe: Registered configuration.
  :param seed: Initialization and shuffle seed.
  :param config: Study and device settings.
  :param fixed_epochs: Disable validation selection and fit this many epochs.
  :return: Completed run with its selected or final checkpoint.
  """
  batch_size = benchmark_neural(data, model, config)
  identity = {
    "data": data.fingerprint,
    "training": data.membership(training),
    "validation": data.membership(validation),
    "model": model,
    "recipe": recipe,
    "settings": neural_settings(model, recipe, config),
    "seed": seed,
    "training_settings": config["training"],
    "batch_size": batch_size,
    "fixed_epochs": fixed_epochs,
    "source": sources(numerical=True),
    "torch": torch.__version__,
    "device": config["execution"]["device"]
  }
  fingerprint = digest(identity)
  directory = work_root(config) / "neural" / fingerprint
  directory.mkdir(parents=True, exist_ok=True)
  result_path = directory / "result.json"
  if result_path.exists():
    saved = read(result_path)
    if saved["status"] in ("early_stopped", "epoch_ceiling", "fixed_duration"):
      if checksum(saved["checkpoint"]) != saved["checkpoint_sha256"]:
        raise ValueError("Neural checkpoint checksum mismatch")
      return saved
  device = configure_device(seed, config["execution"]["device"], config["training"]["cpu_threads"])
  if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(config["execution"]["memory_fraction"], device)
  network, optimizer = make_neural(data, model, recipe, config, device)
  checkpoint = directory / "last.pt"
  state = {
    "epoch": 0,
    "best": None,
    "best_model": None,
    "history": [],
    "patience_reference": -1.0,
    "stale": 0,
    "seconds": 0.0,
    "identity": identity
  }
  if checkpoint.exists():
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["identity"] != identity:
      raise ValueError("Incompatible neural checkpoint")
    network.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
  else:
    state.update({
      "model": network.state_dict(),
      "optimizer": optimizer.state_dict(),
      "rng": rng_state()
    })
    atomic_checkpoint(checkpoint, state)
  write(directory / "identity.json", identity)
  started = time.time()
  previous = state["seconds"]
  limit = fixed_epochs or config["training"]["max_epochs"]
  status = "fixed_duration" if fixed_epochs else "epoch_ceiling"
  if not fixed_epochs and state["epoch"] >= config["training"]["min_epochs"] and state["stale"] >= config["training"]["patience"]:
    status = "early_stopped"
  with Progress(f"{data.name}/{model}/{recipe}/seed={seed}/train/{fingerprint[:10]}", limit, state["epoch"], "epochs") as task:
    epochs = range(state["epoch"] + 1, limit + 1) if status != "early_stopped" else ()
    for epoch in epochs:
      epoch_started = time.perf_counter()
      network.train()
      loss_sum = 0.0
      synchronize(device)
      learning_started = time.perf_counter()
      for batch in track(neural_batches(data, training, batch_size, True), f"{data.name}/{model}/seed={seed}/epoch={epoch}", unit="batches"):
        stop(config)
        values, lengths, labels = padded_batch(data, batch)
        optimizer.zero_grad(set_to_none=True)
        logits = network(torch.from_numpy(values).to(device), torch.from_numpy(lengths).to(device))
        loss = nn.functional.cross_entropy(logits, torch.from_numpy(labels).to(device))
        if not torch.isfinite(loss):
          raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.item() * len(batch)
      synchronize(device)
      learning_seconds = time.perf_counter() - learning_started
      train_metrics = neural_evaluate(network, data, training, batch_size, device, config)
      held_metrics = neural_evaluate(network, data, validation, batch_size, device, config) if len(validation) and not fixed_epochs else None
      row = {
        "epoch": epoch,
        "online_loss": loss_sum / len(training),
        "learning_seconds": learning_seconds,
        "train": train_metrics,
        "validation": held_metrics,
        "epoch_seconds": time.perf_counter() - epoch_started
      }
      improved = bool(fixed_epochs) or state["best"] is None or (held_metrics["macro_f1"], held_metrics["accuracy"]) > (state["best"]["validation"]["macro_f1"], state["best"]["validation"]["accuracy"])
      if improved:
        state["best"] = copy.deepcopy(row)
        state["best_model"] = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
      if not fixed_epochs:
        if held_metrics["macro_f1"] >= state["patience_reference"] + config["training"]["min_delta"]:
          state["stale"] = 0
          state["patience_reference"] = held_metrics["macro_f1"]
        else:
          state["stale"] += 1
      state["history"].append(row)
      state.update({
        "epoch": epoch,
        "seconds": previous + time.time() - started,
        "model": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng_state()
      })
      atomic_checkpoint(checkpoint, state)
      write(directory / "history.json", state["history"])
      task.update()
      task.event("epoch", epoch=epoch, train_accuracy=train_metrics["accuracy"], validation_f1=held_metrics["macro_f1"] if held_metrics else None)
      if not fixed_epochs and epoch >= config["training"]["min_epochs"] and state["stale"] >= config["training"]["patience"]:
        status = "early_stopped"
        break
    task.status = status
  chosen = directory / "best.pt"
  if fixed_epochs and len(validation):
    state["best"]["validation"] = neural_evaluate(network, data, validation, batch_size, device, config)
  atomic_checkpoint(chosen, {
    "identity": identity,
    "epoch": state["best"]["epoch"],
    "model": state["best_model"],
    "best": state["best"]
  })
  result = {
    "status": status,
    "fingerprint": fingerprint,
    "identity": identity,
    "best": state["best"],
    "completed_epoch": state["epoch"],
    "seconds": state["seconds"],
    "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
    "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    "checkpoint": str(chosen),
    "checkpoint_sha256": checksum(chosen)
  }
  result["history"] = scalar_history(state["history"])
  write(result_path, result)
  return result
