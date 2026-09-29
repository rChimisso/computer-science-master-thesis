import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

def configure_device(seed: int, device: str = "cuda", threads: int = 4) -> torch.device:
  """Configure reproducible float32 execution on a fixed device.

  :param seed: Seed for Python, NumPy, PyTorch, and CUDA.
  :param device: Explicit device name; no silent CPU fallback.
  :param threads: Bounded PyTorch CPU thread count.
  :return: Verified execution device.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.set_num_threads(threads)
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  selected = torch.device(device)
  if selected.type == "cuda":
    if not torch.cuda.is_available():
      raise RuntimeError("CUDA requested but unavailable; run with authorized GPU access")
    if selected.index is None:
      selected = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(selected)
  return selected

def synchronize(device: torch.device) -> None:
  """Complete device operations before timing boundaries.

  :param device: Active execution device.
  """
  if device.type == "cuda":
    torch.cuda.synchronize(device)

def rng_state() -> dict[str, Any]:
  """Capture all random streams needed to replay a completed epoch boundary.

  :return: Python, NumPy, CPU, and available CUDA RNG states.
  """
  return {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
  }

def restore_rng(state: dict[str, Any]) -> None:
  """Restore recorded random streams after model and optimizer construction.

  :param state: RNG state from a trusted study checkpoint.
  """
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])
  torch.set_rng_state(state["torch"].cpu())
  if state["cuda"]:
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])

def atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
  """Persist a trusted local checkpoint without exposing partial files.

  :param path: Checkpoint destination.
  :param state: Serializable model, optimizer, protocol, and RNG state.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name + ".tmp")
  torch.save(state, temporary)
  os.replace(temporary, path)
