import copy
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

@dataclass(frozen=True)
class ExecutionSettings:
  """Validated operational settings recorded separately from scientific choices.

  :ivar cpu_threads: Native CPU threads per reservoir or readout task.
  :ivar gpu_threads: Aer native thread limit supported by the implementation.
  :ivar quantum_batch: Complete quantum examples per simulator call.
  :ivar crc_batch: Complete classical examples per extraction batch.
  :ivar job_seconds: Maximum duration of one quantum simulator call.
  :ivar memory_fraction: Allowed fraction of device memory.
  :ivar qrc_small: Aer device for the smaller allocations.
  :ivar qrc_large: Aer device for the largest allocation.
  :ivar crc_device: PyTorch device for classical extraction.
  :ivar crc_graphs: Whether classical extraction captures CUDA graphs.
  :ivar device: PyTorch device for neural fitting and inference.
  """

  cpu_threads: int
  gpu_threads: int
  quantum_batch: int
  crc_batch: int
  job_seconds: float
  memory_fraction: float
  qrc_small: str
  qrc_large: str
  crc_device: str
  crc_graphs: bool
  device: str

  @classmethod
  def from_config(cls, config: dict) -> "ExecutionSettings":
    """Reject unsupported operational values before planning or starting workers.

    :param config: Resolved settings with an execution section.
    :return: Validated execution policy.
    :raises ValueError: If a setting is missing, unknown or outside its supported domain.
    """
    try:
      result = cls(**config["execution"])
    except (KeyError, TypeError) as error:
      raise ValueError("Invalid execution setting fields") from error
    for name in ("cpu_threads", "gpu_threads", "quantum_batch", "crc_batch"):
      value = getattr(result, name)
      if type(value) is not int or value < 1:
        raise ValueError(f"Execution {name} must be a positive integer")
    if result.gpu_threads not in (1, 4):
      raise ValueError("Aer supports one or four native threads")
    for name in ("job_seconds", "memory_fraction"):
      value = getattr(result, name)
      if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Execution {name} must be finite and positive")
    if result.memory_fraction > 1:
      raise ValueError("Execution memory_fraction cannot exceed one")
    if result.qrc_small not in ("CPU", "GPU") or result.qrc_large not in ("CPU", "GPU"):
      raise ValueError("Aer devices must be CPU or GPU")
    if result.crc_device not in ("cpu", "cuda") or result.device not in ("cpu", "cuda"):
      raise ValueError("PyTorch execution devices must be cpu or cuda")
    if type(result.crc_graphs) is not bool or result.crc_graphs and result.crc_device != "cuda":
      raise ValueError("Classical CUDA graphs require crc_device=cuda")
    return result

def load(path: str = "configs/core.yaml") -> dict:
  """Load standalone settings without accessing datasets or past runs.

  :param path: Scientific configuration file.
  :return: Validated independent settings.
  """
  config = yaml.safe_load(Path(path).read_text())
  ExecutionSettings.from_config(config)
  if config["allocations"] != [
    [4, 2],
    [6, 2],
    [6, 4],
    [8, 2],
    [8, 4],
    [8, 6]
  ]:
    raise ValueError("The declared allocation roster must remain explicit")
  if config["readout"]["solver"] != "cholesky" or config["seeds"] != [
    42,
    44,
    46
  ]:
    raise ValueError("Unsupported core readout or seed protocol")
  conventions = {
    "input_transform": "log1p",
    "probability_gain": 0.5,
    "projection_seed": 42,
    "encoding": "amplitude-probabilities",
    "uploads": 1,
    "virtual_nodes": 1,
    "observables": [
      "x",
      "y",
      "z"
    ],
    "entanglement": "cz-ring",
    "reset": "partial",
    "quantum_precision": "complex128",
    "classical_precision": "float64",
    "crc_sparsity": 0.5,
    "summary": "segment_mean_std_final",
    "summary_ddof": 0,
    "standardization": "training-only",
    "lsqr_tolerance": 0.001,
    "lsqr_iterations": [1000, 5000]
  }
  if config["method"] != conventions:
    raise ValueError("The declared numerical conventions differ from this implementation")
  config["neural"] = yaml.safe_load(Path(config["paths"]["reference"]).read_text())
  return config

def bind(config: dict, run: Path, deadline: float = float("inf")) -> dict:
  """Bind portable scientific settings to one execution namespace.

  :param config: Standalone settings.
  :param run: Run directory.
  :param deadline: Absolute stopping time.
  :return: Independent runtime settings.
  """
  value = copy.deepcopy(config)
  from src.runtime.storage import evidence_root
  root = evidence_root(run)
  value["paths"].update({"run": str(run), "results": str(root), "evidence": str(root / "metrics")})
  value["paths"]["temporary"] = str(run / "temporary")
  value["_deadline"] = deadline
  return value
