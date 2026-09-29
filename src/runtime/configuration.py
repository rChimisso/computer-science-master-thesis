import copy
from pathlib import Path

import yaml

def load(path: str = "configs/core.yaml") -> dict:
  """Load standalone settings without accessing datasets or past runs.

  :param path: Scientific configuration file.
  :return: Validated independent settings.
  """
  config = yaml.safe_load(Path(path).read_text())
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
