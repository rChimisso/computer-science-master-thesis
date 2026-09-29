import importlib.metadata
import importlib.util
from pathlib import Path

from src.runtime.records import checksum

def identity(model: str, config: dict) -> dict:
  """Fingerprint numerical extraction independently of run names and reporting settings.

  :param model: Extraction family.
  :param config: Numerical and device policy.
  :return: Content-based compatible-cache identity.
  """
  paths = [
    "src/data/indexed.py",
    "src/features/adapters.py",
    "src/features/extraction.py",
    "src/features/observations.py",
    "src/features/summary.py"
  ]
  paths += ["src/models/quantum_execution.py"] if model == "qrc" else [
    "src/models/classical_reservoir.py",
    "src/models/streaming_crc.py",
    "src/models/torch_classical_reservoir.py"
  ] if model == "crc" else []
  packages = ["numpy", "scipy"] + (["qiskit", "qiskit-aer"] if model == "qrc" else ["torch"] if model == "crc" else [])
  execution_keys = ("qrc_small", "qrc_large", "gpu_threads", "quantum_batch") if model == "qrc" else ("crc_device", "crc_graphs", "cpu_threads", "crc_batch") if model == "crc" else ()
  native = {}
  if model == "qrc":
    package = importlib.util.find_spec("qiskit_aer")
    root = Path(package.origin).parent
    native = {str(path.relative_to(root)): checksum(path) for path in root.rglob("*.so")}
    native.update({str(path): checksum(path) for path in Path("environment/local/native").glob("*.so.*") if path.is_file() and not path.is_symlink()})
  return {
    "schema": 1,
    "source": {name: checksum(name) for name in paths},
    "packages": {name: importlib.metadata.version(name) for name in packages},
    "native": native,
    "method": config["method"],
    "execution": {key: config["execution"][key] for key in execution_keys}
  }

def moments(values) -> dict:
  """Collect mergeable valid-bin statistics without retaining trajectories.

  :param values: Time-coordinate array.
  :return: Counts, moments, ranges and declared near-boundary counts.
  """
  import numpy as np
  values = np.asarray(values, dtype=np.float64)
  return {
    "count": len(values),
    "sum": values.sum(axis=0).tolist(),
    "squares": (values * values).sum(axis=0).tolist(),
    "minimum": values.min(axis=0).tolist(),
    "maximum": values.max(axis=0).tolist(),
    "probability_boundary": ((values < 0.01) | (values > 0.99)).sum(axis=0).tolist(),
    "absolute_boundary": (np.abs(values) > 0.99).sum(axis=0).tolist()
  }

def merge(left: dict, right: dict) -> dict:
  """Merge independent sufficient-statistic records with equal coordinate ordering.

  :param left: Previous accumulator or an empty object.
  :param right: Next valid-bin record.
  :return: Combined accumulator.
  """
  import numpy as np
  if not left:
    return right
  result = {"count": left["count"] + right["count"]}
  for key in ("sum", "squares", "probability_boundary", "absolute_boundary"):
    result[key] = (np.asarray(left[key]) + np.asarray(right[key])).tolist()
  result["minimum"] = np.minimum(left["minimum"], right["minimum"]).tolist()
  result["maximum"] = np.maximum(left["maximum"], right["maximum"]).tolist()
  return result
