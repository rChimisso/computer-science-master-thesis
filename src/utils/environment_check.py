import importlib
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from src.utils.config import load_config
from src.utils.reproducibility import set_global_seed
from src.utils.results import append_json_result

REQUIRED_PACKAGES = {
  "numpy": "numpy",
  "scikit-learn": "sklearn",
  "tonic": "tonic",
  "reservoirpy": "reservoirpy",
  "qiskit": "qiskit"
}
"""Distribution names and their corresponding import names."""

def check_environment() -> dict[str, str]:
  """Verify imports, configuration, deterministic seeds, and result logging.

  :return: Installed versions of all required distributions.
  :raises RuntimeError: If deterministic seeding or JSON logging fails.
  """
  package_versions = {}
  for distribution, module in REQUIRED_PACKAGES.items():
    importlib.import_module(module)
    package_versions[distribution] = version(distribution)

  config = load_config()
  seed = int(config["project"]["seed"])
  first = set_global_seed(seed).random(5)
  second = set_global_seed(seed).random(5)
  if not np.array_equal(first, second):
    raise RuntimeError("Deterministic seed check failed")

  with tempfile.TemporaryDirectory() as temporary_directory:
    log_path = Path(temporary_directory) / "smoke.jsonl"
    append_json_result(log_path, {"seed": seed}, {"ok": True})
    if len(log_path.read_text(encoding="utf-8").splitlines()) != 1:
      raise RuntimeError("JSON result logging check failed")
  return package_versions

def main() -> None:
  """Print the environment report.

  :return: ``None``.
  """
  print(f"Python: {platform.python_version()}")
  for package, package_version in check_environment().items():
    print(f"{package}: {package_version}")
  print("Environment OK")

if __name__ == "__main__":
  main()
