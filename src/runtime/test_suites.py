import ast
import os
import subprocess
import sys
import time
from pathlib import Path

def describe(suite: str) -> dict:
  """List test methods statically without importing tests or touching datasets/devices.

  :param suite: Quick, extended or all checks.
  :return: Explicit inventory and unmeasured planning ranges.
  """
  definitions = {
    "quick": {"paths": sorted(Path("tests/quick").glob("test_*.py")), "minutes": [1, 5], "retention": "permanent scientific integrity"},
    "extended": {"paths": sorted(Path("tests/extended").glob("test_*.py")), "minutes": [8, 25], "retention": "explicit integration and numerical checks"}
  }
  if suite not in (*definitions, "all"):
    raise ValueError("Unknown validation suite")
  result = {}
  for name in definitions if suite == "all" else (suite,):
    definition = definitions[name]
    tests = []
    modules = []
    for path in definition["paths"]:
      module = ".".join(path.with_suffix("").parts)
      modules.append(module)
      tree = ast.parse(path.read_text())
      for node in tree.body:
        if isinstance(node, ast.ClassDef):
          tests.extend(f"{module}.{node.name}.{method.name}" for method in node.body if isinstance(method, ast.FunctionDef) and method.name.startswith("test_"))
    if not tests:
      raise ValueError(f"Empty or missing test suite: {name}")
    result[name] = {
      "modules": modules,
      "tests": tests,
      "count": len(tests),
      "retention": definition["retention"],
      "planning_minutes": definition["minutes"]
    }
  return {"suites": result, "estimates": "Planning allowances, not fresh measurements; see tests/README.md", "tests_executed": False}

def gpu_preflight(destination: Path) -> dict:
  """Run device agreement in a short-lived process to release its CUDA context.

  :param destination: Unique validation namespace receiving the GPU measurement.
  :return: Verified CPU/GPU agreement measurements from the successful child.
  """
  from src.runtime.records import read
  record = destination / 'gpu.json'
  script = "\n".join([
    'import sys',
    'from pathlib import Path',
    'from src.runtime.configuration import bind, load',
    'from src.runtime.records import write',
    'from src.runtime.validation import gpu_agreement',
    'path = Path(sys.argv[1])',
    'write(path, gpu_agreement(bind(load(), path.parent)))'
  ])
  subprocess.run([sys.executable, '-c', script, str(record)], check=True)
  return read(record)

def execute(suite: str, gpu: bool) -> dict:
  """Run only explicitly selected suites and record elapsed time and failure status.

  :param suite: Explicit validation scope.
  :param gpu: Allow actual GPU checks; otherwise hide CUDA devices in children.
  :return: Completed per-suite process results, with no claim about skipped tests.
  """
  inventory = describe(suite)
  environment = dict(os.environ)
  environment["THESIS_VALIDATION_GPU"] = "1" if gpu else "0"
  if not gpu:
    environment["CUDA_VISIBLE_DEVICES"] = ""
  results = {}
  for name, definition in inventory["suites"].items():
    started = time.perf_counter()
    process = subprocess.run([sys.executable, "-m", "unittest", *definition["modules"], "-v"], env=environment)
    results[name] = {
      "exit_code": process.returncode,
      "seconds": time.perf_counter() - started,
      "declared_tests": definition["count"],
      "status": "passed" if process.returncode == 0 else "failed",
      "skips": "Consult unittest output; declared count does not imply every test executed"
    }
    if process.returncode:
      break
  return {"suites": results, "status": "passed" if all(row["exit_code"] == 0 for row in results.values()) else "failed", "gpu_requested": gpu}
