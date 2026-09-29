import datetime
from pathlib import Path

import numpy as np

from src.runtime.configuration import bind, load
from src.runtime.records import digest, write

def fixture(name: str):
  """Create a labelled synthetic training fixture with all four held-group folds.

  :param name: SHD or DVS fixture geometry.
  :return: Small ragged source explicitly marked as synthetic.
  """
  from src.data.indexed import StudyData
  groups = [
    0,
    1,
    2,
    6,
    3,
    11,
    7,
    9,
    8,
    10
  ] if name == "shd" else list(range(1, 13))
  sequences, labels, people, identities = [], [], [], []
  for person in groups:
    for label in range(3):
      rng = np.random.default_rng(person * 100 + label)
      values = rng.poisson(0.2, (3 + (person + label) % 4, 32 if name == "shd" else 128)).astype(np.uint16)
      values[:, label * 4:label * 4 + 4] += 2
      sequences.append(values)
      labels.append(label)
      people.append(person)
      identities.append(f"{name}-train:synthetic-{person}-{label}")
  people = np.asarray(people)
  held = people < 2 if name == "shd" else people <= 4
  return StudyData(name, sequences, np.asarray(labels), people, identities, np.flatnonzero(~held), np.flatnonzero(held), 3, 10000 if name == "shd" else 50000, digest([name, identities]), [""] * len(labels))

def smoke(gpu: bool = False, parallel: bool = False) -> dict:
  """Exercise a small complete workflow and synthetic official-stage fixtures.

  :param gpu: Include explicit device agreement after the CPU workflow.
  :param parallel: Exercise the spawned task scheduler with synthetic prepared data.
  :return: Correctness evidence, never scientific performance results.
  """
  import copy
  import shutil
  from unittest.mock import patch
  from src.reporting.study import report
  from src.runtime.records import checksum
  from src.runtime.storage import cleanup, seal
  from src.studies.design import group
  from src.workflow import evaluate_frozen, run
  name = datetime.datetime.now(datetime.timezone.utc).strftime("validation-%Y%m%dT%H%M%S%fZ")
  root = Path("output/runs") / name
  config = load()
  datasets = {dataset: fixture(dataset) for dataset in ("shd", "dvs")}
  declarations = [
    group("comparison", "fixed:q4-i2-qrc", [{"model": "qrc", "qubits": 4, "inputs": 2, "depth": 1, "view": "all"}]),
    group("comparison", "fixed:q4-i2-crc", [{"model": "crc", "size": 12, "inputs": 2, "radius": 0.9, "scale": 0.03, "tau": 20, "view": "all"}]),
    group("comparison", "projection:i2", [{"model": "projection", "inputs": 2, "view": "all"}])
  ]
  request = {
    "config": config,
    "profile": "core",
    "studies": ["comparison", "generalization"],
    "datasets": list(datasets),
    "models": [],
    "seeds": config["seeds"],
    "resources": False,
    "purpose": "Synthetic correctness fixture, not thesis results",
    'official_test': False
  }
  from contextlib import ExitStack
  import concurrent.futures
  from src.runtime.tasks import Scheduler
  from src.runtime.resources import policy
  manager = ExitStack()
  scheduler = manager.enter_context(Scheduler(root, policy(2, 8, 1), bind(config, root))) if parallel else None
  if scheduler:
    original_submit = scheduler.submit

    def submit(operation, payload, identity, dependencies=()):
      """Supply explicitly synthetic descriptors while executing other tasks normally.

      :param operation: Task operation.
      :param payload: Original task arguments.
      :param identity: Stable task identity.
      :param dependencies: Declared dependencies.
      :return: Prepared synthetic descriptor or a real worker future.
      """
      if operation == 'data':
        future = concurrent.futures.Future()
        future.set_result(datasets[payload['name']])
        return future
      return original_submit(operation, payload, identity, dependencies)

    manager.enter_context(patch.object(scheduler, 'submit', side_effect=submit))
  try:
    with patch('src.workflow.groups', return_value=declarations), patch('src.data.datasets.load', side_effect=lambda name, *args, **kwargs: datasets[name]):
      development = run(request, name)
  except BaseException:
    manager.close()
    raise
  held = {}
  for dataset, data in datasets.items():
    test = copy.deepcopy(data)
    test.identities = [value.replace("-train:", "-test:") for value in data.identities]
    test.groups = data.groups.copy()
    test.groups[-3:] = 99
    test.fingerprint = digest(test.identities)
    held[dataset] = test

  def load_fixture(dataset: str, settings: dict, split: str = "train", official: bool = False):
    """Supply synthetic held records exclusively through the explicit test gate.

    :param dataset: Requested synthetic dataset.
    :param settings: Frozen execution settings.
    :param split: Requested source partition.
    :param official: Explicit test access flag.
    :return: Synthetic training or evaluation data.
    """
    if split == "test" and not official:
      raise ValueError("Synthetic evaluation requires explicit access")
    return held[dataset] if split == "test" else datasets[dataset]

  try:
    with patch('src.data.datasets.load', side_effect=load_fixture):
      final = evaluate_frozen(root / 'frozen.json', True)
  finally:
    manager.close()
  seal(root)
  rendered = report(root)
  cleanup_record = cleanup(root, rendered)
  before = {str(path.relative_to(root / "reports")): checksum(path) for path in (root / "reports").rglob("*.csv")}
  shutil.rmtree(root / "reports")
  with patch("src.data.datasets.load", side_effect=AssertionError("Report loaded datasets")), patch("src.runtime.pipelines.predict", side_effect=AssertionError("Report executed a model")):
    regenerated = report(root)
  after = {str(path.relative_to(root / "reports")): checksum(path) for path in (root / "reports").rglob("*.csv")}
  if before != after or regenerated["status"] != "completed":
    raise ValueError("Incomplete or inconsistent independent report regeneration")
  result = {
    "status": "completed",
    "purpose": "Synthetic correctness only; no official dataset was opened",
    "run": str(root),
    "development": development,
    "official_fixture_jobs": len(final["jobs"]),
    "cleanup": cleanup_record["status"],
    "regenerated_tables": len(after)
  }
  if gpu:
    result["gpu"] = gpu_agreement(bind(config, root))
  write(root / "validation.json", result)
  return result

def gpu_agreement(config: dict) -> dict:
  """Check actual CPU/GPU quantum observations and classical summaries.

  :param config: Execution settings.
  :return: Maximum observed backend differences.
  """
  from src.features.observations import classical_reference
  from src.models.quantum_execution import AerReservoir
  from src.models.streaming_crc import StreamingCRC
  rng = np.random.default_rng(42)
  sequences = [rng.uniform(size=(length, 4)) for length in (3, 8)]
  spec = {
    "qubits": 8,
    "inputs": 4,
    "depth": 1,
    "seed": 42,
    "reset_mode": "partial",
    "observables": [
      "x",
      "y",
      "z"
    ]
  }
  observed = []
  for device in ("CPU", "GPU"):
    reservoir = AerReservoir(spec, device)
    try:
      observed.append(reservoir.transform(sequences)["values"])
    finally:
      reservoir.close()
  quantum = max(float(np.max(np.abs(left - right))) for left, right in zip(*observed))
  if quantum > 1e-10:
    raise ValueError("Quantum backend agreement failed")
  data = fixture("shd")
  spec = {
    "size": 18,
    "inputs": 4,
    "radius": 0.9,
    "scale": 0.03,
    "tau": 20
  }
  reference = classical_reference(data, spec, 42)
  cpu = StreamingCRC(reference, "cpu").extract(sequences, 4, ("all",))[0]["all"]
  errors = {}
  for graphs in (False, True):
    executor = StreamingCRC(reference, "cuda", graphs)
    actual = executor.extract(sequences, 4, ("all",))[0]["all"]
    np.testing.assert_allclose(actual, cpu, atol=1e-10, rtol=1e-10)
    changed = [value[::-1].copy() for value in sequences]
    replay = executor.extract(changed, 4, ("all",))[0]["all"]
    expected = StreamingCRC(reference, "cpu").extract(changed, 4, ("all",))[0]["all"]
    np.testing.assert_allclose(replay, expected, atol=1e-10, rtol=1e-10)
    errors["graphs" if graphs else "eager"] = float(np.max(np.abs(actual - cpu)))
  return {"quantum_maximum_error": quantum, "classical_maximum_errors": errors}

def validate(run_smoke: bool = False, gpu: bool = False, suite: str = "quick", list_only: bool = False) -> dict:
  """Run an explicit verification scope, or list it without executing tests.

  :param run_smoke: Include the end-to-end smoke experiment.
  :param gpu: Include real device agreement checks.
  :param suite: Quick, extended or all suites.
  :param list_only: Inspect the suite without importing tests or executing work.
  :return: Validation report.
  """
  from src.runtime.test_suites import describe, execute, gpu_preflight
  if list_only:
    return describe(suite) | {"gpu_requested": gpu, "additional_smoke": run_smoke and suite not in ("extended", "all")}
  result = {"status": "running", "suite": suite, "gpu_requested": gpu}
  destination = Path("output/runs") / datetime.datetime.now(datetime.timezone.utc).strftime("checks-%Y%m%dT%H%M%S%fZ") / "validation.json"
  write(destination, result)
  try:
    if gpu:
      result["gpu"] = gpu_preflight(destination.parent)
    result.update(execute(suite, gpu))
    if result["status"] != "passed":
      raise RuntimeError("Validation failed; see the selected suite's unittest output")
    if run_smoke and suite not in ("extended", "all"):
      result["smoke"] = smoke(False)
    result["record"] = str(destination)
  except BaseException as error:
    result.update({"status": "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed", "error": repr(error)})
    raise
  finally:
    write(destination, result)
  return result
