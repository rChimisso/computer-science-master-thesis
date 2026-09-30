import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np

from src.runtime.configuration import bind, load
from src.runtime.records import checksum, read
from src.runtime.validation import fixture

from tests.support import SyntheticCase

class WorkflowTests(SyntheticCase):
  """Exercise core execution and optional model inference packages."""

  def test_complete_synthetic_workflow(self):
    """Require serial/parallel selection, prediction and report-value agreement."""
    from src.runtime.validation import smoke
    from src.runtime.records import digest
    from threadpoolctl import threadpool_limits
    snapshots = []
    for parallel in (False, True):
      with threadpool_limits(limits=load()['execution']['cpu_threads']):
        result = smoke(gpu=False, parallel=parallel)
      self.assertEqual(result['status'], 'completed')
      self.assertGreater(result['regenerated_tables'], 0)
      self.assertEqual(result['cleanup'], 'completed')
      root = Path(result['run'])
      self.addCleanup(shutil.rmtree, root)
      frozen = read(root / 'frozen.json')['manifest']
      decisions = {name: {role: {'spec': entry['spec'], 'selection': entry['selection']} for role, entry in entries.items()} for name, entries in frozen['entries'].items()}
      predictions = {}
      for path in (root / 'evidence/metrics/jobs').glob('*/result.json'):
        job = read(path)
        identity = {key: value for key, value in job['identity'].items() if key not in ('source', 'fit')}
        predictions[digest(identity)] = [(row.get('lambda'), row.get('train_predictions'), row.get('predictions')) for row in job['rows']]
      tables = {name: (root / 'reports' / name).read_text() for name in ('performance/tables/performance.csv', 'comparisons/tables/paired-differences.csv')}
      self.assertTrue(all(tables.values()))
      snapshots.append({'decisions': decisions, 'predictions': predictions, 'tables': tables})
      if parallel:
        import tarfile
        from unittest.mock import patch
        from src.runtime.archive import create, verify
        from src.runtime.pipelines import predict
        archive = self.root / "evidence.tar.gz"
        with patch("src.data.datasets.load", side_effect=AssertionError("Archive opened a dataset")):
          packed = create(root, archive)
        self.assertEqual(packed["sha256"], verify(archive)["sha256"])
        restored = self.root / "restored"
        with tarfile.open(archive) as stream:
          stream.extractall(restored, filter="data")
        self.assertEqual((root / "frozen.json").read_bytes(), (restored / "run/frozen.json").read_bytes())
        hidden = root.with_name(root.name + "-unavailable")
        root.rename(hidden)
        try:
          output = restored / "run/reports"
          archived_tables = {str(path.relative_to(output)): checksum(path) for path in output.rglob("*.csv")}
          shutil.rmtree(output)
          environment = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
          environment.update({"THESIS_PROGRESS": "off", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
          process = subprocess.run(
            [sys.executable, "-m", "src", "report", "--run", "../run", "--progress", "off"],
            cwd=restored / "reader", env=environment, capture_output=True, text=True, timeout=120
          )
          self.assertEqual(process.returncode, 0, process.stderr)
          replayed = json.loads(process.stdout)
          self.assertEqual(replayed["status"], "completed")
          self.assertEqual(archived_tables, {str(path.relative_to(output)): checksum(path) for path in output.rglob("*.csv")})
          self.assertEqual(replayed["reporting_source"]["src/models/protocol.py"], checksum(restored / "reader/src/models/protocol.py"))
          self.assertEqual(tables, {name: (restored / "run/reports" / name).read_text() for name in tables})
          packages = read(restored / "pipelines.json")
          package = next(value for key, value in packages.items() if key.startswith("confirmation:shd:comparison:projection:"))
          package["directory"] = str(restored / package["directory"])
          data = fixture("shd")
          actual = predict(package, data, data.development[:3], bind(load(), restored / "run"))
          self.assertEqual(actual.tolist(), package["verified_predictions"]["predictions"])
        finally:
          hidden.rename(root)
    self.assertEqual(snapshots[0], snapshots[1])

  def test_optional_models_fit_and_replay(self):
    """Fit every optional architecture and retain only replayable inference state."""
    from src.runtime.pipelines import export, predict
    from src.studies.experiments import experiment
    devices = ("cpu", "cuda") if os.environ.get("THESIS_VALIDATION_GPU") == "1" else ("cpu",)
    for device in devices:
      for name in ("shd", "dvs"):
        data = fixture(name)
        config = bind(load(), self.root / name / device)
        config["paths"]["cache"] = str(self.root / "shared")
        config["execution"].update({"device": device, "crc_device": device, "cpu_threads": 1})
        config["training"].update({"max_epochs": 1, "min_epochs": 1, "batch_size": 8, "cpu_threads": 1})
        config["_identity"] = "optional-fixture-" + name + device
        fold = {"fold": "confirmation", "train": data.development.tolist(), "validation": data.confirmation.tolist()}
        specs = [config["supporting"]["practical"][name]["large_crc"] | {"model": "crc", "inputs": 0, "adapter": "identity", "solver": "lsqr", "view": "all"}]
        specs += [{"model": model, "recipe": "baseline", "fixed_epochs": 1} for model in ("lstm", "transformer")]
        for spec in specs:
          job = experiment(data, fold, spec, 42, config, [0.1], "confirmation")
          package = export(data, data.development, spec, 42, "confirmation", job, config)
          actual = predict(package, data, data.confirmation, config)
          np.testing.assert_array_equal(actual, job["rows"][0]["predictions"])
          if spec['model'] in ('lstm', 'transformer'):
            with np.load(Path(package['directory']) / 'weights.npz') as stored:
              self.assertFalse(any('optimizer' in key for key in stored.files))
            from src.runtime.records import write
            request = self.root / "fresh-inference.json"
            write(request, {"package": package, "config": {key: value for key, value in config.items() if not key.startswith("_")}, "dataset": name})
            script = "\n".join([
              "import json, sys, torch",
              "from src.runtime.records import read",
              "from src.runtime.pipelines import predict",
              "from src.runtime.validation import fixture",
              "request = read(sys.argv[1])",
              "data = fixture(request['dataset'])",
              "predictions = predict(request['package'], data, data.confirmation, request['config'])",
              "print(json.dumps({'predictions': predictions.tolist(), 'deterministic': torch.are_deterministic_algorithms_enabled(), "
              "'cudnn_deterministic': torch.backends.cudnn.deterministic, 'tf32': torch.backends.cudnn.allow_tf32, "
              "'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'threads': torch.get_num_threads()}))"
            ])
            process = subprocess.run([sys.executable, "-c", script, str(request)], capture_output=True, text=True, check=True, timeout=120, env=os.environ | {"THESIS_PROGRESS": "off"})
            fresh = json.loads(process.stdout)
            self.assertEqual(fresh["predictions"], actual.tolist())
            self.assertTrue(fresh["deterministic"])
            self.assertTrue(fresh["cudnn_deterministic"])
            self.assertFalse(fresh["tf32"])
            self.assertFalse(fresh["matmul_tf32"])
            self.assertEqual(fresh["threads"], config["training"]["cpu_threads"])
            self.assertTrue(job['neural']['history'])
            self.assertNotIn('predictions', job['neural']['history'][0]['train'])

  def test_parallel_neural_calibration_and_rng(self):
    """Keep representation-specific calibration separate and replay seeded neural fits."""
    import copy
    from src.runtime.resources import policy
    from src.runtime.tasks import Scheduler
    from src.studies.experiments import experiment
    config = copy.deepcopy(self.config)
    config['execution'].update({'cpu_threads': 1, 'device': 'cuda' if os.environ.get('THESIS_VALIDATION_GPU') == '1' else 'cpu'})
    config['training'].update({'max_epochs': 1, 'min_epochs': 1, 'batch_size': 8})
    changed = copy.deepcopy(self.data)
    changed.fingerprint += ':longer-fixture'
    changed.sequences = [np.concatenate((values, values), axis=0) for values in changed.sequences]
    fold = {'fold': 'confirmation', 'train': self.data.development.tolist(), 'validation': self.data.confirmation.tolist()}
    cases = [(data, {'model': 'lstm', 'recipe': recipe, 'fixed_epochs': 1}) for data in (self.data, changed) for recipe in ('baseline', 'dropout')]
    expected = [experiment(data, fold, spec, 42, bind(config, self.root / 'serial'), phase='confirmation') for data, spec in cases]
    with Scheduler(self.root, policy(2, 4, 1), config) as scheduler:
      futures = [scheduler.experiment(data, fold, spec, 42, config, phase='confirmation') for data, spec in cases]
      actual = [future.result(timeout=120) for future in futures]
      benchmarks = [row for row in scheduler.records.values() if row['operation'] == 'benchmark']
      self.assertEqual(len(benchmarks), 2)
      self.assertTrue(all(row['resources']['exclusive'] for row in benchmarks))
    paths = list((self.root / 'evidence/metrics/neural-calibration').glob('*/shd/benchmark-lstm.json'))
    self.assertEqual(len(paths), 2)
    for left, right in zip(expected, actual):
      self.assertEqual(left['rows'][0]['predictions'], right['rows'][0]['predictions'])
      self.assertEqual(left['rows'][0]['train_predictions'], right['rows'][0]['train_predictions'])
