import os
import shutil
from pathlib import Path

import numpy as np

from src.runtime.configuration import bind, load
from src.runtime.records import read
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
