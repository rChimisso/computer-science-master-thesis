import csv
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data.indexed import study_folds
from src.reporting.evidence import collect
from src.reporting.products import contrasts, export
from src.reporting.study import report
from src.runtime.records import checksum, read, write
from src.runtime.storage import cleanup
from src.studies.design import group

from tests.support import EvidenceCase

class ReportTests(EvidenceCase):
  """Verify offline regeneration, retained models and mapping source semantics."""

  def test_offline_publication(self):
    """Replay reports from evidence and preserve previous publication on render failure."""
    with patch("src.data.datasets.load", side_effect=AssertionError("Dataset access")), patch("src.runtime.pipelines.predict", side_effect=AssertionError("Inference")):
      first = report(self.run)
      self.assertEqual(first["development_status"], "completed")
      self.assertEqual(first["official_status"], "pending")
      self.assertEqual(first["reporting_source"]["src/models/protocol.py"], checksum("src/models/protocol.py"))
      self.assertEqual(first["reporting_source"]["src/models/protocol.py"], checksum("src/models/protocol.py"))
      output = self.run / "reports"
      hashes = {str(path.relative_to(output)): checksum(path) for path in output.rglob("*.csv")}
      shutil.rmtree(output)
      report(self.run)
      self.assertEqual(hashes, {str(path.relative_to(output)): checksum(path) for path in output.rglob("*.csv")})
      (output / "stale.csv").write_text("obsolete\n")
      report(self.run)
      self.assertFalse((output / "stale.csv").exists())
      self.assertTrue((output / "resources/tables/reservoir-structure.csv").exists())
      self.assertTrue((output / "data/tables/dataset-settings.csv").exists())
      self.assertFalse(list(output.rglob("*.tex")))
      self.assertFalse(list(output.rglob("*.pdf")))
      self.assertIn("macro_f1_seed_sd", (output / "performance/tables/performance.csv").read_text())
      self.assertFalse((output / "findings").exists())
      self.assertFalse(list(output.rglob("documents")))
      self.assertTrue((output / "comparisons/README.md").exists())
      self.assertIn("comparisons/README.md", read(output / "products.json")["products"])
      relocated = self.root / "evidence-only"
      shutil.copytree(self.run / "evidence", relocated / "evidence")
      for name in ("manifest.json", "frozen.json"):
        shutil.copy2(self.run / name, relocated / name)
      report(relocated)
      self.assertEqual(hashes, {str(path.relative_to(relocated / "reports")): checksum(path) for path in (relocated / "reports").rglob("*.csv")})
    output = self.run / "reports"
    output.mkdir(exist_ok=True)
    (output / "keep.txt").write_text("previous successful publication")
    with patch("src.reporting.study.render", side_effect=RuntimeError("render failure")):
      with self.assertRaises(RuntimeError):
        report(self.run)
    self.assertTrue((output / "keep.txt").exists())
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      rows = [{
        'dataset': 'shd',
        'phase': 'development',
        'role': 'comparison:tuned:q8-i6-qrc',
        'population': 'held',
        'seed': seed,
        'fold': fold,
        'accuracy': 0.8,
        'macro_f1': 0.79,
        'macro_precision': 0.81,
        'macro_recall': 0.8
      } for seed, fold in ((42, 0), (44, 1))]
      export(root, 'performance', 'source', rows)
      self.assertFalse(list(root.rglob('*.tex')))
      with (root / 'performance/tables/source.csv').open() as stream:
        saved = list(csv.DictReader(stream))
      self.assertEqual([int(row['seed']) for row in saved], [42, 44])
      self.assertEqual([int(row['fold']) for row in saved], [0, 1])
      self.assertTrue(all(row['dataset'] == 'shd' and row['phase'] == 'development' for row in saved))

  def test_final_pipeline_replay_and_safe_cleanup(self):
    """Verified inference packages survive cleanup and failed work is retained."""
    from src.runtime.pipelines import export, predict
    from src.studies.experiments import experiment
    spec = self.specs["comparison:fixed:q4-i2-crc"]
    fold = {"fold": "confirmation", "train": self.data.development.tolist(), "validation": self.data.confirmation.tolist()}
    job = experiment(self.data, fold, spec, 42, self.config, [0.1], "confirmation")
    package = export(self.data, self.data.development, spec, 42, "confirmation", job, self.config)
    actual = predict(package, self.data, self.data.confirmation, self.config)
    np.testing.assert_array_equal(actual, job["rows"][0]["predictions"])
    write(self.run / "evidence/pipelines.json", {"confirmation:shd:test:42": package})
    (self.run / "evidence/metrics/confirmation.json").unlink()
    write(self.run / "progress.json", {"status": "failed"})
    self.assertEqual(cleanup(self.run, {"development_status": "completed"})["status"], "deferred")
    self.assertTrue((self.run / "temporary").exists())
    write(self.run / "progress.json", {"status": "completed"})
    self.assertEqual(cleanup(self.run, {"development_status": "completed"})["status"], "completed")
    self.assertFalse((self.run / "temporary").exists())
    self.assertTrue(Path(package["directory"]).exists())
    self.assertEqual(experiment(self.data, fold, spec, 42, self.config, [0.1], "confirmation")["status"], "completed")

  def test_mapping_evidence_and_views(self):
    """Preserve mapping pairs, source populations and projection-seed semantics."""
    with self.subTest(case='mapping_robustness_and_explicit_pairs'):
      self.manifest["studies"].append("mapping")
      write(self.run / "manifest.json", self.manifest)
      selections = read(self.run / "evidence/metrics/selections.json")
      paired = {"groups": {}, "status": "completed"}
      robustness = {"groups": {}, "status": "partial"}
      for adapter in ("legacy", "pooling"):
        for family in ("qrc", "crc"):
          role = f"mapping-paired:i2:{adapter}:{family}"
          self.specs[role] = self.specs[f"comparison:fixed:q4-i2-{family}"] | {"adapter": adapter}
          paths = [self.job(role, seed, fold, "development") for seed in (42, 44, 46) for fold in study_folds(self.data, self.config)]
          chosen = {"spec": self.specs[role], "jobs": paths, "selection": {"lambda": 0.1, "mean_macro_f1": 1.0}}
          paired["groups"][role] = {"declaration": group("mapping-paired", role.split(":", 1)[1], [self.specs[role]], final=False), "selected": chosen, "screen": [], "replication": {}}
      for projection_seed in (42, 44, 46):
        role = f"mapping-robustness:i2:legacy:qrc:p{projection_seed}"
        self.specs[role] = self.specs["comparison:fixed:q4-i2-qrc"] | {"adapter": "legacy", "projection_seed": projection_seed}
        paths = [self.job(role, 42, fold, "development") for fold in study_folds(self.data, self.config)]
        chosen = {"spec": self.specs[role], "jobs": paths, "selection": {"lambda": 0.1, "mean_macro_f1": 1.0}}
        robustness["groups"][role] = {"declaration": group("mapping-robustness", role.split(":", 1)[1], [self.specs[role]], final=False), "selected": chosen, "screen": [], "replication": {}}
      selections["shd"]["mapping"] = {"advanced": {"2": "pooling"}, "paired": paired, "projection_robustness": robustness}
      write(self.run / "evidence/metrics/selections.json", selections)
      evidence = collect(self.run)
      rows = [row for row in evidence["cells"] if row["phase"] == "mapping-robustness" and row["population"] == "held"]
      self.assertEqual(len(rows), 12)
      self.assertEqual({row["seed"] for row in rows}, {42})
      individual, _, _ = contrasts(evidence)
      self.assertTrue(any(row["contrast"] == "mapping-QRC-minus-CRC" for row in individual))
    with self.subTest(case='mapping_views_keep_dataset_specific_comparisons_and_seed_semantics'):
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
      self.addCleanup(plt.close, 'all')
      from src.reporting.figures import Figures, mapping_performance
      with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        for dataset, adapter in (('shd', 'pooling'), ('dvs', 'pca')):
          for width in (2, 4):
            for mode in ('legacy', adapter):
              for model in ('qrc', 'crc', 'projection'):
                rows.append({'dataset': dataset, 'phase': 'mapping-development', 'role': f'mapping-paired:i{width}:{mode}:{model}', 'macro_f1': 0.6, 'macro_f1_seed_sd': 0.02})
            for seed in (42, 44, 46):
              rows.append({'dataset': dataset, 'phase': 'mapping-robustness', 'role': f'mapping-robustness:i{width}:legacy:qrc:p{seed}', 'macro_f1': 0.5, 'macro_f1_seed_sd': ''})
        export(root, 'performance', 'performance', rows)
        writer = Figures(root)
        source = writer.rows('performance/tables/performance.csv')
        with patch.object(writer, 'publish') as publish:
          mapping_performance(writer, source)
        self.assertEqual(publish.call_count, 2)
        for call in publish.call_args_list:
          _, name, figure, used, caption, recipe = call.args
          self.assertEqual({row['_row'] for row in used}, {row['_row'] for row in source if row['phase'] == recipe['phase']})
          self.assertEqual(recipe['seed_axis'], 'projection' if recipe['phase'] == 'mapping-robustness' else None)
          self.assertEqual({panel[0] for panel in recipe['panels']}, {row['dataset'] for row in used})
          points = [container.lines[0].get_ydata() for axis in figure.axes for container in axis.containers]
          np.testing.assert_allclose(np.concatenate(points).astype(float), 0.5 if recipe['phase'] == 'mapping-robustness' else 0.6)
          if recipe['phase'] == 'mapping-robustness':
            self.assertIn('reservoir initialization fixed at $42$', caption)
            self.assertTrue(all(not container.has_yerr for axis in figure.axes for container in axis.containers))
        plt.close('all')
    with self.subTest(case='mapping_effects_separate_contrasts_and_keep_all_pairs'):
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
      self.addCleanup(plt.close, 'all')
      from src.reporting.figures import Figures, mapping_effects
      with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        for dataset, mode in (('shd', 'pooling'), ('dvs', 'pca')):
          for contrast, right in (('mapping-QRC-minus-CRC', f'mapping-paired:i2:{mode}:crc'), ('mapping-minus-anchor', 'mapping-paired:i2:legacy:qrc')):
            rows.append({'dataset': dataset, 'contrast': contrast, 'left': f'mapping-paired:i2:{mode}:qrc', 'right': right, 'macro_f1': 0.02, 'macro_f1_seed_sd': 0.01})
        export(root, 'comparisons', 'paired-differences', rows)
        writer = Figures(root)
        source = writer.rows('comparisons/tables/paired-differences.csv')
        with patch.object(writer, 'publish') as publish:
          mapping_effects(writer, source, 'mapping-development')
        figure = publish.call_args.args[2]
        self.assertEqual(publish.call_args.args[3], source)
        recipe = publish.call_args.args[5]
        self.assertEqual(set(recipe['panels']), {(row['contrast'], row['dataset']) for row in source})
        points = [container.lines[0].get_xdata() for axis in figure.axes for container in axis.containers]
        np.testing.assert_allclose(np.concatenate(points).astype(float), 0.02)
        plt.close('all')
