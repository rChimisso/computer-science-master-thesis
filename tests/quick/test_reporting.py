import copy
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.reporting.evidence import collect
from src.reporting.products import aggregate, contrasts, export
from src.reporting.study import report
from src.runtime.records import read, write
from src.runtime.storage import save_job, seal

from tests.support import EvidenceCase

class EvidenceTests(EvidenceCase):
  """Protect comparison eligibility and compact evidence integrity."""

  def test_report_eligibility(self):
    """Reject incomplete, failed, inconsistent or mismatched evidence cells."""
    with self.subTest(case='test_incomplete_official_roster_is_not_completed_by_a_pair'):
      self.reset_case()
      evidence = collect(self.run)
      evidence["coverage"] = [row for row in evidence["coverage"] if row["phase"] != "official"]
      evidence["coverage"] += [
        {"dataset": "all", "phase": "official", "role": "all", "status": "incomplete"},
        {"dataset": "shd", "phase": "official", "role": "completed-pair", "status": "completed"}
      ]
      with patch("src.reporting.study.collect", return_value=evidence), patch("src.reporting.study.render", return_value={"captions": {}, "notes": {}}):
        result = report(self.run)
      self.assertEqual(result["status"], "partial")
      self.assertEqual(result["official_status"], "pending")
    with self.subTest(case='test_official_fit_requires_all_frozen_training_members'):
      self.reset_case()
      index = read(self.run / "evidence/metrics/confirmation.json")
      entry = index["jobs"][0]
      job = read(entry["path"])
      job["identity"]["phase"] = "official"
      path = self.run / "evidence/metrics/official-job.json"
      save_job(path, job, self.config)
      entry["path"] = str(path)
      index["jobs"] = [entry]
      write(self.run / "evidence/metrics/official.json", index)
      with self.assertRaisesRegex(ValueError, "official-training population"):
        collect(self.run)
    with self.subTest(case='test_failed_child_and_frozen_changes_rejected'):
      self.reset_case()
      index = read(self.run / "evidence/metrics/confirmation.json")
      path = Path(index["jobs"][0]["path"])
      job = read(path)
      job["status"] = "failed"
      save_job(path, job, self.config)
      with self.assertRaisesRegex(ValueError, "child"):
        collect(self.run)
    with self.subTest(case='test_missing_peer_and_equal_missing_folds'):
      self.reset_case()
      evidence = collect(self.run)
      cells = [row for row in evidence["cells"] if row["phase"] == "development" and row["fold"] == 0]
      with self.assertRaisesRegex(ValueError, "Incomplete"):
        aggregate(cells, [42, 44, 46])
      evidence["cells"] = [row for row in evidence["cells"] if row["model"] != "crc"]
      _, summaries, coverage = contrasts(evidence)
      self.assertFalse(summaries)
      self.assertTrue(all(row["status"] == "incomplete" for row in coverage))
    with self.subTest(case='test_penalty_grid_rejects_incomplete_and_conflicting_cells'):
      self.reset_case()
      from src.reporting.figures import penalty_grid
      rows = [{'role': 'crc', 'configuration': 'fixed', 'seed': '42', 'fold': str(fold), 'lambda': str(penalty), 'status': 'completed', 'macro_f1': str(0.6 if penalty == 0.1 else 0.7)} for penalty in (0.1, 1) for fold in range(4)]
      _, _, values, regret = penalty_grid(rows + [rows[0]])
      np.testing.assert_allclose(values, [[0.6, 0.7]])
      np.testing.assert_allclose(regret, [[0.1, 0]])
      _, _, values, _ = penalty_grid(rows[1:])
      self.assertTrue(np.isnan(values[0, 0]))
      with self.assertRaisesRegex(ValueError, 'Conflicting'):
        penalty_grid(rows + [rows[0] | {'macro_f1': '0.8'}])
    with self.subTest(case='matched-membership'):
      self.reset_case()
      evidence = collect(self.run)
      evidence["cells"][0]["membership"] = "different"
      target = next(row for row in evidence["cells"] if row["population"] == "held")
      target["membership"] = "different"
      with self.assertRaisesRegex(ValueError, "memberships"):
        contrasts(evidence)

  def test_compact_predictions_and_integrity(self):
    """Shared compact references detect tampering without loading models."""
    path = next((self.run / "evidence/metrics/jobs").glob("*/result.json"))
    raw = json.loads(path.read_text())
    self.assertEqual(raw["evidence_format"], 1)
    self.assertIn("$evidence", raw["record"]["membership"]["train"])
    reference = raw["record"]["rows"][0]["predictions"]
    target = path.parent / reference["path"]
    target.write_bytes(b"invalid")
    with self.assertRaisesRegex(ValueError, "checksum"):
      read(path)

  def resource_evidence(self):
    """Construct complete compact resource evidence without numerical execution.

    :return: Frozen one-family evidence covering all declared seeds.
    """
    role = "practical:lstm"
    spec = {"model": "lstm", "recipe": "baseline", "fixed_epochs": 2}
    rows = [{
      "identity": identity,
      "label": int(self.data.labels[index]),
      "group": int(self.data.groups[index]),
      "steps": len(self.data.counts(index)),
      "partition": "confirmation" if index in self.data.confirmation else "development"
    } for index, identity in enumerate(self.data.identities)]
    indices = np.asarray(sorted(self.data.confirmation, key=lambda index: (len(self.data.counts(index)), int(index))))
    records = [{
      "dataset": "shd",
      "role": role,
      "model": "lstm",
      "seed": seed,
      "configuration": spec,
      "membership": self.data.membership(indices),
      "measurements": [{"repeat": repeat, "workload": workload, "samples": 1 if workload == "single" else len(indices), "seconds": 0.1} for repeat in range(3) for workload in ("single", "batch")]
    } for seed in self.config["seeds"]]
    return {
      "manifest": {"datasets": ["shd"], "models": ["lstm"], "resources": True, "config": self.config},
      "frozen": {"manifest": {"entries": {"shd": {role: {"spec": spec}}}}},
      "datasets": [{"dataset": "shd", "population": "train", "rows": rows}],
      "resources": {"status": "completed", "practical": records}
    }

  def test_resource_evidence(self):
    """Require valid complete resource comparisons and preserve measurement boundaries."""
    with self.subTest(case='resource_coverage_requires_models_seeds_and_repetitions'):
      self.reset_case()
      from src.reporting.evidence import resource_coverage
      complete = self.resource_evidence()
      self.assertTrue(all(row["status"] == "completed" for row in resource_coverage(complete)))
      for missing in ("all", "seed", "repeat", "model"):
        evidence = copy.deepcopy(complete)
        if missing == "all":
          evidence["resources"]["practical"] = []
        elif missing == "seed":
          evidence["resources"]["practical"].pop()
        elif missing == "repeat":
          evidence["resources"]["practical"][0]["measurements"].pop()
        else:
          evidence["manifest"]["models"].append("transformer")
        with self.subTest(missing=missing):
          self.assertEqual(resource_coverage(evidence)[-1]["status"], "incomplete")
    with self.subTest(case='resource_coverage_rejects_invalid_identities_and_measurements'):
      self.reset_case()
      from src.reporting.evidence import resource_coverage
      for damage in ("duplicate", "seed", "configuration", "membership", "repeat", "samples", "seconds"):
        evidence = self.resource_evidence()
        records = evidence["resources"]["practical"]
        if damage == "duplicate":
          records.append(copy.deepcopy(records[0]))
        elif damage == "seed":
          records[0]["seed"] = 99
        elif damage == "configuration":
          records[0]["configuration"] = {"model": "transformer"}
        elif damage == "membership":
          records[0]["membership"]["identities"].reverse()
        elif damage == "repeat":
          records[0]["measurements"].append(copy.deepcopy(records[0]["measurements"][0]))
        elif damage == "samples":
          records[0]["measurements"][0]["samples"] = 20
        else:
          records[0]["measurements"][0]["seconds"] = float("nan")
        with self.subTest(damage=damage), self.assertRaises(ValueError):
          resource_coverage(evidence)
    with self.subTest(case='empty_completed_resource_record_keeps_report_incomplete'):
      self.reset_case()
      self.manifest.update({"models": ["lstm"], "resources": True})
      write(self.run / "manifest.json", self.manifest)
      write(self.run / "evidence/metrics/resources.json", {"status": "completed", "practical": []})
      seal(self.run)
      result = report(self.run)
      self.assertEqual(result["development_status"], "incomplete")
      self.assertFalse((self.run / "reports/resources/tables/practical-costs.csv").exists())
    with self.subTest(case='practical_table_discloses_cache_and_devices'):
      self.reset_case()
      with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        export(root, 'resources', 'practical-costs', [{
          'dataset': 'shd',
          'model': 'crc',
          'seed': 42,
          'device': 'cpu',
          'fit_seconds': 0.5,
          'features_reused': True,
          'inference_seconds_per_example': 0.001,
          'storage_mib': 1,
          'host_peak_mib': 800,
          'gpu_allocated_mib': None,
          'accuracy': 0.8,
          'macro_f1': 0.79,
          'macro_precision': 0.81,
          'macro_recall': 0.8
        }])
        with (root / 'resources/tables/practical-costs.csv').open() as stream:
          row = next(csv.DictReader(stream))
        self.assertEqual(row['device'], 'cpu')
        self.assertEqual(row['features_reused'], 'True')
        self.assertEqual(float(row['inference_seconds_per_example']), 0.001)
        self.assertFalse(list(root.rglob('*.tex')))

  def test_fitting_methods_follow_model_declarations(self):
    """Infer legacy optimizer labels and reject contradictory fitting records."""
    from src.models.protocol import FittingProtocol
    from src.reporting.evidence import job_rows
    path = next((self.run / "evidence/metrics/jobs").glob("*/result.json"))
    source = read(path)
    for model, solver in (("lstm", "adamax"), ("transformer", "adamw")):
      for phase in ("development", "confirmation", "official"):
        with self.subTest(model=model, phase=phase):
          job = copy.deepcopy(source)
          spec = {"model": model, "recipe": "baseline"}
          job["identity"].update({"spec": spec, "phase": phase})
          job["rows"] = [job["rows"][0]]
          row = job["rows"][0]
          row.pop("solver")
          row["lambda"] = None
          result = {key: [] for key in ("cells", "confusions", "classes", "subjects", "history", "diagnostics")}
          result["specs"] = {}
          job_rows(job, "shd", "practical:" + model, job["identity"]["seed"], None, phase, spec, result)
          self.assertEqual({cell["solver"] for cell in result["cells"]}, {solver})
          row["solver"] = "cholesky"
          with self.assertRaisesRegex(ValueError, "fitting method"):
            job_rows(job, "shd", "practical:" + model, job["identity"]["seed"], None, phase, spec, result)
    for family in ("crc", "qrc", "projection"):
      for solver in ("cholesky", "lsqr"):
        self.assertEqual(FittingProtocol.from_spec({"model": family, "solver": solver}).solver, solver)
    with self.assertRaisesRegex(ValueError, "Unknown model"):
      FittingProtocol.from_spec({"model": "unknown"})
