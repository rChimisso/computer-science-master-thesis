import copy
from unittest.mock import patch

from src.runtime.records import checksum, digest, frozen_record, read, write
from src.studies.design import group
from src.workflow import run

from tests.support import FrozenCase, SyntheticCase

class OfficialTests(FrozenCase):
  """Verify synthetic official boundaries and failure recovery."""

  def test_official_lifecycle_and_recovery(self):
    """Refit before official access, retain access provenance and resume safely."""
    held = copy.deepcopy(self.data)
    held.identities = [value.replace('-train:', '-test:') for value in self.data.identities]
    held.fingerprint = digest(held.identities)
    self.request['official_test'] = True

    def loader(name, config, split='train', official=False):
      """Return synthetic held data through the internally explicit gate.

      :param name: Dataset name.
      :param config: Resolved settings.
      :param split: Population to load.
      :param official: Gate authorizing test access.
      :return: Synthetic population descriptor.
      """
      if split == 'test':
        self.assertTrue(official)
        self.assertTrue((self.root / 'frozen.json').exists())
        fitted = read(self.root / 'evidence/models/official-fit.json')
        self.assertEqual(len(fitted), len(self.config['seeds']))
        self.assertTrue(all(value['status'] == 'completed' for value in fitted.values()))
        return held
      return self.data

    with patch('src.data.datasets.load', side_effect=loader), patch('src.workflow.groups', return_value=self.declarations), patch('src.workflow.finish', return_value={}):
      with patch('src.studies.final.evaluate', side_effect=RuntimeError('Synthetic inference failure')):
        with self.assertRaisesRegex(RuntimeError, 'Synthetic inference failure'):
          run(self.request, self.root.name)
      self.assertTrue(read(self.root / 'progress.json')['official_test_loaded'])
      frozen_hash = checksum(self.root / 'frozen.json')
      result = run(self.request, self.root.name, resume=True)
      self.assertTrue(result['official_test_loaded'])
      self.assertEqual(result['status'], 'completed')
      self.assertEqual(checksum(self.root / 'frozen.json'), frozen_hash)
    from src.workflow import evaluate_run
    root = self.root / "run"
    write(root / "progress.json", {"status": "completed", "completed": ["shd"]})
    checkpoint = root / "temporary/checkpoint.bin"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"resumable fixture")
    with patch("src.workflow.frozen_record", return_value={}), patch("src.workflow.evaluate_frozen", side_effect=TimeoutError("deadline")), patch("src.workflow.finish") as finish:
      with self.assertRaises(TimeoutError):
        evaluate_run(root / "frozen.json", True)
    finish.assert_not_called()
    self.assertEqual(read(root / "progress.json")["status"], "interrupted")
    self.assertEqual(read(root / "progress.json")["phase"], "official")
    self.assertEqual(checkpoint.read_bytes(), b"resumable fixture")

class ContinuationTests(SyntheticCase):
  """Verify seed-filtered continuation and frozen reproduction."""

  def test_seed_filtered_rounds_resume_same_manifest(self):
    """Seed filters extend a run without changing its scientific identity."""
    from src.main import parser
    from src.workflow import resolve
    declaration = group("comparison", "projection:i2", [{"model": "projection", "inputs": 2}])
    for seed in (42, 44, 46):
      args = parser().parse_args([
        "run",
        "--development-only",
        "--profile",
        "core",
        "--dataset",
        "shd",
        "--seed",
        str(seed)
      ])
      request = resolve(args)
      request["config"]["execution"].update({"qrc_large": "CPU", "device": "cpu"})
      request["config"]["paths"]["runs"] = str(self.root / "runs")
      request["config"]["paths"]["cache"] = str(self.root / "cache")
      with patch("src.workflow.groups", return_value=[declaration]), patch("src.data.datasets.load", return_value=self.data):
        result = run(request, "rounds", resume=seed != 42)
      self.assertEqual(result["status"], "completed" if seed == 46 else "partial")
    frozen = self.root / "runs/rounds/frozen.json"
    body = frozen_record(frozen)
    self.assertEqual(body["manifest"]["decisions"]["shd"]["seeds"], [
      42,
      44,
      46
    ])
    from src.runtime.reproduction import prepare
    from src.workflow import evaluate_frozen
    from src.reporting.study import report
    replay = prepare(frozen, "replay")
    evaluate_frozen(replay, False, datasets={"shd": self.data})
    original_jobs = self.root / "runs/rounds/evidence/metrics/jobs"
    original_jobs.rename(original_jobs.with_name("unavailable-original-jobs"))
    self.assertEqual(report(replay.parent)["rendering_status"], "completed")
    changed = copy.deepcopy(self.data)
    changed.fingerprint = "changed-numerical-data"
    with self.assertRaisesRegex(ValueError, "contents changed"):
      evaluate_frozen(frozen, False, datasets={"shd": changed})
    self.assertEqual(body["manifest"]["parameter_register"]["parameters"]["execution.qrc_large"]["value"], "CPU")
    self.assertEqual(body["manifest"]["parameter_register"]["parameters"]["execution.qrc_large"]["baseline_value"], "GPU")
    role = next(iter(body["manifest"]["entries"]["shd"]))
    body["manifest"]["entries"]["shd"][role]["spec"]["inputs"] = 4
    body["fingerprint"] = digest(body["manifest"])
    write(frozen, body)
    with self.assertRaisesRegex(ValueError, "development decision"):
      frozen_record(frozen)
