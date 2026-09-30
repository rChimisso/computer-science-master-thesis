import copy
from pathlib import Path
from unittest.mock import patch

from src.data.acquisition import acquire
from src.methodology.parameters import validate_register
from src.runtime.configuration import load
from src.runtime.provenance import protected_syntax, validate_source
from src.runtime.records import frozen_record, read, sources, write
from src.studies.design import groups

from tests.support import SyntheticCase

class ProtocolTests(SyntheticCase):
  """Protect declared scope, source compatibility and official-data gates."""

  def test_study_definitions_and_provenance(self):
    """Keep practical models optional and declared settings justified."""
    with self.subTest(case='standalone_profiles_and_optional_models'):
      rows = groups("shd", ["comparison"], [], self.config)
      self.assertTrue(rows)
      self.assertEqual({spec["model"] for row in rows for spec in row["candidates"]}, {"qrc", "crc", "projection"})
      self.assertFalse(any(spec["model"] in ("lstm", "transformer") or spec.get("size") == 256 for row in rows for spec in row["candidates"]))
      self.assertFalse(any(spec.get("qubits", 0) > 8 for row in rows for spec in row["candidates"]))
    with self.subTest(case='parameter_coverage'):
      config = load()
      register = read("configs/parameters.json")
      validate_register(config, register)
      altered = copy.deepcopy(config)
      altered["search"]["depths"] = [1]
      with self.assertRaisesRegex(ValueError, "stale"):
        validate_register(altered, register)

  def test_frozen_source_policy(self):
    """Accept operational changes while rejecting changed scientific implementations."""
    with self.subTest(case='frozen_presentation_changes_are_compatible'):
      original = sources()
      for name in ('src/reporting/figures.py', 'src/runtime/monitor.py', 'src/main.py', 'src/runtime/test_suites.py'):
        changed = original | {name: 'different-producer-hash'}
        self.assertEqual(validate_source(changed, Path('/nonexistent')), [name])
    with self.subTest(case='frozen_science_and_unknown_files_are_rejected'):
      original = sources()
      for name in ('src/models/quantum_execution.py', 'src/data/datasets.py', 'src/studies/selection.py', 'src/runtime/configuration.py', 'src/runtime/storage.py'):
        with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Scientific source'):
          validate_source(original | {name: 'different'}, Path('/nonexistent'))
      with self.assertRaisesRegex(ValueError, 'added or removed'):
        validate_source(original | {'src/new_science.py': 'new'}, Path('/nonexistent'))
    with self.subTest(case='mixed_module_scientific_functions_remain_protected'):
      before = 'def finish():\n  return 1\ndef freeze():\n  return 2\n'
      allowed = before.replace('return 1', 'return 3')
      rejected = before.replace('return 2', 'return 3')
      self.assertEqual(protected_syntax(before, {'finish'}), protected_syntax(allowed, {'finish'}))
      self.assertNotEqual(protected_syntax(before, {'finish'}), protected_syntax(rejected, {'finish'}))

  def test_official_access_gate(self):
    """Reject unauthorized test access and malformed freezes before side effects."""
    with self.subTest(case='official_gate_precedes_io'):
      with self.assertRaisesRegex(ValueError, "explicit"):
        acquire("shd", "test", self.config)
      write(self.root / "frozen.json", {
        "status": "frozen",
        "manifest": {},
        "fingerprint": "invalid"
      })
      with self.assertRaises(ValueError):
        frozen_record(self.root / "frozen.json")
    with self.subTest(case='invalid_freeze_precedes_action_writes'):
      from src.workflow import evaluate_run
      with patch("src.workflow.frozen_record", side_effect=ValueError("invalid freeze")), patch("src.workflow.evaluate_frozen") as evaluate:
        with self.assertRaisesRegex(ValueError, "invalid freeze"):
          evaluate_run(self.root / "absent/frozen.json", True)
      evaluate.assert_not_called()
      self.assertFalse((self.root / "absent").exists())

  def test_execution_overrides_keep_scientific_provenance(self):
    """Accept the documented CPU recipe and reject unregistered scientific changes."""
    import yaml
    from src.main import parser
    from src.methodology.parameters import resolve_register
    from src.workflow import plan, resolve
    config = load()
    baseline = read("configs/parameters.json")
    original = copy.deepcopy(baseline)
    config["execution"].update({"qrc_small": "CPU", "qrc_large": "CPU", "crc_device": "cpu", "crc_graphs": False, "device": "cpu"})
    path = self.root / "cpu.yaml"
    path.write_text(yaml.safe_dump({key: value for key, value in config.items() if key != "neural"}))
    request = resolve(parser().parse_args(["plan", "--config", str(path), "--profile", "core"]))
    self.assertTrue(plan(request)["datasets"])
    registered = resolve_register(request["config"], baseline)
    self.assertEqual(validate_register(config, registered), [])
    entry = registered["parameters"]["execution.qrc_large"]
    self.assertEqual(entry["value"], "CPU")
    self.assertEqual(entry["baseline_value"], "GPU")
    self.assertIn("execution-override", entry["origins"])
    self.assertEqual(baseline, original)
    config["search"]["depths"] = [1]
    with self.assertRaisesRegex(ValueError, "stale"):
      resolve_register(config, baseline)
    for key, value in (("device", "cpU"), ("cpu_threads", 0), ("crc_graphs", True), ("memory_fraction", float("nan")), ("gpu_threads", 2)):
      altered = load()
      altered["execution"][key] = value
      with self.subTest(key=key), self.assertRaises(ValueError):
        resolve_register(altered, baseline)
