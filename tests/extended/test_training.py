from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.runtime.configuration import bind
from src.runtime.records import read

from tests.support import SyntheticCase

class TrainingTests(SyntheticCase):
  """Protect neural checkpoints and calibrated inference batches."""

  def test_checkpoint_commit_interruptions_replay_best_weights(self):
    """Committed current/best weights resume together, including a stopping boundary."""
    import torch
    from src.models import training
    original_save = training.atomic_checkpoint
    original_evaluate = training.neural_evaluate

    def stable_metrics(*args):
      """Hold selection scores constant to exercise a known patience boundary.

      :param args: Original evaluation arguments.
      :return: Real predictions with fixture-only constant stopping scores.
      """
      result = original_evaluate(*args)
      return result | {"macro_f1": 0.5, "accuracy": 0.5}

    for fixed in (None, 2):
      for filename, after in (("last.pt", False), ("last.pt", True), ("best.pt", True)):
        with self.subTest(fixed=fixed, filename=filename, after=after):
          directory = self.root / f"case-{fixed}-{filename}-{after}"
          local = bind(self.config, directory)
          interrupted = []

          def save(path, state):
            """Interrupt once immediately around a target checkpoint commit.

            :param path: Checkpoint destination.
            :param state: Current atomic record.
            """
            selected = not interrupted and path.name == filename and (filename == "best.pt" or state["epoch"] == 2)
            if selected and not after:
              interrupted.append(True)
              raise TimeoutError("Fixture before checkpoint commit")
            original_save(path, state)
            if selected:
              interrupted.append(True)
              raise TimeoutError("Fixture after checkpoint commit")

          with patch.object(training, "benchmark_neural", return_value=8), patch.object(training, "neural_evaluate", side_effect=stable_metrics):
            with patch.object(training, "atomic_checkpoint", side_effect=save), self.assertRaises(TimeoutError):
              training.train_neural(self.data, self.data.development, self.data.confirmation, "lstm", "baseline", 42, local, fixed)
            resumed = training.train_neural(self.data, self.data.development, self.data.confirmation, "lstm", "baseline", 42, local, fixed)
            fresh = training.train_neural(self.data, self.data.development, self.data.confirmation, "lstm", "baseline", 42, bind(self.config, directory / "fresh"), fixed)
          self.assertEqual(resumed["completed_epoch"], 2)
          self.assertEqual(resumed["status"], "fixed_duration" if fixed else "early_stopped")
          self.assertEqual(resumed["best"]["epoch"], fresh["best"]["epoch"])
          self.assertEqual(resumed["best"]["validation"]["predictions"], fresh["best"]["validation"]["predictions"])
          actual = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=False)
          expected = torch.load(fresh["checkpoint"], map_location="cpu", weights_only=False)
          self.assertEqual(actual["epoch"], resumed["best"]["epoch"])
          self.assertNotIn("optimizer", actual)
          for name, values in actual["model"].items():
            torch.testing.assert_close(values, expected["model"][name], rtol=0, atol=0)

  def test_batch_calibration_and_replay(self):
    """Retain feasible complete batches across calibration, inference and measurements."""
    with self.subTest(case='calibration_retries_smaller_complete_batches'):
      self.reset_case()
      import torch
      from src.models.training import benchmark_neural
      self.config["training"]["batch_size"] = 16
      attempts = []

      def attempt(data, indices, model, config, device):
        """Simulate capacity without exhausting the actual machine.

        :param data: Synthetic source.
        :param indices: Candidate batch.
        :param model: Requested family.
        :param config: Execution settings.
        :param device: Selected CPU fixture device.
        :return: Bounded measurement for a fitting candidate.
        """
        attempts.append(len(indices))
        if len(indices) > 8:
          raise torch.cuda.OutOfMemoryError("Synthetic capacity")
        return {"two_updates_seconds": 0.01, "maximum_steps": 6, "peak_gpu_bytes": 0}

      with patch("src.models.training.benchmark_batch", side_effect=attempt):
        self.assertEqual(benchmark_neural(self.data, "lstm", self.config), 8)
      self.assertEqual(attempts, [16, 8])
      with patch("src.models.training.benchmark_batch", side_effect=AssertionError("Repeated calibration")):
        self.assertEqual(benchmark_neural(self.data, "lstm", self.config), 8)
    with self.subTest(case='inference_and_resource_batches_retain_calibration'):
      self.reset_case()
      from src.models.training import neural_evaluate
      from src.runtime.pipelines import export, predict
      from src.studies.experiments import experiment
      from src.studies.resources import resource_worker
      from src.data.indexed import padded_batch
      spec = {"model": "lstm", "recipe": "baseline", "fixed_epochs": 1}
      fold = {"fold": "confirmation", "train": self.data.development.tolist(), "validation": self.data.confirmation.tolist()}
      with patch("src.models.training.benchmark_neural", return_value=2):
        job = experiment(self.data, fold, spec, 42, self.config, phase="confirmation")
      package = export(self.data, self.data.development, spec, 42, "confirmation", job, self.config)
      self.assertEqual(read(Path(package["directory"]) / "pipeline.json")["batch_size"], 2)
      with patch("src.models.training.neural_evaluate", wraps=neural_evaluate) as evaluate:
        predictions = predict(package, self.data, self.data.confirmation, self.config)
      self.assertEqual(evaluate.call_args.args[3], 2)
      np.testing.assert_array_equal(predictions, job["rows"][0]["predictions"])
      messages = []
      connection = SimpleNamespace(send=messages.append, close=lambda: None)
      entry = {"spec": spec, "role": "practical:lstm", "resource_job": job["path"], "pipeline": package}
      with patch("src.data.indexed.padded_batch", wraps=padded_batch) as batches:
        resource_worker(connection, self.data, entry, 42, self.config)
      self.assertNotIn("error", messages[0])
      self.assertTrue(all(len(call.args[1]) <= 2 for call in batches.call_args_list))
      self.assertEqual(len(messages[0]["measurements"]), 6)
      self.assertEqual(messages[0]["batch_size"], 2)
      self.assertTrue(messages[0]["deterministic_algorithms"])
