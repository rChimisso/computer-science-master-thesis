import os
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.data.indexed import study_folds
from src.features.extraction import extract
from src.features.summary import summarize_sequence_features
from src.models.classical_reservoir import ClassicalReservoir, ClassicalReservoirParameters
from src.models.torch_classical_reservoir import TorchClassicalReservoir
from src.runtime.records import read
from src.runtime.validation import fixture

from tests.support import SyntheticCase

class ReservoirTests(SyntheticCase):
  """Verify long sequences, independent trajectories and quantum recovery."""

  def test_numerical_backends(self):
    """Compare complete ragged trajectories, summaries and changed-input graph replay."""
    from src.features.observations import classical_reference
    from src.features.extraction import summarize
    from src.models.quantum_execution import AerReservoir
    from src.models.streaming_crc import StreamingCRC
    from src.models.training_support import configure_device
    gpu = os.environ.get("THESIS_VALIDATION_GPU") == "1"
    configure_device(42, "cuda" if gpu else "cpu", 1)
    rng = np.random.default_rng(42)
    for name, length in (("shd", 140), ("dvs", 370)):
      data = fixture(name)
      segments = 8 if name == "shd" else 4
      for inputs in (2, 4, 6):
        sequences = [rng.uniform(size=(length, inputs)), rng.uniform(size=(3, inputs))]
        spec = {
          "qubits": 8,
          "inputs": inputs,
          "depth": 3,
          "seed": 42,
          "observables": ["x", "y", "z"]
        }
        observed = []
        for device in (("CPU", "GPU") if gpu else ("CPU",)):
          executor = AerReservoir(spec, device)
          try:
            batched = executor.transform(sequences)["values"]
            alone = executor.transform([sequences[1]])["values"][0]
          finally:
            executor.close()
          np.testing.assert_allclose(batched[1], alone, atol=1e-10, rtol=0)
          self.assertEqual(batched[0].shape, (length, 24))
          observed.append(batched)
        if gpu:
          for left, right in zip(*observed):
            np.testing.assert_allclose(left, right, atol=1e-10, rtol=0)
        reference = classical_reference(data, {"size": 24, "inputs": inputs, "radius": 0.9, "scale": 0.03, "tau": 20}, 42)
        expected = np.asarray([summarize(reference.transform_sequential(value[None])[0], segments) for value in sequences])
        for device, graphs in (("cpu", False), ("cuda", False), ("cuda", True)) if gpu else (("cpu", False),):
          executor = StreamingCRC(reference, device, graphs)
          actual = executor.extract(sequences, segments, ("all",))[0]["all"]
          np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
          changed = [value[::-1].copy() for value in sequences]
          expected_replay = np.asarray([summarize(reference.transform_sequential(value[None])[0], segments) for value in changed])
          replay = executor.extract(changed, segments, ("all",))[0]["all"]
          np.testing.assert_allclose(replay, expected_replay, atol=1e-10, rtol=1e-10)
    for device in (("cpu", "cuda") if gpu else ("cpu",)):
      self.addCleanup(torch.set_num_threads, torch.get_num_threads())
      torch.set_num_threads(1)
      torch.backends.cuda.matmul.allow_tf32 = False
      with threadpool_limits(limits=1):
        reference = ClassicalReservoir(4, ClassicalReservoirParameters(18, 0.9, 0.3, 0.1, 0.5), 42)
        values = np.random.default_rng(44).normal(size=(4, 370, 4))
        lengths = np.asarray([1, 7, 140, 370])
        trajectory = reference.transform_sequential(values, lengths)
        for row, length in enumerate(lengths):
          values[row, length:] = np.nan
        backend = TorchClassicalReservoir(reference, device)
        batch = backend.prepare(values, lengths)
        for mode in ("segment_mean_final", "segment_mean_std_final"):
          for segments in (1, 4, 8):
            expected = summarize_sequence_features(trajectory, lengths, segments, mode)
            actual = backend.summarize(batch, segments, mode).cpu().numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=1e-9)
            repeated = backend.summarize(batch, segments, mode).cpu().numpy()
            np.testing.assert_array_equal(actual, repeated)
        padded = np.pad(values, ((0, 0), (0, 15), (0, 0)), constant_values=np.inf)
        actual = backend.summarize(batch, 8).cpu().numpy()
        extra = backend.summarize(backend.prepare(padded, lengths), 8).cpu().numpy()
        np.testing.assert_array_equal(actual, extra)
        for row, length in enumerate(lengths):
          single = backend.summarize(backend.prepare(values[row:row + 1, :length], lengths[row:row + 1]), 8).cpu().numpy()
          np.testing.assert_allclose(single[0], actual[row], atol=2e-8, rtol=1e-9)

  def test_quantum_extraction_interruption(self):
    """An interrupted extraction preserves complete samples and resumes identically."""
    from src.features.extraction import summarize
    fold = study_folds(self.data, self.config)[0]
    train = np.asarray(fold["train"])
    indices = train[:2]
    self.config["execution"]["quantum_batch"] = 1
    spec = {
      "model": "qrc",
      "inputs": 2,
      "qubits": 4,
      "depth": 1
    }
    calls = []
    def interrupted(values, segments):
      """Interrupt the second sample before its atomic progress publication.

      :param values: Valid trajectory.
      :param segments: Summary count.
      :return: Completed first-sample view.
      """
      calls.append(1)
      if len(calls) == 3:
        raise TimeoutError("Synthetic interruption")
      return summarize(values, segments)
    with patch("src.features.extraction.summarize", side_effect=interrupted):
      with self.assertRaises(TimeoutError):
        extract(self.data, train, indices, spec, 42, self.config)
    records = list((self.root / "shared/features").glob("*/metadata.json"))
    self.assertEqual(read(records[0])["completed_samples"], 1)
    resumed, _, _ = extract(self.data, train, indices, spec, 42, self.config)
    fresh_config = self.config | {"paths": self.config["paths"] | {"cache": str(self.root / "independent-cache")}}
    fresh, _, _ = extract(self.data, train, indices, spec, 42, fresh_config)
    np.testing.assert_allclose(resumed, fresh, atol=1e-10, rtol=0)
