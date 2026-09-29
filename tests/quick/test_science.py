import copy
from unittest.mock import patch

import numpy as np

from src.data.indexed import nested_indices, study_folds
from src.features.adapters import fit_adapter
from src.features.extraction import extract
from src.reporting.study import aggregate
from src.runtime.validation import fixture
from src.studies.experiments import experiment
from src.studies.selection import score, winner

from tests.support import SyntheticCase

class ScientificTests(SyntheticCase):
  """Protect data, matched reservoirs, selection and metric arithmetic."""

  def test_event_counts_and_temporal_boundaries(self):
    """Preserve events, channel/polarity ordering and complete clip endpoints."""
    from src.data.gesture import bin_dvs_clip
    from src.data.spikes import PreprocessingSpec, bin_shd_events
    events = np.zeros(4, dtype=[("t", np.int64), ("x", np.int64)])
    events["t"] = [0, 9999, 10000, 25000]
    events["x"] = [0, 699, 350, 10]
    counts, length = bin_shd_events(events, PreprocessingSpec(10000, 2, "count", 30000))
    np.testing.assert_array_equal(counts, [[1, 1], [0, 1], [1, 0]])
    self.assertEqual(length, 3)
    gesture = np.asarray([
      [0, 0, 0, 0],
      [127, 127, 1, 49.999],
      [127, 127, 1, 50],
      [0, 0, 1, 5000]
    ])
    counts = bin_dvs_clip(gesture)
    self.assertEqual(counts.shape, (101, 128))
    self.assertEqual(int(counts.sum()), 4)
    self.assertEqual(int(counts[100, 64]), 1)

  def test_membership_and_nested_subsets(self):
    """All fold memberships are isolated and subsets are nested."""
    for name in ("shd", "dvs"):
      data = fixture(name)
      for fold in study_folds(data, self.config):
        train, held = np.asarray(fold["train"]), np.asarray(fold["validation"])
        self.assertFalse(set(data.groups[train]) & set(data.groups[held]))
        smaller = nested_indices(data, train, 15)
        larger = nested_indices(data, train, 18)
        self.assertTrue(set(smaller) <= set(larger))

  def test_train_only_adapter(self):
    """Changing held examples cannot change fitted adapter statistics."""
    for name, mode in (("shd", "pooling"), ("dvs", "pca")):
      data = fixture(name)
      fold = study_folds(data, self.config)[0]
      train = np.asarray(fold["train"])
      left = fit_adapter(data, train, mode, 2, 42, self.config)
      modified = copy.deepcopy(data)
      for index in fold["validation"]:
        modified.sequences[index] *= 100
      modified.fingerprint = "changed-held-only"
      right = fit_adapter(modified, train, mode, 2, 42, self.config)
      for key in ("weights", "means", "scales"):
        np.testing.assert_array_equal(left[key], right[key])

  def test_matched_inputs_and_feature_dimensions(self):
    """Both reservoirs receive the identical fitted probability stream exactly once."""
    from src.features.adapters import apply_adapter
    from src.models.quantum_execution import AerReservoir
    from src.models.streaming_crc import StreamingCRC
    train = self.data.development
    indices = train[:2]
    quantum_inputs, classical_inputs = [], []
    quantum_transform = AerReservoir.transform
    classical_extract = StreamingCRC.extract

    def quantum(executor, sequences, *args, **kwargs):
      """Capture the actual quantum stream before native execution.

      :param executor: Quantum executor.
      :param sequences: Unpadded probability arrays.
      :param args: Positional execution options.
      :param kwargs: Named execution options.
      :return: Native observations.
      """
      quantum_inputs.extend(value.copy() for value in sequences)
      return quantum_transform(executor, sequences, *args, **kwargs)

    def classical(executor, sequences, *args, **kwargs):
      """Capture the actual classical stream before recurrent execution.

      :param executor: Classical executor.
      :param sequences: Unpadded probability arrays.
      :param args: Positional execution options.
      :param kwargs: Named execution options.
      :return: Classical summaries.
      """
      classical_inputs.extend(value.copy() for value in sequences)
      return classical_extract(executor, sequences, *args, **kwargs)

    qrc = {"model": "qrc", "qubits": 4, "inputs": 2, "depth": 1}
    crc = {"model": "crc", "size": 12, "inputs": 2, "radius": 0.9, "scale": 0.03, "tau": 20}
    with patch.object(AerReservoir, "transform", quantum), patch.object(StreamingCRC, "extract", classical):
      quantum_features, _, adapter = extract(self.data, train, indices, qrc, 42, self.config)
      classical_features, _, _ = extract(self.data, train, indices, crc, 42, self.config)
    self.assertEqual(quantum_features.shape, classical_features.shape)
    self.assertEqual(len(quantum_inputs), len(indices))
    self.assertEqual(len(classical_inputs), len(indices))
    for index, left, right in zip(indices, quantum_inputs, classical_inputs):
      np.testing.assert_array_equal(left, right)
      np.testing.assert_array_equal(left, apply_adapter(self.data.values(int(index)), adapter))

  def test_known_quantum_state_and_independent_sequences(self):
    """Zero-angle CZ dynamics preserve boundary basis states and reset each example."""
    from src.models.quantum_execution import AerReservoir
    spec = {
      "qubits": 4,
      "inputs": 2,
      "depth": 1,
      "seed": 42,
      "gate_angles": np.zeros((1, 4, 2)).tolist(),
      "observables": ["x", "y", "z", "zz_ring"]
    }
    executor = AerReservoir(spec, "CPU")
    try:
      observed = executor.transform([np.ones((2, 2)), np.zeros((1, 2))], diagnostic=True)
    finally:
      executor.close()
    expected = np.zeros((2, 16))
    expected[:, 8:12] = [-1, -1, 1, 1]
    expected[:, 12:] = [1, -1, 1, -1]
    np.testing.assert_allclose(observed["values"][0], expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(observed["values"][1][0, 8:12], 1, atol=1e-10, rtol=0)
    for state in observed["states"]:
      np.testing.assert_allclose(np.trace(state), 1, atol=1e-10, rtol=0)
      np.testing.assert_allclose(state, state.conj().T, atol=1e-10, rtol=0)
      self.assertGreaterEqual(float(np.linalg.eigvalsh(state).min()), -1e-10)
    spec.pop("gate_angles")
    histories = [np.asarray([[0.0, 0.0], [0.5, 0.5]]), np.asarray([[1.0, 1.0], [0.5, 0.5]])]
    for reset in ("partial", "all"):
      executor = AerReservoir(spec | {"reset_mode": reset}, "CPU")
      try:
        trajectories = executor.transform(histories)["values"]
      finally:
        executor.close()
      difference = float(np.max(np.abs(trajectories[0][-1] - trajectories[1][-1])))
      if reset == "all":
        self.assertLess(difference, 1e-10)
      else:
        self.assertGreater(difference, 1e-7)

  def test_padding_and_summary_order(self):
    """Padded cells cannot change summary statistics or optional neural outputs."""
    import torch
    from src.features.summary import summarize_sequence_features
    from src.models.temporal_reference import TemporalLSTM, TemporalTransformer
    from src.models.training_support import configure_device
    configure_device(42, "cpu", 1)
    values = np.asarray([[[1.0], [3.0], [np.nan]]])
    summary = summarize_sequence_features(values, np.asarray([2]), 1, "segment_mean_std_final")
    np.testing.assert_array_equal(summary, [[2, 1, 3]])
    inputs = torch.randn(2, 4, 3)
    lengths = torch.tensor([2, 4])
    poisoned = inputs.clone()
    poisoned[0, 2:] = torch.nan
    extended = torch.cat((poisoned, torch.full((2, 3, 3), torch.nan)), dim=1)
    for model in (TemporalLSTM(3, hidden_size=4), TemporalTransformer(3, width=8, heads=2, feedforward_size=16)):
      model.eval()
      with torch.no_grad():
        torch.testing.assert_close(model(inputs, lengths), model(extended, lengths), atol=1e-6, rtol=1e-5)

  def test_readout_and_four_metric_arithmetic(self):
    """Ridge uses the mean-loss penalty and subgroup metrics are recomputed."""
    from src.evaluation.metrics import prediction_metrics
    from src.evaluation.readout import direct_readout
    labels = np.asarray([0, 0, 1, 1])
    features = np.asarray([[0.0], [1.0], [3.0], [4.0]])
    result, state, _ = direct_readout(features, features[:2], labels, labels[:2], 0.1, 2)
    self.assertAlmostEqual(result["alpha"], 0.4)
    np.testing.assert_allclose(state["means"], features.mean(axis=0))
    metrics, matrix = prediction_metrics(labels, [0, 1, 1, 1], 2)
    np.testing.assert_array_equal(matrix, [[1, 1], [0, 2]])
    self.assertAlmostEqual(metrics["accuracy"], 0.75)
    self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.8) / 2)
    self.assertAlmostEqual(metrics["macro_precision"], (1 + 2 / 3) / 2)
    self.assertAlmostEqual(metrics["macro_recall"], 0.75)
    subgroup, _ = prediction_metrics(labels[:2], [0, 1], 2)
    self.assertAlmostEqual(subgroup["macro_f1"], 1 / 3)

  def test_selection_coverage_and_tie(self):
    """Missing folds are rejected and shallow QRC wins within equivalence."""
    fold = study_folds(self.data, self.config)[0]
    job = experiment(self.data, fold, {"model": "projection", "inputs": 2}, 42, self.config)
    with self.assertRaisesRegex(ValueError, "complete"):
      score([job], self.config)
    candidates = [{"spec": {"model": "qrc", "depth": depth}, "selection": {"mean_macro_f1": value}} for depth, value in ((1, 0.701), (3, 0.704))]
    self.assertEqual(winner(candidates, self.config)["spec"]["depth"], 1)

  def test_protocol_separation_and_missing_seeds(self):
    """Reported uncertainty requires every declared fold and seed."""
    rows = []
    for phase in ("development", "official"):
      for seed in (42, 44, 46):
        for fold in range(4) if phase == "development" else ["official"]:
          rows.append({
            "dataset": "shd",
            "phase": phase,
            "role": "qrc",
            "population": "held",
            "seed": seed,
            "fold": fold,
            "accuracy": 0.5,
            "macro_f1": 0.4 + (seed - 42) * 0.05,
            "macro_precision": 0.4,
            "macro_recall": 0.5
          })
    summaries = aggregate(rows, [42, 44, 46])
    self.assertEqual(len(summaries), 2)
    for summary in summaries:
      self.assertAlmostEqual(summary["macro_f1_seed_sd"], 0.1)
    with self.assertRaisesRegex(ValueError, "Incomplete"):
      aggregate(rows[:-1], [42, 44, 46])
