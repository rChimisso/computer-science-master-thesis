import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.feasibility import analyze_dataset

class FakeEventDataset:
  """Minimal SHD-like dataset used by feasibility unit tests."""

  sensor_size = (700, 1, 1)
  """Native sensor dimensions."""

  dtype = np.dtype([("t", np.int64), ("x", np.int64), ("p", np.int64)])
  """Structured event-array data type."""

  targets = [0, 1]
  """Class labels for the synthetic samples."""

  location_on_system: str
  """Directory containing the synthetic raw file."""

  samples: list[tuple[np.ndarray, int]]
  """Synthetic event samples and their labels."""

  def __init__(self, location: str):
    """Create two deterministic event samples.

    :param location: Directory reported as the dataset location.
    """
    self.location_on_system = location
    first = np.zeros(2, dtype=self.dtype)
    first["t"] = [0, 10_000]
    first["x"] = [0, 699]
    first["p"] = [1, 1]
    second = np.zeros(1, dtype=self.dtype)
    second["t"] = 5_000
    second["x"] = 350
    second["p"] = 1
    self.samples = [(first, 0), (second, 1)]

  def __len__(self) -> int:
    """Return the number of synthetic samples.

    :return: Number of event samples.
    """
    return len(self.samples)

  def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
    """Return one synthetic event sample.

    :param index: Position of the requested sample.
    :return: Event array and integer class label.
    """
    return self.samples[index]

class FeasibilityTests(unittest.TestCase):
  """Validate feasibility summaries without external datasets."""

  def test_summary_reports_measured_shape_and_cost_assumptions(self) -> None:
    """Verify measured shapes, disk usage, and QRC assumptions.

    :return: ``None``.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
      dataset = FakeEventDataset(temporary_directory)
      Path(temporary_directory, "raw.bin").write_bytes(b"1234")
      summary = analyze_dataset(dataset, "shd", "train", sample_limit=2)
    self.assertEqual(summary.number_of_samples, 2)
    self.assertEqual(summary.number_of_classes, 2)
    self.assertEqual(summary.event_count.mean, 1.5)
    self.assertEqual(summary.preprocessing_output_features, 32)
    self.assertEqual(summary.disk_usage_bytes, 4)
    self.assertEqual(summary.qrc_cost_proxy["statevector_complex_amplitudes"], 64)

if __name__ == "__main__":
  unittest.main()
