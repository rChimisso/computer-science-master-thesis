import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.splits import speaker_disjoint_indices
from src.utils.config import load_config
from src.utils.reproducibility import set_global_seed
from src.utils.results import append_json_result

class FoundationTests(unittest.TestCase):
  """Validate the reproducible software foundation."""

  def test_default_config_loads(self) -> None:
    """Verify the frozen dataset and validation-speaker configuration.

    :return: ``None``.
    """
    config = load_config()
    self.assertEqual(config["dataset"]["primary"], "shd")
    self.assertEqual(config["dataset"]["shd"]["validation_speakers"], [0, 1])

  def test_seed_is_deterministic(self) -> None:
    """Verify that repeated seeds produce identical random values.

    :return: ``None``.
    """
    first = set_global_seed(42).integers(
      0,
      100,
      size=10
    )
    second = set_global_seed(42).integers(
      0,
      100,
      size=10
    )
    np.testing.assert_array_equal(first, second)

  def test_json_result_log_is_append_only(self) -> None:
    """Verify that result records are appended rather than overwritten.

    :return: ``None``.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
      path = Path(temporary_directory) / "results.jsonl"
      append_json_result(
        path,
        {"seed": 42},
        {"accuracy": 0.5}
      )
      append_json_result(
        path,
        {"seed": 43},
        {"accuracy": 0.6}
      )
      records = [json.loads(line) for line in path.read_text().splitlines()]
      self.assertEqual(len(records), 2)
      self.assertEqual(records[1]["parameters"]["seed"], 43)

  def test_speaker_split_has_no_overlap(self) -> None:
    """Verify that training and validation speaker partitions do not overlap.

    :return: ``None``.
    """
    speakers = np.asarray(
      [
        0,
        1,
        2,
        0,
        3,
        1
      ]
    )
    train, validation = speaker_disjoint_indices(speakers, [0, 1])
    self.assertEqual(set(train).intersection(validation), set())
    self.assertEqual(speakers[train].tolist(), [2, 3])
    self.assertEqual(
      speakers[validation].tolist(),
      [
        0,
        1,
        0,
        1
      ]
    )

if __name__ == "__main__":
  unittest.main()
