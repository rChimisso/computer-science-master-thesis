from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data.indexed import study_folds
from src.features.extraction import extract
from src.runtime.configuration import bind, load
from src.runtime.records import checksum

from tests.support import SyntheticCase

class CacheTests(SyntheticCase):
  """Protect reusable features and numerical cache identities."""

  def test_reuse_and_integrity(self):
    """Reuse compatible banks and reject corruption or different settings."""
    with self.subTest(case='cache_resume_and_corruption'):
      self.reset_case()
      fold = study_folds(self.data, self.config)[0]
      train = np.asarray(fold["train"])
      indices = np.asarray(fold["train"] + fold["validation"])
      spec = {
        "model": "crc",
        "inputs": 2,
        "size": 12,
        "radius": 0.9,
        "scale": 0.03,
        "tau": 20
      }
      left, record, adapter = extract(self.data, train, indices, spec, 42, self.config)
      right, replay_record, _ = extract(self.data, train, indices, spec | {"solver": "lsqr"}, 42, self.config)
      self.assertEqual(record["directory"], replay_record["directory"])
      np.testing.assert_array_equal(left, right)
      path = Path(record["directory"]) / "all.npy"
      values = np.load(path)
      values[0, 0] += 1
      np.save(path, values)
      with self.assertRaisesRegex(ValueError, "changed"):
        extract(self.data, train, indices, spec, 42, self.config)
    with self.subTest(case='cache_reuse_and_numerical_rejection'):
      self.reset_case()
      fold = study_folds(self.data, self.config)[0]
      train = np.asarray(fold["train"])
      indices = np.asarray(fold["train"] + fold["validation"])
      spec = {"model": "projection", "inputs": 2}
      first = extract(self.data, train, indices, spec, 42, self.config)
      other = bind(load(), self.root / "other")
      other["paths"]["cache"] = self.config["paths"]["cache"]
      other["_identity"] = "another-run-name"
      second = extract(self.data, train, indices, spec, 42, other)
      self.assertEqual(first[1]["directory"], second[1]["directory"])
      self.assertTrue(second[1]["cache_hit"])
      changed = extract(self.data, train, indices, spec | {"inputs": 4}, 42, other)
      self.assertNotEqual(changed[1]["directory"], first[1]["directory"])
    with self.subTest(case='projection_cache_and_lock_are_shared_across_dynamics_seeds'):
      self.reset_case()
      from src.runtime.numerical import identity
      indices = self.data.development[:6]
      spec = {"model": "projection", "inputs": 2}
      with patch("src.features.extraction.numerical_identity", wraps=identity):
        first, metadata, _ = extract(self.data, indices, indices, spec, 42, self.config)
      locks = {path.name for path in (self.root / "shared/locks").iterdir()}
      second, repeated, _ = extract(self.data, indices, indices, spec, 44, self.config)
      np.testing.assert_array_equal(first, second)
      self.assertEqual(metadata["directory"], repeated["directory"])
      self.assertTrue(repeated["cache_hit"])
      self.assertEqual(locks, {path.name for path in (self.root / "shared/locks").iterdir()})

  def test_numerical_invalidation(self):
    """Changed transformations and loaders must invalidate numerical caches."""
    with self.subTest(case='input_code_changes_reject_feature_and_adapter_caches'):
      self.reset_case()
      spec = {"model": "projection", "inputs": 2, "adapter": "legacy"}
      indices = self.data.development[:6]
      first, before, adapter = extract(self.data, indices, indices, spec, 42, self.config)

      def changed(path):
        """Represent a numerical input implementation change without editing source.

        :param path: Source or artifact to fingerprint.
        :return: Real checksum except for the deliberately changed input implementation.
        """
        return "changed-input-transform" if str(path) == "src/data/indexed.py" else checksum(path)

      with patch("src.runtime.numerical.checksum", side_effect=changed), patch("src.features.adapters.checksum", side_effect=changed):
        second, after, updated = extract(self.data, indices, indices, spec, 42, self.config)
      self.assertFalse(after["cache_hit"])
      self.assertNotEqual(before["directory"], after["directory"])
      self.assertNotEqual(adapter["fingerprint"], updated["fingerprint"])
      np.testing.assert_array_equal(first, second)
    with self.subTest(case='inline_dataset_loader_changes_reject_count_cache'):
      self.reset_case()
      import h5py
      from src.data.datasets import load as load_data
      path = self.root / "fixture.h5"
      with h5py.File(path, "w") as data:
        data.create_dataset("labels", data=[0])
        data.create_dataset("extra/speaker", data=[2])
        data.create_dataset("spikes/times", data=[[0.001]])
        data.create_dataset("spikes/units", data=[[1]])
      for version in ("before", "after"):
        with patch("src.data.datasets.acquire", return_value=path), patch("src.data.datasets.checksum", side_effect=lambda name: version if str(name).endswith("src/data/datasets.py") else checksum(name)):
          with self.assertRaisesRegex(ValueError, "Incomplete shd/train"):
            load_data("shd", self.config)
      self.assertEqual(len(list((self.root / "shared/datasets").iterdir())), 2)
