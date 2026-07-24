from collections.abc import Iterable

import numpy as np

def speaker_disjoint_indices(speakers: np.ndarray, validation_speakers: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
  """Split training indices by speaker without random sample leakage.

  :param speakers: Speaker identifier for every sample.
  :param validation_speakers: Speaker identifiers reserved for validation.
  :return: Training indices followed by validation indices.
  """
  speaker_ids = np.asarray(speakers)
  validation_ids = np.asarray(list(validation_speakers), dtype=speaker_ids.dtype)
  validation_mask = np.isin(speaker_ids, validation_ids)
  return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)
