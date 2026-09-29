import numpy as np

def summarize_sequence_features(features: np.ndarray, lengths: np.ndarray, temporal_segments: int, mode: str) -> np.ndarray:
  """Summarize sequential features with one normalized-time strategy.

  Empty normalized segments remain zero. Standard deviations use the population definition.

  :param features: Sequential features shaped ``(samples, time, features)``.
  :param lengths: Number of valid time steps per sample.
  :param temporal_segments: Number of normalized temporal segments.
  :param mode: Requested summary strategy.
  :return: Flattened fixed-length feature matrix.
  :raises ValueError: If shapes, lengths, or the segment count are invalid.
  """
  values = np.asarray(features)
  if not np.issubdtype(values.dtype, np.floating):
    values = values.astype(np.float64)
  valid_lengths = np.asarray(lengths, dtype=np.int64)
  if values.ndim != 3:
    raise ValueError("Sequential features must have three dimensions")
  if valid_lengths.shape != (len(values),):
    raise ValueError("Lengths must contain one value per sample")
  if temporal_segments <= 0:
    raise ValueError("Temporal segment count must be positive")
  if mode not in ("segment_mean_final", "segment_mean_std_final"):
    raise ValueError("Unsupported temporal summary")
  sample_count, time_steps, feature_dimension = values.shape
  if np.any(valid_lengths <= 0) or np.any(valid_lengths > time_steps):
    raise ValueError("Sequence lengths lie outside the feature tensor")
  segment_sums = np.zeros((sample_count, temporal_segments, feature_dimension), dtype=values.dtype)
  segment_squares = np.zeros((sample_count, temporal_segments, feature_dimension), dtype=values.dtype) if mode == "segment_mean_std_final" else None
  segment_counts = np.zeros((sample_count, temporal_segments), dtype=np.int64)
  for time_step in range(time_steps):
    active = time_step < valid_lengths
    if not np.any(active):
      break
    active_indices = np.flatnonzero(active)
    segments = np.minimum(time_step * temporal_segments // valid_lengths[active], temporal_segments - 1)
    np.add.at(segment_sums, (active_indices, segments), values[active, time_step])
    if segment_squares is not None:
      np.add.at(segment_squares, (active_indices, segments), values[active, time_step] ** 2)
    np.add.at(segment_counts, (active_indices, segments), 1)
  means = segment_sums / np.maximum(segment_counts[:, :, np.newaxis], 1)
  final = values[np.arange(sample_count), valid_lengths - 1]
  if mode == "segment_mean_std_final":
    second_moments = segment_squares / np.maximum(segment_counts[:, :, np.newaxis], 1)
    standard_deviations = np.sqrt(np.maximum(second_moments - means**2, 0.0))
    return np.concatenate((means.reshape(sample_count, -1), standard_deviations.reshape(sample_count, -1), final), axis=1)
  return np.concatenate((means.reshape(sample_count, -1), final), axis=1)
