import numpy as np

def bin_dvs_clip(events: np.ndarray, bin_us: int = 50_000, grid_size: int = 8) -> np.ndarray:
  """Bin one complete Tonic-exported gesture clip into polarity-separated counts.

  Tonic's local unstructured arrays store columns x, y, polarity, and milliseconds. Their loader multiplies time by a thousand and casts to integer microseconds.

  :param events: Local unstructured event matrix with four columns.
  :param bin_us: Positive temporal bin width in microseconds.
  :param grid_size: Spatial grid width and height dividing the sensor width.
  :return: Unpadded uint32 time-feature count matrix covering every event.
  """
  values = np.asarray(events)
  if values.ndim != 2 or values.shape[1] != 4 or not len(values) or not np.all(np.isfinite(values)):
    raise ValueError("DVS clip must contain finite x/y/p/t events")
  if bin_us <= 0 or grid_size <= 0 or 128 % grid_size:
    raise ValueError("Unsupported DVS time or grid resolution")
  coordinates = values[:, :3].astype(np.int64)
  if np.any(coordinates != values[:, :3]) or np.any(coordinates[:, :2] < 0) or np.any(coordinates[:, :2] >= 128) or np.any((coordinates[:, 2] < 0) | (coordinates[:, 2] > 1)):
    raise ValueError("DVS sensor coordinates or polarities are invalid")
  times = (values[:, 3] * 1_000).astype(np.int64)
  if np.any(times < 0) or np.any(np.diff(times) < 0):
    raise ValueError("DVS clip times must be nonnegative and ordered")
  time_bins = times // bin_us
  cell_width = 128 // grid_size
  channels = coordinates[:, 2] * grid_size * grid_size + (coordinates[:, 1] // cell_width) * grid_size + coordinates[:, 0] // cell_width
  counts = np.zeros((int(time_bins[-1]) + 1, 2 * grid_size * grid_size), dtype=np.uint32)
  np.add.at(counts, (time_bins, channels), 1)
  if int(counts.sum(dtype=np.uint64)) != len(values):
    raise ValueError("DVS binning did not conserve every event")
  return counts
