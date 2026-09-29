from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class PreprocessingSpec:
  """Configuration for deterministic SHD event binning.

  :ivar temporal_bin_us: Width of one time step in microseconds.
  :ivar pooled_channels: Number of adjacent cochlear channel groups.
  :ivar value_mode: Event aggregation mode.
  :ivar fixed_duration_us: Padded sequence duration in microseconds.
  """

  temporal_bin_us: int
  """Width of one time step in microseconds."""

  pooled_channels: int
  """Number of adjacent cochlear channel groups."""

  value_mode: str
  """Event aggregation mode: ``binary``, ``count``, or ``normalized_count``."""

  fixed_duration_us: int
  """Padded sequence duration in microseconds."""

  def validate(self) -> None:
    """Validate preprocessing parameters.

    :raises ValueError: If any parameter is unsupported or nonpositive.
    """
    if self.temporal_bin_us <= 0:
      raise ValueError("Temporal bin width must be positive")
    if self.pooled_channels <= 0:
      raise ValueError("Pooled channel count must be positive")
    if self.fixed_duration_us <= 0:
      raise ValueError("Fixed duration must be positive")
    if self.value_mode not in {
      "binary",
      "count",
      "normalized_count"
    }:
      raise ValueError(f"Unsupported event value mode: {self.value_mode}")

  @property
  def time_steps(self) -> int:
    """Return the number of padded time steps.

    :return: Ceiling of fixed duration divided by temporal-bin width.
    """
    return (self.fixed_duration_us + self.temporal_bin_us - 1) // self.temporal_bin_us

  @property
  def cache_name(self) -> str:
    """Return a stable directory name for this representation.

    :return: Human-readable cache key.
    """
    bin_ms = self.temporal_bin_us / 1_000
    duration_ms = self.fixed_duration_us / 1_000
    return f"shd_bin-{bin_ms:g}ms_channels-{self.pooled_channels}_value-{self.value_mode}_duration-{duration_ms:g}ms"

def bin_shd_events(events: np.ndarray, spec: PreprocessingSpec) -> tuple[np.ndarray, int]:
  """Convert one SHD event stream into a padded time-feature tensor.

  Events beyond the configured duration are deterministically truncated. The default duration is longer than every sample in the current official SHD train and test files.

  :param events: Structured SHD events with ``t`` and ``x`` fields.
  :param spec: Validated preprocessing configuration.
  :return: Binned tensor and number of non-padding time steps.
  """
  spec.validate()
  counts = np.zeros((spec.time_steps, spec.pooled_channels), dtype=np.uint16)
  if len(events) == 0:
    return _convert_event_values(counts, spec.value_mode), 1
  time_indices = events["t"].astype(np.int64) // spec.temporal_bin_us
  valid = time_indices < spec.time_steps
  valid_time_indices = time_indices[valid]
  channels = events["x"][valid].astype(np.int64)
  pooled_indices = np.minimum(channels * spec.pooled_channels // 700, spec.pooled_channels - 1)
  np.add.at(counts, (valid_time_indices, pooled_indices), 1)
  observed_steps = min(spec.time_steps, int(time_indices.max()) + 1)
  return _convert_event_values(counts, spec.value_mode), max(1, observed_steps)

def _convert_event_values(counts: np.ndarray, value_mode: str) -> np.ndarray:
  """Convert raw event counts to the configured value representation.

  :param counts: Unsigned integer event counts.
  :param value_mode: Requested event aggregation mode.
  :return: Converted event tensor.
  """
  if value_mode == "binary":
    return (counts > 0).astype(np.uint8)
  if value_mode == "normalized_count":
    maximum = int(counts.max())
    return counts.astype(np.float32) / maximum if maximum else counts.astype(np.float32)
  return counts
