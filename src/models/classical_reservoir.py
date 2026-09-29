from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class ClassicalReservoirParameters:
  """Fixed classical reservoir parameters.

  :ivar reservoir_size: Number of recurrent state units.
  :ivar spectral_radius: Target magnitude of the largest recurrent eigenvalue.
  :ivar leak_rate: Fraction of each candidate state accepted per step.
  :ivar input_scaling: Scale of fixed input weights.
  :ivar sparsity: Fraction of recurrent weights set to zero.
  :ivar temporal_segments: Number of normalized temporal state summaries.
  """

  reservoir_size: int
  """Number of recurrent state units."""

  spectral_radius: float
  """Target magnitude of the largest recurrent eigenvalue."""

  leak_rate: float
  """Fraction of each candidate state accepted per step."""

  input_scaling: float
  """Scale of fixed input weights."""

  sparsity: float
  """Fraction of recurrent weights set to zero."""

  temporal_segments: int = 8
  """Number of normalized temporal state summaries."""

  def validate(self) -> None:
    """Validate reservoir parameters.

    :raises ValueError: If a parameter lies outside its supported range.
    """
    if self.reservoir_size <= 0:
      raise ValueError("Reservoir size must be positive")
    if self.spectral_radius <= 0:
      raise ValueError("Spectral radius must be positive")
    if not 0 < self.leak_rate <= 1:
      raise ValueError("Leak rate must be in the interval (0, 1]")
    if self.input_scaling <= 0:
      raise ValueError("Input scaling must be positive")
    if not 0 <= self.sparsity < 1:
      raise ValueError("Sparsity must be in the interval [0, 1)")
    if self.temporal_segments <= 0:
      raise ValueError("Temporal segment count must be positive")

class ClassicalReservoir:
  """Fixed Echo State Network reservoir with temporal state summaries.

  The recurrent update is ``x(t) = (1-a)x(t-1) + a*tanh(W_in*u(t) + W*x(t-1) + b)``, where ``a`` is the leak rate. Only the downstream linear readout is trained.
  """

  parameters: ClassicalReservoirParameters
  """Validated reservoir parameters."""

  input_dimension: int
  """Number of input features per time step."""

  seed: int
  """Seed controlling all fixed reservoir weights."""

  input_weights: np.ndarray
  """Fixed input projection matrix."""

  recurrent_weights: np.ndarray
  """Fixed recurrent reservoir matrix."""

  bias: np.ndarray
  """Fixed reservoir bias vector."""

  def __init__(self, input_dimension: int, parameters: ClassicalReservoirParameters, seed: int):
    """Initialize deterministic fixed reservoir weights.

    :param input_dimension: Number of features in each input time step.
    :param parameters: Reservoir hyperparameters.
    :param seed: Random seed for fixed weights.
    """
    parameters.validate()
    if input_dimension <= 0:
      raise ValueError("Input dimension must be positive")
    self.parameters = parameters
    self.input_dimension = input_dimension
    self.seed = seed
    generator = np.random.default_rng(seed)
    size = parameters.reservoir_size
    self.input_weights = generator.uniform(-parameters.input_scaling, parameters.input_scaling, size=(size, input_dimension))
    recurrent = generator.uniform(-1.0, 1.0, size=(size, size))
    recurrent[generator.random((size, size)) < parameters.sparsity] = 0.0
    radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
    if radius == 0:
      raise ValueError("Recurrent matrix has zero spectral radius; reduce sparsity or change seed")
    self.recurrent_weights = recurrent * parameters.spectral_radius / radius
    self.bias = generator.uniform(-0.1, 0.1, size=size)

  @property
  def output_dimension(self) -> int:
    """Return the segmented-state and final-state summary dimension.

    :return: Reservoir size times the temporal segments plus final state.
    """
    return (self.parameters.temporal_segments + 1) * self.parameters.reservoir_size

  def transform(self, sequences: np.ndarray, lengths: np.ndarray | None = None) -> np.ndarray:
    """Generate fixed reservoir features for padded sequences.

    :param sequences: Input array shaped ``(samples, time, features)``.
    :param lengths: Optional number of valid time steps per sample.
    :return: Concatenated segment-mean and final reservoir states.
    :raises ValueError: If input shapes are incompatible.
    """
    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != self.input_dimension:
      raise ValueError(f"Expected input shape (samples, time, {self.input_dimension})")
    sample_count, time_steps, _ = values.shape
    valid_lengths = np.full(sample_count, time_steps, dtype=np.int64) if lengths is None else np.asarray(lengths, dtype=np.int64)
    if valid_lengths.shape != (sample_count,) or np.any(valid_lengths <= 0) or np.any(valid_lengths > time_steps):
      raise ValueError("Sequence lengths must contain one valid length per sample")
    states = np.zeros((sample_count, self.parameters.reservoir_size), dtype=np.float64)
    segment_sums = np.zeros((sample_count, self.parameters.temporal_segments, self.parameters.reservoir_size), dtype=np.float64)
    segment_counts = np.zeros((sample_count, self.parameters.temporal_segments), dtype=np.int64)
    for time_step in range(time_steps):
      active = time_step < valid_lengths
      if not np.any(active):
        break
      candidate = np.tanh(values[active, time_step] @ self.input_weights.T + states[active] @ self.recurrent_weights.T + self.bias)
      states[active] = (1.0 - self.parameters.leak_rate) * states[active] + self.parameters.leak_rate * candidate
      active_indices = np.flatnonzero(active)
      segments = np.minimum(time_step * self.parameters.temporal_segments // valid_lengths[active], self.parameters.temporal_segments - 1)
      np.add.at(segment_sums, (active_indices, segments), states[active])
      np.add.at(segment_counts, (active_indices, segments), 1)
    segment_means = segment_sums / np.maximum(segment_counts[:, :, np.newaxis], 1)
    return np.concatenate((segment_means.reshape(sample_count, -1), states), axis=1)

  def transform_sequential(self, sequences: np.ndarray, lengths: np.ndarray | None = None) -> np.ndarray:
    """Generate every valid fixed-reservoir state while preserving zero padding.

    :param sequences: Input array shaped ``(samples, time, features)``.
    :param lengths: Optional number of valid time steps per sample.
    :return: Sequential state tensor with zero values after every valid length.
    :raises ValueError: If input shapes or lengths are incompatible.
    """
    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != self.input_dimension:
      raise ValueError(f"Expected input shape (samples, time, {self.input_dimension})")
    sample_count, time_steps, _ = values.shape
    valid_lengths = np.full(sample_count, time_steps, dtype=np.int64) if lengths is None else np.asarray(lengths, dtype=np.int64)
    if valid_lengths.shape != (sample_count,) or np.any(valid_lengths <= 0) or np.any(valid_lengths > time_steps):
      raise ValueError("Sequence lengths must contain one valid length per sample")
    states = np.zeros((sample_count, self.parameters.reservoir_size), dtype=np.float64)
    sequential = np.zeros((sample_count, time_steps, self.parameters.reservoir_size), dtype=np.float64)
    for time_step in range(time_steps):
      active = time_step < valid_lengths
      if not np.any(active):
        break
      candidate = np.tanh(values[active, time_step] @ self.input_weights.T + states[active] @ self.recurrent_weights.T + self.bias)
      states[active] = (1.0 - self.parameters.leak_rate) * states[active] + self.parameters.leak_rate * candidate
      sequential[active, time_step] = states[active]
    return sequential
