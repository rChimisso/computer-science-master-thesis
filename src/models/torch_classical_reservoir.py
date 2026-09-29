from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from src.models.classical_reservoir import ClassicalReservoir

@dataclass(frozen=True)
class ResidentBatch:
  """Validated padded inputs retained on one device.

  :ivar values: Sanitized batch-time-channel values.
  :ivar lengths: Device-resident valid lengths.
  :ivar steps: Maximum valid length, obtained once during preparation.
  """

  values: torch.Tensor
  lengths: torch.Tensor
  steps: int

class TorchClassicalReservoir:
  """Inference-only device implementation using identical NumPy-generated weights.

  :ivar input_weights: Fixed input projection on the chosen device.
  :ivar recurrent_weights: Fixed recurrent matrix on the chosen device.
  :ivar bias: Fixed reservoir bias on the chosen device.
  :ivar leak: Fraction of the candidate state accepted at each step.
  """

  def __init__(self, reference: ClassicalReservoir, device: str = "cpu", dtype: torch.dtype = torch.float64):
    """Copy existing weights without changing initialization or frozen sources.

    :param reference: CPU reservoir defining the exact model.
    :param device: CPU or CUDA device.
    :param dtype: Explicit float64 reference precision or float32 diagnostic precision.
    """
    if dtype not in (torch.float32, torch.float64):
      raise ValueError("Only float32 and float64 are supported")
    self.input_weights = torch.tensor(reference.input_weights, device=device, dtype=dtype)
    self.recurrent_weights = torch.tensor(reference.recurrent_weights, device=device, dtype=dtype)
    self.bias = torch.tensor(reference.bias, device=device, dtype=dtype)
    self.leak = reference.parameters.leak_rate

  def prepare(self, values: np.ndarray | torch.Tensor, lengths: np.ndarray | torch.Tensor) -> ResidentBatch:
    """Validate and transfer a batch once, removing even nonfinite padded values.

    :param values: Padded inputs with the reference input width.
    :param lengths: Integer valid lengths, preferably supplied on CPU.
    :return: Resident inputs requiring no host transfers during recurrent execution.
    """
    host_lengths = lengths.detach().cpu().numpy() if isinstance(lengths, torch.Tensor) else np.asarray(lengths)
    if len(values.shape) != 3 or values.shape[2] != self.input_weights.shape[1] or values.shape[0] == 0:
      raise ValueError("Invalid input shape")
    if not np.issubdtype(host_lengths.dtype, np.integer) or host_lengths.shape != (len(values),):
      raise ValueError("One integer length per sample is required")
    if np.any(host_lengths <= 0) or np.any(host_lengths > values.shape[1]):
      raise ValueError("Lengths lie outside the padded input")
    steps = int(host_lengths.max())
    device = self.input_weights.device
    resident = torch.as_tensor(values, device=device, dtype=self.input_weights.dtype)[:, :steps]
    valid_lengths = torch.as_tensor(host_lengths, device=device, dtype=torch.int64)
    valid = torch.arange(steps, device=device)[None, :] < valid_lengths[:, None]
    resident = torch.where(valid[:, :, None], resident, 0.0)
    if not torch.isfinite(resident).all().item():
      raise ValueError("Valid inputs must be finite")
    return ResidentBatch(resident, valid_lengths, steps)

  @torch.inference_mode()
  def summarize(self, batch: ResidentBatch, segments: int, mode: str = "segment_mean_std_final", cancellation: Callable | None = None) -> torch.Tensor:
    """Accumulate compact summaries without retaining complete trajectories.

    :param batch: Inputs prepared by this backend on its device and precision.
    :param segments: Number of normalized temporal segments.
    :param mode: Segment means, optional population deviations, then final state.
    :param cancellation: Optional host-only cancellation check at every time step.
    :return: Device-resident summary matrix in the historical feature ordering.
    """
    if segments <= 0 or mode not in ("segment_mean_final", "segment_mean_std_final"):
      raise ValueError("Unsupported summary")
    if batch.values.device != self.input_weights.device or batch.values.dtype != self.input_weights.dtype:
      raise ValueError("Prepare the batch with this backend's device and dtype")
    samples = len(batch.values)
    width = len(self.bias)
    states = batch.values.new_zeros((samples, width))
    sums = batch.values.new_zeros((samples, segments, width))
    squares = torch.zeros_like(sums) if mode == "segment_mean_std_final" else None
    counts = batch.values.new_zeros((samples, segments, 1))
    for step in range(batch.steps):
      if cancellation is not None:
        cancellation()
      active = (step < batch.lengths)[:, None]
      candidate = torch.tanh(batch.values[:, step] @ self.input_weights.T + states @ self.recurrent_weights.T + self.bias)
      states = torch.where(active, (1.0 - self.leak) * states + self.leak * candidate, states)
      positions = torch.clamp(step * segments // batch.lengths, max=segments - 1)[:, None, None]
      indices = positions.expand(-1, 1, width)
      contribution = torch.where(active, states, 0.0)[:, None, :]
      sums.scatter_add_(1, indices, contribution)
      counts.scatter_add_(1, positions, active[:, None, :].to(counts.dtype))
      if squares is not None:
        squares.scatter_add_(1, indices, contribution.square())
    denominator = counts.clamp_min(1)
    means = sums / denominator
    if squares is None:
      return torch.cat((means.flatten(1), states), dim=1)
    deviations = (squares / denominator - means.square()).clamp_min(0).sqrt()
    return torch.cat((means.flatten(1), deviations.flatten(1), states), dim=1)
