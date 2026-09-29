import math
import time
from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import torch

from src.models.classical_reservoir import ClassicalReservoir
from src.models.torch_classical_reservoir import ResidentBatch, TorchClassicalReservoir

class StreamingCRC(TorchClassicalReservoir):
  """Streaming reservoir summaries with optional nonlinear controls and CUDA graphs.

  :ivar graphs: Bounded graph cache keyed by static operation shapes.
  :ivar use_graphs: Whether CUDA capture is requested explicitly.
  """

  def __init__(self, reference: ClassicalReservoir, device: str, graphs: bool = False):
    """Reuse historical weights and retain a bounded set of graph buffers.

    :param reference: Historical seeded reservoir.
    :param device: CPU or CUDA.
    :param graphs: Request captured execution, only on CUDA.
    """
    super().__init__(reference, device, torch.float64)
    if graphs and device != "cuda":
      raise ValueError("Graph execution requires CUDA")
    self.graphs = OrderedDict()
    self.use_graphs = graphs

  @torch.inference_mode()
  def summaries(self, batch: ResidentBatch, segments: int, views: tuple[str, ...], cancellation: Callable | None = None) -> dict[str, torch.Tensor]:
    """Accumulate requested views before temporal aggregation.

    :param batch: Device-resident padded values and valid lengths.
    :param segments: Normalized temporal segments.
    :param views: All states or states plus six cyclic products of the last six units.
    :param cancellation: Host stop check between steps, disabled during graph capture.
    :return: Device-resident mean/deviation/final summaries for every requested view.
    """
    if segments <= 0 or not views or any(view not in ("all", "products") for view in views):
      raise ValueError("Unsupported summary view")
    if "products" in views and len(self.bias) < 6:
      raise ValueError("Product controls require at least six states")
    states = batch.values.new_zeros((len(batch.values), len(self.bias)))
    total = torch.zeros_like(states)
    squared = torch.zeros_like(states)
    boundary = torch.zeros_like(states)
    minimum = torch.full_like(states, float("inf"))
    maximum = torch.full_like(states, float("-inf"))
    accumulators = {}
    for view in views:
      width = len(self.bias) + (6 if view == "products" else 0)
      shape = (len(states), segments, width)
      accumulators[view] = (batch.values.new_zeros(shape), batch.values.new_zeros(shape))
    counts = batch.values.new_zeros((len(states), segments, 1))
    final = {}
    for step in range(batch.steps):
      if cancellation is not None:
        cancellation()
      active = (step < batch.lengths)[:, None]
      candidate = torch.tanh(batch.values[:, step] @ self.input_weights.T + states @ self.recurrent_weights.T + self.bias)
      states = torch.where(active, (1 - self.leak) * states + self.leak * candidate, states)
      total += torch.where(active, states, 0)
      squared += torch.where(active, states.square(), 0)
      boundary += (active & (states.abs() > 0.99)).to(states.dtype)
      minimum = torch.where(active, torch.minimum(minimum, states), minimum)
      maximum = torch.where(active, torch.maximum(maximum, states), maximum)
      positions = (step * segments // batch.lengths).clamp_max(segments - 1)[:, None, None]
      counts.scatter_add_(1, positions, active[:, None, :].to(counts.dtype))
      for view, (sums, squares) in accumulators.items():
        values = states if view == "all" else torch.cat((states, states[:, -6:] * states[:, -6:].roll(-1, dims=1)), dim=1)
        contribution = torch.where(active, values, 0.0)[:, None, :]
        indices = positions.expand(-1, 1, values.shape[1])
        sums.scatter_add_(1, indices, contribution)
        squares.scatter_add_(1, indices, contribution.square())
        final[view] = values
    output = {}
    for view, (sums, squares) in accumulators.items():
      means = sums / counts.clamp_min(1)
      deviations = (squares / counts.clamp_min(1) - means.square()).clamp_min(0).sqrt()
      output[view] = torch.cat((means.flatten(1), deviations.flatten(1), final[view]), dim=1)
    output["_moments"] = torch.stack((total, squared, minimum, maximum, boundary), dim=1)
    return output

  @torch.inference_mode()
  def extract(self, sequences: list[np.ndarray], segments: int, views: tuple[str, ...], cancellation: Callable | None = None) -> tuple[dict, dict]:
    """Prepare bounded resident inputs and transfer only completed summaries back.

    :param sequences: Complete unpadded examples in requested output order.
    :param segments: Summary segment count.
    :param views: Requested feature transformations.
    :param cancellation: Stop callback before, during eager execution, and after replay.
    :return: Host summary matrices and measured preparation/execution/download costs.
    """
    if cancellation is not None:
      cancellation()
    started = time.perf_counter()
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    steps = int(lengths.max())
    steps = int(math.ceil(steps / 16) * 16) if self.use_graphs else steps
    values = np.zeros((len(sequences), steps, self.input_weights.shape[1]), dtype=np.float64)
    for index, sequence in enumerate(sequences):
      values[index, :len(sequence)] = sequence
    prepared = self.prepare(values, lengths)
    if prepared.steps != steps:
      prepared = ResidentBatch(torch.nn.functional.pad(prepared.values, (0, 0, 0, steps - prepared.steps)), prepared.lengths, steps)
    cuda = self.input_weights.is_cuda
    if cuda:
      torch.cuda.synchronize()
    preparation = time.perf_counter() - started
    started = time.perf_counter()
    capture = 0.0
    if self.use_graphs:
      key = (len(sequences), steps, segments, views)
      if key not in self.graphs:
        capture_started = time.perf_counter()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
          for _ in range(3):
            self.summaries(prepared, segments, views)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
          output = self.summaries(prepared, segments, views)
        self.graphs[key] = (graph, prepared, output, stream)
        if len(self.graphs) > 2:
          self.graphs.popitem(last=False)
        capture = time.perf_counter() - capture_started
      self.graphs.move_to_end(key)
      graph, resident, output, _ = self.graphs[key]
      resident.values.copy_(prepared.values)
      resident.lengths.copy_(prepared.lengths)
      graph.replay()
    else:
      output = self.summaries(prepared, segments, views, cancellation)
    if cuda:
      torch.cuda.synchronize()
    execution = time.perf_counter() - started
    started = time.perf_counter()
    result = {name: values.cpu().numpy().copy() for name, values in output.items()}
    statistics = result.pop("_moments")
    if cancellation is not None:
      cancellation()
    return result, {
      "preparation_seconds": preparation,
      "execution_seconds": execution,
      "capture_seconds": capture,
      "download_seconds": time.perf_counter() - started,
      "state_statistics": [{
        "count": int(length),
        "sum": row[0].tolist(),
        "squares": row[1].tolist(),
        "minimum": row[2].tolist(),
        "maximum": row[3].tolist(),
        "absolute_boundary": row[4].tolist(),
        "probability_boundary": np.zeros(row.shape[1]).tolist()
      } for length, row in zip(lengths, statistics)]
    }
