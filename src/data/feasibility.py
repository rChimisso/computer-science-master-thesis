import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MATPLOTLIB_CONFIG_DIRECTORY = "/tmp/qrc-spiking-matplotlib"
"""Temporary writable directory used by Matplotlib for its runtime configuration."""

os.environ.setdefault("MPLCONFIGDIR", MATPLOTLIB_CONFIG_DIRECTORY)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tonic

DVS_GESTURE_TRAIN_URL = "https://zenodo.org/records/8060604/files/ibmGestureTrain.tar.gz?download=1"
"""Maintained URL for the DVS Gesture training archive."""

DVS_GESTURE_TEST_URL = "https://zenodo.org/records/8060604/files/ibmGestureTest.tar.gz?download=1"
"""Maintained URL for the DVS Gesture test archive."""

@dataclass(frozen=True)
class DistributionSummary:
  """Summary statistics for one measured distribution."""

  minimum: float
  """Minimum observed value."""

  median: float
  """Median observed value."""

  mean: float
  """Arithmetic mean of the observed values."""

  p95: float
  """Ninety-fifth percentile of the observed values."""

  maximum: float
  """Maximum observed value."""

@dataclass(frozen=True)
class DatasetSummary:
  """Measured feasibility information for one dataset split."""

  dataset: str
  """Canonical dataset name."""

  split: str
  """Dataset split that was measured."""

  number_of_samples: int
  """Total number of samples in the split."""

  number_of_classes: int
  """Total number of represented classes."""

  sensor_size: tuple[int, ...]
  """Native sensor dimensions reported by Tonic."""

  raw_input_dimensions: int
  """Product of the native sensor dimensions."""

  event_fields: tuple[str, ...]
  """Names of the fields in the structured event arrays."""

  inspected_samples: int
  """Number of samples used for estimated distributions."""

  event_count: DistributionSummary
  """Distribution of events per inspected sample."""

  duration_ms: DistributionSummary
  """Distribution of inspected sample durations in milliseconds."""

  class_balance: dict[str, int]
  """Exact number of samples belonging to each class."""

  disk_usage_bytes: int
  """Local disk usage of the dataset in bytes."""

  preprocessing_seconds: float
  """Total preprocessing probe time in seconds."""

  preprocessing_seconds_per_sample: float
  """Mean preprocessing probe time per inspected sample."""

  preprocessing_output_features: int
  """Number of features produced by the preprocessing probe."""

  qrc_cost_proxy: dict[str, float | int | str]
  """Assumptions and values for the relative QRC simulation cost proxy."""

def _distribution(values: list[float]) -> DistributionSummary:
  """Calculate fixed descriptive statistics.

  :param values: Measured values to summarize.
  :return: Summary containing the minimum, median, mean, ninety-fifth percentile, and maximum.
  :raises ValueError: If ``values`` is empty.
  """
  array = np.asarray(values, dtype=np.float64)
  if not len(array):
    raise ValueError("At least one measured sample is required")
  return DistributionSummary(
    minimum=float(array.min()),
    median=float(np.median(array)),
    mean=float(array.mean()),
    p95=float(np.percentile(array, 95)),
    maximum=float(array.max())
  )

def _sample_indices(length: int, limit: int) -> np.ndarray:
  """Select deterministic indices distributed across a dataset.

  :param length: Number of available samples.
  :param limit: Maximum number of indices to return.
  :return: Evenly spaced integer indices.
  :raises ValueError: If ``length`` or ``limit`` is not positive.
  """
  if length <= 0 or limit <= 0:
    raise ValueError("Dataset length and sample limit must be positive")
  count = min(length, limit)
  return np.linspace(0, length - 1, num=count, dtype=np.int64)

def _directory_size(path: Path) -> int:
  """Calculate recursive disk usage.

  :param path: Directory whose files should be measured.
  :return: Total size of all files below ``path`` in bytes.
  """
  return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())

def _labels(dataset: Any) -> np.ndarray:
  """Load all labels without materializing every event array.

  :param dataset: Tonic dataset with either ``targets`` or an HDF5 label dataset.
  :return: One integer label per sample.
  """
  if hasattr(dataset, "targets") and len(dataset.targets):
    return np.asarray(dataset.targets, dtype=np.int64)
  data_path = Path(dataset.location_on_system) / dataset.data_filename
  with h5py.File(data_path, "r") as data_file:
    return np.asarray(data_file["labels"], dtype=np.int64)

def _preprocess_probe(events: np.ndarray, dataset_name: str, temporal_bin_us: int, shd_pooled_channels: int, dvs_spatial_bins: int) -> np.ndarray:
  """Bin one event stream into a compact feasibility representation.

  :param events: Structured event array supplied by Tonic.
  :param dataset_name: Canonical dataset name.
  :param temporal_bin_us: Width of each temporal bin in microseconds.
  :param shd_pooled_channels: Number of adjacent SHD channel groups.
  :param dvs_spatial_bins: Number of DVS spatial bins along each axis.
  :return: Integer event-count tensor indexed by time and feature.
  """
  if len(events) == 0:
    feature_count = shd_pooled_channels if dataset_name == "shd" else 2 * dvs_spatial_bins**2
    return np.zeros((1, feature_count), dtype=np.int32)
  time_steps = max(1, int(events["t"].max()) // temporal_bin_us + 1)
  time_indices = np.minimum(events["t"] // temporal_bin_us, time_steps - 1)
  if dataset_name == "shd":
    feature_indices = np.minimum(events["x"] * shd_pooled_channels // 700, shd_pooled_channels - 1)
    feature_count = shd_pooled_channels
  else:
    x_bins = np.minimum(events["x"] * dvs_spatial_bins // 128, dvs_spatial_bins - 1)
    y_bins = np.minimum(events["y"] * dvs_spatial_bins // 128, dvs_spatial_bins - 1)
    polarities = events["p"].astype(np.int64)
    feature_indices = (polarities * dvs_spatial_bins + y_bins) * dvs_spatial_bins + x_bins
    feature_count = 2 * dvs_spatial_bins**2
  binned = np.zeros((time_steps, feature_count), dtype=np.int32)
  np.add.at(binned, (time_indices, feature_indices), 1)
  return binned

def load_dataset(name: str, raw_data_path: str | Path, train: bool = True) -> Any:
  """Load a supported candidate through Tonic.

  Tonic ``1.6.0`` references retired Figshare downloader identifiers for DVS Gesture. The maintained Zenodo archives are byte-identical and retain the hashes validated by Tonic.

  :param name: Candidate name, such as ``shd`` or ``dvs``.
  :param raw_data_path: Directory used for raw downloads.
  :param train: Whether to load the training split.
  :return: Loaded Tonic dataset.
  :raises ValueError: If the candidate name is unsupported.
  """
  root = str(raw_data_path)
  normalized_name = name.lower().replace("_", "-")
  if normalized_name == "shd":
    return tonic.datasets.SHD(save_to=root, train=train)
  if normalized_name in {"dvs", "dvs-gesture", "dvsgesture"}:
    tonic.datasets.DVSGesture.train_url = DVS_GESTURE_TRAIN_URL
    tonic.datasets.DVSGesture.test_url = DVS_GESTURE_TEST_URL
    return tonic.datasets.DVSGesture(save_to=root, train=train)
  raise ValueError(f"Unsupported candidate dataset: {name}")

def analyze_dataset(dataset: Any, name: str, split: str, sample_limit: int = 100, temporal_bin_us: int = 10_000, shd_pooled_channels: int = 32, dvs_spatial_bins: int = 8, qrc_qubits: int = 6, qrc_circuit_depth: int = 2) -> DatasetSummary:
  """Measure a deterministic subset and report all feasibility assumptions.

  :param dataset: Loaded Tonic dataset.
  :param name: Candidate dataset name.
  :param split: Name of the measured split.
  :param sample_limit: Maximum number of samples used for estimated distributions.
  :param temporal_bin_us: Width of each preprocessing probe bin in microseconds.
  :param shd_pooled_channels: Number of adjacent SHD channel groups.
  :param dvs_spatial_bins: Number of DVS spatial bins along each axis.
  :param qrc_qubits: Number of simulated qubits assumed by the cost proxy.
  :param qrc_circuit_depth: Circuit depth assumed by the cost proxy.
  :return: Complete measured dataset summary.
  """
  normalized_name = "shd" if name.lower() == "shd" else "dvs_gesture"
  indices = _sample_indices(len(dataset), sample_limit)
  event_counts: list[float] = []
  durations_ms: list[float] = []
  elapsed = 0.0
  output_features = 0
  for index in indices:
    events, _ = dataset[int(index)]
    event_counts.append(float(len(events)))
    duration_us = float(events["t"].max() - events["t"].min()) if len(events) else 0.0
    durations_ms.append(duration_us / 1_000)
    start = time.perf_counter()
    binned = _preprocess_probe(events, normalized_name, temporal_bin_us, shd_pooled_channels, dvs_spatial_bins)
    elapsed += time.perf_counter() - start
    output_features = binned.shape[1]

  labels = _labels(dataset)
  class_balance = {str(label): count for label, count in sorted(Counter(labels.tolist()).items())}
  sensor_size = tuple(int(size) for size in dataset.sensor_size)
  raw_dimensions = int(np.prod(sensor_size))
  mean_steps = max(1.0, _distribution(durations_ms).mean * 1_000 / temporal_bin_us)
  statevector_size = 2**qrc_qubits
  work_per_sample = mean_steps * qrc_circuit_depth * qrc_qubits * statevector_size
  disk_path = Path(dataset.location_on_system)
  return DatasetSummary(
    dataset=normalized_name,
    split=split,
    number_of_samples=len(dataset),
    number_of_classes=len(np.unique(labels)),
    sensor_size=sensor_size,
    raw_input_dimensions=raw_dimensions,
    event_fields=tuple(dataset.dtype.names),
    inspected_samples=len(indices),
    event_count=_distribution(event_counts),
    duration_ms=_distribution(durations_ms),
    class_balance=class_balance,
    disk_usage_bytes=_directory_size(disk_path),
    preprocessing_seconds=elapsed,
    preprocessing_seconds_per_sample=elapsed / len(indices),
    preprocessing_output_features=output_features,
    qrc_cost_proxy={
      "description": "mean_bins * depth * qubits * 2**qubits; relative simulator work, not wall time",
      "qubits": qrc_qubits,
      "circuit_depth": qrc_circuit_depth,
      "statevector_complex_amplitudes": statevector_size,
      "mean_time_bins": mean_steps,
      "work_units_per_sample": work_per_sample,
      "work_units_full_split": work_per_sample * len(dataset)
    }
  )

def save_summary(summary: DatasetSummary, path: str | Path) -> None:
  """Save a summary as stable, human-readable JSON.

  :param summary: Dataset summary to serialize.
  :param path: Destination JSON path.
  :return: ``None``.
  """
  output_path = Path(path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

def plot_event_samples(dataset: Any, name: str, path: str | Path, count: int = 10) -> None:
  """Create a visual inspection sheet from samples distributed across a split.

  :param dataset: Loaded Tonic dataset.
  :param name: Candidate dataset name.
  :param path: Destination image path.
  :param count: Number of samples to display.
  :return: ``None``.
  """
  indices = _sample_indices(len(dataset), count)
  figure, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
  for axis, index in zip(axes.flat, indices, strict=True):
    events, label = dataset[int(index)]
    if name.lower() == "shd":
      axis.scatter(events["t"] / 1_000, events["x"], s=0.3, alpha=0.4)
      axis.set_xlabel("time (ms)")
      axis.set_ylabel("channel")
    else:
      display_events = events[::max(1, len(events) // 20_000)]
      colors = np.where(display_events["p"], "tab:red", "tab:blue")
      axis.scatter(display_events["x"], display_events["y"], c=colors, s=0.2, alpha=0.25)
      axis.set_xlim(0, 127)
      axis.set_ylim(127, 0)
      ticks = np.arange(0, 129, 32)
      axis.set_xticks(ticks)
      axis.set_yticks(ticks)
      axis.set_aspect("equal")
      axis.set_xlabel("x")
      axis.set_ylabel("y")
    axis.set_title(f"sample {int(index)}, class {int(label)}")
  output_path = Path(path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=160)
  plt.close(figure)

def plot_shd_diagnostics(dataset: Any, path: str | Path, sample_limit: int = 100) -> None:
  """Plot SHD channel counts, durations, event timing, and speaker metadata.

  :param dataset: Loaded SHD training dataset.
  :param path: Destination image path.
  :param sample_limit: Maximum samples used for channel and temporal-event counts.
  :return: ``None``.
  """
  indices = _sample_indices(len(dataset), sample_limit)
  channel_counts = np.zeros(700, dtype=np.int64)
  normalized_event_times: list[np.ndarray] = []
  for index in indices:
    events, _ = dataset[int(index)]
    channel_counts += np.bincount(events["x"], minlength=700)
    if len(events) and events["t"].max() > 0:
      normalized_event_times.append(events["t"] / events["t"].max())

  data_path = Path(dataset.location_on_system) / dataset.data_filename
  with h5py.File(data_path, "r") as data_file:
    timestamps = data_file["spikes/times"]
    durations_ms = np.fromiter((float(times[-1]) * 1_000 if len(times) else 0.0 for times in timestamps), dtype=np.float64, count=len(timestamps))
    speakers = np.asarray(data_file["extra/speaker"], dtype=np.int64)

  figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
  axes[0, 0].plot(channel_counts)
  axes[0, 0].set(title="Spike counts by cochlear channel", xlabel="channel", ylabel="events")
  axes[0, 1].hist(durations_ms, bins=40)
  axes[0, 1].set(title="Full training duration distribution", xlabel="duration (ms)", ylabel="samples")
  axes[1, 0].hist(np.concatenate(normalized_event_times), bins=50)
  axes[1, 0].set(title="Events over normalized sample time", xlabel="relative time", ylabel="events")
  speaker_ids, speaker_counts = np.unique(speakers, return_counts=True)
  axes[1, 1].bar(speaker_ids.astype(str), speaker_counts)
  axes[1, 1].set(title="Official speaker metadata", xlabel="speaker ID", ylabel="samples")
  output_path = Path(path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=160)
  plt.close(figure)
