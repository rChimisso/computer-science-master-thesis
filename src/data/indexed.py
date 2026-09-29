from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass
class StudyData:
  """Indexed ragged official-training data.

  :ivar name: Dataset identifier.
  :ivar sequences: Raw count arrays or paths, one per complete example.
  :ivar labels: Integer class identities.
  :ivar groups: Speaker or subject identities.
  :ivar identities: Globally unique source sample identities.
  :ivar development: Indices eligible for inner-fold fitting.
  :ivar confirmation: Existing held-out development indices.
  :ivar classes: Registered class count.
  :ivar bin_us: Physical time-bin width.
  :ivar fingerprint: Source cache provenance hash.
  :ivar lighting: Lighting metadata, or empty strings for SHD.
  :ivar raw_fingerprint: Content identity of the original training files, when available.
  """

  name: str
  sequences: list
  labels: np.ndarray
  groups: np.ndarray
  identities: list[str]
  development: np.ndarray
  confirmation: np.ndarray
  classes: int
  bin_us: int
  fingerprint: str
  lighting: list[str]
  raw_fingerprint: str | None = None

  def counts(self, index: int) -> np.ndarray:
    """Read one complete count sequence.

    :param index: Global dataset row.
    :return: Unpadded time-channel counts.
    """
    value = self.sequences[int(index)]
    return np.load(value, mmap_mode="r") if isinstance(value, (str, Path)) else value

  def values(self, index: int) -> np.ndarray:
    """Transform a single complete sequence without fitted statistics.

    :param index: Global dataset row.
    :return: Float32 log-count sequence.
    """
    return np.log1p(np.asarray(self.counts(index), dtype=np.float32))

  def membership(self, indices: np.ndarray) -> dict:
    """Serialize explicit source membership for provenance.

    :param indices: Global dataset rows in their persisted order.
    :return: Labels, groups, and unique source identities.
    """
    return {
      "identities": [self.identities[int(i)] for i in indices],
      "labels": self.labels[indices].tolist(),
      "groups": self.groups[indices].tolist()
    }

def study_folds(data: StudyData, config: dict) -> list[dict]:
  """Construct the frozen inner group folds and validate coverage.

  :param data: Official-training dataset.
  :param config: Study configuration containing SHD held groups.
  :return: Explicit training and validation indices for every fold.
  """
  groups = sorted(np.unique(data.groups[data.development]).tolist())
  held = config["folds"] if data.name == "shd" else [groups[i::4] for i in range(4)]
  folds = []
  covered = []
  for number, validation_groups in enumerate(held):
    validation = data.development[np.isin(data.groups[data.development], validation_groups)]
    training = data.development[~np.isin(data.groups[data.development], validation_groups)]
    validate_partition(data, training, validation)
    covered.extend(validation.tolist())
    folds.append({
      "fold": number,
      "held_groups": validation_groups,
      "train": training.tolist(),
      "validation": validation.tolist()
    })
  if sorted(covered) != sorted(data.development.tolist()):
    raise ValueError("Inner folds do not partition the development examples exactly once")
  return folds

def validate_partition(data: StudyData, training: np.ndarray, validation: np.ndarray) -> None:
  """Reject example, group, or class violations in a fitted partition.

  :param data: Dataset metadata.
  :param training: Fitting indices.
  :param validation: Held group indices.
  """
  if len(np.unique(training)) != len(training) or len(np.unique(validation)) != len(validation) or np.intersect1d(training, validation).size:
    raise ValueError("Duplicate or overlapping sample membership")
  if set(data.groups[training]) & set(data.groups[validation]):
    raise ValueError("Speaker or subject leakage")
  for indices in (training, validation):
    if set(data.labels[indices]) != set(range(data.classes)):
      raise ValueError("A partition lacks a registered class")

def nested_indices(data: StudyData, training: np.ndarray, count: int, seed: int = 42) -> np.ndarray:
  """Select a nested proportional class-and-group-stratified prefix.

  :param data: Dataset labels and groups.
  :param training: Eligible source indices.
  :param count: Requested prefix length, capped at available examples.
  :param seed: Membership random seed independent of model initialization.
  :return: Deterministic nested source membership in sorted order.
  """
  generator = np.random.default_rng(seed)
  cells = sorted({(int(data.labels[i]), int(data.groups[i])) for i in training})
  queues = [generator.permutation([i for i in training if (data.labels[i], data.groups[i]) == cell]).tolist() for cell in cells]
  sizes = np.asarray([len(queue) for queue in queues])
  used = np.zeros(len(queues), dtype=np.int64)
  selected = []
  for step in range(min(count, len(training))):
    deficits = (step + 1) * sizes / len(training) - used
    deficits[used >= sizes] = -np.inf
    cell = int(np.argmax(deficits))
    selected.append(queues[cell][used[cell]])
    used[cell] += 1
  result = np.sort(np.asarray(selected, dtype=np.int64))
  if set(data.labels[result]) != set(range(data.classes)):
    raise ValueError("Nested subset does not preserve class coverage")
  return result

def padded_batch(data: StudyData, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Materialize only one ragged float32 minibatch.

  :param data: Sequence source.
  :param indices: Requested global rows.
  :return: Log-count values, lengths, and labels.
  """
  sequences = [data.values(int(i)) for i in indices]
  lengths = np.asarray([len(value) for value in sequences], dtype=np.int64)
  values = np.zeros((len(indices), int(lengths.max()), sequences[0].shape[1]), dtype=np.float32)
  for row, sequence in enumerate(sequences):
    values[row, :len(sequence)] = sequence
  return values, lengths, data.labels[indices]
