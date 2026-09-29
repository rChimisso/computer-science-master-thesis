from pathlib import Path

import numpy as np

from src.runtime.progress import stage, track
from src.data.indexed import StudyData
from src.runtime.records import checksum, digest, read, stop, write

@stage("features.adapters.fit_adapter")
def fit_adapter(data: StudyData, training: np.ndarray, mode: str, width: int, seed: int, config: dict) -> dict:
  """Fit one fold-local input adapter using valid training bins only.

  :param data: Raw count source.
  :param training: Fitting membership.
  :param mode: Legacy, orthogonal, pooling, PCA, or identity mapping.
  :param width: Number of probability coordinates.
  :param seed: Projection seed independent of reservoir dynamics.
  :param config: Study paths and cancellation settings.
  :return: Serializable affine statistics and transformation metadata.
  """
  channels = data.counts(int(training[0])).shape[1]
  identity = {
    "data": data.fingerprint,
    "training": data.membership(training),
    "mode": mode,
    "width": width,
    "seed": seed,
    "source": {"adapter": checksum(__file__), "inputs": checksum("src/data/indexed.py")}
  }
  path = Path(config["paths"]["results"]) / "adapters" / f"{digest(identity)}.json"
  if path.exists():
    saved = read(path)
    if saved.get("identity") != identity or saved.get("fingerprint") != digest({key: value for key, value in saved.items() if key != "fingerprint"}):
      raise ValueError("Fitted adapter identity or checksum changed")
    return saved
  result = {
    "mode": mode,
    "width": width,
    "seed": seed,
    "identity": identity
  }
  if mode == "identity":
    result["fingerprint"] = digest(result)
    write(path, result)
    return result
  if not 0 < width <= channels:
    raise ValueError("Projection width must fit the input channels")
  generator = np.random.default_rng(seed)
  weights = np.linalg.qr(generator.normal(0.0, 1.0, size=(channels, width)))[0].T.copy()
  bias = generator.uniform(-np.pi, np.pi, size=width)
  if mode == "pooling":
    weights = np.zeros((width, channels), dtype=np.float64)
    if data.name == "dvs":
      for polarity in range(2):
        for side in range(width // 2):
          columns = [polarity * 64 + y * 8 + x for y in range(8) for x in range(8) if width == 2 or x // 4 == side]
          weights[polarity * (width // 2) + side, columns] = 1.0 / len(columns)
    else:
      for row, columns in enumerate(np.array_split(np.arange(channels), width)):
        weights[row, columns] = 1.0 / len(columns)
  if mode not in ("legacy", "orthogonal", "pooling", "pca"):
    raise ValueError("Unknown adapter")
  if mode != "legacy":
    count = 0
    sums = np.zeros(channels, dtype=np.float64)
    products = np.zeros((channels, channels), dtype=np.float64)
    for index in track(training, f"{data.name}/{mode}/width={width}/adapter-statistics", unit="samples"):
      stop(config)
      values = data.values(int(index)).astype(np.float64)
      count += len(values)
      sums += values.sum(axis=0)
      products += values.T @ values
    means = sums / count
    covariance = products / count - np.outer(means, means)
    if mode == "pca":
      eigenvalues, vectors = np.linalg.eigh(covariance)
      weights = vectors[:, np.argsort(eigenvalues)[::-1][:width]].T
      for row in weights:
        if row[np.argmax(np.abs(row))] < 0:
          row *= -1
    projected_means = weights @ means
    deviations = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", weights, covariance, weights), 0))
    result["means"] = projected_means.tolist()
    result["scales"] = np.where(deviations > 1e-8, deviations, 1.0).tolist()
    result["valid_training_bins"] = count
    bias = np.zeros(width)
  result["weights"] = weights.tolist()
  result["bias"] = bias.tolist()
  result["fingerprint"] = digest(result)
  write(path, result)
  return result

def apply_adapter(values: np.ndarray, adapter: dict) -> np.ndarray:
  """Apply the persisted mapping exactly once to valid sequence values.

  :param values: Unpadded log-count inputs.
  :param adapter: Serializable fitted adapter.
  :return: Shared probabilities or unchanged direct inputs.
  """
  if adapter["mode"] == "identity":
    return np.asarray(values, dtype=np.float64)
  affine = np.asarray(values, dtype=np.float64) @ np.asarray(adapter["weights"]).T + np.asarray(adapter["bias"])
  if adapter["mode"] != "legacy":
    affine = (affine - np.asarray(adapter["means"])) / np.asarray(adapter["scales"])
  return 1.0 / (1.0 + np.exp(-np.clip(0.5 * affine, -700, 700)))
