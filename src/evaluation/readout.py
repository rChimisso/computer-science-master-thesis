import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import classification_metrics
from src.runtime.records import checksum, stop

def predict_readout(features: np.ndarray, state) -> np.ndarray:
  """Apply stored training normalization and ridge coefficients in class order.

  :param features: Ordered temporal-summary rows.
  :param state: Mapping containing fitted normalization, coefficients and classes.
  :return: Predicted class identifiers with binary RidgeClassifier semantics.
  """
  scores = ((features - state["means"]) / state["scales"]) @ state["coefficients"].T + state["intercept"]
  classes = state["classes"]
  return classes[(scores.ravel() > 0).astype(int)] if len(classes) == 2 else classes[scores.argmax(axis=1)]

def direct_readout(training: np.ndarray, validation: np.ndarray, train_labels: np.ndarray, held_labels: np.ndarray, penalty: float, classes: int) -> tuple:
  """Solve the identical standardized RidgeClassifier objective by dense Cholesky.

  :param training: Raw training summary features.
  :param validation: Raw held-group summary features.
  :param train_labels: Training classes.
  :param held_labels: Held-group classes.
  :param penalty: Mean-loss ridge regularization.
  :param classes: Dataset output class count.
  :return: Diagnostics, persisted state and held decision scores.
  """
  started = time.perf_counter()
  scaler = StandardScaler()
  fitted = scaler.fit_transform(np.asarray(training, dtype=np.float64))
  held = scaler.transform(validation)
  classifier = RidgeClassifier(alpha=len(training) * penalty, solver="cholesky")
  classifier.fit(fitted, train_labels)
  seconds = time.perf_counter() - started
  scores = classifier.decision_function(held)
  predicted = classifier.predict(held)
  centered = fitted - fitted.mean(axis=0)
  target = np.where(train_labels[:, None] == classifier.classes_[None], 1.0, -1.0)
  if len(classifier.classes_) == 2:
    target = target[:, 1:]
  target -= target.mean(axis=0)
  right = centered.T @ target
  coefficients = np.atleast_2d(classifier.coef_).T
  residual = centered.T @ (centered @ coefficients) + len(training) * penalty * coefficients - right
  relative = float(np.linalg.norm(residual) / max(np.linalg.norm(right), np.finfo(float).tiny))
  if not np.isfinite(scores).all() or relative > 1e-8:
    raise ValueError(f"Direct ridge failed normal-equation residual check: {relative}")
  result = {
    "status": "completed",
    "solver": "cholesky",
    "lambda": penalty,
    "alpha": len(training) * penalty,
    "relative_normal_equation_residual": relative,
    "feature_dimension": training.shape[1],
    "readout_norm": float(np.linalg.norm(classifier.coef_)),
    "feature_variance": np.var(training, axis=0).tolist(),
    "fitting_seconds": seconds,
    "train": classification_metrics(train_labels, classifier.predict(fitted), classes),
    "validation": classification_metrics(held_labels, predicted, classes),
    "predictions": predicted.tolist(),
    "train_predictions": classifier.predict(fitted).tolist()
  }
  state = {
    "coefficients": classifier.coef_,
    "intercept": classifier.intercept_,
    "classes": classifier.classes_,
    "means": scaler.mean_,
    "scales": scaler.scale_
  }
  return result, state, scores

def save_state(directory: Path, label: str, state: dict) -> dict:
  """Atomically persist a compact fitted readout.

  :param directory: Existing destination directory.
  :param label: Stable artifact basename.
  :param state: Numerical readout arrays.
  :return: Artifact path and checksum.
  """
  temporary = directory / "state.tmp.npz"
  path = directory / f"{label}.npz"
  np.savez(temporary, **state)
  temporary.replace(path)
  return {"state_path": str(path), "state_sha256": checksum(path)}

def fit_readout(training: np.ndarray, validation: np.ndarray, train_labels: np.ndarray, validation_labels: np.ndarray, penalty: float, classes: int, config: dict, maximum: int | None = None) -> tuple[dict, dict]:
  """Fit sample-normalized standardized ridge and report convergence.

  :param training: Fitting features.
  :param validation: Held-group features.
  :param train_labels: Fitting labels.
  :param validation_labels: Held-group labels.
  :param penalty: Regularization in the mean-loss convention.
  :param classes: Registered output count.
  :param config: Common solver settings.
  :param maximum: Explicit common iteration ceiling.
  :return: Metrics and serializable fitted coefficients/statistics.
  """
  stop(config)
  started = time.perf_counter()
  scaler = StandardScaler()
  transformed = scaler.fit_transform(training)
  held = scaler.transform(validation)
  limit = maximum or config["readout"]["max_iterations"]
  classifier = RidgeClassifier(alpha=len(training) * penalty, solver="lsqr", tol=config["readout"]["tolerance"], max_iter=limit)
  classifier.fit(transformed, train_labels)
  iterations = np.asarray(classifier.n_iter_).astype(int)
  fitting = time.perf_counter() - started
  inference_started = time.perf_counter()
  predicted = classifier.predict(held)
  inference = time.perf_counter() - inference_started
  result = {
    "status": "completed" if np.all(iterations < limit) else "solver_limit",
    "solver": "lsqr",
    "lambda": penalty,
    "alpha": len(training) * penalty,
    "max_iterations": limit,
    "iterations": iterations.tolist(),
    "feature_dimension": training.shape[1],
    "readout_norm": float(np.linalg.norm(classifier.coef_)),
    "feature_variance": np.var(training, axis=0).tolist(),
    "fitting_seconds": fitting,
    "inference_seconds": inference,
    "train": classification_metrics(train_labels, classifier.predict(transformed), classes),
    "validation": classification_metrics(validation_labels, predicted, classes),
    "predictions": predicted.tolist()
  }
  state = {
    "coefficients": classifier.coef_,
    "intercept": classifier.intercept_,
    "classes": classifier.classes_,
    "means": scaler.mean_,
    "scales": scaler.scale_
  }
  result["train_predictions"] = predict_readout(training, state).tolist()
  return result, state
