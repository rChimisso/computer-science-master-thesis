from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def classification_metrics(labels: np.ndarray, predictions: np.ndarray, classes: int = 20) -> dict[str, Any]:
  """Compute comparable full-class metrics including absent-class handling.

  :param labels: True integer classes.
  :param predictions: Predicted integer classes.
  :param classes: Number of registered classes.
  :return: Accuracy, macro-F1, per-class report, and confusion matrix.
  """
  class_ids = list(range(classes))
  return {
    "accuracy": float(accuracy_score(labels, predictions)),
    "macro_f1": float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)),
    "per_class": classification_report(labels, predictions, labels=class_ids, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist()
  }

def metric_names() -> tuple:
  """Define the four audited metrics.

  :return: Stable column order.
  """
  return (
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall"
  )

def confusion_metrics(matrix: np.ndarray) -> dict:
  """Compute metrics directly from integer counts, independently of old reporters.

  :param matrix: True-label by predicted-label counts with every dataset class.
  :return: Accuracy and uniformly macro-averaged metrics; zero denominators yield zero.
  """
  matrix = np.asarray(matrix)
  if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0) or matrix.sum() == 0:
    raise ValueError("Invalid confusion matrix")
  diagonal = matrix.diagonal().astype(float)
  support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
  precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted != 0)
  recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support != 0)
  f1 = np.divide(2 * diagonal, support + predicted, out=np.zeros_like(diagonal), where=support + predicted != 0)
  return {
    "accuracy": float(diagonal.sum() / matrix.sum()),
    "macro_f1": float(f1.mean()),
    "macro_precision": float(precision.mean()),
    "macro_recall": float(recall.mean())
  }

def prediction_metrics(labels: list, predictions: list, classes: int) -> tuple:
  """Rebuild a complete confusion matrix from saved sample predictions.

  :param labels: Membership-bound true classes.
  :param predictions: Predictions in the identical sample order.
  :param classes: Dataset class count.
  :return: Four metrics and the reconstructed confusion matrix.
  """
  labels, predictions = np.asarray(labels), np.asarray(predictions)
  if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
    raise ValueError("Prediction membership mismatch")
  if np.any(labels < 0) or np.any(labels >= classes) or np.any(predictions < 0) or np.any(predictions >= classes):
    raise ValueError("Prediction class outside dataset")
  matrix = np.bincount((classes * labels + predictions).astype(int), minlength=classes * classes).reshape(classes, classes)
  return confusion_metrics(matrix), matrix

def assert_metrics(actual: dict, expected: dict) -> None:
  """Require tight agreement with saved scalar or classification-report metrics.

  :param actual: Independently reconstructed metrics.
  :param expected: Prior scalar row or full report.
  """
  for name in metric_names():
    if name in expected:
      value = expected[name]
    else:
      value = expected["per_class"]["macro avg"][name.removeprefix("macro_")]
    np.testing.assert_allclose(actual[name], float(value), atol=1e-13, rtol=1e-12)
