import numpy as np

from tests.support import SyntheticCase

class ReadoutTests(SyntheticCase):
  """Compare exported ridge inference with an independent estimator."""

  def test_readout_inference_matches_estimator_for_both_class_layouts(self):
    """Shared inference preserves binary and multiclass ridge decisions."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    from src.evaluation.readout import direct_readout, fit_readout, predict_readout
    generator = np.random.default_rng(42)
    training = generator.normal(size=(30, 7))
    held = generator.normal(size=(9, 7))
    for classes in (2, 3):
      labels = np.arange(len(training)) % classes
      targets = np.arange(len(held)) % classes
      scaler = StandardScaler().fit(training)
      for solver in ("cholesky", "lsqr"):
        with self.subTest(classes=classes, solver=solver):
          options = {"alpha": len(training) * 0.1, "solver": solver}
          if solver == "lsqr":
            options.update({"tol": 0.001, "max_iter": 1000})
            local = self.config | {"readout": {"tolerance": 0.001, "max_iterations": 1000}}
            result, state = fit_readout(training, held, labels, targets, 0.1, classes, local)
          else:
            result, state, _ = direct_readout(training, held, labels, targets, 0.1, classes)
          estimator = RidgeClassifier(**options).fit(scaler.transform(training), labels)
          np.testing.assert_array_equal(predict_readout(held, state), estimator.predict(scaler.transform(held)))
          np.testing.assert_array_equal(result["train_predictions"], predict_readout(training, state))
