import os
import random

import numpy as np

def set_global_seed(seed: int) -> np.random.Generator:
  """Seed supported process-level random-number generators.

  :param seed: Integer seed shared by Python and NumPy.
  :return: Explicit NumPy random-number generator initialized with ``seed``.
  """
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  return np.random.default_rng(seed)
