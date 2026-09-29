import ctypes
import os
from pathlib import Path

def bootstrap() -> list[str]:
  """Preload relocated Aer dependencies without changing the installed package.

  :return: Explicitly loaded native library paths.
  """
  root = Path(__file__).resolve().parents[2] / "environment/local/native"
  loaded = []
  for name in ("libquadmath.so.0", "libgfortran.so.5", "libopenblas.so.0"):
    matches = sorted(root.rglob(name)) if root.exists() else []
    if matches:
      ctypes.CDLL(str(matches[0]), mode=os.RTLD_GLOBAL)
      loaded.append(str(matches[0]))
  return loaded
