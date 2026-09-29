import math

import numpy as np

from src.models.classical_reservoir import ClassicalReservoir, ClassicalReservoirParameters

def classical_reference(data, spec: dict, seed: int) -> ClassicalReservoir:
  """Create the exact historical CRC initialization with physical leak conversion.

  :param data: Dataset bin width and channel metadata.
  :param spec: Reservoir size and dynamics.
  :param seed: Initialization seed.
  :return: Fixed NumPy reference reservoir.
  """
  parameters = ClassicalReservoirParameters(spec["size"], spec["radius"], -math.expm1(-data.bin_us / (1000 * spec["tau"])), spec["scale"], 0.5)
  return ClassicalReservoir(spec["inputs"] or data.counts(0).shape[1], parameters, seed)

def quantum_views(values: np.ndarray, spec: dict) -> dict[str, np.ndarray]:
  """Derive declared observable controls without reconstructing missing correlations.

  :param values: Full exact observation bank in XYZ/ring-ZZ order.
  :param spec: Quantum allocation.
  :return: Sequential view arrays to summarize independently.
  """
  qubits = spec["qubits"]
  inputs = spec["inputs"]
  selected = [group * qubits + qubit for group in range(3) for qubit in (0, 1, inputs, inputs + 1)]
  result = {"all": values[:, :3 * qubits], "fixed12": values[:, selected]}
  if qubits == 6:
    local = values[:, 2 * qubits:3 * qubits]
    if values.shape[1] != 4 * qubits:
      raise ValueError("Joint observations require an extracted quantum superset")
    result.update({"zz": values, "products": np.concatenate((result["all"], local * np.roll(local, -1, axis=1)), axis=1)})
  return result
