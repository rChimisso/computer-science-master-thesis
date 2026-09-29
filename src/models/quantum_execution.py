import multiprocessing as mp
import time
from collections import OrderedDict
from collections.abc import Callable

import numpy as np

def quantum_worker(connection, specification: dict, device: str, threads: int) -> None:
  """Run isolated exact Aer jobs and retain bounded parameterized templates.

  :param connection: Duplex process connection owned by the parent executor.
  :param specification: Quantum size, input width, depth, seed and observation groups.
  :param device: Explicit Aer CPU or GPU device.
  :param threads: Maximum native execution threads.
  """
  from src.runtime.native import bootstrap
  bootstrap()
  from qiskit import QuantumCircuit
  from qiskit.circuit import ParameterVector
  from qiskit.quantum_info import Pauli
  from qiskit_aer import AerSimulator
  from qiskit_aer.library import save_density_matrix, save_expectation_value
  try:
    qubits = specification["qubits"]
    inputs = specification["inputs"]
    generator = np.random.default_rng(specification["seed"])
    evolution = QuantumCircuit(qubits)
    angles = np.asarray(specification.get("gate_angles", generator.uniform(-np.pi, np.pi, (specification["depth"], qubits, 2))))
    for layer in range(specification["depth"]):
      for qubit in range(qubits):
        evolution.ry(float(angles[layer, qubit, 0]), qubit)
        evolution.rz(float(angles[layer, qubit, 1]), qubit)
      for qubit in range(qubits):
        evolution.cz(qubit, (qubit + 1) % qubits)
    operators = []
    labels = []
    for group in specification["observables"]:
      for qubit in range(qubits):
        symbols = ["I"] * qubits
        symbols[qubits - qubit - 1] = "Z" if group == "zz_ring" else group.upper()
        if group == "zz_ring":
          symbols[qubits - ((qubit + 1) % qubits) - 1] = "Z"
        operators.append(Pauli("".join(symbols)))
        labels.append(f"{group}:{qubit}")
    if "observable_indices" in specification:
      operators = [operators[index] for index in specification["observable_indices"]]
      labels = [labels[index] for index in specification["observable_indices"]]
    simulator = AerSimulator(method="density_matrix", device=device, precision="double", enable_truncation=False, max_parallel_threads=threads, max_parallel_experiments=1, zero_threshold=0)
    templates = OrderedDict()
    connection.send({"ready": True, "labels": labels})
    while True:
      request = connection.recv()
      if request is None:
        break
      sequences, diagnostic = request
      circuits = []
      construction = 0.0
      binding = 0.0
      for values in sequences:
        key = (len(values), diagnostic)
        started = time.perf_counter()
        if key not in templates:
          parameters = ParameterVector("input", len(values) * inputs)
          circuit = QuantumCircuit(qubits)
          for step in range(len(values)):
            circuit.reset(list(range(qubits if specification["reset_mode"] == "all" else inputs)))
            for qubit in range(inputs):
              circuit.ry(parameters[step * inputs + qubit], qubit)
            circuit.compose(evolution, inplace=True)
            for index, operator in enumerate(operators):
              save_expectation_value(circuit, operator, list(range(qubits)), label=f"t{step}_o{index}")
          if diagnostic:
            save_density_matrix(circuit, label="final")
          templates[key] = (circuit, parameters)
          if len(templates) > 16:
            templates.popitem(last=False)
        templates.move_to_end(key)
        circuit, parameters = templates[key]
        construction += time.perf_counter() - started
        started = time.perf_counter()
        angles = 2 * np.arcsin(np.sqrt(values)).ravel()
        circuits.append(circuit.assign_parameters(dict(zip(parameters, angles))))
        binding += time.perf_counter() - started
      started = time.perf_counter()
      result = simulator.run(circuits, shots=1).result()
      execution = time.perf_counter() - started
      if not result.success:
        raise RuntimeError(str(result.status))
      started = time.perf_counter()
      outputs = []
      states = []
      metadata = []
      for index, values in enumerate(sequences):
        details = result.results[index].metadata
        if details.get("device") != device or details.get("method") != "density_matrix":
          raise RuntimeError("Aer executed on a different device or method")
        record = result.data(index)
        outputs.append(np.asarray([[record[f"t{step}_o{j}"] for j in range(len(operators))] for step in range(len(values))], dtype=np.float64))
        states.append(np.asarray(record["final"]) if diagnostic else None)
        metadata.append({
          "device": details["device"],
          "method": details["method"],
          "parallel_experiments": details.get("parallel_experiments")
        })
      connection.send({
        "values": outputs,
        "states": states,
        "metadata": metadata,
        "construction_seconds": construction,
        "binding_seconds": binding,
        "execution_seconds": execution,
        "recovery_seconds": time.perf_counter() - started
      })
  except BaseException as error:
    try:
      connection.send({"error": f"{type(error).__name__}: {error}"})
    except (BrokenPipeError, EOFError, OSError):
      pass
  finally:
    connection.close()

class AerReservoir:
  """Cancellable process-isolated exact reservoir with reusable circuit templates.

  :ivar specification: Validated exact quantum configuration.
  :ivar connection: Parent endpoint of the worker connection.
  :ivar process: Isolated simulator process, never shared with another job.
  :ivar labels: Stable observable column labels.
  """

  def __init__(self, specification: dict, device: str = "CPU", threads: int = 1, cancellation: Callable | None = None):
    """Start an explicitly selected Aer worker without any device fallback.

    :param specification: Total/input qubits, depth, seed and observable groups.
    :param device: CPU or GPU.
    :param threads: Positive native thread limit.
    :param cancellation: Optional stop check while the worker initializes.
    """
    if (specification["qubits"], specification["inputs"]) not in ((4, 2), (6, 2), (6, 4), (8, 2), (8, 4), (8, 6)):
      raise ValueError("Only declared allocations are permitted")
    if specification["depth"] not in (1, 2, 3) or device not in ("CPU", "GPU") or threads not in (1, 4):
      raise ValueError("Unsupported execution settings")
    if not specification["observables"] or any(group not in ("x", "y", "z", "zz_ring") for group in specification["observables"]):
      raise ValueError("Unsupported observables")
    self.specification = {"reset_mode": "partial"} | dict(specification)
    if self.specification["reset_mode"] not in ("partial", "all"):
      raise ValueError("Unsupported reset mode")
    context = mp.get_context("spawn")
    self.connection, child = context.Pipe()
    self.process = context.Process(target=quantum_worker, args=(child, self.specification, device, threads), daemon=True)
    self.process.start()
    child.close()
    try:
      self.labels = self.receive(cancellation)["labels"]
    except BaseException:
      self.close()
      raise

  def receive(self, cancellation: Callable | None = None) -> dict:
    """Wait with bounded polling and abort the isolated worker when cancelled.

    :param cancellation: Optional deadline or stop-file callback.
    :return: Complete worker response.
    """
    try:
      while not self.connection.poll(0.1):
        if cancellation is not None:
          cancellation()
        if not self.process.is_alive():
          raise RuntimeError("Aer worker exited without a complete result")
      result = self.connection.recv()
      if "error" in result:
        raise RuntimeError(result["error"])
      if cancellation is not None:
        cancellation()
      return result
    except BaseException:
      self.close()
      raise

  def transform(self, sequences: list[np.ndarray], diagnostic: bool = False, cancellation: Callable | None = None) -> dict:
    """Extract complete independent ragged sequences without padded evolution.

    :param sequences: Valid probability arrays, with no fitted transformation here.
    :param diagnostic: Include final density matrices for correctness checks only.
    :param cancellation: Stop callback checked during native execution.
    :return: Exact ordered trajectories, optional states, and timing components.
    """
    values = [np.asarray(sequence, dtype=np.float64) for sequence in sequences]
    for value in values:
      if value.ndim != 2 or len(value) == 0 or value.shape[1] != self.specification["inputs"] or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError("Expected finite complete encoding probabilities")
    if not values:
      raise ValueError("An empty simulator job is not allowed")
    if cancellation is not None:
      cancellation()
    self.connection.send((values, diagnostic))
    return self.receive(cancellation)

  def close(self) -> None:
    """Stop the worker and release its native CPU/GPU allocations."""
    if self.process.is_alive():
      self.process.terminate()
      self.process.join(timeout=5)
      if self.process.is_alive():
        self.process.kill()
        self.process.join(timeout=5)
    self.connection.close()
