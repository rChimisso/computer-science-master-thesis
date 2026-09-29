import csv
from pathlib import Path

import numpy as np

def metrics() -> tuple:
  """Name classification quantities in a stable presentation order.

  :return: Four metric keys.
  """
  return ("accuracy", "macro_f1", "macro_precision", "macro_recall")

def export(root: Path, topic: str, name: str, rows: list) -> None:
  """Write exhaustive CSV without discarding numerical precision.

  :param root: Staged report root.
  :param topic: Scientific topic.
  :param name: Stable product name.
  :param rows: Flat numerical records.
  """
  if not rows:
    return
  directory = root / topic / "tables"
  directory.mkdir(parents=True, exist_ok=True)
  columns = list(dict.fromkeys(key for row in rows for key in row))
  with (directory / f"{name}.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

def aggregate(cells: list, seeds: list, strict: bool = True) -> list:
  """Average folds within seeds and reject missing declared replication cells.

  :param cells: Individual numerical rows.
  :param seeds: Default required initialization seeds.
  :param strict: Raise on incomplete groups, otherwise leave them out of summaries.
  :return: Complete seed summaries with separate fold variability.
  """
  grouped = {}
  for row in cells:
    keys = ("dataset", "phase", "role", "population") + (("group", "kind") if "group" in row else ())
    identity = tuple((key, row[key]) for key in keys)
    grouped.setdefault(identity, []).append(row)
  result = []
  for identity, rows in grouped.items():
    common = dict(identity)
    phase = common["phase"]
    required = [42] if phase in ("mapping-screen", "mapping-robustness") else seeds
    folds = set(range(4)) if phase in ("development", "mapping-screen", "mapping-development", "mapping-robustness", "learning-development") else {rows[0]["fold"]}
    expected = {(str(fold), seed) for fold in folds for seed in required}
    actual = [(str(row["fold"]), row["seed"]) for row in rows]
    if set(actual) != expected or len(set(actual)) != len(actual):
      if strict:
        raise ValueError(f"Incomplete or duplicate required seed/fold cells: {common}")
      continue
    common.update({"seeds": len(required), "folds": len(folds)})
    for name in metrics():
      values = [float(np.mean([row[name] for row in rows if row["seed"] == seed])) for seed in required]
      folded = [float(np.mean([row[name] for row in rows if str(row["fold"]) == str(fold)])) for fold in folds]
      common[name] = float(np.mean(values))
      common[name + "_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
      common[name + "_fold_sd"] = float(np.std(folded, ddof=1)) if len(folded) > 1 else None
    result.append(common)
  return result

def contrasts(evidence: dict) -> tuple:
  """Compute explicitly declared intervention pairs with membership and input checks.

  :param evidence: Validated run cells and declarations.
  :return: Individual differences, aggregate differences and coverage rows.
  """
  cells = evidence["cells"]
  config = evidence["manifest"]["config"]
  declarations = []
  for dataset, roles in evidence["declarations"].items():
    for qubits, inputs in config["allocations"]:
      for mode in ("fixed", "fixed12", "tuned"):
        left = f"comparison:{mode}:q{qubits}-i{inputs}-qrc"
        right = f"comparison:{mode}:q{qubits}-i{inputs}-crc"
        if left in roles and right in roles:
          declarations.append((dataset, left, right, "QRC-minus-CRC", True))
      for family in ("qrc", "crc"):
        fixed = f"comparison:fixed:q{qubits}-i{inputs}-{family}"
        tuned = f"comparison:tuned:q{qubits}-i{inputs}-{family}"
        control = f"comparison:fixed12:q{qubits}-i{inputs}-{family}"
        for left, right, label in ((tuned, fixed, "tuned-minus-fixed"), (fixed, control, "natural-minus-fixed12"), (fixed, f"comparison:projection:i{inputs}", "reservoir-minus-projection")):
          if left in roles and right in roles:
            declarations.append((dataset, left, right, label, True))
    for family in ("qrc", "crc"):
      left, right = f"recurrence:{family}:partial", f"recurrence:{family}:all"
      if left in roles and right in roles:
        declarations.append((dataset, left, right, "partial-minus-reset-all", True))
    if dataset == "shd" and "learning-curves" in evidence["manifest"]["studies"]:
      for width in (2, 4):
        for size in config["supporting"]["learning_sizes"] + ["full"]:
          left = f"comparison:tuned:q6-i{width}-qrc:n={size}"
          right = f"comparison:tuned:q6-i{width}-crc:n={size}"
          declarations.append((dataset, left, right, "learning-QRC-minus-CRC", True))
    for width in (2, 4):
      for left, right, label in (
        (f"observations:qrc{width}:zz", f"observations:qrc{width}:all", "joint-minus-local"),
        (f"observations:qrc{width}:zz", f"observations:qrc{width}:products", "joint-minus-products"),
        (f"observations:crc{width}:products", f"observations:crc{width}:all", "products-minus-local"),
        (f"observations:crc{width}:width24", f"observations:crc{width}:all", "width24-minus-width18"),
        (f"observations:qrc{width}:zz", f"observations:crc{width}:width24", "joint-QRC-minus-CRC24")
      ):
        if left in roles and right in roles:
          declarations.append((dataset, left, right, label, True))
    mapping = evidence["selections"].get(dataset, {}).get("mapping", {})
    for width, mode in mapping.get("advanced", {}).items():
      for adapter in ("legacy", mode):
        declarations.append((dataset, f"mapping-paired:i{width}:{adapter}:qrc", f"mapping-paired:i{width}:{adapter}:crc", "mapping-QRC-minus-CRC", True))
      for family in ("qrc", "crc", "projection"):
        declarations.append((dataset, f"mapping-paired:i{width}:{mode}:{family}", f"mapping-paired:i{width}:legacy:{family}", "mapping-minus-anchor", False))
  if "preprocessing" in evidence["manifest"]["studies"]:
    from itertools import combinations
    for left, right in combinations(config["supporting"]["preprocessing"]["representations"], 2):
      declarations.append(("shd", left, right, "representation-difference", False))
  lookup = {}
  for row in cells:
    key = (row["dataset"], row["phase"], row["role"], row["population"], row["seed"], str(row["fold"]))
    if key in lookup:
      raise ValueError("Duplicate report cell")
    lookup[key] = row
  differences, coverage = [], []
  for dataset, left, right, label, equal_inputs in declarations:
    phases = sorted({row["phase"] for row in cells if row["dataset"] == dataset and row["role"] in (left, right)})
    for phase in phases:
      left_rows = [row for row in cells if row["dataset"] == dataset and row["phase"] == phase and row["role"] == left and row["population"] == "held"]
      right_rows = [row for row in cells if row["dataset"] == dataset and row["phase"] == phase and row["role"] == right and row["population"] == "held"]
      paired = []
      for row in left_rows:
        peer = lookup.get((dataset, phase, right, "held", row["seed"], str(row["fold"])))
        if peer is None:
          continue
        if row["membership"] != peer["membership"] or row["training_membership"] != peer["training_membership"]:
          raise ValueError("Paired models have different sample memberships")
        if equal_inputs and row["adapter"] != peer["adapter"]:
          raise ValueError("Controlled pair has different input adapters")
        paired.append({
          "dataset": dataset,
          "phase": phase,
          "role": f"{label}:{left}:{right}",
          "population": "held",
          "contrast": label,
          "left": left,
          "right": right,
          "seed": row["seed"],
          "fold": row["fold"]
        } | {name: row[name] - peer[name] for name in metrics()})
      complete = aggregate(paired, config["seeds"], strict=False)
      coverage.append({
        "dataset": dataset,
        "phase": phase,
        "role": f"{label}:{left}:{right}",
        "status": "completed" if complete and len(paired) == len(left_rows) == len(right_rows) else "incomplete",
        "observed_cells": len(paired),
        "expected_cells": max(len(left_rows), len(right_rows))
      })
      differences.extend(paired)
  summaries = aggregate(differences, config["seeds"], strict=False)
  for summary in summaries:
    rows = [row for row in differences if all(row[key] == summary[key] for key in ("dataset", "phase", "role"))]
    summary.update({key: rows[0][key] for key in ("contrast", "left", "right")})
    fold_gains = [np.mean([row["macro_f1"] for row in rows if str(row["fold"]) == fold]) for fold in sorted({str(row["fold"]) for row in rows})]
    summary["positive_folds"] = sum(value > 0 for value in fold_gains)
    summary["supported_development_gain"] = summary["phase"] == "development" and summary["macro_f1"] >= config["selection"]["supported_gain"] and summary["positive_folds"] >= config["selection"]["positive_folds"]
  return differences, summaries, coverage

def structure(evidence: dict) -> list:
  """Derive logical reservoir requirements without simulator cost measurements.

  :param evidence: Resolved configurations and dataset descriptions.
  :return: Structure rows with explicit hardware limits.
  """
  rows = []
  config = evidence["manifest"]["config"]
  classes = {row["dataset"]: row["classes"] for row in evidence["datasets"]}
  for (dataset, role), spec in sorted(evidence["specs"].items()):
    if spec["model"] not in ("crc", "qrc") or role.startswith("practical:"):
      continue
    quantum = spec["model"] == "qrc"
    width = 12 if spec.get("view") == "fixed12" else 24 if spec.get("view") in ("zz", "products") else 3 * spec["qubits"] if quantum else spec["size"]
    segments = spec.get("segments", config["datasets"][dataset]["segments"])
    features = ((1 if spec.get("summary") == "segment_mean_final" else 2) * segments + 1) * width
    outputs = classes.get(dataset, 20 if dataset == "shd" else 11)
    rows.append({
      "dataset": dataset,
      "role": role,
      "model": spec["model"],
      "inputs": spec["inputs"],
      "qubits": spec.get("qubits"),
      "memory_qubits": spec["qubits"] - spec["inputs"] if quantum else None,
      "classical_units": spec.get("size"),
      "depth": spec.get("depth"),
      "encoding": "amplitude probabilities" if quantum else "same numerical probability stream",
      "interaction": "CZ ring" if quantum else "fixed recurrent matrix",
      "reset": spec.get("reset", "partial"),
      "uploads_per_step": 1,
      "rotations_per_step": 2 * spec["qubits"] * spec["depth"] + spec["inputs"] if quantum else None,
      "cz_per_step": spec["qubits"] * spec["depth"] if quantum else None,
      "observation": spec.get("view", "all"),
      "sequential_features": width,
      "readout_features": features,
      "readout_parameters": (features + 1) * (1 if outputs == 2 else outputs),
      "local_measurement_bases": 3 if quantum else None,
      "finite_shot_accuracy": "unmeasured" if quantum else "not applicable",
      "hardware_note": "Exact expectations omit disturbance and shot noise; hardware requires repeated preparations; ZZ uses the Z basis" if quantum else "Classical recurrent state and fixed weights"
    })
  return rows

def topic(role: str, phase: str = "") -> str:
  """Choose the scientific section from explicit study role and phase.

  :param role: Declared role.
  :param phase: Population boundary.
  :return: Section identifier.
  """
  if phase == "learning-development":
    return "training-size"
  if phase == "preprocessing-diagnostic":
    return "preprocessing"
  prefix = role.split(":")[0]
  return "mapping" if prefix.startswith("mapping") else "comparisons" if prefix == "comparison" else prefix if prefix in ("observations", "recurrence", "practical") else "performance"
