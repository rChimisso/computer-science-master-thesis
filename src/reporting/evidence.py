from pathlib import Path

import numpy as np

from src.evaluation.metrics import assert_metrics, prediction_metrics
from src.models.protocol import FittingProtocol
from src.runtime.records import checksum, digest, read

def resource_coverage(evidence: dict) -> list:
  """Validate practical measurement identities and every requested repetition.

  :param evidence: Collected run declarations, frozen settings and durable measurements.
  :return: Per-pipeline coverage plus the resource-action completion boundary.
  """
  manifest = evidence["manifest"]
  records = evidence["resources"].get("practical", [])
  if not manifest.get("resources"):
    if records:
      raise ValueError("Resource measurements were not requested")
    return [{"dataset": "all", "phase": "resources", "role": "practical-costs", "status": "not-requested", "observed_cells": 0, "expected_cells": 0}]
  required = {(dataset, f"practical:{model}", seed) for dataset in manifest["datasets"] for model in manifest["models"] for seed in manifest["config"]["seeds"]}
  observed = {}
  repetitions = {(repeat, workload) for repeat in range(3) for workload in ("single", "batch")}
  for entry in records:
    key = (entry["dataset"], entry["role"], entry["seed"])
    if key not in required or key in observed:
      raise ValueError("Duplicate or undeclared resource pipeline")
    frozen = evidence.get("frozen")
    if not frozen:
      raise ValueError("Resource measurements require frozen settings")
    spec = frozen["manifest"]["entries"][entry["dataset"]][entry["role"]]["spec"]
    if entry["configuration"] != spec or entry["model"] != spec["model"]:
      raise ValueError("Resource configuration differs from the frozen pipeline")
    snapshot = next(row for row in evidence["datasets"] if row["dataset"] == entry["dataset"] and row["population"] == "train")
    candidates = [(index, row) for index, row in enumerate(snapshot["rows"]) if row["partition"] == "confirmation"]
    candidates.sort(key=lambda value: (value[1]["steps"], value[0]))
    selected = [candidates[int(index)][1] for index in np.linspace(0, len(candidates) - 1, min(64, len(candidates)), dtype=int)]
    expected_membership = {plural: [row[singular] for row in selected] for plural, singular in (("identities", "identity"), ("labels", "label"), ("groups", "group"))}
    if not selected or entry["membership"] != expected_membership:
      raise ValueError("Resource benchmark membership differs from the declared population")
    seen = set()
    for measurement in entry["measurements"]:
      identity = (measurement["repeat"], measurement["workload"])
      if identity not in repetitions or identity in seen:
        raise ValueError("Duplicate or undeclared resource repetition")
      expected_samples = 1 if measurement["workload"] == "single" else len(selected)
      if measurement["samples"] != expected_samples or not np.isfinite(measurement["seconds"]) or measurement["seconds"] < 0:
        raise ValueError("Invalid resource workload or duration")
      seen.add(identity)
    observed[key] = seen
  rows = [{
    "dataset": dataset,
    "phase": "resources",
    "role": role,
    "seed": seed,
    "status": "completed" if observed.get((dataset, role, seed)) == repetitions else "incomplete",
    "observed_cells": len(observed.get((dataset, role, seed), set())),
    "expected_cells": len(repetitions)
  } for dataset, role, seed in sorted(required)]
  rows.append({
    "dataset": "all",
    "phase": "resources",
    "role": "practical-costs",
    "status": "not-applicable" if not required else "completed" if evidence["resources"].get("status") == "completed" and all(row["status"] == "completed" for row in rows) else "incomplete",
    "observed_cells": sum(len(value) for value in observed.values()),
    "expected_cells": len(required) * len(repetitions)
  })
  return rows

def read_source(path: str | Path, hashes: dict) -> dict:
  """Read a source record while collecting report dependency identities.

  :param path: Evidence record.
  :param hashes: Mutable source inventory.
  :return: Decoded record.
  """
  path = Path(path)
  hashes[str(path)] = checksum(path)
  return read(path)

def job_rows(job: dict, dataset: str, role: str, seed: int, penalty, phase: str, expected_spec: dict, result: dict) -> None:
  """Validate one fitting cell and recompute scalar, class and subject metrics.

  :param job: Completed numerical record.
  :param dataset: Dataset identity.
  :param role: Explicit comparison role.
  :param seed: Expected initialization seed.
  :param penalty: Selected penalty or None for neural models.
  :param phase: Reporting population boundary.
  :param expected_spec: Declared numerical settings.
  :param result: Mutable typed reporting evidence.
  """
  if job["status"] != "completed" or job["identity"]["seed"] != seed:
    raise ValueError("Incomplete or misidentified child fitting record")
  actual_spec = job["identity"]["spec"]
  expected = {key: value for key, value in expected_spec.items() if key != "fixed_epochs" or phase in ("confirmation", "official")}
  actual = {key: value for key, value in actual_spec.items() if key != "fixed_epochs" or phase in ("confirmation", "official")}
  if actual != expected:
    raise ValueError("Fitting settings differ from the selected declaration")
  if job["identity"].get("dataset", dataset) != dataset:
    raise ValueError("Dataset identity differs")
  members = job["membership"]
  for member in members.values():
    if len(set(member["identities"])) != len(member["identities"]) or not len(member["labels"]) == len(member["groups"]) == len(member["identities"]):
      raise ValueError("Duplicate or malformed sample membership")
  if set(members["train"]["identities"]) & set(members["held"]["identities"]):
    raise ValueError("Training and evaluation samples overlap")
  if phase != "official" and set(members["train"]["groups"]) & set(members["held"]["groups"]):
    raise ValueError("Held-subject isolation failed")
  classes = job.get("classes", len(set(members["train"]["labels"])))
  row = next(item for item in job["rows"] if item["lambda"] == penalty)
  if row.get("status", "completed") != "completed":
    raise ValueError("Unresolved readout cannot enter reporting")
  spec_id = digest(actual_spec)
  base = {
    "dataset": dataset,
    "phase": phase,
    "role": role,
    "seed": seed,
    "fold": job["identity"]["fold"]["fold"],
    "configuration": spec_id,
    "model": expected_spec["model"],
    "protocol": digest({"spec": expected_spec, "phase": phase}),
    "adapter": job.get("adapter", {}).get("fingerprint") if job.get("adapter") else "rich-input",
    "training_membership": digest(members["train"]),
    "solver": FittingProtocol.from_spec(expected_spec, row).solver,
    "lambda": penalty
  }
  for population, prediction_key, metric_key in (("train", "train_predictions", "train"), ("held", "predictions", "validation")):
    membership = members[population]
    labels = np.asarray(membership["labels"])
    predictions = np.asarray(row[prediction_key])
    actual, matrix = prediction_metrics(labels, predictions, classes)
    assert_metrics(actual, row[metric_key])
    common = base | {"population": population, "membership": digest(membership), "samples": len(labels)}
    result["cells"].append(common | actual)
    result["confusions"].append(common | {"matrix": matrix.tolist()})
    for label in range(classes):
      support = int(matrix[label].sum())
      predicted = int(matrix[:, label].sum())
      correct = int(matrix[label, label])
      result["classes"].append(common | {
        "class": label,
        "support": support,
        "precision": correct / predicted if predicted else 0.0,
        "recall": correct / support if support else 0.0,
        "f1": 2 * correct / (support + predicted) if support + predicted else 0.0
      })
    groups = np.asarray(membership["groups"])
    masks = [("subject", str(group), groups == group) for group in sorted(set(groups))]
    if dataset == "shd" and phase == "official" and population == "held":
      seen = np.isin(groups, members["train"]["groups"])
      masks += [("speaker-subset", "seen", seen), ("speaker-subset", "unseen", ~seen), ("speaker-subset", "whole", np.ones(len(groups), dtype=bool))]
    for kind, group, mask in masks:
      if mask.any():
        metrics, _ = prediction_metrics(labels[mask], predictions[mask], classes)
        result["subjects"].append(common | {"kind": kind, "group": group, "samples": int(mask.sum())} | metrics)
  result["specs"][(dataset, role)] = expected_spec
  if job.get("neural"):
    for epoch in job["neural"].get("history", []):
      for population in ("train", "validation"):
        if epoch.get(population):
          metrics = epoch[population]
          result["history"].append(base | {
            "population": population,
            "epoch": epoch["epoch"],
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "learning_seconds": epoch["learning_seconds"],
            "epoch_seconds": epoch["epoch_seconds"],
            "selected_epoch": job["best_epoch"],
            "stopping": job["neural_status"]
          })
  for population, diagnostics in job.get("features", {}).get("diagnostics", {}).items():
    for kind, record in diagnostics.items():
      count = record["count"]
      for coordinate in range(len(record["sum"])):
        mean = record["sum"][coordinate] / count
        result["diagnostics"].append(base | {
          "population": population,
          "kind": kind,
          "coordinate": coordinate,
          "valid_bins": count,
          "mean": mean,
          "std": max(0, record["squares"][coordinate] / count - mean * mean) ** 0.5,
          "minimum": record["minimum"][coordinate],
          "maximum": record["maximum"][coordinate],
          "near_boundary_fraction": record["probability_boundary" if kind == "input" else "absolute_boundary"][coordinate] / count
        })

def collect(run: Path) -> dict:
  """Collect declared run evidence and independently validate frozen evaluation coverage.

  :param run: Unified run root.
  :return: Typed numerical evidence and explicit coverage records.
  """
  hashes = {}
  root = run / "evidence"

  def source(path: str | Path, hashes: dict) -> dict:
    """Resolve persisted evidence references within this run, including relocated copies.

    :param path: Persisted or current record path.
    :param hashes: Source inventory.
    :return: Run-local source record.
    """
    candidate = Path(path)
    if "evidence" in candidate.parts:
      position = len(candidate.parts) - 1 - candidate.parts[::-1].index("evidence")
      candidate = root.joinpath(*candidate.parts[position + 1:])
    if not candidate.resolve().is_relative_to(run.resolve()):
      raise ValueError("A report cannot read evidence outside its run")
    return read_source(candidate, hashes)

  manifest = source(run / "manifest.json", hashes)
  if manifest.get("evidence_version") != 1:
    raise ValueError("Reporting requires the authored workflow evidence format")
  result = {key: [] for key in ("cells", "classes", "subjects", "confusions", "history", "diagnostics", "candidates", "candidate_scores", "decisions", "coverage")}
  result.update({"manifest": manifest, "sources": hashes, "specs": {}, "datasets": [], "resources": {}})
  for path in sorted((root / "datasets").glob("*.json")):
    result["datasets"].append(source(path, hashes))
  selections_path = root / "metrics/selections.json"
  selections = source(selections_path, hashes) if selections_path.exists() else {}
  result["selections"] = selections
  expected = {}
  from src.studies.design import groups
  for dataset in manifest.get("datasets", []):
    declarations = manifest.get("declarations", {}).get(dataset)
    if declarations is None:
      declarations = groups(dataset, manifest["studies"], manifest.get("models", []), manifest["config"])
    expected[dataset] = {item["id"]: item for item in declarations}
  result["declarations"] = expected
  for dataset, declarations in expected.items():
    if dataset not in selections:
      for path in sorted((root / "metrics").glob(f"{dataset}-selection-*.json")):
        candidate = source(path, hashes)
        declared = {row["group"] for row in candidate.get("declared_candidates", [])}
        if declared == set(declarations):
          selections[dataset] = candidate
          break
  if not manifest.get("reproduction"):
    for dataset in expected:
      if not any(row["dataset"] == dataset and row["population"] == "train" for row in result["datasets"]):
        result["coverage"].append({"dataset": dataset, "phase": "data", "role": "dataset-statistics", "status": "incomplete", "observed_cells": 0, "expected_cells": 1})

  def add_selection(dataset: str, selection: dict, phase: str, required_seeds: list) -> None:
    """Load completed candidate evidence without concealing partial seed rounds.

    :param dataset: Dataset key.
    :param selection: Selection record, possibly partial.
    :param phase: Distinct reporting protocol.
    :param required_seeds: Declared replication coverage.
    """
    for role, item in selection.get("groups", {}).items():
      declaration = item["declaration"]
      required = [42] if declaration.get("screen_only") else required_seeds
      actual_phase = "mapping-screen" if declaration.get("screen_only") else phase
      chosen = item["selected"]
      candidate = {key: value for key, value in chosen["spec"].items() if key != "fixed_epochs"}
      if candidate not in declaration["candidates"]:
        raise ValueError("Selected configuration is outside the declared candidate set")
      observed = set()
      for path in chosen["jobs"]:
        job = source(path, hashes)
        if job["identity"]["phase"] != "development":
          raise ValueError("Selection includes a non-development fitting record")
        fold = job["identity"]["fold"]
        source_data = next((row for row in result["datasets"] if row["dataset"] == dataset and row["population"] == "train"), None)
        if source_data:
          for population, key in (("train", "train"), ("held", "validation")):
            identifiers = [source_data["rows"][index]["identity"] for index in fold[key]]
            if identifiers != job["membership"][population]["identities"]:
              raise ValueError("Fold membership differs from saved dataset identities")
        observed.add((job["identity"]["fold"]["fold"], job["identity"]["seed"]))
        job_rows(job, dataset, role, job["identity"]["seed"], chosen["selection"]["lambda"], actual_phase, chosen["spec"], result)
      needed = {(fold, seed) for fold in range(4) for seed in required}
      result["coverage"].append({"dataset": dataset, "phase": actual_phase, "role": role, "status": "completed" if observed == needed else "incomplete", "observed_cells": len(observed), "expected_cells": len(needed)})
      result["decisions"].append({
        "dataset": dataset,
        "phase": actual_phase,
        "role": role,
        "configuration": digest(chosen["spec"]),
        "candidate_count": len(declaration["candidates"]),
        "lambda": chosen["selection"]["lambda"],
        "macro_f1": chosen["selection"]["mean_macro_f1"],
        "boundary_values": str(chosen.get("decision", {}).get("boundary_values", {})),
        "equivalence": manifest["config"]["selection"]["equivalence"],
        "adoption": str(item.get("adoption", {})),
        "settings": str(chosen["spec"])
      })
      for candidate in item.get("screen", []):
        result["candidate_scores"].append({
          "dataset": dataset,
          "phase": actual_phase,
          "role": role,
          "configuration": digest(candidate["spec"]),
          "seed": 42,
          "status": candidate.get("status", "completed"),
          "macro_f1": candidate["selection"]["mean_macro_f1"],
          "lambda": candidate["selection"]["lambda"],
          "settings": str(candidate["spec"])
        })
      paths = {path for candidate in item.get("screen", []) for path in candidate["jobs"]}
      paths.update(path for values in item.get("replication", {}).values() for path in values)
      for path in sorted(paths):
        job = source(path, hashes)
        for row in job["rows"]:
          metrics = row["validation"]
          result["candidates"].append({
            "dataset": dataset,
            "phase": actual_phase,
            "role": role,
            "configuration": digest(job["identity"]["spec"]),
            "seed": job["identity"]["seed"],
            "fold": job["identity"]["fold"]["fold"],
            "lambda": row["lambda"],
            "status": row.get("status", job["status"]),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"] if "macro_precision" in metrics else metrics["per_class"]["macro avg"]["precision"],
            "macro_recall": metrics["macro_recall"] if "macro_recall" in metrics else metrics["per_class"]["macro avg"]["recall"],
            "selected": job["identity"]["spec"] == {key: value for key, value in chosen["spec"].items() if key != "fixed_epochs"} and row["lambda"] == chosen["selection"]["lambda"],
            "feature_dimension": row.get("feature_dimension"),
            "readout_norm": row.get("readout_norm"),
            "solver": FittingProtocol.from_spec(job["identity"]["spec"], row).solver,
            "iterations": str(row.get("iterations", []))
          })

  for dataset in expected:
    selection = selections.get(dataset, {})
    for candidate in selection.get("declared_candidates", []):
      if candidate["status"] != "completed":
        result["candidate_scores"].append({
          "dataset": dataset,
          "phase": "development",
          "role": candidate["group"],
          "configuration": digest(candidate["spec"]),
          "seed": 42,
          "status": candidate["status"],
          "macro_f1": None,
          "lambda": None,
          "settings": str(candidate["spec"])
        })
    unknown = set(selection.get("groups", {})) - set(expected[dataset])
    if unknown:
      raise ValueError(f"Undeclared selection roles: {unknown}")
    add_selection(dataset, selection, "development", manifest["config"]["seeds"])
    for role in set(expected[dataset]) - set(selection.get("groups", {})):
      result["coverage"].append({"dataset": dataset, "phase": "development", "role": role, "status": "incomplete", "observed_cells": 0, "expected_cells": 12})
    if "mapping" in manifest["studies"]:
      mapping = selection.get("mapping", {})
      for key, phase, seeds in (("paired", "mapping-development", manifest["config"]["seeds"]), ("projection_robustness", "mapping-robustness", [42])):
        add_selection(dataset, mapping.get(key, {}), phase, seeds)
        if not mapping.get(key, {}).get("groups"):
          result["coverage"].append({"dataset": dataset, "phase": phase, "role": "mapping", "status": "incomplete", "observed_cells": 0, "expected_cells": None})
    if "learning-curves" in manifest["studies"] and dataset == "shd":
      curves = selection.get("learning_curves", {})
      memberships = {}
      observed_curves = set()
      roles = [f"comparison:tuned:q6-i{width}-{family}" for width in (2, 4) for family in ("qrc", "crc")]
      roles += [f"comparison:projection:i{width}" for width in (2, 4)]
      roles += [role for role in selection.get("groups", {}) if role.startswith("practical:")]
      needed_curves = {(role, str(size), fold, seed) for role in roles for size in manifest["config"]["supporting"]["learning_sizes"] + ["full"] for fold in range(4) for seed in manifest["config"]["seeds"]}
      for entry in curves.get("rows", []):
        job = source(entry["job"], hashes)
        size = str(entry["size"]) if entry["size"] in manifest["config"]["supporting"]["learning_sizes"] else "full"
        key = (entry["role"], size, entry["fold"], entry["seed"])
        if key in observed_curves or key not in needed_curves:
          raise ValueError("Duplicate or undeclared learning-curve cell")
        observed_curves.add(key)
        member_key = (entry["fold"], size)
        membership = job["membership"]["train"]
        if member_key in memberships and memberships[member_key] != membership:
          raise ValueError("Learning-curve membership differs across models or seeds")
        memberships[member_key] = membership
        job_rows(job, dataset, f"{entry['role']}:n={size}", entry["seed"], job["rows"][0]["lambda"], "learning-development", job["identity"]["spec"], result)
      for fold in range(4):
        previous = set()
        for size in [str(value) for value in manifest["config"]["supporting"]["learning_sizes"]] + ["full"]:
          if (fold, size) in memberships:
            current = set(memberships[(fold, size)]["identities"])
            if not previous <= current:
              raise ValueError("Training-size subsets are not nested")
            previous = current
      complete = curves.get("status") == "completed" and observed_curves == needed_curves
      result["coverage"].append({"dataset": dataset, "phase": "learning-development", "role": "learning-curves", "status": "completed" if complete else "incomplete", "observed_cells": len(observed_curves), "expected_cells": len(needed_curves)})
  if "preprocessing" in manifest["studies"]:
    path = root / "metrics/preprocessing.json"
    preprocessing = source(path, hashes) if path.exists() else {}
    observed_preprocessing = set()
    shared_membership = None
    for entry in preprocessing.get("rows", []):
      job = source(entry["job"], hashes)
      key = (entry["representation"], entry["seed"])
      if key in observed_preprocessing or shared_membership is not None and shared_membership != job["membership"]:
        raise ValueError("Preprocessing cells are duplicated or use different memberships")
      observed_preprocessing.add(key)
      shared_membership = job["membership"]
      job_rows(job, "shd", entry["representation"], entry["seed"], None, "preprocessing-diagnostic", job["identity"]["spec"], result)
    needed = {(name, seed) for name in manifest["config"]["supporting"]["preprocessing"]["representations"] for seed in manifest["config"]["seeds"]}
    complete = preprocessing.get("status") == "completed" and observed_preprocessing == needed
    result["coverage"].append({"dataset": "shd", "phase": "preprocessing-diagnostic", "role": "preprocessing", "status": "completed" if complete else "incomplete", "observed_cells": len(observed_preprocessing), "expected_cells": len(needed)})
  frozen_path = run / "frozen.json"
  frozen = source(frozen_path, hashes) if frozen_path.exists() else None
  if frozen:
    if frozen.get("status") != "frozen" or frozen["fingerprint"] != digest(frozen["manifest"]):
      raise ValueError("Invalid frozen evidence signature")
    for dataset, declarations in expected.items():
      roles = {role for role, declaration in declarations.items() if declaration["final"]}
      if not manifest.get("reproduction") and set(frozen["manifest"]["entries"].get(dataset, {})) != roles:
        raise ValueError("Frozen evaluation roster differs from declared comparisons")
    for dataset, entries in frozen["manifest"]["entries"].items():
      for role, entry in entries.items():
        if entry != frozen["manifest"]["decisions"][dataset]["groups"][role]["selected"]:
          raise ValueError("Frozen configuration differs from selection")
  result["frozen"] = frozen
  for phase in ("confirmation", "official"):
    path = root / "metrics" / f"{phase}.json"
    if not path.exists():
      result["coverage"].append({"dataset": "all", "phase": phase, "role": "all", "status": "not-applicable" if manifest.get("diagnostic_only") else "pending", "observed_cells": 0, "expected_cells": None})
      continue
    if not frozen:
      raise ValueError("Evaluation records require frozen configuration evidence")
    phase_record = source(path, hashes)
    if phase_record.get("frozen") != frozen["fingerprint"]:
      raise ValueError("Evaluation freeze identity differs")
    entries = frozen["manifest"]["entries"]
    needed = {(name, role, seed) for name, roles in entries.items() for role in roles for seed in manifest["config"]["seeds"]}
    observed = set()
    for entry in phase_record["jobs"]:
      key = (entry["dataset"], entry["role"], entry["seed"])
      if key not in needed or key in observed:
        raise ValueError("Duplicate or undeclared frozen evaluation cell")
      observed.add(key)
      chosen = entries[entry["dataset"]][entry["role"]]
      job = source(entry["path"], hashes)
      if job["identity"]["phase"] != phase or job["rows"][0]["lambda"] != chosen["selection"]["lambda"]:
        raise ValueError("Frozen evaluation phase or penalty changed")
      memberships = frozen["manifest"]["memberships"][entry["dataset"]]
      if phase == "confirmation" and job["membership"] != {"train": memberships["development"], "held": memberships["confirmation"]}:
        raise ValueError("Confirmation membership differs from frozen selection")
      if phase == "official":
        required = {identity: (label, group) for member in memberships.values() for identity, label, group in zip(member["identities"], member["labels"], member["groups"])}
        actual = job["membership"]["train"]
        observed_training = dict(zip(actual["identities"], zip(actual["labels"], actual["groups"])))
        if observed_training != required:
          raise ValueError("Official fitting membership differs from frozen official-training population")
        snapshot = next((item for item in result["datasets"] if item["dataset"] == entry["dataset"] and item["population"] == "test"), None)
        if snapshot is None:
          raise ValueError("Official evaluation requires saved test-population metadata")
        held = {"identities": [item["identity"] for item in snapshot["rows"]], "labels": [item["label"] for item in snapshot["rows"]], "groups": [item["group"] for item in snapshot["rows"]]}
        if job["membership"]["held"] != held:
          raise ValueError("Official evaluation membership differs from saved test population")
      job_rows(job, entry["dataset"], entry["role"], entry["seed"], chosen["selection"]["lambda"], phase, chosen["spec"], result)
    complete = phase_record["status"] == "completed" and observed == needed
    result["coverage"].append({"dataset": "all", "phase": phase, "role": "all", "status": "completed" if complete else "incomplete", "observed_cells": len(observed), "expected_cells": len(needed)})
  if (root / "metrics/resources.json").exists():
    result["resources"] = source(root / "metrics/resources.json", hashes)
  result["coverage"].extend(resource_coverage(result))
  for path in sorted((root / "predictions").glob("*.npy")):
    hashes[str(path)] = checksum(path)
  for path in sorted((root / "memberships").glob("*.json")):
    hashes[str(path)] = checksum(path)
  return result
