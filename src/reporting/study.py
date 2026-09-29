import json
import shutil
import tempfile
from pathlib import Path

from src.reporting.evidence import collect
from src.reporting.products import aggregate, contrasts, export, metrics, structure, topic
from src.runtime.progress import stage
from src.runtime.records import checksum, read, write

@stage("reporting.study.report")
def report(run: Path, pdf: bool = False, details: dict | None = None) -> dict:
  """Validate durable evidence and transactionally publish factual report products.

  :param run: Unified execution namespace.
  :param pdf: Include vector PDF figures in addition to PNG.
  :param details: Optional diagnostic kinds and identity filters; canonical tables remain complete.
  :return: Rendering and scientific coverage status, separately identified.
  """
  evidence = collect(run)
  integrity_path = run / "evidence/integrity.json"
  if integrity_path.exists():
    for name, expected in read(integrity_path)["artifacts"].items():
      if checksum(run / "evidence" / name) != expected:
        raise ValueError(f"Run evidence changed after sealing: {name}")
  else:
    evidence["coverage"].append({"dataset": "all", "phase": "integrity", "role": "evidence", "status": "incomplete", "observed_cells": 0, "expected_cells": 1})
  cells = evidence["cells"]
  summaries = aggregate(cells, evidence["manifest"]["config"]["seeds"], strict=False)
  differences, paired, pair_coverage = contrasts(evidence)
  evidence["coverage"].extend(pair_coverage)
  output = run / "reports"
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = Path(tempfile.mkdtemp(prefix=".reports-", dir=output.parent))
  try:
    documents = render(temporary, evidence, summaries, differences, paired, pdf=pdf, details=details)
    coverage = evidence["coverage"]
    development = [row for row in coverage if row["phase"] != "official"]
    development_complete = bool(development) and all(row["status"] in ("completed", "not-applicable", "not-requested") for row in development)
    official = [row for row in coverage if row["phase"] == "official"]
    official_complete = bool(official) and all(row["status"] == "completed" for row in official)
    record = {
      "status": "completed" if development_complete and (official_complete or evidence["manifest"].get("diagnostic_only")) else "partial",
      "rendering_status": "completed",
      "formats": {"tables": ["csv"], "figures": ["png", "pdf"] if pdf else ["png"]},
      "development_status": "completed" if development_complete else "incomplete",
      "official_status": "completed" if official_complete else "pending",
      "cells": len(cells),
      "presentation": {"mode": "canonical tables and compact overviews", "details": details or {}},
      "sources": evidence["sources"],
      "reporting_source": {str(path): checksum(path) for path in sorted(Path("src/reporting").glob("*.py"))},
      "root": str(output),
      "coverage": coverage,
      "uncertainty": "Sample SD across seeds after averaging folds within each seed; fold variability is descriptive, not independent replication"
    }
    write(temporary / "overview/metadata/report.json", record)
    from src.reporting.catalog import guides, index
    guides(temporary, documents["captions"], documents["notes"])
    index(temporary, run, record)
    backup = output.with_name(".reports-previous")
    if backup.exists() and not output.exists():
      backup.replace(output)
    if backup.exists():
      shutil.rmtree(backup)
    if output.exists():
      output.replace(backup)
    try:
      temporary.replace(output)
    except BaseException:
      if backup.exists():
        backup.replace(output)
      raise
    if backup.exists():
      shutil.rmtree(backup)
    return record
  finally:
    if temporary.exists():
      shutil.rmtree(temporary)

def render(root: Path, evidence: dict, summaries: list, differences: list, paired: list, pdf: bool = False, details: dict | None = None) -> dict:
  """Generate numerical tables, factual metadata and figures for enabled studies.

  :param root: Staged destination.
  :param evidence: Validated durable records.
  :param summaries: Complete performance aggregates.
  :param differences: Individual paired differences.
  :param paired: Complete paired summaries.
  :param pdf: Include vector PDF figures.
  :param details: Optional detailed diagnostic selection.
  :return: Captions and measurement notes for topic guides.
  """
  export(root, "performance", "performance-cells", evidence["cells"])
  export(root, "performance", "performance", summaries)
  export(root, "generalization", "per-class", evidence["classes"])
  export(root, "generalization", "per-subject", [row for row in evidence["subjects"] if row["kind"] == "subject"])
  speakers = [row for row in evidence["subjects"] if row["kind"] == "speaker-subset"]
  export(root, "generalization", "speaker-cells", speakers)
  export(root, "generalization", "speaker-subsets", aggregate(speakers, evidence["manifest"]["config"]["seeds"], strict=False))
  write(root / "generalization/metadata/confusions.json", evidence["confusions"])
  gaps = []
  cells = evidence["cells"]
  for row in cells:
    if row["population"] != "held":
      continue
    train = next(item for item in cells if item["population"] == "train" and all(item[key] == row[key] for key in ("dataset", "phase", "role", "seed", "fold")))
    gaps.append({key: row[key] for key in ("dataset", "phase", "role", "seed", "fold")} | {key: train[key] - row[key] for key in metrics()})
  export(root, "generalization", "training-minus-held", gaps)
  export(root, "comparisons", "paired-cells", differences)
  export(root, "comparisons", "paired-differences", paired)
  export(root, "methodology", "candidate-cells", evidence["candidates"])
  export(root, "methodology", "candidate-scores", evidence["candidate_scores"])
  export(root, "methodology", "selection-decisions", evidence["decisions"])
  export(root, "mapping", "conditioning", evidence["diagnostics"])
  export(root, "practical", "epoch-history", evidence["history"])
  export(root, "resources", "reservoir-structure", structure(evidence))
  protocols = []
  for (dataset, role), spec in sorted(evidence["specs"].items()):
    protocols.append({"dataset": dataset, "role": role, "settings": json.dumps(spec, sort_keys=True), "solver": spec.get("solver", "neural" if spec["model"] in ("lstm", "transformer") else "cholesky"), "selection_population": "development only"})
  export(root, "methodology", "protocols", protocols)
  register = evidence["manifest"]["parameter_register"]
  from src.methodology.parameters import validate_register
  validate_register(evidence["manifest"]["config"], register)
  parameters = [{
    "parameter": name,
    "value": json.dumps(entry["value"]),
    "origin": ", ".join(entry["origins"]),
    "usage": "declared setting; selected values are in the selection table",
    "reference": entry.get("reference", ""),
    "reference_location": entry.get("location", "")
  } for name, entry in register["parameters"].items()]
  export(root, "methodology", "parameters", parameters)
  write(root / "methodology/metadata/selection.json", evidence["selections"])
  export(root, "data", "dataset-settings", [{key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in data.items() if key != "rows"} for data in evidence["datasets"]])
  export(root, "data", "fitting-memberships", [{key: row[key] for key in ("dataset", "phase", "role", "seed", "fold", "population", "samples", "membership", "training_membership")} for row in cells])
  samples, coverage = [], []
  for data in evidence["datasets"]:
    identity = {"dataset": data["dataset"], "population": data["population"]}
    samples.extend(identity | row for row in data["rows"])
    counts = {}
    for row in data["rows"]:
      key = (row["partition"], row["group"], row["label"])
      counts[key] = counts.get(key, 0) + 1
    coverage.extend(identity | {"partition": key[0], "group": key[1], "class": key[2], "samples": value} for key, value in sorted(counts.items()))
  export(root, "data", "samples", samples)
  export(root, "data", "coverage", coverage)
  resources = []
  costs = []
  for entry in evidence["resources"].get("practical", []):
    common = {key: value for key, value in entry.items() if not isinstance(value, (dict, list))}
    role = entry["role"]
    matched = next((row for row in cells if row["dataset"] == entry["dataset"] and row["role"] == role and row["seed"] == entry["seed"] and row["phase"] == "official" and row["population"] == "held"), None)
    if matched:
      from src.runtime.records import digest
      if entry.get("configuration") and digest(entry["configuration"]) != matched["configuration"]:
        raise ValueError("Resource and test configurations differ")
      common.update({"test_" + key: matched[key] for key in metrics()})
    for repetition in entry["measurements"]:
      resources.append(common | repetition)
    coverage = next(row for row in evidence["coverage"] if row["phase"] == "resources" and row["dataset"] == entry["dataset"] and row["role"] == role and row.get("seed") == entry["seed"])
    if coverage["status"] != "completed":
      continue
    batch = [row for row in entry["measurements"] if row["workload"] == "batch"]
    import numpy as np
    per_example = [row["seconds"] / row["samples"] for row in batch]
    costs.append({
      "dataset": entry["dataset"],
      "model": entry["model"],
      "seed": entry["seed"],
      "device": entry.get("device", "unrecorded"),
      "fitting_mode": "cached features" if entry.get("features_reused") else "recorded fit; see measurement boundaries",
      "fit_seconds": entry.get("complete_fitting_seconds"),
      "inference_seconds_per_example": float(np.mean(per_example)) if per_example else None,
      "inference_repeat_sd": float(np.std(per_example, ddof=1)) if len(per_example) > 1 else None,
      "storage_mib": entry["inference_artifact_bytes"] / 1024 ** 2,
      "host_peak_mib": entry["peak_host_rss_bytes"] / 1024 ** 2,
      "gpu_allocated_mib": entry["torch_peak_allocated_bytes"] / 1024 ** 2 if entry.get("torch_peak_allocated_bytes") is not None else None,
      "features_reused": entry.get("features_reused"),
      "pipeline": entry.get("pipeline_id"),
      "test_status": "completed" if matched else "pending"
    } | {name: matched[name] if matched else None for name in metrics()})
  export(root, "resources", "practical-measurements", resources)
  export(root, "resources", "practical-costs", costs)
  notes = {}
  if costs:
    notes["resources"] = (
      "# Practical cost measurement boundaries\n\n"
      "Recorded fit time includes extraction or cache loading, fitting and evaluation. Rows marked cached features are not cold training-from-scratch costs. "
      "Fitting timings retain their original execution conditions; only inference repetitions were isolated.\n\n"
      "Inference uses warmed, length-stratified development-confirmation workloads and confirmation-fit packages. Joined official-test metrics use separately refitted models with identical settings and seeds. "
      "Device is explicit: CPU and CUDA timings are implementation/device measurements, not hardware-neutral architecture costs.\n\n"
      "Storage measures the inference package. RAM is whole-process inference peak RSS including runtime and metadata. VRAM is PyTorch allocated memory, not total GPU use or peak training memory. "
      "No whole-system energy comparison is available. Exhaustive boundaries and measurements are retained in practical-measurements.csv.\n"
    )
  for section in ("mapping", "observations", "recurrence", "learning-curves", "preprocessing"):
    if section not in evidence["manifest"]["studies"]:
      evidence["coverage"].append({"dataset": "all", "phase": "scope", "role": section, "status": "not-requested", "observed_cells": 0, "expected_cells": 0})
  export(root, "overview", "coverage", evidence["coverage"])
  from src.reporting.figures import render as figures
  return {"captions": figures(root, evidence, pdf=pdf, details=details), "notes": notes}
