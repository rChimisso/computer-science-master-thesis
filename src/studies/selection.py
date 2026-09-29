from contextlib import nullcontext
from pathlib import Path

import numpy as np

from src.runtime.progress import stage, track
from src.data.indexed import study_folds
from src.evaluation.selection import select_penalty
from src.runtime.records import digest, write
from src.studies.experiments import experiment

@stage("studies.selection.execute_round")
def execute_round(data, specs: list, seed: int, config: dict) -> None:
  """Complete one deterministic seed round through resource-bounded task futures.

  :param data: Development source.
  :param specs: Declared candidate settings, deduplicated before execution.
  :param seed: Common initialization round.
  :param config: Worker budgets and deadlines.
  """
  from src.runtime.tasks import current
  scheduler = current(config)
  if scheduler is not None:
    unique = {digest(spec): spec for spec in specs}
    futures = [scheduler.experiment(data, fold, spec, seed, config | {'_study': config.get('_study_labels', {}).get(digest(spec), config.get('_study', []))}) for spec in unique.values() for fold in study_folds(data, config)]
    for future in futures:
      future.result()
    return
  for spec in {digest(spec): spec for spec in specs}.values():
    for fold in study_folds(data, config):
      experiment(data, fold, spec, seed, config)

def score(jobs: list, config: dict) -> dict:
  """Select penalties or summarize neural fits with complete paired coverage.

  :param jobs: One configuration's completed cells.
  :param config: Selection rules.
  :return: Selection evidence including candidate curves.
  """
  cells = [(job["identity"]["fold"]["fold"], job["identity"]["seed"]) for job in jobs]
  seeds = {seed for _, seed in cells}
  if any(job["status"] != "completed" for job in jobs) or len(set(cells)) != len(cells) or set(cells) != {(fold, seed) for fold in range(4) for seed in seeds}:
    raise ValueError("Selection requires complete distinct fold/seed cells")
  if jobs[0]["identity"]["spec"]["model"] not in ("lstm", "transformer"):
    selected = select_penalty(jobs, config)
    return selected | {"penalty_boundary": selected["lambda"] in (min(config["readout"]["lambdas"]), max(config["readout"]["lambdas"])), "penalty_rule": "Strongest regularization within the declared macro-F1 equivalence band"}
  rows = [job["rows"][0] for job in jobs]
  return {
    "lambda": None,
    "mean_macro_f1": float(np.mean([row["validation"]["macro_f1"] for row in rows])),
    "cells": [{
      "fold": fold,
      "seed": seed,
      "macro_f1": row["validation"]["macro_f1"]
    } for (fold, seed), row in zip(cells, rows)]
  }

def winner(candidates: list, config: dict) -> dict:
  """Apply declared equivalence and deterministic candidate preference.

  :param candidates: Complete scored configurations.
  :param config: Equivalence band.
  :return: Winning candidate with its evidence.
  """
  best = max(row["selection"]["mean_macro_f1"] for row in candidates)
  eligible = [row for row in candidates if row["selection"]["mean_macro_f1"] >= best - config["selection"]["equivalence"]]
  def preference(row: dict) -> tuple:
    """Rank equivalent settings without using confirmation outcomes.

    :param row: Scored candidate.
    :return: Declared complexity and stable identity ordering.
    """
    spec = row["spec"]
    if spec["model"] in ("lstm", "transformer"):
      return (int(spec["recipe"] == "baseline"), 0, spec["recipe"])
    if spec["model"] == "qrc":
      return (spec["depth"], 0, digest(spec))
    if "segments" in spec:
      multiplier = 2 if spec.get("summary") == "segment_mean_std_final" else 1
      return (spec["size"] * (multiplier * spec["segments"] + 1), -(row["selection"]["lambda"] or 0), digest(spec))
    return (0, 0, digest(spec))
  selected = min(eligible, key=preference)
  boundary = {}
  for key, value in selected["spec"].items():
    if isinstance(value, (int, float)) and not isinstance(value, bool):
      values = sorted({row["spec"][key] for row in candidates if key in row["spec"]})
      if len(values) > 1 and value in (values[0], values[-1]):
        boundary[key] = "lower" if value == values[0] else "upper"
  return selected | {"decision": {
    "best_candidate_score": best,
    "equivalence": config["selection"]["equivalence"],
    "eligible": [digest(row["spec"]) for row in eligible],
    "selected": digest(selected["spec"]),
    "boundary_values": boundary
  }}

@stage("studies.selection.develop")
def develop(data, groups: list, seeds: list, config: dict) -> dict:
  """Screen every declared group before replication, retaining all candidate evidence.

  :param data: Development-only source.
  :param groups: Finite comparison declarations.
  :param seeds: Requested seed round coverage.
  :param config: Bound settings.
  :return: Explicit partial or complete selections.
  """
  if 42 not in seeds:
    raise ValueError("Selection requires the declared screening seed 42")
  labels = {}
  for group in groups:
    for spec in group['candidates']:
      labels.setdefault(digest(spec), set()).add(group['study'])
  config = config | {'_study_labels': {key: sorted(value) for key, value in labels.items()}}
  folds = study_folds(data, config)
  destination = Path(config["paths"]["results"]) / "metrics" / f"{data.name}-selection-{digest([item['id'] for item in groups])[:12]}.json"
  result = {
    "status": "running",
    "phase": "development",
    "dataset": data.name,
    "groups": {},
    "seeds": seeds
  }
  result["declared_candidates"] = [{
    "group": row["id"],
    "spec": spec,
    "status": "pending"
  } for row in groups for spec in row["candidates"]]
  write(destination, result)
  try:
    execute_round(data, [spec for row in groups for spec in row["candidates"]], 42, config)
  except BaseException as error:
    from src.runtime.records import read
    jobs = [read(path) for path in (Path(config["paths"]["results"]) / "metrics/jobs").glob("*/result.json")]
    for candidate in result["declared_candidates"]:
      matches = [job for job in jobs if job["identity"]["spec"] == candidate["spec"] and job["identity"]["dataset"] == data.name and job["identity"]["seed"] == 42]
      candidate["cells"] = [{
        "fold": job["identity"]["fold"]["fold"],
        "status": job["status"],
        "path": job["path"]
      } for job in matches]
      candidate["status"] = "failed" if any(job["status"] == "failed" for job in matches) else "incomplete"
    result.update({"status": "interrupted", "error": repr(error)})
    write(destination, result)
    raise
  from src.runtime.tasks import current
  scheduler = current(config)
  for group in track(groups, f"{data.name}/selection", unit="groups"):
    context = scheduler.coordinator_task('selection', {'group': group['id'], 'source': config['_identity']}, data.name) if scheduler else nullcontext()
    with context:
      candidates = []
      for spec in group["candidates"]:
        jobs = [experiment(data, fold, spec, 42, config) for fold in folds]
        candidates.append({
          "spec": spec,
          "selection": score(jobs, config),
          "jobs": [job["path"] for job in jobs],
          "status": "completed"
        })
      chosen = winner(candidates, config)
      baseline = next((row for row in candidates if row["spec"].get("recipe") == "baseline"), None)
      shortlisted = [chosen]
      if chosen["spec"]["model"] == "crc" and chosen["spec"].get("size") == 256:
        shortlisted = sorted(candidates, key=lambda row: (-row["selection"]["mean_macro_f1"], digest(row["spec"])))[:2]
        incumbent = next(row for row in candidates if row["spec"].get("segments") == 8 and row["spec"].get("summary") == "segment_mean_std_final")
        shortlisted = list({digest(row["spec"]): row for row in shortlisted + [incumbent]}.values())
      result["groups"][group["id"]] = {
        "declaration": group,
        "screen": candidates,
        "selected": chosen,
        "shortlisted": shortlisted,
        "baseline": baseline,
        "replication": {}
      }
      write(destination, result)
  for seed in track(seeds, f"{data.name}/replication", unit="seeds"):
    execute_round(data, [row["spec"] for item in result["groups"].values() if not item["declaration"].get("screen_only") for row in item["shortlisted"]], seed, config)
    for item in result["groups"].values():
      if item["declaration"].get("screen_only"):
        continue
      variants = item["shortlisted"] + ([item["baseline"]] if item["baseline"] else [])
      for candidate in variants:
        key = digest(candidate["spec"])
        jobs = [experiment(data, fold, candidate["spec"], seed, config) for fold in folds]
        item["replication"].setdefault(key, []).extend(job["path"] for job in jobs)
      write(destination, result)
  from src.runtime.records import read
  for item in result["groups"].values():
    if item["declaration"].get("screen_only"):
      continue
    for key, paths in item["replication"].items():
      item["replication"][key] = list(dict.fromkeys(paths))
    chosen = item["selected"]
    if len(item["shortlisted"]) > 1:
      candidates = []
      for candidate in item["shortlisted"]:
        jobs = [read(path) for path in item["replication"][digest(candidate["spec"])]]
        candidates.append(candidate | {"selection": score(jobs, config), "jobs": [job["path"] for job in jobs]})
      chosen = winner(candidates, config)
    jobs = [read(path) for path in item["replication"][digest(chosen["spec"])]]
    chosen = chosen | {"selection": score(jobs, config), "jobs": [job["path"] for job in jobs]}
    if item["baseline"]:
      baseline = item["baseline"]
      base_jobs = [read(path) for path in item["replication"][digest(baseline["spec"])]]
      base = score(base_jobs, config)
      left = {(row["fold"], row["seed"]): row["macro_f1"] for row in chosen["selection"]["cells"]}
      right = {(row["fold"], row["seed"]): row["macro_f1"] for row in base["cells"]}
      gains = [float(np.mean([left[(fold, seed)] - right[(fold, seed)] for seed in seeds])) for fold in range(4)]
      accepted = np.mean(gains) >= config["selection"]["supported_gain"] and sum(value > 0 for value in gains) >= config["selection"]["positive_folds"]
      item["adoption"] = {
        "mean_gain": float(np.mean(gains)),
        "fold_gains": gains,
        "accepted": bool(accepted)
      }
      if not accepted:
        chosen = baseline | {"selection": base, "jobs": [job["path"] for job in base_jobs]}
        jobs = base_jobs
      chosen["spec"] = chosen["spec"] | {"fixed_epochs": int(np.rint(np.median([job["best_epoch"] for job in jobs])))}
    item["selected"] = chosen
  for candidate in result["declared_candidates"]:
    candidate["status"] = "completed"
  result["status"] = "completed" if seeds == config["seeds"] else "partial"
  write(destination, result)
  return result
