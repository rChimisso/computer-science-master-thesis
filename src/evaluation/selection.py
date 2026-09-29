import numpy as np

def select_penalty(jobs: list[dict], config: dict) -> dict:
  """Choose a common penalty over complete shared fold/seed cells.

  :param jobs: Complete jobs for exactly one numerical configuration.
  :param config: Equivalence band and ridge grid.
  :return: Selected aggregate with explicit paired cells.
  """
  if not jobs or any(job["status"] != "completed" for job in jobs):
    raise ValueError("Incomplete comparisons cannot enter selection")
  cells = [(job["identity"]["fold"]["fold"], job["identity"]["seed"]) for job in jobs]
  if len(set(cells)) != len(cells):
    raise ValueError("Duplicated fold/seed cells")
  seeds = {seed for _, seed in cells}
  if set(cells) != {(fold, seed) for fold in range(4) for seed in seeds}:
    raise ValueError("Every selected seed requires all four held-group folds")
  candidates = []
  for penalty in config["readout"]["lambdas"]:
    results = [next(row for row in job["rows"] if row["lambda"] == penalty) for job in jobs]
    candidates.append({
      "lambda": penalty,
      "mean_macro_f1": float(np.mean([row["validation"]["macro_f1"] for row in results])),
      "mean_accuracy": float(np.mean([row["validation"]["accuracy"] for row in results])),
      "mean_train_macro_f1": float(np.mean([row["train"]["macro_f1"] for row in results])),
      "mean_train_accuracy": float(np.mean([row["train"]["accuracy"] for row in results])),
      "cells": [{
        "fold": fold,
        "seed": seed,
        "macro_f1": row["validation"]["macro_f1"],
        "accuracy": row["validation"]["accuracy"]
      } for (fold, seed), row in zip(cells, results)]
    })
  best = max(row["mean_macro_f1"] for row in candidates)
  selected = max((row for row in candidates if row["mean_macro_f1"] >= best - config["selection"]["equivalence"]), key=lambda row: row["lambda"])
  return selected | {"curve": [{key: value for key, value in row.items() if key != "cells"} for row in candidates]}
