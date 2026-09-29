from pathlib import Path

import numpy as np

from src.runtime.progress import stage, track
from src.data.indexed import nested_indices, study_folds
from src.runtime.records import write
from src.studies.design import group
from src.studies.experiments import experiment
from src.studies.selection import develop

@stage("studies.supporting.mappings")
def mappings(data, selected: dict, seeds: list, config: dict) -> dict:
  """Advance the best projection-only mapping and anchor without changing core inputs.

  :param data: Development data.
  :param selected: Completed seed-round projection screens.
  :param seeds: Dynamics robustness seeds.
  :param config: Main settings and explicit six-qubit anchors.
  :return: Paired mapping evidence separated from core selection.
  """
  declarations = []
  advanced = {}
  for width in config["supporting"]["mapping"]["widths"]:
    candidates = {mode: selected["groups"][f"mapping:screen:i{width}:{mode}"]["screen"][0] for mode in ("orthogonal", "pooling", "pca")}
    best = max(row["selection"]["mean_macro_f1"] for row in candidates.values())
    mode = next(mode for mode in ("pooling", "orthogonal", "pca") if candidates[mode]["selection"]["mean_macro_f1"] >= best - config["selection"]["equivalence"])
    advanced[str(width)] = mode
    for adapter in ("legacy", mode):
      for family in ("qrc", "crc", "projection"):
        base = config["supporting"]["mapping"]["anchors"][data.name][f"{family}{width}"] if family != "projection" else {"model": family, "inputs": width}
        spec = base | {"adapter": adapter, "solver": "lsqr"}
        declarations.append(group("mapping-paired", f"i{width}:{adapter}:{family}", [spec], final=False))
        if adapter in ("legacy", "orthogonal"):
          for projection_seed in config["supporting"]["mapping"]["projection_seeds"]:
            robustness = spec | {"projection_seed": projection_seed}
            declarations.append(group("mapping-robustness", f"i{width}:{adapter}:{family}:p{projection_seed}", [robustness], final=False))
  paired = develop(data, [row for row in declarations if row["study"] == "mapping-paired"], seeds, config)
  robustness = develop(data, [row for row in declarations if row["study"] == "mapping-robustness"], [42], config)
  result = {
    "status": paired["status"],
    "advanced": advanced,
    "paired": paired,
    "projection_robustness": robustness,
    "core_adapter_unchanged": config["datasets"][data.name]["adapter"]
  }
  write(Path(config["paths"]["results"]) / "metrics" / f"{data.name}-mapping.json", result)
  return result

@stage("studies.supporting.learning_curves")
def learning_curves(data, selected: dict, seeds: list, config: dict) -> dict:
  """Refit fixed selected pipelines on nested SHD class-and-speaker subsets.

  :param data: Official-training SHD source.
  :param selected: Core selected dynamics and penalties.
  :param seeds: Declared dynamics seeds.
  :param config: Subset sizes and shared settings.
  :return: Complete paired development learning-curve records.
  """
  if data.name != "shd":
    return {"status": "not_applicable", "reason": "The declared training-size study is SHD only"}
  roles = [f"comparison:tuned:q6-i{width}-{family}" for width in (2, 4) for family in ("qrc", "crc")]
  roles += [f"comparison:projection:i{width}" for width in (2, 4)]
  roles += [role for role in selected["groups"] if role.startswith("practical:")]
  from src.runtime.tasks import current
  scheduler = current(config)
  rows = []
  for seed in seeds:
    pending = []
    for fold in study_folds(data, config):
      training = np.asarray(fold["train"])
      for size in config["supporting"]["learning_sizes"] + [len(training)]:
        indices = nested_indices(data, training, size)
        subset = fold | {"train": indices.tolist(), "subset": size}
        for role in track(roles, f"{data.name}/learning/fold={fold['fold']}/size={size}/seed={seed}", unit="models"):
          entry = selected["groups"][role]["selected"]
          spec = {key: value for key, value in entry["spec"].items() if key != "fixed_epochs"}
          penalty = entry["selection"]["lambda"]
          metadata = {'role': role, 'size': size, 'fold': fold['fold'], 'seed': seed, 'membership': data.membership(indices)}
          penalties = [penalty] if penalty is not None else None
          if scheduler:
            pending.append((metadata, scheduler.experiment(data, subset, spec, seed, config, penalties)))
          else:
            job = experiment(data, subset, spec, seed, config, penalties)
            rows.append(metadata | {'job': job['path']})
    for metadata, future in pending:
      job = future.result()
      rows.append(metadata | {'job': job['path']})
      write(Path(config['paths']['results']) / 'metrics/learning-curves.json', {'status': 'running', 'rows': rows})
    write(Path(config['paths']['results']) / 'metrics/learning-curves.json', {'status': 'running', 'rows': rows})
  result = {
    "status": "completed" if seeds == config["seeds"] else "partial",
    "rows": rows,
    "hyperparameters": "Selected on complete development folds; transformations and readouts refitted on each subset"
  }
  write(Path(config["paths"]["results"]) / "metrics/learning-curves.json", result)
  return result
