import itertools

from src.runtime.records import digest

def anchor(name: str, family: str, qubits: int, inputs: int, config: dict) -> dict:
  """Declare common fixed reservoir dynamics.

  :param name: Dataset.
  :param family: CRC or QRC.
  :param qubits: Allocation size.
  :param inputs: Shared input width.
  :param config: Scientific settings.
  :return: Numeric specification.
  """
  spec = {
    "model": family,
    "inputs": inputs,
    "view": "all"
  }
  return spec | ({"qubits": qubits, "depth": config["datasets"][name]["depth"]} if family == "qrc" else {"size": 3 * qubits} | config["anchors"]["crc"])

def group(study: str, role: str, candidates: list, final: bool = True) -> dict:
  """Describe an independently selected, fully paired comparison row.

  :param study: Scientific question identifier.
  :param role: Stable row label.
  :param candidates: Explicit finite numerical specifications.
  :param final: Include the selected pipeline in frozen evaluation.
  :return: Queue declaration.
  """
  return {
    "id": f"{study}:{role}",
    "study": study,
    "role": role,
    "candidates": candidates,
    "final": final
  }

def groups(name: str, studies: list, models: list, config: dict) -> list:
  """Enumerate independent finite studies without opening previous artifacts.

  :param name: Dataset.
  :param studies: Enabled study identifiers.
  :param models: Optional practical families.
  :param config: Standalone scientific settings.
  :return: Declared groups with no data-dependent winners.
  """
  rows = []
  if "comparison" in studies:
    for qubits, inputs in config["allocations"]:
      for family in ("qrc", "crc"):
        label = f"q{qubits}-i{inputs}-{family}"
        fixed = anchor(name, family, qubits, inputs, config)
        rows.append(group("comparison", f"fixed:{label}", [fixed]))
        control = fixed | ({"view": "fixed12"} if family == "qrc" else {"size": 12})
        rows.append(group("comparison", f"fixed12:{label}", [control]))
        variations = [{"depth": value} for value in config["search"]["depths"]] if family == "qrc" else [
          {
            "radius": radius,
            "scale": scale,
            "tau": tau
          }
          for radius, scale, tau in itertools.product(config["search"]["radii"], config["search"]["scales"], config["search"]["timescales_ms"])
        ]
        rows.append(group("comparison", f"tuned:{label}", [fixed | value for value in variations]))
    for width in (2, 4, 6):
      rows.append(group("comparison", f"projection:i{width}", [{
        "model": "projection",
        "inputs": width,
        "view": "all"
      }]))
  if "observations" in studies:
    for key, fixed in config["supporting"]["observations"][name].items():
      for view in (("all", "zz", "products") if fixed["model"] == "qrc" else ("all", "products")):
        rows.append(group("observations", f"{key}:{view}", [fixed | {"view": view, "solver": "lsqr"}]))
      if fixed["model"] == "crc":
        rows.append(group("observations", f"{key}:width24", [fixed | {"size": 24, "solver": "lsqr"}]))
    for width in (2, 4):
      rows.append(group("observations", f"projection:i{width}", [{
        "model": "projection",
        "inputs": width,
        "solver": "lsqr"
      }]))
  if "recurrence" in studies:
    for family, spec in config["supporting"]["recurrence"][name].items():
      for reset in ("partial", "all"):
        rows.append(group("recurrence", f"{family}:{reset}", [spec | {"reset": reset}]))
  if "mapping" in studies:
    for width in config["supporting"]["mapping"]["widths"]:
      for mode in config["supporting"]["mapping"]["modes"]:
        spec = {
          "model": "projection",
          "inputs": width,
          "adapter": mode,
          "solver": "lsqr"
        }
        rows.append(group("mapping", f"screen:i{width}:{mode}", [spec], final=False) | {"screen_only": True})
  if "large-crc" in models:
    base = config["supporting"]["practical"][name]["large_crc"]
    spec = dict(base) | {
      "model": "crc",
      "inputs": 0,
      "adapter": "identity",
      "solver": "lsqr"
    }
    candidates = [spec | {"segments": segments, "summary": mode} for segments in (1, 4, 8) for mode in ("segment_mean_final", "segment_mean_std_final")]
    rows.append(group("practical", "large-crc", candidates))
  for model in ("lstm", "transformer"):
    if model in models:
      rows.append(group("practical", model, [{"model": model, "recipe": recipe} for recipe in ("baseline", "dropout", "regularization")]))
  for row in rows:
    row["id"] = f"{row['study']}:{row['role']}"
    row["candidate_ids"] = [digest(spec) for spec in row["candidates"]]
  return rows
