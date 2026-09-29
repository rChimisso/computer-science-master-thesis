def parameters(value: dict, prefix: str = "") -> dict:
  """Flatten configuration leaves while retaining candidate arrays as values.

  :param value: Resolved configuration.
  :param prefix: Qualified parameter prefix.
  :return: Stable path-to-value mapping.
  """
  result = {}
  for key, item in value.items():
    path = f"{prefix}.{key}" if prefix else key
    if key.startswith("_") or path.startswith("paths"):
      continue
    if isinstance(item, dict):
      result.update(parameters(item, path))
    else:
      result[path] = item
  return result

def validate_register(config: dict, register: dict) -> list:
  """Require explicit origins and exact values for every scientific setting.

  :param config: Resolved scientific configuration.
  :param register: Authored parameter provenance.
  :return: Entries with unresolved provenance.
  """
  actual = parameters(config)
  entries = register["parameters"]
  if set(actual) != set(entries):
    raise ValueError(f"Parameter justification coverage mismatch: {sorted(set(actual) ^ set(entries))}")
  unresolved = []
  for name, value in actual.items():
    entry = entries[name]
    if entry["value"] != value or not entry["origins"] or not entry["rationale"]:
      raise ValueError(f"Missing or stale justification: {name}")
    if "literature-informed" in entry["origins"] and not all(entry.get(key) for key in ("reference", "location", "departures")):
      raise ValueError(f"Literature attribution lacks a location or implementation differences: {name}")
    if "unresolved" in entry["origins"]:
      unresolved.append(name)
  return unresolved

