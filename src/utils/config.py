from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
"""Absolute path to the repository root."""

def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
  """Load a YAML configuration.

  :param path: Absolute path or repository-relative configuration path.
  :return: Parsed configuration mapping with its resolved source path.
  :raises ValueError: If the parsed configuration is not a mapping.
  """
  config_path = Path(path)
  if not config_path.is_absolute():
    config_path = REPOSITORY_ROOT / config_path
  with config_path.open(encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)
  if not isinstance(config, dict):
    raise ValueError(f"Configuration must be a mapping: {config_path}")
  config["_config_path"] = str(config_path)
  return config

def repository_path(configured_path: str | Path) -> Path:
  """Resolve a configured repository path.

  :param configured_path: Absolute path or path relative to the repository root.
  :return: Absolute path for the configured value.
  """
  path = Path(configured_path)
  return path if path.is_absolute() else REPOSITORY_ROOT / path
