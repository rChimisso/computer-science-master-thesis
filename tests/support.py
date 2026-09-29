import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data.indexed import study_folds
from src.evaluation.metrics import classification_metrics
from src.runtime.configuration import bind, load
from src.runtime.records import digest, read, write
from src.runtime.storage import save_job, seal, snapshot_dataset
from src.runtime.validation import fixture
from src.studies.design import group
from src.workflow import run

class SyntheticCase(unittest.TestCase):
  """Provide isolated, bounded CPU inputs without saved dataset fixtures.

  :ivar temporary: Owned temporary directory.
  :ivar case_number: Counter separating subcase artifacts.
  :ivar root: Current subcase directory.
  :ivar config: CPU-only execution settings.
  :ivar data: Synthetic SHD descriptor.
  """

  def setUp(self):
    """Create test-owned storage and register cleanup even when setup fails."""
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.case_number = 0
    self.reset_case()

  def reset_case(self):
    """Start an independent subcase with fresh data and execution settings."""
    self.case_number += 1
    self.root = Path(self.temporary.name) / str(self.case_number)
    self.root.mkdir()
    self.config = bind(load(), self.root)
    self.config['paths']['cache'] = str(self.root / 'shared')
    self.config['execution'].update({'device': 'cpu', 'crc_device': 'cpu', 'crc_graphs': False, 'qrc_small': 'CPU', 'cpu_threads': 1})
    self.config['training'].update({'cpu_threads': 1, 'max_epochs': 5, 'min_epochs': 1, 'patience': 1})
    self.config['neural']['lstm']['hidden_size'] = 4
    self.config['_identity'] = 'synthetic-contract'
    self.data = fixture('shd')

class EvidenceCase(SyntheticCase):
  """Provide complete compact reporting evidence for each independent subcase.

  :ivar run: Current evidence namespace.
  :ivar specs: Matched reservoir definitions.
  :ivar manifest: Declared evidence population and settings.
  """

  def reset_case(self):
    """Build a fresh matched evidence population and frozen selection."""
    super().reset_case()
    self.run = self.root / "run"
    self.data = fixture("shd")
    config = load()
    self.specs = {
      "comparison:fixed:q4-i2-qrc": {"model": "qrc", "inputs": 2, "qubits": 4, "depth": 1, "view": "all"},
      "comparison:fixed:q4-i2-crc": {"model": "crc", "inputs": 2, "size": 12, "radius": 0.9, "scale": 0.03, "tau": 20, "view": "all"}
    }
    declarations = [group("comparison", role.removeprefix("comparison:"), [spec]) for role, spec in self.specs.items()]
    self.manifest = {
      "evidence_version": 1,
      "purpose": "Synthetic reporting fixture, not a scientific result",
      "config": config,
      "datasets": ["shd"],
      "studies": ["comparison", "generalization"],
      "models": [],
      "resources": False,
      "declarations": {"shd": declarations},
      "parameter_register": read("configs/parameters.json")
    }
    write(self.run / "manifest.json", self.manifest)
    self.config = bind(config, self.run)
    self.config["paths"]["cache"] = str(self.root / "shared")
    self.config["execution"]["device"] = "cpu"
    self.config["_identity"] = "synthetic"
    snapshot_dataset(self.data, self.config)
    selection = {"status": "completed", "phase": "development", "seeds": config["seeds"], "groups": {}}
    for declaration in declarations:
      role = declaration["id"]
      paths = []
      for seed in config["seeds"]:
        for fold in study_folds(self.data, config):
          paths.append(self.job(role, seed, fold, "development"))
      chosen = {"spec": self.specs[role], "jobs": paths, "selection": {"lambda": 0.1, "mean_macro_f1": 1.0, "cells": []}}
      selection["groups"][role] = {"declaration": declaration, "selected": chosen, "screen": [chosen], "replication": {}}
    write(self.run / "evidence/metrics/selections.json", {"shd": selection})
    body = {
      "config": config,
      "entries": {"shd": {role: item["selected"] for role, item in selection["groups"].items()}},
      "decisions": {"shd": selection},
      "memberships": {"shd": {"development": self.data.membership(self.data.development), "confirmation": self.data.membership(self.data.confirmation)}}
    }
    frozen = {"status": "frozen", "manifest": body, "fingerprint": digest(body)}
    write(self.run / "frozen.json", frozen)
    jobs = []
    fold = {"fold": "confirmation", "train": self.data.development.tolist(), "validation": self.data.confirmation.tolist()}
    for role in self.specs:
      for seed in config["seeds"]:
        jobs.append({"dataset": "shd", "role": role, "seed": seed, "path": self.job(role, seed, fold, "confirmation")})
    write(self.run / "evidence/metrics/confirmation.json", {"status": "completed", "frozen": frozen["fingerprint"], "jobs": jobs})
    seal(self.run)

  def job(self, role: str, seed: int, fold: dict, phase: str) -> str:
    """Persist deterministic predictions with actual synthetic memberships.

    :param role: Declared comparison role.
    :param seed: Initialization identifier.
    :param fold: Sample indices and fold identifier.
    :param phase: Population boundary.
    :return: Saved record path.
    """
    membership = {"train": self.data.membership(np.asarray(fold["train"])), "held": self.data.membership(np.asarray(fold["validation"]))}
    rows = []
    for penalty in self.config["readout"]["lambdas"] if phase == "development" else [0.1]:
      rows.append({
        "lambda": penalty,
        "status": "completed",
        "solver": "cholesky",
        "train": classification_metrics(np.asarray(membership["train"]["labels"]), np.asarray(membership["train"]["labels"]), 3),
        "validation": classification_metrics(np.asarray(membership["held"]["labels"]), np.asarray(membership["held"]["labels"]), 3),
        "predictions": membership["held"]["labels"],
        "train_predictions": membership["train"]["labels"]
      })
    path = self.run / "evidence/metrics/jobs" / digest([role, seed, fold, phase]) / "result.json"
    job = {
      "identity": {"dataset": "shd", "spec": self.specs[role], "seed": seed, "fold": fold, "phase": phase},
      "status": "completed",
      "classes": 3,
      "path": str(path),
      "membership": membership,
      "rows": rows,
      "adapter": {"fingerprint": "shared-input-stream"}
    }
    save_job(path, job, self.config)
    return str(path)

class FrozenCase(unittest.TestCase):
  """Provide a tiny frozen workflow with explicit memberships.

  :ivar temporary: Owned storage.
  :ivar root: Run namespace.
  :ivar config: Scientific settings.
  :ivar data: Synthetic training descriptor.
  :ivar declarations: Projection-only fitting scope.
  :ivar request: Resolved workflow request.
  """

  def setUp(self):
    """Prepare a bounded workflow and register cleanup."""
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name) / 'resume-fixture'
    self.config = load()
    self.config['paths'].update({'runs': self.temporary.name, 'cache': str(Path(self.temporary.name) / 'cache')})
    self.data = fixture('shd')
    self.declarations = [group('comparison', 'projection:i2', [{'model': 'projection', 'inputs': 2, 'view': 'all'}])]
    self.request = {
      'config': self.config,
      'profile': 'core',
      'studies': ['comparison'],
      'datasets': ['shd'],
      'models': [],
      'seeds': self.config['seeds'],
      'official_test': False,
      'resources': False
    }

  def initial_run(self):
    """Fit and freeze a real tiny comparison with only final rendering substituted.

    :return: Completed workflow result.
    """
    with patch('src.data.datasets.load', return_value=self.data), patch('src.workflow.groups', return_value=self.declarations), patch('src.workflow.finish', return_value={}):
      return run(self.request, self.root.name)
