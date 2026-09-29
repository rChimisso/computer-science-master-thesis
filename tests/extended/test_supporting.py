from unittest.mock import patch

from src.data.indexed import study_folds
from src.runtime.records import write
from src.runtime.resources import policy
from src.runtime.tasks import Scheduler
from src.runtime.validation import fixture
from src.studies.design import groups
from src.studies.experiments import experiment

from tests.support import SyntheticCase

class SupportingTests(SyntheticCase):
  """Protect declared supporting-study execution and seed budgets."""

  def test_supporting_tasks_and_isolated_measurements(self):
    """Execute finite supporting controls and keep resource measurements isolated."""
    with self.subTest(case='supporting_views_and_resource_measurement'):
      self.reset_case()
      from src.studies.resources import measure
      for name in ("shd", "dvs"):
        data = fixture(name)
        fold = study_folds(data, self.config)[0]
        declarations = groups(name, [
          "observations",
          "recurrence",
          "mapping"
        ], [], self.config)
        for declaration in declarations:
          spec = declaration["candidates"][0]
          if spec["model"] == "qrc":
            self.assertTrue({
              "depth",
              "qubits",
              "inputs"
            } <= set(spec))
            continue
          job = experiment(data, fold, spec, 42, self.config)
          self.assertEqual(job["status"], "completed")
      spec = {
        "model": "crc",
        "inputs": 2,
        "size": 12,
        "radius": 0.9,
        "scale": 0.03,
        "tau": 20
      }
      entry = {"spec": spec, "selection": {"lambda": 0.1}}
      frozen = {"manifest": {"entries": {"shd": {"practical:crc-fixture": entry}}}}
      config = self.config | {"seeds": [42]}
      from src.runtime.pipelines import export
      fold = {"fold": "confirmation", "train": self.data.development.tolist(), "validation": self.data.confirmation.tolist()}
      fitted = experiment(self.data, fold, spec, 42, config, [0.1], "confirmation")
      package = export(self.data, self.data.development, spec, 42, "confirmation", fitted, config)
      write(self.root / "evidence/metrics/confirmation.json", {"jobs": [{"dataset": "shd", "role": "practical:crc-fixture", "seed": 42, "path": fitted["path"]}]})
      write(self.root / "evidence/pipelines.json", {"confirmation:shd:practical:crc-fixture:42": package})
      result = measure({"shd": self.data}, frozen, config)
      self.assertEqual(len(result["practical"][0]["measurements"]), 6)
      self.assertFalse(result["quantum_simulator_costs_are_hardware_costs"])
    with self.subTest(case='waiting_resource_measurement_drains_all_workers'):
      self.reset_case()
      import time
      from src.runtime.resources import request

      def requirements(operation, spec, config):
        """Use fixture work while preserving real isolated-measurement admission.

        :param operation: Fixture operation.
        :param spec: Optional measurement marker.
        :param config: CPU fixture configuration.
        :return: Production resource reservations.
        """
        return request('resources' if spec.get('measurement') else operation, spec, config)

      payload = {
        'config': self.config,
        'steps': 20,
        'delay': 0.1
      }
      with patch('src.runtime.tasks.request', side_effect=requirements), Scheduler(self.root, policy(2, 2, 1), self.config) as scheduler:
        active = scheduler.submit('fixture', payload, {'test': 'active'})
        deadline = time.monotonic() + 10
        while active.task_id not in scheduler.running and time.monotonic() < deadline:
          time.sleep(0.02)
        self.assertIn(active.task_id, scheduler.running)
        with scheduler.mutex:
          measurement = scheduler.submit('fixture', payload | {'spec': {'measurement': True}}, {'test': 'measurement'})
          queued = scheduler.submit('fixture', payload, {'test': 'queued'})
        for future in (active, measurement, queued):
          self.assertEqual(future.result(timeout=30), 1)
        records = scheduler.records
        self.assertGreaterEqual(records[measurement.task_id]['started_utc'], records[active.task_id]['finished_utc'])
        self.assertGreaterEqual(records[queued.task_id]['started_utc'], records[measurement.task_id]['finished_utc'])

  def test_supporting_selection_budgets(self):
    """Keep mapping screening seeds and learning-curve seed barriers explicit."""
    with self.subTest(case='mapping_screen_uses_declared_single_seed'):
      self.reset_case()
      from src.studies.selection import develop
      declarations = groups("shd", ["mapping"], [], self.config)
      result = develop(self.data, declarations, [
        42,
        44,
        46
      ], self.config)
      for item in result["groups"].values():
        self.assertTrue(item["declaration"]["screen_only"])
        self.assertEqual({cell["seed"] for cell in item["selected"]["selection"]["cells"]}, {42})
        self.assertEqual(item["replication"], {})
    with self.subTest(case='learning_curve_seed_round_barriers'):
      self.reset_case()
      from unittest.mock import Mock
      from src.studies.supporting import learning_curves
      roles = [f'comparison:tuned:q6-i{width}-{family}' for width in (2, 4) for family in ('qrc', 'crc')]
      roles += [f'comparison:projection:i{width}' for width in (2, 4)]
      selected = {'groups': {role: {'selected': {'spec': {'model': 'projection', 'inputs': 2}, 'selection': {'lambda': 0.1}}} for role in roles}}
      pending = []
      rounds = []

      def submit(data, fold, spec, seed, config, penalties):
        """Expose unfinished futures so advancing a seed too early is observable.

        :param data: Synthetic data.
        :param fold: Fixed validation and subset membership.
        :param spec: Fixture specification.
        :param seed: Initialization round.
        :param config: Bound fixture settings.
        :param penalties: Frozen regularization.
        :return: Lazily completed fitting future.
        """
        if not rounds or rounds[-1] != seed:
          self.assertFalse(pending)
          rounds.append(seed)
        key = object()
        pending.append(key)

        def result():
          """Mark this fixture task consumed.

          :return: Minimal saved-result reference.
          """
          pending.remove(key)
          return {'path': 'synthetic-job'}

        return Mock(result=result)

      scheduler = Mock(experiment=submit)
      with patch('src.runtime.tasks.current', return_value=scheduler):
        record = learning_curves(fixture('shd'), selected, [
          42,
          44,
          46
        ], self.config)
      self.assertEqual(rounds, [
        42,
        44,
        46
      ])
      self.assertEqual(record['status'], 'completed')
