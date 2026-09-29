import copy
from unittest.mock import patch

from src.runtime.configuration import bind, load
from src.runtime.records import checksum, digest, read, write
from src.runtime.resources import admission, policy, request
from src.runtime.tasks import Scheduler
from src.workflow import run

from tests.support import FrozenCase, SyntheticCase

class SchedulingTests(SyntheticCase):
  """Protect resource bounds, single ownership and failure propagation."""

  def test_resource_admission(self):
    """Bound worker reservations, threads, GPU exclusivity and native memory."""
    with self.subTest(case='budgets_and_exclusive_admission'):
      limits = policy(2, 4, 1)
      memory = {
        'host_available': 1000,
        'host_total': 1000,
        'gpu_free': 1000,
        'gpu_total': 1000
      }
      cpu = {
        'cpu_workers': 1,
        'cpu_threads': 2,
        'gpu_workers': 0,
        'exclusive': False,
        'host_bytes': 100,
        'gpu_bytes': 0
      }
      gpu = cpu | {
        'cpu_workers': 0,
        'gpu_workers': 1,
        'gpu_bytes': 100
      }
      self.assertIsNone(admission(gpu, [cpu], limits, memory))
      self.assertEqual(admission(cpu, [cpu, gpu], limits, memory), 'cpu threads budget')
      self.assertEqual(admission(gpu | {'exclusive': True}, [cpu], limits, memory), 'exclusive task')
      self.assertIsNone(admission(gpu | {'gpu_exclusive': True}, [cpu], limits, memory))
      self.assertEqual(admission(gpu, [gpu | {'gpu_exclusive': True}], policy(2, 8, 2), memory), 'exclusive GPU')
      self.assertEqual(admission(gpu, [], limits, memory | {'gpu_free': None}), 'GPU memory unavailable')
    with self.subTest(case='cuda_crc_reservations_and_device_labels'):
      config = load()
      config['execution']['crc_device'] = 'cuda'
      for operation in ('feature', 'export', 'final-fit', 'final-evaluate'):
        with self.subTest(operation=operation):
          spec = {'model': 'crc', 'size': 18}
          wanted = request(operation, spec, config)
          self.assertEqual(wanted['gpu_workers'], 1)
          self.assertEqual(wanted['cpu_workers'], 0)
          self.assertGreater(wanted['gpu_bytes'], 0)
    with self.subTest(case='cpu_stages_do_not_reserve_gpu'):
      config = load()
      config['execution'].update({'device': 'cuda', 'crc_device': 'cuda', 'qrc_small': 'GPU'})
      for model in ('qrc', 'crc', 'lstm'):
        for operation in ('adapter', 'readout', 'data'):
          wanted = request(operation, {'model': model}, config)
          self.assertEqual(wanted['gpu_workers'], 0)
          self.assertEqual(wanted['gpu_bytes'], 0)
    with self.subTest(case='gpu_crc_respects_neural_exclusivity'):
      config = load()
      config['execution'].update({'device': 'cuda', 'crc_device': 'cuda'})
      neural = request('neural', {'model': 'lstm'}, config)
      crc = request('feature', {'model': 'crc'}, config)
      available = {'host_available': 100 * 1024 ** 3, 'host_total': 100 * 1024 ** 3, 'gpu_free': 10 * 1024 ** 3, 'gpu_total': 10 * 1024 ** 3}
      self.assertEqual(admission(crc, [neural], policy(2, 12, 2), available), 'exclusive GPU')
    with self.subTest(case='pending_memory_is_reserved_against_stale_free_sample'):
      wanted = {
        'cpu_workers': 0,
        'cpu_threads': 1,
        'gpu_workers': 1,
        'exclusive': False,
        'host_bytes': 100,
        'gpu_bytes': 600
      }
      available = {'host_available': 900, 'host_total': 8000, 'gpu_free': 900, 'gpu_total': 8000}
      self.assertIsNone(admission(wanted, [], policy(2, 12, 2), available))
      self.assertEqual(admission(wanted, [wanted], policy(2, 12, 2), available), 'GPU memory')
      cpu = wanted | {'cpu_workers': 1, 'gpu_workers': 0, 'gpu_bytes': 0, 'host_bytes': 400}
      self.assertIsNone(admission(cpu, [], policy(2, 12, 2), available))
      self.assertEqual(admission(cpu, [cpu], policy(2, 12, 2), available), 'host memory')
      self.assertIsNone(admission(wanted, [], policy(2, 12, 2), available))

  def test_task_ownership_and_restart(self):
    """Deduplicate task and final-fit writes and reuse committed work on restart."""
    with self.subTest(case='parallel_tasks_deduplicate_and_resume'):
      self.reset_case()
      payload = {
        'config': self.config,
        'steps': 20,
        'delay': 0.03,
        'value': 7
      }
      with Scheduler(self.root, policy(2, 2, 1), self.config) as scheduler:
        left = scheduler.submit('fixture', payload, {'test': 'a'})
        self.assertIs(left, scheduler.submit('fixture', payload, {'test': 'a'}))
        right = scheduler.submit('fixture', payload | {'value': 8}, {'test': 'b'})
        self.assertEqual((left.result(timeout=30), right.result(timeout=30)), (7, 8))
        rows = list(scheduler.records.values())
      with Scheduler(self.root, policy(1, 2, 1), self.config) as scheduler:
        self.assertEqual(scheduler.submit('fixture', payload, {'test': 'a'}).result(timeout=5), 7)
        self.assertTrue(scheduler.records[left.task_id]['reused'])
    with self.subTest(case='identical_final_models_share_one_writer_across_roles'):
      self.reset_case()
      from src.workflow import final_fit_identity
      config = bind(self.config, self.root)
      spec = {'model': 'projection', 'inputs': 2, 'view': 'all'}
      left = {'spec': spec, 'selection': {'lambda': 0.1}, 'decision': 'fixed'}
      right = left | {'decision': 'tuned'}
      first = final_fit_identity(self.data, left, 42, 'frozen')
      second = final_fit_identity(self.data, right, 42, 'frozen')
      self.assertEqual(first, second)
      scheduler = Scheduler(self.root, policy(2, 12, 1), config)
      payload = {'data': self.data, 'entry': left, 'spec': spec, 'seed': 42, 'phase': 'official', 'config': config}
      one = scheduler.submit('final-fit', payload, first)
      two = scheduler.submit('final-fit', payload | {'entry': right}, second)
      self.assertIs(one, two)
      self.assertEqual(len(scheduler.records), 1)
      self.assertNotEqual(first, final_fit_identity(self.data, right | {'selection': {'lambda': 1}}, 42, 'frozen'))
      self.assertNotEqual(first, final_fit_identity(self.data, right, 44, 'frozen'))

  def test_failure_stops_dependent_work(self):
    """The first unexpected failure blocks consumers and pauses submission."""
    payload = {'config': self.config, 'fail': True}
    with Scheduler(self.root, policy(1, 2, 1), self.config) as scheduler:
      left = scheduler.submit('fixture', payload, {'test': 'failure'})
      right = scheduler.submit('fixture', payload | {'fail': False}, {'test': 'consumer'}, (left,))
      with self.assertRaises(RuntimeError):
        left.result(timeout=30)
      with self.assertRaises(RuntimeError):
        right.result(timeout=30)
    self.assertTrue((self.root / 'STOP').exists())

class FrozenTests(FrozenCase):
  """Protect frozen resumption and declared populations."""

  def test_frozen_resume_contract(self):
    """Resume without selection and reject changed frozen roles or memberships."""
    import src.workflow as workflow
    self.initial_run()
    frozen_hash = checksum(self.root / 'frozen.json')
    jobs = {str(path): read(path) for path in (self.root / 'evidence/metrics/jobs').glob('*/result.json')}
    with patch('src.data.datasets.load', return_value=self.data) as loader, patch('src.workflow.groups', return_value=self.declarations):
      with patch('src.studies.selection.develop', side_effect=AssertionError('Development rerun')), patch('src.studies.supporting.mappings', side_effect=AssertionError('Mapping rerun')):
        with patch('src.workflow.evaluate_frozen', wraps=workflow.evaluate_frozen) as evaluate, patch('src.workflow.finish', return_value={}) as finish:
          result = run(self.request, self.root.name, resume=True)
    self.assertEqual(result['status'], 'completed')
    self.assertEqual(result['completed'], ['shd'])
    self.assertEqual(checksum(self.root / 'frozen.json'), frozen_hash)
    self.assertEqual(jobs, {str(path): read(path) for path in (self.root / 'evidence/metrics/jobs').glob('*/result.json')})
    self.assertFalse(evaluate.call_args.args[1])
    self.assertEqual(loader.call_args.args[0], 'shd')
    self.assertEqual(len(loader.call_args.args), 2)
    self.assertEqual(loader.call_args.kwargs, {})
    self.assertEqual(loader.call_count, 1)
    finish.assert_called_once()
    original = read(self.root / 'frozen.json')
    for problem in ('roles', 'membership'):
      record = copy.deepcopy(original)
      if problem == 'roles':
        record['manifest']['entries']['shd'].clear()
      else:
        record['manifest']['memberships']['shd']['development']['identities'][0] += '-changed'
      record['fingerprint'] = digest(record['manifest'])
      write(self.root / 'frozen.json', record)
      with patch('src.data.datasets.load', return_value=self.data), patch('src.workflow.groups', return_value=self.declarations):
        with patch('src.studies.selection.develop', side_effect=AssertionError('Development rerun')):
          with self.assertRaisesRegex(ValueError, 'coverage|membership'):
            run(self.request, self.root.name, resume=True)
    write(self.root / 'frozen.json', original)
