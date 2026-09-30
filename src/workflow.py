import copy
import datetime
import importlib.metadata
import importlib.util
import platform
from contextlib import nullcontext
from pathlib import Path

from src.runtime.progress import stage, track
from src.runtime.configuration import bind, load
from src.runtime.records import checksum, digest, frozen_record, read, sources, write
from src.studies.design import groups

def resolve(args) -> dict:
  """Resolve requested studies, optional models and required dependencies.

  :param args: Parsed command arguments.
  :return: Explicit scientific and execution request.
  """
  config = load(args.config)
  from src.methodology.parameters import resolve_register
  resolve_register(config, read("configs/parameters.json"))
  for values in (args.models, args.seed or [], args.dataset or [], args.study or []):
    if len(values) != len(set(values)):
      raise ValueError("Duplicate filters are not permitted")
  studies = args.study or config["profiles"][args.profile]
  if any(study not in {
    "comparison",
    "generalization",
    "mapping",
    "observations",
    "recurrence",
    "learning-curves",
    "preprocessing"
  } for study in studies):
    raise ValueError("Unknown study")
  diagnostic_only = studies == ["preprocessing"]
  if "comparison" not in studies and not diagnostic_only:
    studies = ["comparison"] + list(studies)
  if "preprocessing" in studies and "lstm" not in args.models:
    raise ValueError("The separately requested preprocessing study requires --models lstm")
  return {
    "config": config,
    "profile": args.profile,
    "studies": studies,
    "datasets": args.dataset or (["shd"] if diagnostic_only else ["shd", "dvs"]),
    "models": args.models,
    "seeds": sorted(set(args.seed) | {42}) if args.seed else config["seeds"],
    "resources": args.resources,
    "diagnostic_only": diagnostic_only,
    "official_test": not getattr(args, "development_only", False) and not diagnostic_only
  }

def plan(request: dict) -> dict:
  """Describe the finite queue without loading data or numerical backends.

  :param request: Resolved settings.
  :return: Deduplicated screening counts and conditional work descriptions.
  """
  result = {
    "profile": request["profile"],
    "studies": request["studies"],
    "models": request["models"],
    "datasets": {},
    "official_test": "Included after frozen confirmation and all final training fits" if request.get("official_test", True) else "Disabled for this development-only run"
  }
  for name in track(request["datasets"], "workflow datasets", unit="datasets"):
    if request.get("diagnostic_only"):
      result["datasets"][name] = {"diagnostic": "Three SHD representations on the original previously used development split", "fits": 3 * len(request["seeds"]) if name == "shd" else 0}
      continue
    declarations = groups(name, request["studies"], request["models"], request["config"])
    unique = {digest(spec): spec for row in declarations for spec in row["candidates"]}
    result["datasets"][name] = {
      "groups": len(declarations),
      "screening_configurations": len(unique),
      "screening_cells": 4 * len(unique),
      "replication_cells_upper_bound": 4 * max(0, len(request["seeds"]) - 1) * 2 * len(declarations),
      "queue": declarations
    }
  result["conditional_work"] = {
    "mapping": "Projection winner and anchor paired replication; independent random-projection robustness",
    "learning-curves": "SHD selected six-qubit pairs, projection controls and enabled practical models on nested subsets",
    "resources": request["resources"]
  }
  result['dependencies'] = [
    'Each dataset: preparation -> adapters -> features -> readouts -> selection -> replication',
    'Completed selections release mapping, learning curves and independent preprocessing work',
    'Complete development -> freeze -> confirmation -> isolated resources -> frozen official refits and evaluation unless development-only -> reports',
    'Official action: frozen training fits all complete -> official data loading -> inference'
  ]
  return result

def environment() -> dict:
  """Record installed runtime versions without initializing GPU backends.

  :return: Portable environment identity.
  """
  result = {"python": platform.python_version(), "platform": platform.platform()}
  for name in ("numpy", "scipy", "scikit-learn", "torch", "qiskit", "qiskit-aer", "tonic", "h5py"):
    result[name] = importlib.metadata.version(name)
  package = importlib.util.find_spec("qiskit_aer")
  root = Path(package.origin).parent
  result["aer_native"] = {str(path.relative_to(root)): checksum(path) for path in root.rglob("*.so")}
  native = Path("environment/local/native")
  result["native_runtime"] = {str(path.relative_to(native)): checksum(path) for path in native.rglob("*.so.*") if path.is_file() and not path.is_symlink()}
  return result

def finish(root: Path, progress: dict) -> dict:
  """Publish validated reports and apply retention after recording execution status.

  :param root: Run namespace.
  :param progress: Current authoritative execution state.
  :return: Report and cleanup outcomes.
  """
  from src.reporting.study import report
  from src.runtime.storage import cleanup, seal
  from src.runtime.tasks import current
  scheduler = current({'paths': {'run': str(root)}})
  context = scheduler.coordinator_task('report', {'run': str(root)}) if scheduler else nullcontext()
  with context:
    if progress.get("status") in ("completed", "diagnostic_completed"):
      progress.pop("error", None)
    write(root / "progress.json", progress)
    seal(root)
    generated = report(root)
    return {"report": generated, "cleanup": cleanup(root, generated)}

def evaluate_run(path: str | Path, official: bool, deadline: float = float("inf"), run_id: str | None = None) -> dict:
  """Coordinate a frozen evaluation action, its progress, reporting and retention.

  :param path: Authored frozen manifest.
  :param official: Explicit official-test action, otherwise frozen confirmation.
  :param deadline: Absolute stop time.
  :param run_id: Optional independent frozen-confirmation namespace.
  :return: Evaluation result and validated report/cleanup outcomes.
  """
  if official and run_id is not None:
    raise ValueError("Official evaluation uses its existing frozen run")
  from src.runtime.reproduction import prepare
  path = Path(path) if official else prepare(path, run_id)
  frozen_record(path)
  root = path.parent
  progress_path = root / "progress.json"
  progress = read(progress_path) if progress_path.exists() else {}
  progress.update({"status": "running", "phase": "official" if official else "confirmation"})
  write(progress_path, progress)
  try:
    result = evaluate_frozen(path, official, deadline)
    progress["status"] = "completed"
    progress["official_test_loaded"] = official
    return result | finish(root, progress)
  except BaseException as error:
    saved = read(root / "progress.json") if (root / "progress.json").exists() else {}
    progress["official_test_loaded"] = bool(progress.get("official_test_loaded") or saved.get("official_test_loaded"))
    progress.update({"status": "interrupted" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "failed", "error": repr(error)})
    write(progress_path, progress)
    raise

def confirm(root: Path, frozen: dict, datasets: dict, resources: bool, deadline: float) -> None:
  """Finish frozen confirmation and optional isolated resource measurements.

  :param root: Current run namespace.
  :param frozen: Verified frozen settings.
  :param datasets: Mutable prepared-data registry, populated during evaluation if empty.
  :param resources: Include the declared practical resource measurements.
  :param deadline: Absolute cancellation deadline.
  """
  evaluate_frozen(root / "frozen.json", False, deadline, datasets)
  if resources:
    from src.runtime.tasks import current
    from src.studies.resources import measure
    local = bind(frozen["manifest"]["config"], root, deadline)
    local["_identity"] = frozen["fingerprint"]
    scheduler = current(local)
    if scheduler:
      scheduler.submit('resources', {'datasets': datasets, 'frozen': frozen, 'config': local}, {'frozen': frozen['fingerprint']}).result()
    else:
      measure(datasets, frozen, local)

def resume_frozen(root: Path, identity: dict, deadline: float) -> dict:
  """Resume a fully frozen run without repeating development orchestration.

  :param root: Existing run namespace.
  :param identity: Verified current run manifest and unchanged requested scope.
  :param deadline: Absolute cancellation deadline.
  :return: Completion or resumable failure through the ordinary reporting lifecycle.
  """
  from src.runtime.storage import compact
  from src.runtime.tasks import current
  config = bind(identity["config"], root, deadline)
  scheduler = current(config)
  progress = {
    "status": "running",
    "run": root.name,
    "phase": "confirmation",
    "official_test_loaded": read(root / "progress.json").get("official_test_loaded", False) if (root / "progress.json").exists() else False,
    "requested_seeds": identity["seeds"],
    "completed": []
  }
  try:
    context = scheduler.coordinator_task('restore-frozen', {'source': digest(identity)}) if scheduler else nullcontext()
    with context:
      frozen = frozen_record(root / "frozen.json")
      body = frozen["manifest"]
      for field in ("source", "environment", "config", "parameter_register"):
        if body[field] != identity[field]:
          raise ValueError("Frozen resume differs from its run manifest: " + field)
      expected = set(identity["datasets"])
      for field in ("entries", "decisions", "memberships", "data_fingerprints", "raw_fingerprints"):
        if set(body[field]) != expected:
          raise ValueError("Frozen dataset coverage differs from the requested run")
      for name in identity["datasets"]:
        declarations = identity["declarations"][name]
        decision = body["decisions"][name]
        if set(body["entries"][name]) != {row["id"] for row in declarations if row["final"]}:
          raise ValueError("Frozen final-model coverage differs from the declared comparison")
        if any(decision["groups"].get(row["id"], {}).get("declaration") != row for row in declarations):
          raise ValueError("Frozen selection declarations differ from the run")
        for study, key in (("mapping", "mapping"), ("learning-curves", "learning_curves")):
          if study in identity["studies"] and decision.get(key, {}).get("status") not in ("completed", "not_applicable"):
            raise ValueError("Frozen supporting study is incomplete: " + study)
      if "preprocessing" in identity["studies"] and "shd" in expected:
        preprocessing = read(root / "evidence/metrics/preprocessing.json")
        if preprocessing.get("status") != "completed":
          raise ValueError("Frozen preprocessing study is incomplete")
    progress["completed"] = list(identity["datasets"])
    write(root / "progress.json", progress)
    confirm(root, frozen, {}, identity["resources"], deadline)
    if identity.get("official_test", True):
      progress["phase"] = "official"
      evaluate_frozen(root / "frozen.json", True, deadline)
      progress["official_test_loaded"] = True
    progress["status"] = "completed"
    compact(root, config)
    finish(root, progress)
  except BaseException as error:
    saved = read(root / "progress.json") if (root / "progress.json").exists() else {}
    progress["official_test_loaded"] = bool(progress.get("official_test_loaded") or saved.get("official_test_loaded"))
    progress.update({"status": "interrupted" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "failed", "error": repr(error)})
    write(root / "progress.json", progress)
    raise
  write(root / "progress.json", progress)
  return progress

@stage("workflow.run")
def run(request: dict, run_id: str | None = None, resume: bool = False, deadline: float = float("inf")) -> dict:
  """Execute selection and confirmation, then frozen official evaluation by default.

  :param request: Fully resolved study request.
  :param run_id: Stable output identity or an automatically assigned identifier.
  :param resume: Require compatible existing work when the run already exists.
  :param deadline: Absolute stopping time.
  :return: Run completion or resumable interruption record.
  """
  from src.data.datasets import load as load_data
  from src.methodology.parameters import resolve_register
  from src.runtime.storage import compact, snapshot_dataset
  from src.studies.selection import develop
  from src.studies.supporting import learning_curves, mappings
  register = resolve_register(request["config"], read("configs/parameters.json"))
  run_id = run_id or datetime.datetime.now(datetime.timezone.utc).strftime("study-%Y%m%dT%H%M%S%fZ")
  if run_id in (".", "..") or Path(run_id).name != run_id:
    raise ValueError("Run identifier must be a single directory name")
  root = Path(request["config"]["paths"]["runs"]) / run_id
  identity = request | {
    "evidence_version": 1,
    "declarations": {name: [] if request.get("diagnostic_only") else groups(name, request["studies"], request["models"], request["config"]) for name in request["datasets"]},
    "seeds": request["config"]["seeds"],
    "source": sources(),
    "environment": environment(),
    "parameter_register": register
  }
  manifest_path = root / "manifest.json"
  if manifest_path.exists():
    if not resume or read(manifest_path) != identity:
      raise ValueError("Existing run requires --resume and unchanged source/settings")
  else:
    write(manifest_path, identity)
  from src.runtime.provenance import capture
  capture(root, identity, deadline)
  if resume and (root / "frozen.json").exists():
    return resume_frozen(root, identity, deadline)
  config = bind(request["config"], root, deadline)
  config["_identity"] = digest(identity)
  progress = {
    "status": "running",
    "run": run_id,
    "official_test_loaded": False,
    "requested_seeds": request["seeds"],
    "completed": []
  }
  write(root / "progress.json", progress)
  evidence = Path(config["paths"]["results"])
  selection_path = evidence / "metrics/selections.json"
  selections = read(selection_path) if resume and selection_path.exists() else {}
  data_sources = {}
  try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.runtime.tasks import current
    scheduler = current(config)

    def dataset_work(name: str) -> tuple:
      """Advance one dataset through its declared selection dependencies.

      :param name: Dataset identifier.
      :return: Prepared data and completed development decisions.
      """
      if scheduler is not None:
        scheduler.conditional(name, 'Selection-dependent replication, supporting studies, confirmation and reporting')
        data = scheduler.submit('data', {'name': name, 'config': config | {'_study': ['preparation']}}, {'name': name, 'settings': config['data'][name], 'source': config['_identity']}).result()
      else:
        data = load_data(name, config)
      snapshot_dataset(data, config)
      seeds = sorted(set(request['seeds']) | set(selections.get(name, {}).get('seeds', [])))
      if request.get('diagnostic_only'):
        from src.studies.preprocessing import run_preprocessing
        run_preprocessing(data, seeds, config)
        return data, None
      declarations = groups(name, request['studies'], request['models'], config)
      selected = develop(data, declarations, seeds, config)
      from src.studies.preprocessing import run_preprocessing
      operations = {}
      if 'mapping' in request['studies']:
        operations['mapping'] = lambda: mappings(data, selected, seeds, config | {'_study': ['mapping']})
      if 'learning-curves' in request['studies']:
        operations['learning_curves'] = lambda: learning_curves(data, selected, seeds, config | {'_study': ['learning-curves']})
      if 'preprocessing' in request['studies']:
        operations['preprocessing'] = lambda: run_preprocessing(data, seeds, config | {'_study': ['preprocessing']})
      with ThreadPoolExecutor(max_workers=max(1, len(operations)) if scheduler else 1) as supporting:
        tasks = {key: supporting.submit(function) for key, function in operations.items()}
        for key, task in tasks.items():
          value = task.result()
          if key != 'preprocessing':
            selected[key] = value
      return data, selected

    with ThreadPoolExecutor(max_workers=len(request['datasets']) if scheduler else 1) as datasets_pool:
      tasks = {datasets_pool.submit(dataset_work, name): name for name in request['datasets']}
      try:
        for task in as_completed(tasks):
          name = tasks[task]
          data, selected = task.result()
          data_sources[name] = data
          if selected is not None:
            selections[name] = selected
            write(selection_path, selections)
          progress['completed'].append(name)
          write(root / 'progress.json', progress)
          if scheduler:
            scheduler.conditional(name, None)
      except BaseException:
        if scheduler:
          (root / 'STOP').touch()
        raise
    if request.get("diagnostic_only"):
      progress["status"] = "diagnostic_completed"
    elif any(value["status"] != "completed" for value in selections.values()):
      progress["status"] = "partial"
    else:
      context = scheduler.coordinator_task('freeze', {'source': config['_identity']}) if scheduler else nullcontext()
      with context:
        frozen = freeze(root, identity, selections, data_sources)
      confirm(root, frozen, data_sources, request["resources"], deadline)
      if request.get("official_test", True):
        progress["phase"] = "official"
        write(root / "progress.json", progress)
        evaluate_frozen(root / "frozen.json", True, deadline, data_sources)
        progress["official_test_loaded"] = True
      progress["status"] = "completed"
    if progress["status"] in ("completed", "diagnostic_completed"):
      compact(root, config)
    finish(root, progress)
  except BaseException as error:
    saved = read(root / "progress.json") if (root / "progress.json").exists() else {}
    progress["official_test_loaded"] = bool(progress.get("official_test_loaded") or saved.get("official_test_loaded"))
    progress.update({"status": "interrupted" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "failed", "error": repr(error)})
    write(root / "progress.json", progress)
    raise
  write(root / "progress.json", progress)
  return progress

@stage("workflow.freeze")
def freeze(root: Path, identity: dict, selections: dict, datasets: dict) -> dict:
  """Freeze complete settings and their development decision records.

  :param root: Run directory.
  :param identity: Immutable source/settings record.
  :param selections: Complete declared development selections.
  :param datasets: Training sources with explicit memberships.
  :return: Signed final-evaluation manifest.
  """
  if any(value["status"] != "completed" or value["seeds"] != identity["config"]["seeds"] for value in selections.values()):
    raise ValueError("Incomplete seed rounds cannot be frozen")
  entries = {}
  for name, selection in selections.items():
    entries[name] = {role: copy.deepcopy(item["selected"]) for role, item in selection["groups"].items() if item["declaration"]["final"]}
  memberships = {name: {"development": data.membership(data.development), "confirmation": data.membership(data.confirmation)} for name, data in datasets.items()}
  body = {
    "source": identity["source"],
    "environment": identity["environment"],
    "config": identity["config"],
    "parameter_register": identity["parameter_register"],
    "entries": entries,
    "memberships": memberships,
    "data_fingerprints": {name: data.fingerprint for name, data in datasets.items()},
    "raw_fingerprints": {name: data.raw_fingerprint for name, data in datasets.items()},
    "decisions": selections,
    "official_benchmark_status": "previously inspected"
  }
  record = {
    "status": "frozen",
    "manifest": body,
    "fingerprint": digest(body)
  }
  path = root / "frozen.json"
  if path.exists() and read(path) != record:
    raise ValueError("Cannot replace an existing frozen protocol")
  write(path, record)
  return record

def final_fit_identity(data, entry: dict, seed: int, fingerprint: str) -> dict:
  """Share one final-fit writer across roles selecting identical numerical settings.

  :param data: Complete official-training descriptor.
  :param entry: Frozen selected candidate, including nonnumerical decision evidence.
  :param seed: Frozen initialization seed.
  :param fingerprint: Frozen protocol identity.
  :return: Scheduler identity matching the fitted numerical artifact population.
  """
  return {
    'data': data.fingerprint,
    'spec': entry['spec'],
    'penalty': entry['selection']['lambda'],
    'seed': seed,
    'frozen': fingerprint
  }

@stage("workflow.evaluate_frozen")
def evaluate_frozen(path: str | Path, official: bool, deadline: float = float("inf"), datasets: dict | None = None) -> dict:
  """Refit frozen settings on fixed memberships without any model selection.

  :param path: Explicit frozen manifest.
  :param official: Permit official-test loading for this action only.
  :param deadline: Absolute stop time.
  :param datasets: Optional already loaded training data for confirmation.
  :return: Fixed-configuration evaluation results.
  """
  from src.data.datasets import load as load_data
  from src.studies.experiments import experiment
  from src.runtime.storage import save_job, snapshot_dataset
  from src.runtime.pipelines import export
  frozen = frozen_record(path)
  body = frozen["manifest"]
  if body["environment"] != environment():
    raise ValueError("Frozen runtime versions differ")
  root = Path(path).parent
  from src.runtime.provenance import capture
  capture(root, body, deadline)
  config = bind(body["config"], root, deadline)
  config["_identity"] = frozen["fingerprint"]
  phase = "official" if official else "confirmation"
  result = {
    "status": "running",
    "phase": phase,
    "frozen": frozen["fingerprint"],
    "benchmark_status": body["official_benchmark_status"],
    "jobs": []
  }
  evidence = Path(config["paths"]["results"])
  destination = evidence / "metrics" / f"{phase}.json"
  packages_path = evidence / "pipelines.json"
  packages = read(packages_path) if packages_path.exists() else {}
  from src.runtime.tasks import current
  scheduler = current(config)
  fitted = {}
  training_sources = {}
  pending = []
  for name, entries in body['entries'].items():
    if datasets is not None and name in datasets:
      data = datasets[name]
    elif scheduler:
      data = scheduler.submit('data', {'name': name, 'config': config | {'_study': ['preparation']}}, {'name': name, 'settings': config['data'][name], 'source': config['_identity']}).result()
    else:
      data = load_data(name, config)
    if datasets is not None:
      datasets[name] = data
    training_sources[name] = data
    snapshot_dataset(data, config)
    expected = body['memberships'][name]
    if name in body.get('data_fingerprints', {}) and data.fingerprint != body['data_fingerprints'][name]:
      raise ValueError('Frozen dataset contents changed')
    if name in body.get('raw_fingerprints', {}) and data.raw_fingerprint != body['raw_fingerprints'][name]:
      raise ValueError('Frozen raw dataset contents changed')
    if data.membership(data.development) != expected['development'] or data.membership(data.confirmation) != expected['confirmation']:
      raise ValueError('Frozen dataset membership changed')
    fold = {'fold': phase, 'train': data.development.tolist(), 'validation': data.confirmation.tolist()}
    for seed in config['seeds']:
      for role, entry in entries.items():
        local = config | {'_study': [role.split(':', 1)[0]]}
        payload = {'data': data, 'entry': entry, 'spec': entry['spec'], 'seed': seed, 'phase': phase, 'config': local}
        if official:
          from src.studies.final import fit
          future = scheduler.submit('final-fit', payload, final_fit_identity(data, entry, seed, frozen['fingerprint'])) if scheduler else None
          value = None if future else fit(data, entry, seed, config)
        else:
          penalty = entry['selection']['lambda']
          penalties = [penalty] if penalty is not None else None
          future = scheduler.experiment(data, fold, entry['spec'], seed, local, penalties, phase) if scheduler else None
          value = None if future else experiment(data, fold, entry['spec'], seed, config, penalties, phase)
        pending.append((name, role, entry, seed, fold, future, value))
  exports = []
  for name, role, entry, seed, fold, future, value in pending:
    job = future.result() if future else value
    data = training_sources[name]
    if official:
      fitted[f'{name}:{role}:{seed}'] = job
      continue
    package_key = f'{phase}:{name}:{role}:{seed}'
    if package_key not in packages:
      payload = {'data': data, 'train': fold['train'], 'spec': entry['spec'], 'seed': seed, 'phase': phase, 'job': job['path'], 'config': config}
      if scheduler:
        task = scheduler.submit('export', payload, {'job': job['path'], 'phase': phase})
        exports.append((package_key, task))
      else:
        import numpy as np
        packages[package_key] = export(data, np.asarray(fold['train']), entry['spec'], seed, phase, job, config)
    result['jobs'].append({'dataset': name, 'role': role, 'seed': seed, 'path': job['path']})
    write(destination, result)
  for key, future in exports:
    packages[key] = future.result()
    write(packages_path, packages)
  write(packages_path, packages)
  if official:
    save_job(evidence / 'models/official-fit.json', fitted, config)
    from src.studies.final import evaluate
    pending = []
    progress_path = root / 'progress.json'
    progress = read(progress_path) if progress_path.exists() else {}
    progress.update({'official_test_loaded': True, 'phase': 'official'})
    write(progress_path, progress)
    for name, entries in body['entries'].items():
      held = load_data(name, config, 'test', official=True)
      snapshot_dataset(held, config, 'test')
      for seed in config['seeds']:
        for role, entry in entries.items():
          fit = fitted[f'{name}:{role}:{seed}']
          payload = {'data': training_sources[name], 'held': held, 'entry': entry, 'spec': entry['spec'], 'fitted': fit, 'seed': seed, 'phase': phase, 'config': config | {'_study': [role.split(':', 1)[0]]}}
          task = scheduler.submit('final-evaluate', payload, {'fit': digest(fit), 'held': held.fingerprint, 'seed': seed}) if scheduler else None
          value = None if task else evaluate(training_sources[name], held, entry, fit, seed, config)
          pending.append((name, role, seed, task, value))
    for name, role, seed, task, value in pending:
      job = task.result() if task else value
      packages[f'{phase}:{name}:{role}:{seed}'] = fitted[f'{name}:{role}:{seed}']['pipeline']
      write(packages_path, packages)
      result['jobs'].append({'dataset': name, 'role': role, 'seed': seed, 'path': job['path']})
      write(destination, result)
  result["status"] = "completed"
  write(destination, result)
  return result
