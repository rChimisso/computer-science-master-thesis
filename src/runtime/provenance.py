import ast
import datetime
import hashlib
import sys
import tarfile
from pathlib import Path

from src.runtime.records import checksum, digest, read, sources, write

def validate_source(original: dict, root: Path) -> list:
  """Reject scientific changes while allowing identified presentation and execution changes.

  :param original: Complete producer source checksums retained in the frozen manifest.
  :param root: Run namespace containing its verified producer snapshot.
  :return: Changed paths, including permitted execution and presentation edits.
  :raises ValueError: If an unclassified or scientific implementation differs.
  """
  current = sources()
  changed = sorted(name for name in original.keys() | current.keys() if original.get(name) != current.get(name))
  operational = {
    'src/main.py',
    'src/reporting/catalog.py',
    'src/reporting/figures.py',
    'src/reporting/products.py',
    'src/reporting/study.py',
    'src/runtime/monitor.py',
    'src/runtime/estimates.py',
    'src/runtime/forecast.py',
    'src/runtime/progress.py',
    'src/runtime/test_suites.py',
    'src/runtime/execution.py',
    'src/runtime/resources.py',
    'src/runtime/provenance.py',
    'src/runtime/reproduction.py'
  }
  omitted = {
    'src/workflow.py': {'finish'},
    'src/runtime/validation.py': {'validate'},
    'src/runtime/task_worker.py': {'worker', 'configure_threads'},
    'src/runtime/tasks.py': {'Scheduler.launch', 'Scheduler.loop'},
    'src/runtime/records.py': {'frozen_record'}
  }
  for name in changed:
    if name not in original or name not in current:
      raise ValueError('Frozen source file added or removed: ' + name)
    if name in operational:
      continue
    if name not in omitted:
      raise ValueError('Scientific source differs from the frozen implementation: ' + name)
    archive = root / 'evidence/provenance/implementation.tar.gz'
    inventory = read(archive.with_name('implementation.json'))
    if inventory['source'] != original or checksum(archive) != inventory['sha256']:
      raise ValueError('Frozen producer source snapshot is not intact')
    with tarfile.open(archive) as stream:
      member = stream.extractfile(name)
      if member is None:
        raise ValueError('Frozen producer source is missing: ' + name)
      before = member.read()
    if hashlib.sha256(before).hexdigest() != original[name]:
      raise ValueError('Frozen producer source checksum differs: ' + name)
    if protected_syntax(before.decode(), omitted[name]) != protected_syntax(Path(name).read_text(), omitted[name]):
      raise ValueError('Scientific source differs from the frozen implementation: ' + name)
  return changed

def protected_syntax(text: str, omitted: set) -> str:
  """Compare executable syntax outside explicitly separated operational functions.

  :param text: Verified source text.
  :param omitted: Qualified operational functions excluded from scientific identity.
  :return: Syntax representation retaining imports, constants and scientific functions.
  """
  tree = ast.parse(text)
  tree.body = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in omitted]
  for node in tree.body:
    if isinstance(node, ast.ClassDef):
      node.body = [child for child in node.body if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name + '.' + child.name not in omitted]
  return ast.dump(tree, include_attributes=False)

def snapshot(directory: Path, source: dict) -> None:
  """Capture the actual executing implementation without relabelling producer code.

  :param directory: Destination within execution provenance.
  :param source: Current complete source hashes, checked before archival.
  """
  if source != sources():
    raise ValueError('Cannot snapshot producer hashes from a different executing source')
  directory.mkdir(parents=True, exist_ok=True)
  archive = directory / 'implementation.tar.gz'
  inventory_path = directory / 'implementation.json'
  if archive.exists():
    inventory = read(inventory_path)
    if checksum(archive) != inventory['sha256'] or inventory['source'] != source:
      raise ValueError('Implementation snapshot changed')
    return
  files = [Path(name) for name in source]
  files += [path for name in ('configs', 'environment') for path in Path(name).glob('*') if path.is_file()]
  temporary = archive.with_name('implementation.tmp.tar.gz')
  with tarfile.open(temporary, 'w:gz') as stream:
    for path in sorted(set(files)):
      stream.add(path, arcname=str(path), recursive=False)
  temporary.replace(archive)
  write(inventory_path, {
    'source': source,
    'sha256': checksum(archive),
    'files': {str(path): checksum(path) for path in files},
    'native_binaries': 'Restore the separately checksummed environment/local wheel and native libraries'
  })

def capture(root: Path, identity: dict, deadline: float) -> None:
  """Preserve reconstructable source and record each execution or resume invocation.

  :param root: Run namespace.
  :param identity: Immutable source and environment settings.
  :param deadline: Absolute deadline, including an unbounded value.
  """
  directory = root / "evidence/provenance"
  directory.mkdir(parents=True, exist_ok=True)
  archive = directory / "implementation.tar.gz"
  inventory_path = directory / "implementation.json"
  if archive.exists():
    inventory = read(inventory_path)
    if checksum(archive) != inventory["sha256"] or inventory["source"] != identity["source"]:
      raise ValueError("Run implementation snapshot changed")
  else:
    snapshot(directory, identity['source'])
  current = sources()
  changed = validate_source(identity['source'], root)
  execution_directory = directory
  if changed:
    execution_directory = directory / 'executions' / digest(current)
    snapshot(execution_directory, current)
  path = root / "evidence/logs/invocations.json"
  record = read(path) if path.exists() else {"invocations": []}
  record["invocations"].append({
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "arguments": sys.argv,
    "executing_source": current,
    "producer_source": identity["source"],
    "changed_operational_paths": changed,
    "executing_snapshot": str(execution_directory.relative_to(root)),
    "deadline": deadline if deadline != float("inf") else None
  })
  write(path, record)
