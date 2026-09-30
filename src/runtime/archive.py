import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from src.runtime.records import checksum, digest, read, sources

def relative_name(name: str) -> str:
  """Validate one canonical archive path without filesystem access.

  :param name: Relative POSIX member name.
  :return: Unchanged safe member name.
  :raises ValueError: If a path is absolute, ambiguous or traverses a parent.
  """
  path = PurePosixPath(name)
  if not name or path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name or name == ".":
    raise ValueError("Unsafe archive path: " + name)
  return name

def verified_files(root: Path, inventory: dict) -> dict:
  """Resolve an existing checksummed inventory without following symbolic links.

  :param root: Artifact directory boundary.
  :param inventory: Relative filenames and SHA-256 digests.
  :return: Verified relative filenames and source paths.
  """
  result = {}
  for name, expected in inventory.items():
    relative_name(name)
    path = root / name
    if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.resolve().is_relative_to(root.resolve()):
      raise ValueError("Artifact path escapes its directory: " + name)
    if not path.is_file() or checksum(path) != expected:
      raise ValueError("Archive source checksum mismatch: " + str(path))
    result[name] = path
  return result

def verify(archive: Path) -> dict:
  """Verify a complete archive without extraction, datasets or numerical backends.

  :param archive: Evidence archive with its versioned inventory as the first member.
  :return: Validated inventory, content identity and archive checksum.
  :raises ValueError: If coverage, member type, path, size or checksum is inconsistent.
  """
  with tarfile.open(archive, "r|gz") as stream:
    first = stream.next()
    if first is None or first.name != "bundle.json" or not first.isfile() or first.size > 64 * 1024 ** 2:
      raise ValueError("Archive requires a bounded leading bundle.json inventory")
    record = json.load(stream.extractfile(first))
    body = {key: value for key, value in record.items() if key != "fingerprint"}
    if record.get("format") != "thesis-evidence" or record.get("version") != 1 or record.get("fingerprint") != digest(body):
      raise ValueError("Unsupported or damaged evidence archive inventory")
    expected = record.get("files")
    if not isinstance(expected, dict) or not expected:
      raise ValueError("Evidence archive inventory is empty")
    for name, entry in expected.items():
      relative_name(name)
      if name == "bundle.json" or not isinstance(entry, dict) or set(entry) != {"sha256", "bytes"}:
        raise ValueError("Invalid archive file record")
      if type(entry["bytes"]) is not int or entry["bytes"] < 0:
        raise ValueError("Invalid archive member size")
      if not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64 or any(value not in "0123456789abcdef" for value in entry["sha256"]):
        raise ValueError("Invalid archive member checksum")
    observed = set()
    for member in stream:
      if member is first:
        continue
      name = relative_name(member.name)
      if not member.isfile() or name not in expected or name in observed:
        raise ValueError("Duplicate, undeclared or nonregular archive member: " + name)
      if member.size != expected[name]["bytes"]:
        raise ValueError("Archive member size mismatch: " + name)
      hashed = hashlib.sha256()
      with stream.extractfile(member) as incoming:
        for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
          hashed.update(block)
      if hashed.hexdigest() != expected[name]["sha256"]:
        raise ValueError("Archive member checksum mismatch: " + name)
      observed.add(name)
    if observed != set(expected):
      raise ValueError("Archive is missing inventoried files")
  return {"status": "verified", "archive": str(archive), "sha256": checksum(archive), "inventory": record}

def create(run: Path, destination: Path) -> dict:
  """Archive sealed evidence, reports, source snapshots and retained inference packages.

  Original evidence bytes are preserved. A separate portable pipeline index uses paths relative to the extracted archive, leaving frozen historical references intact.

  :param run: Completed, inactive run with sealed evidence and published reports.
  :param destination: New archive filename; an existing file is never replaced.
  :return: Verified archive identity, pipeline count and scientific coverage status.
  """
  from src.runtime.monitor import status
  run, destination = Path(run), Path(destination)
  if status(run)["alive"]:
    raise ValueError("Pause the run before archiving its evidence")
  if destination.exists() or destination.resolve().is_relative_to(run.resolve()):
    raise ValueError("Archive destination must be new and outside its source run")
  manifest = read(run / "manifest.json")
  progress = read(run / "progress.json")
  if progress.get("status") not in ("completed", "diagnostic_completed"):
    raise ValueError("Only successfully completed runs can be archived")
  evidence = run / "evidence"
  seal = read(evidence / "integrity.json")
  sealed = verified_files(evidence, seal["artifacts"])
  inventory = read(evidence / "provenance/implementation.json")
  producer = evidence / "provenance/implementation.tar.gz"
  if inventory["source"] != manifest["source"] or checksum(producer) != inventory["sha256"]:
    raise ValueError("Historical producer source snapshot changed")
  report = read(run / "reports/overview/metadata/report.json")
  if report.get("rendering_status") != "completed" or report.get("development_status") != "completed":
    raise ValueError("Archive requires complete development evidence and a published report")
  products = read(run / "reports/products.json")
  reports = verified_files(run / "reports", products["products"])
  for name, expected in report["sources"].items():
    parts = Path(name).parts
    if "evidence" in parts:
      position = len(parts) - 1 - parts[::-1].index("evidence")
      relative = "evidence/" + "/".join(parts[position + 1:])
    elif Path(name).name in ("manifest.json", "frozen.json"):
      relative = Path(name).name
    else:
      raise ValueError("Unrecognized report dependency: " + name)
    verified_files(run, {relative: expected})
  files = {"run/evidence/" + name: path for name, path in sealed.items()}
  files.update({"run/reports/" + name: path for name, path in reports.items()})
  for name in ("manifest.json", "progress.json", "evidence/integrity.json", "evidence/cleanup.json", "reports/products.json", "reports/index.md", "frozen.json"):
    path = run / name
    if path.is_file():
      files["run/" + name] = path
  if (run / "frozen.json").exists():
    frozen = read(run / "frozen.json")
    if frozen.get("status") != "frozen" or frozen.get("fingerprint") != digest(frozen.get("manifest")):
      raise ValueError("Frozen archive evidence changed")
  packages_path = evidence / "pipelines.json"
  packages = read(packages_path) if packages_path.exists() else {}
  for phase in ("confirmation", "official"):
    path = evidence / "metrics" / (phase + ".json")
    if path.exists():
      for job in read(path)["jobs"]:
        key = f"{phase}:{job['dataset']}:{job['role']}:{job['seed']}"
        if key not in packages:
          raise ValueError("Archive is missing a retained inference package: " + key)
  portable = {}
  for key, package in packages.items():
    identifier = relative_name(package["id"])
    if "/" in identifier or set(package["artifacts"]) != {"pipeline.json", "weights.npz"}:
      raise ValueError("Invalid inference package identity or artifacts")
    for name, path in verified_files(Path(package["directory"]), package["artifacts"]).items():
      target = "models/" + identifier + "/" + name
      if target in files and checksum(files[target]) != checksum(path):
        raise ValueError("Conflicting inference package identities")
      files[target] = path
    portable[key] = package | {"directory": "models/" + identifier}
  for directory in ("src", "configs", "environment"):
    paths = Path(directory).rglob("*.py") if directory == "src" else Path(directory).glob("*")
    for path in sorted(paths):
      if path.is_file() and not path.is_symlink():
        files["reader/" + path.as_posix()] = path
  files["reader/README.md"] = Path("README.md")
  destination.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix=".evidence-archive-", dir=destination.parent) as temporary:
    work = Path(temporary)
    pipeline_index = work / "pipelines.json"
    pipeline_index.write_text(json.dumps(portable, indent=2, sort_keys=True, allow_nan=False) + "\n")
    files["pipelines.json"] = pipeline_index
    record = {
      "format": "thesis-evidence",
      "version": 1,
      "run": run.name,
      "scientific_status": {key: report[key] for key in ("status", "development_status", "official_status")},
      "producer_source": manifest["source"],
      "reader_source": sources(),
      "pipelines": len(portable),
      "paths": {"run": "run", "reader": "reader", "pipelines": "pipelines.json"},
      "scope": "Sealed evidence, reports, original producer source and retained inference packages; raw datasets, shared feature caches and native binaries require separate restoration.",
      "replay": (
        "Regenerate reports from the reader directory with python -m src report --run ../run. "
        "Portable pipeline directories in pipelines.json are relative to the archive root. "
        "Historical numerical reproduction uses the preserved producer source and recorded environment."
      ),
      "files": {name: {"sha256": checksum(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}
    }
    record["fingerprint"] = digest(record)
    encoded = json.dumps(record, indent=2, sort_keys=True, allow_nan=False).encode()
    archive = work / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as stream:
      header = tarfile.TarInfo("bundle.json")
      header.size = len(encoded)
      stream.addfile(header, io.BytesIO(encoded))
      for name, path in sorted(files.items()):
        relative_name(name)
        if path.is_symlink():
          raise ValueError("Cannot archive a symbolic link")
        stream.add(path, arcname=name, recursive=False)
    checked = verify(archive)
    os.link(archive, destination)
  return {"status": "verified", "archive": str(destination), "sha256": checked["sha256"], "fingerprint": record["fingerprint"], "files": len(files), "pipelines": len(portable), "scientific_status": record["scientific_status"]}
