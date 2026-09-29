import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

def checksum(path: Path) -> str:
  """Hash one archive or build output without loading it into memory.

  :param path: File to inspect.
  :return: SHA256 digest.
  """
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()

def save(path: Path, record: dict) -> None:
  """Atomically preserve completed build operations.

  :param path: Build record destination.
  :param record: JSON-compatible evidence.
  """
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(record, indent=2) + "\n")
  temporary.replace(path)

def download(entry: dict, directory: Path) -> Path:
  """Download a pinned upstream archive and validate its content.

  :param entry: Name, URL and required SHA256 from the build manifest.
  :param directory: Download cache.
  :return: Verified archive path.
  """
  target = directory / entry["name"]
  if not target.exists():
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(entry["url"], timeout=120) as response, temporary.open("wb") as stream:
      shutil.copyfileobj(response, stream)
    if checksum(temporary) != entry["sha256"]:
      temporary.unlink()
      raise ValueError(f"Downloaded checksum differs: {target.name}")
    temporary.replace(target)
  if checksum(target) != entry["sha256"]:
    raise ValueError(f"Cached checksum differs: {target.name}")
  print(f"Verified {target.name}", flush=True)
  return target

def execute(command: list[str], directory: Path, environment: dict, record: dict, destination: Path) -> None:
  """Run a build command and record success or failure before returning.

  :param command: Explicit argument vector.
  :param directory: Working directory.
  :param environment: Compiler and build environment.
  :param record: Mutable build evidence.
  :param destination: Atomic evidence file.
  """
  print(json.dumps(command), flush=True)
  started = time.monotonic()
  completed = subprocess.run(command, cwd=directory, env=environment)
  record["commands"].append({
    "command": command,
    "cwd": str(directory),
    "seconds": time.monotonic() - started,
    "returncode": completed.returncode
  })
  save(destination, record)
  completed.check_returncode()

def build(workspace: Path, architecture: str, jobs: int) -> dict:
  """Rebuild unmodified Aer using the recorded CUDA and native dependencies.

  :param workspace: Separate native, wheel and temporary build namespace.
  :param architecture: CUDA compute capability, such as ``8.6``.
  :param jobs: Maximum parallel compiler jobs.
  :return: Build record including resulting wheel and runtime identities.
  """
  if sys.version_info[:2] != (3, 11) or platform.system() != "Linux" or platform.machine() != "x86_64":
    raise ValueError("This recipe targets Linux x86_64 with Python 3.11")
  pieces = architecture.split(".")
  if len(pieces) != 2 or not all(part.isdigit() for part in pieces) or jobs < 1:
    raise ValueError("Use a major.minor CUDA architecture and a positive job count")
  workspace = workspace.resolve()
  wheels = workspace / "wheels"
  if wheels.exists() and any(wheels.iterdir()):
    raise FileExistsError("Preserve the existing wheel; select an empty --workspace for another build")
  for tool in ("cmake", "ninja", "gcc", "g++", "dpkg-deb"):
    if not shutil.which(tool):
      raise RuntimeError(f"Missing build prerequisite: {tool}; see environment/BUILD.md")
  manifest = json.loads((Path(__file__).resolve().parents[2] / "environment/aer-build.json").read_text())
  working = workspace / "build"
  archives, sources, cuda, native = working / "archives", working / "sources", working / "cuda", workspace / "native"
  for directory in (archives, sources, cuda, native, wheels):
    directory.mkdir(parents=True, exist_ok=True)
  record = {
    "status": "building",
    "manifest_sha256": checksum(Path(__file__).resolve().parents[2] / "environment/aer-build.json"),
    "python": sys.version,
    "platform": platform.platform(),
    "architecture": architecture,
    "commands": [],
    "downloads": manifest["downloads"]
  }
  destination = working / "build.json"
  save(destination, record)
  environment = dict(os.environ)
  environment.update({
    "CC": shutil.which("gcc"),
    "CXX": shutil.which("g++"),
    "CUDACXX": str(cuda / "bin/nvcc"),
    "CMAKE_BUILD_PARALLEL_LEVEL": str(jobs),
    "QISKIT_ADD_CUDA_REQUIREMENTS": "false"
  })
  for entry in manifest["downloads"]:
    archive = download(entry, archives)
    if archive.suffix == ".deb":
      execute(["dpkg-deb", "-x", str(archive), str(native)], working, environment, record, destination)
    else:
      with tarfile.open(archive) as stream:
        stream.extractall(sources, filter="data")
      if archive.name.startswith("cuda_"):
        component = sources / archive.name.removesuffix(".tar.xz")
        shutil.copytree(component, cuda, dirs_exist_ok=True, symlinks=True)
  if not (cuda / "lib64").exists():
    (cuda / "lib64").symlink_to("lib")
  for source, flags in (
    ("json-3.1.1", ["-DJSON_BuildTests=OFF"]),
    ("spdlog-1.9.2", ["-DSPDLOG_BUILD_EXAMPLE=OFF", "-DSPDLOG_BUILD_TESTS=OFF"])
  ):
    build_directory = working / "native-build" / source
    execute([
      "cmake", "-S", str(sources / source), "-B", str(build_directory), "-G", "Ninja",
      f"-DCMAKE_INSTALL_PREFIX={native}", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON", "-DCMAKE_BUILD_TYPE=Release"
    ] + flags, working, environment, record, destination)
    execute(["cmake", "--build", str(build_directory), "--target", "install", "-j", str(jobs)], working, environment, record, destination)
  configuration = native / "lib/cmake/spdlog/spdlogConfig.cmake"
  alias = "\nif(NOT TARGET spdlog)\n  add_library(spdlog ALIAS spdlog::spdlog)\nendif()\n"
  if alias not in configuration.read_text():
    configuration.write_text(configuration.read_text() + alias)
  blas = native / "usr/lib/x86_64-linux-gnu/openblas-pthread"
  runtime_paths = ";".join([str(blas), str(blas.parent), str(cuda / "lib64")])
  aer = sources / "qiskit_aer-0.17.2"
  execute([
    sys.executable, "setup.py", "bdist_wheel", "--", "-DAER_THRUST_BACKEND=CUDA", "-DDISABLE_CONAN=ON", "-DAER_ENABLE_CUQUANTUM=OFF",
    f"-DAER_CUDA_ARCH={architecture}", f"-DCMAKE_CUDA_ARCHITECTURES={''.join(pieces)}", f"-DCMAKE_CUDA_COMPILER={cuda / 'bin/nvcc'}",
    f"-DCMAKE_CUDA_HOST_COMPILER={shutil.which('g++')}", f"-DCUDA_TOOLKIT_ROOT_DIR={cuda}", f"-DCMAKE_PREFIX_PATH={native}",
    f"-DAER_BLAS_LIB_PATH={blas}", "-DBLA_VENDOR=OpenBLAS", "-UBLAS_openblas_WORKS", f"-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{blas.parent}",
    f"-DCMAKE_INSTALL_RPATH={runtime_paths}", f"-DCMAKE_BUILD_RPATH={runtime_paths}", "-DCMAKE_MODULE_LINKER_FLAGS=-Wl,--disable-new-dtags", "-DCMAKE_BUILD_TYPE=Release"
  ], aer, environment, record, destination)
  generated = list((aer / "dist").glob("*.whl"))
  if len(generated) != 1:
    raise RuntimeError("Expected exactly one Aer wheel")
  wheel = wheels / generated[0].name
  shutil.copy2(generated[0], wheel)
  record.update({
    "status": "completed",
    "wheel": {"path": str(wheel), "sha256": checksum(wheel)},
    "native": {str(path.relative_to(native)): checksum(path) for path in sorted(native.rglob("*.so*")) if path.is_file() and not path.is_symlink()}
  })
  save(destination, record)
  save(workspace / "build-result.json", record)
  return record

def main() -> None:
  """Build Aer explicitly without installing or altering an existing environment."""
  parser = argparse.ArgumentParser(description="Rebuild the recorded Aer CUDA wheel without installing it")
  parser.add_argument("--workspace", type=Path, default=Path("environment/local"))
  parser.add_argument("--architecture", default="8.6")
  parser.add_argument("--jobs", type=int, default=2)
  args = parser.parse_args()
  print(json.dumps(build(args.workspace, args.architecture, args.jobs), indent=2))

if __name__ == "__main__":
  main()
