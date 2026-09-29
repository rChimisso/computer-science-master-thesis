import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from src.runtime.progress import Progress, stage, track
from src.runtime.records import stop

def verify_md5(path: Path, expected: str) -> None:
  """Check the archive identity published in the installed dataset metadata.

  :param path: Archive path.
  :param expected: Expected MD5.
  """
  result = hashlib.md5()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      result.update(block)
  if result.hexdigest() != expected:
    raise ValueError(f"Dataset archive checksum mismatch: {path}")

@stage("data.acquisition.download")
def download(url: str, destination: Path, expected: str, config: dict) -> None:
  """Download or resume an archive, validating it before publication.

  :param url: Dataset provider URL.
  :param destination: Final archive path.
  :param expected: Published MD5 checksum.
  :param config: Cancellation settings.
  """
  if destination.exists():
    verify_md5(destination, expected)
    return
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_suffix(destination.suffix + ".partial")
  offset = temporary.stat().st_size if temporary.exists() else 0
  request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
  with urllib.request.urlopen(request, timeout=60) as incoming:
    resumed = offset and incoming.status == 206
    if resumed and not incoming.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
      raise ValueError("Server returned an incompatible resume range")
    initial = offset if resumed else 0
    size = incoming.headers.get("Content-Length")
    total = initial + int(size) if size is not None else None
    with Progress(f"download/{destination.name}", total, initial, "bytes") as task:
      with temporary.open("ab" if resumed else "wb") as outgoing:
        while block := incoming.read(1024 * 1024):
          stop(config)
          outgoing.write(block)
          task.update(len(block))
  verify_md5(temporary, expected)
  temporary.replace(destination)

@stage("data.acquisition.safe_extract")
def safe_extract(archive: Path, root: Path, config: dict) -> None:
  """Extract regular members only, rejecting traversal and links.

  :param archive: Verified tar or zip archive.
  :param root: Destination boundary.
  :param config: Cancellation settings.
  """
  root = root.resolve()
  package = zipfile.ZipFile(archive) if zipfile.is_zipfile(archive) else tarfile.open(archive)
  with package:
    members = package.infolist() if isinstance(package, zipfile.ZipFile) else package.getmembers()
    for member in track(members, f"extract/{archive.name}", unit="files"):
      stop(config)
      zipped = isinstance(package, zipfile.ZipFile)
      name = member.filename if zipped else member.name
      target = (root / name).resolve()
      directory = member.is_dir() if zipped else member.isdir()
      regular = ((member.external_attr >> 16) & 0o170000) in (0, 0o040000, 0o100000) if zipped else member.isfile() or directory
      if not target.is_relative_to(root) or not regular:
        raise ValueError(f"Unsafe archive member: {name}")
      if directory:
        target.mkdir(parents=True, exist_ok=True)
        continue
      target.parent.mkdir(parents=True, exist_ok=True)
      incoming = package.open(member) if zipped else package.extractfile(member)
      temporary = target.with_name(target.name + ".partial")
      with incoming, temporary.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
      temporary.replace(target)

@stage("data.acquisition.acquire")
def acquire(name: str, split: str, config: dict, official: bool = False) -> Path:
  """Resolve raw data without invoking automatic loader downloads.

  :param name: Dataset name.
  :param split: Train or explicitly authorized test split.
  :param config: Storage and cancellation configuration.
  :param official: Permission supplied only after frozen-manifest validation.
  :return: HDF5 file or gesture clip directory.
  """
  if split not in ("train", "test") or split == "test" and not official:
    raise ValueError("Official-test access requires the explicit evaluation action")
  from tonic.datasets import DVSGesture, SHD
  root = Path(config["paths"]["raw"]) / ("SHD" if name == "shd" else "DVSGesture")
  if name == "shd":
    target = root / f"shd_{split}.h5"
    filename = getattr(SHD, f"{split}_zip")
    url = SHD.base_url + filename
    expected = getattr(SHD, f"{split}_md5")
  elif name == "dvs":
    target = root / ("ibmGestureTrain" if split == "train" else "ibmGestureTest")
    filename = getattr(DVSGesture, f"{split}_filename")
    url = f"https://zenodo.org/records/8060604/files/{filename}?download=1"
    expected = getattr(DVSGesture, f"{split}_md5")
  else:
    raise ValueError("Unknown dataset")
  archive = root / filename
  present = target.exists() and (name == "shd" or len(list(target.glob("user*_*/*.npy"))) == (1077 if split == "train" else 264))
  if present:
    if archive.exists():
      verify_md5(archive, expected)
    return target
  download(url, archive, expected, config)
  safe_extract(archive, root, config)
  if not target.exists():
    raise ValueError("Archive does not contain the expected dataset schema")
  return target
