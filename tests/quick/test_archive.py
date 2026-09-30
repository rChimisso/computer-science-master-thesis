import copy
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.runtime.archive import create, verify, verified_files
from src.runtime.records import digest, write

class ArchiveTests(unittest.TestCase):
  """Protect evidence archive integrity and source preservation without model execution.

  :ivar temporary: Test-owned directory removed after every case.
  :ivar root: Temporary archive and source directory.
  """

  def setUp(self):
    """Create isolated archive storage and register its cleanup."""
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)

  def archive(self, members: list, inventory: dict | None = None) -> Path:
    """Write a small archive with independently controllable bytes and inventory.

    :param members: Member names, payload bytes and tar member types.
    :param inventory: Optional authored body for corruption scenarios.
    :return: Replaced temporary archive path.
    """
    body = inventory or {
      "format": "thesis-evidence",
      "version": 1,
      "files": {name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)} for name, payload, kind in members}
    }
    record = body | {"fingerprint": digest(body)}
    path = self.root / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
      payload = json.dumps(record).encode()
      header = tarfile.TarInfo("bundle.json")
      header.size = len(payload)
      stream.addfile(header, io.BytesIO(payload))
      for name, payload, kind in members:
        header = tarfile.TarInfo(name)
        header.type = kind
        header.size = len(payload)
        stream.addfile(header, io.BytesIO(payload))
    return path

  def test_archive_integrity_and_member_boundaries(self):
    """Reject missing, changed, duplicate, unsafe and nonregular archived files."""
    members = [("run/manifest.json", b"original", tarfile.REGTYPE)]
    path = self.archive(members)
    checked = verify(path)
    self.assertEqual(checked["status"], "verified")
    inventory = {key: value for key, value in checked["inventory"].items() if key != "fingerprint"}
    for damage in ("missing", "changed", "duplicate", "traversal", "absolute", "link", "undeclared", "version"):
      body = copy.deepcopy(inventory)
      altered = list(members)
      if damage == "missing":
        altered = []
      elif damage == "changed":
        altered = [("run/manifest.json", b"modified", tarfile.REGTYPE)]
      elif damage == "duplicate":
        altered += members
      elif damage in ("traversal", "absolute"):
        name = "../escape" if damage == "traversal" else "/escape"
        body["files"] = {name: body["files"]["run/manifest.json"]}
        altered = [(name, b"original", tarfile.REGTYPE)]
      elif damage == "link":
        altered = [("run/manifest.json", b"original", tarfile.SYMTYPE)]
      elif damage == "undeclared":
        altered += [("extra", b"extra", tarfile.REGTYPE)]
      else:
        body["version"] = 99
      with self.subTest(damage=damage), self.assertRaises(ValueError):
        verify(self.archive(altered, body))

  def test_archive_sources_and_active_runs(self):
    """Refuse live or incomplete runs and reject changed or linked source artifacts."""
    run = self.root / "run"
    target = self.root / "new.tar.gz"
    with patch("src.runtime.monitor.status", return_value={"alive": True}), self.assertRaisesRegex(ValueError, "Pause"):
      create(run, target)
    self.assertFalse(target.exists())
    write(run / "manifest.json", {})
    write(run / "progress.json", {"status": "failed"})
    with self.assertRaisesRegex(ValueError, "completed runs"):
      create(run, target)
    source = run / "source"
    source.write_bytes(b"original")
    expected = {"source": hashlib.sha256(b"original").hexdigest()}
    self.assertEqual(verified_files(run, expected), {"source": source})
    source.write_bytes(b"modified")
    with self.assertRaisesRegex(ValueError, "checksum"):
      verified_files(run, expected)
    source.unlink()
    source.symlink_to(run / "manifest.json")
    with self.assertRaisesRegex(ValueError, "escapes"):
      verified_files(run, expected)
