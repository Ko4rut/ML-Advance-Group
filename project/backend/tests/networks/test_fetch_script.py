"""scripts/fetch_neural_checkpoint.py against a local file:// mirror (no network, no real weights)."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .support import BACKEND

SCRIPT = BACKEND / "scripts" / "fetch_neural_checkpoint.py"


class FetchScriptTests(unittest.TestCase):
    def run_script(self, base: Path, out: Path, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), "--model", "distill-v4", "--directory", str(out),
                               "--base-url", base.as_uri(), *extra], capture_output=True, text=True)

    def mirror(self, tmp: Path, payload: bytes) -> Path:
        manifest = json.loads((BACKEND / "models/neural/manifest.json").read_text())
        spec = manifest["models"]["distill-v4"]
        target = tmp / "mirror" / manifest["source"]["revision"] / spec["upstream_path"]
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        return tmp / "mirror"

    def test_wrong_content_is_rejected_and_leaves_nothing_behind(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            base = self.mirror(tmp, b"definitely not the pinned checkpoint")
            result = self.run_script(base, tmp / "out")
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("FAILED", result.stderr)
            self.assertEqual(list((tmp / "out").glob("*")), [], "no partial/unverified file may remain")

    def test_oversized_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            base = self.mirror(tmp, b"x" * 15_500_554)
            result = self.run_script(base, tmp / "out")
            self.assertEqual(result.returncode, 1)
            self.assertIn("exceeds", result.stderr)
            self.assertEqual(list((tmp / "out").glob("*")), [])

    def test_existing_unverified_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            out = tmp / "out"
            out.mkdir()
            (out / "distill-v4.pt").write_bytes(b"local experiment")
            result = self.run_script(self.mirror(tmp, b"whatever"), out)
            self.assertEqual(result.returncode, 1)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual((out / "distill-v4.pt").read_bytes(), b"local experiment")

    def test_a_correct_download_is_installed(self):
        real = BACKEND / "models" / "neural" / "distill-v4.pt"
        if not real.is_file():
            self.skipTest("distill-v4 checkpoint not provisioned; the success path is not exercised here")
        data = real.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         "1be52664b216529ce56240043406878a93f95f57e4153ff73c4d63afd772fba4")
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            result = self.run_script(self.mirror(tmp, data), tmp / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((tmp / "out" / "distill-v4.pt").read_bytes(), data)
            again = self.run_script(tmp / "mirror", tmp / "out")  # idempotent
            self.assertIn("already present", again.stdout)
