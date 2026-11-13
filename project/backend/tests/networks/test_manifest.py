"""Manifest, settings, integrity and import-isolation tests (no torch needed)."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.networks.pretrained import (
    CheckpointIntegrityError, CheckpointNotFoundError, NeuralConfigurationError,
    NeuralModelSettings, UnsupportedCheckpointError, load_manifest,
)
from src.networks.pretrained.manifest import (
    BACKEND_ROOT, DEFAULT_MANIFEST_PATH, ModelSpec, parse_manifest, read_verified_bytes,
)

from .support import BACKEND

PINNED = {
    "distill-v4-large": ("05927ccd874c4d97588ba7d06266fda9ba17700d436fdc03ff6844a0d61d0dff", 38033337),
    "distill-v4": ("1be52664b216529ce56240043406878a93f95f57e4153ff73c4d63afd772fba4", 15500553),
}


def document():
    return json.loads(DEFAULT_MANIFEST_PATH.read_text())


class CommittedManifestTests(unittest.TestCase):
    def test_pins_match_the_phase1_verified_hashes(self):
        manifest = load_manifest()
        self.assertEqual(manifest.default_model, "distill-v4-large")
        self.assertEqual(set(manifest.models), set(PINNED))
        for name, (sha, size) in PINNED.items():
            spec = manifest.get(name)
            self.assertEqual((spec.sha256, spec.bytes), (sha, size))
            self.assertEqual(spec.source_revision, "7d305c0e5adbfae4e23db40997603d8dc86b078b")
        self.assertEqual(manifest.get("distill-v4-large").architecture.channels, 192)
        self.assertEqual(manifest.get("distill-v4").architecture.channels, 128)

    def test_phase1_artifact_manifest_agrees_with_the_model_manifest(self):
        artifacts = json.loads((BACKEND / "experiments/neural_mcts/phase1/artifacts.json").read_text())
        self.assertEqual(artifacts["checkpoint"]["sha256"], PINNED["distill-v4"][0])
        self.assertEqual(artifacts["alternate_checkpoints"]["distill-v4-large"]["sha256"],
                         PINNED["distill-v4-large"][0])

    def test_gitignore_excludes_weights_but_not_the_manifest(self):
        def ignored(name):
            return subprocess.run(["git", "check-ignore", "-q", f"models/neural/{name}"], cwd=BACKEND,
                                  capture_output=True)
        probe = ignored("distill-v4-large.pt")
        if probe.returncode not in (0, 1):
            self.skipTest("not a git checkout")
        self.assertEqual(probe.returncode, 0, "weights must be git-ignored")
        self.assertEqual(ignored("anything.bin").returncode, 0)
        self.assertEqual(ignored("manifest.json").returncode, 1, "manifest must stay tracked")
        self.assertEqual(ignored("README.md").returncode, 1)


class ManifestValidationTests(unittest.TestCase):
    def mutated(self, mutate):
        data = document()
        mutate(data)
        return data

    def test_rejections(self):
        cases = {
            "schema": lambda d: d.update(schema_version=2),
            "bad sha": lambda d: d["models"]["distill-v4"].update(sha256="ABC"),
            "upper-case sha": lambda d: d["models"]["distill-v4"].update(
                sha256=PINNED["distill-v4"][0].upper()),
            "path traversal": lambda d: d["models"]["distill-v4"].update(file="../x.pt"),
            "bad size": lambda d: d["models"]["distill-v4"].update(bytes=0),
            "missing arch": lambda d: d["models"]["distill-v4"].pop("architecture"),
            "bad default": lambda d: d.update(default_model="nope"),
            "empty models": lambda d: d.update(models={}),
            "logit contract": lambda d: d["contract"].update(policy_output="probabilities"),
        }
        for name, mutate in cases.items():
            with self.subTest(name), self.assertRaises(NeuralConfigurationError):
                parse_manifest(self.mutated(mutate))
        with self.assertRaises(NeuralConfigurationError):
            parse_manifest([])

    def test_a_manifest_for_a_different_contract_is_unsupported(self):
        for key, value in (("observation", "other_v2"), ("action_vocabulary", "ccz_2086")):
            with self.subTest(key), self.assertRaises(UnsupportedCheckpointError):
                parse_manifest(self.mutated(lambda d: d["contract"].update({key: value})))

    def test_missing_or_corrupt_manifest_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(NeuralConfigurationError):
                load_manifest(Path(tmp) / "missing.json")
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json")
            with self.assertRaises(NeuralConfigurationError):
                load_manifest(bad)


class SettingsTests(unittest.TestCase):
    def test_defaults_select_the_large_model_on_cpu_under_models_neural(self):
        resolved = NeuralModelSettings.from_env({}).resolve()
        self.assertEqual(resolved.spec.name, "distill-v4-large")
        self.assertEqual(resolved.path, (BACKEND_ROOT / "models" / "neural" / "distill-v4-large.pt").resolve())
        self.assertEqual(NeuralModelSettings.from_env({}).device, "cpu")

    def test_model_selection_v4_vs_large(self):
        for name in ("distill-v4", "distill-v4-large"):
            resolved = NeuralModelSettings.from_env({"XIANGQI_NEURAL_MODEL": name}).resolve()
            self.assertEqual((resolved.spec.name, resolved.spec.sha256), (name, PINNED[name][0]))
            self.assertEqual(resolved.path.name, f"{name}.pt")
        with self.assertRaises(NeuralConfigurationError):
            NeuralModelSettings.from_env({"XIANGQI_NEURAL_MODEL": "distill-v9"}).resolve()

    def test_paths_absolute_and_backend_relative_never_cwd_relative(self):
        absolute = NeuralModelSettings.from_env({"XIANGQI_NEURAL_MODEL_PATH": "/opt/m/x.pt"}).resolve()
        self.assertEqual(absolute.path, Path("/opt/m/x.pt").resolve())
        relative = NeuralModelSettings.from_env({"XIANGQI_NEURAL_MODEL_PATH": "models/neural/alt.pt"}).resolve()
        self.assertEqual(relative.path, (BACKEND_ROOT / "models/neural/alt.pt").resolve())

    def test_blank_variables_count_as_unset(self):
        settings = NeuralModelSettings.from_env({"XIANGQI_NEURAL_MODEL": " ", "XIANGQI_NEURAL_MODEL_PATH": "",
                                                 "XIANGQI_NEURAL_DEVICE": "", "XIANGQI_NEURAL_THREADS": ""})
        self.assertEqual((settings.model, settings.path, settings.device, settings.threads),
                         (None, None, "cpu", None))

    def test_sha_variable_is_only_an_assertion_never_a_trust_override(self):
        good = NeuralModelSettings.from_env({"XIANGQI_NEURAL_MODEL_SHA256": PINNED["distill-v4-large"][0].upper()})
        self.assertEqual(good.resolve().spec.sha256, PINNED["distill-v4-large"][0])
        with self.assertRaises(NeuralConfigurationError):
            NeuralModelSettings.from_env({"XIANGQI_NEURAL_MODEL_SHA256": "0" * 64}).resolve()
        with self.assertRaises(NeuralConfigurationError):  # v4's hash does not describe the default (large)
            NeuralModelSettings.from_env({"XIANGQI_NEURAL_MODEL_SHA256": PINNED["distill-v4"][0]}).resolve()

    def test_unsupported_device_and_threads(self):
        for device in ("cuda", "cuda:0", "mps", "gpu"):
            with self.subTest(device), self.assertRaises(NeuralConfigurationError):
                NeuralModelSettings.from_env({"XIANGQI_NEURAL_DEVICE": device}).resolve()
        for threads in ("0", "-2", "two"):
            with self.subTest(threads), self.assertRaises(NeuralConfigurationError):
                NeuralModelSettings.from_env({"XIANGQI_NEURAL_THREADS": threads}).resolve()
        self.assertEqual(NeuralModelSettings.from_env({"XIANGQI_NEURAL_THREADS": "3"}).threads, 3)


class IntegrityTests(unittest.TestCase):
    def spec_for(self, payload: bytes) -> ModelSpec:
        base = load_manifest().get("distill-v4")
        return ModelSpec(**{**{f: getattr(base, f) for f in base.__slots__},
                            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})

    def test_verified_bytes_are_returned_only_for_an_exact_match(self):
        payload = b"checkpoint-bytes" * 100
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            path.write_bytes(payload)
            self.assertEqual(read_verified_bytes(path, self.spec_for(payload)), payload)

    def test_missing_file_wrong_size_and_wrong_hash_fail_loudly(self):
        payload = b"x" * 64
        spec = self.spec_for(payload)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            with self.assertRaises(CheckpointNotFoundError):
                read_verified_bytes(path, spec)
            with self.assertRaises(CheckpointNotFoundError):  # a directory is not a checkpoint
                read_verified_bytes(Path(tmp), spec)
            path.write_bytes(payload + b"!")
            with self.assertRaisesRegex(CheckpointIntegrityError, "size"):
                read_verified_bytes(path, spec)
            path.write_bytes(b"y" * 64)  # same size, different content
            with self.assertRaisesRegex(CheckpointIntegrityError, "SHA-256"):
                read_verified_bytes(path, spec)


class ImportIsolationTests(unittest.TestCase):
    def run_blocked(self, code: str):
        # sys.modules['torch'] = None makes any `import torch` raise ImportError.
        prelude = "import sys; sys.modules['torch'] = None; sys.modules['numpy'] = None; "
        return subprocess.run([sys.executable, "-c", prelude + code], cwd=BACKEND, capture_output=True, text=True)

    def test_engine_encoding_classic_mcts_and_neural_package_import_without_torch(self):
        result = self.run_blocked(
            "import src.environment, src.encoding, src.search, src.agents.mcts_agent, "
            "src.networks.pretrained; print('ok')")
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "ok"), result.stderr)

    def test_loading_without_torch_raises_the_dedicated_error_and_names_the_remedy(self):
        result = self.run_blocked(
            "from src.networks.pretrained import load_adapter, NeuralRuntimeUnavailableError\n"
            "try:\n    load_adapter()\nexcept NeuralRuntimeUnavailableError as e:\n    print('unavailable:', e)\n"
            "else:\n    raise SystemExit('loaded without torch?!')\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("unavailable:", result.stdout)
        self.assertIn("README", result.stdout)
