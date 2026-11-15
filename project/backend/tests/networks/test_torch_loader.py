"""Loader / torch-backend mechanics. REQUIRES torch (skipped, with a reason, when absent).

These tests build a TINY RANDOMLY INITIALISED network purely to exercise loading,
integrity, safety and runtime invariants. Random weights are never presented as
pretrained: real-weight evidence lives in test_real_checkpoint.py.
"""

import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from src.encoding import ACTION_SPACE_SIZE, ObservationEncoder
from src.environment import GameState
from src.networks.pretrained import (
    CheckpointIntegrityError, CheckpointNotFoundError, InvalidModelOutputError, NeuralConfigurationError,
    NeuralModelError, NeuralModelSettings, PolicyValueAdapter, UnsupportedCheckpointError, load_adapter,
)
from src.networks.pretrained.manifest import ArchitectureSpec, DEFAULT_MANIFEST_PATH, ModelSpec, load_manifest

from .support import ENGINE, assert_valid_priors, black_to_move_initial, fake_info, golden_state

HAVE_TORCH = importlib.util.find_spec("torch") is not None
SKIP_REASON = "PyTorch not installed (optional 'neural' extra); loader mechanics not exercised"

if HAVE_TORCH:
    import torch
    from torch import nn
    from src.networks.pretrained.torch_backend import CCLitePolicyValueNet, TorchPolicyValueBackend, load_torch_backend

TINY = {"channels": 8, "blocks": 1, "input_planes": 16, "policy_size": 8100}


def tiny_payload(seed=0, **overrides):
    torch.manual_seed(seed)
    arch = ArchitectureSpec(parameter_count=1, **TINY)
    model = CCLitePolicyValueNet(arch)
    payload = {"model_state": model.state_dict(), "model_config": dict(TINY), "step": 0, "metadata": {}}
    payload.update(overrides)
    return payload, sum(p.numel() for p in model.parameters())


def serialize(obj) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def spec_for(data: bytes, parameter_count: int, name="tiny", **arch_overrides) -> ModelSpec:
    base = load_manifest().get("distill-v4")
    arch = ArchitectureSpec(parameter_count=parameter_count, **{**TINY, **arch_overrides})
    fields = {f: getattr(base, f) for f in base.__slots__}
    fields.update(name=name, file=f"{name}.pt", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                  architecture=arch)
    return ModelSpec(**fields)


@unittest.skipUnless(HAVE_TORCH, SKIP_REASON)
class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, data: bytes, name="tiny.pt") -> Path:
        path = self.dir / name
        path.write_bytes(data)
        return path

    def load(self, data, count, **arch_overrides):
        return load_torch_backend(spec_for(data, count, **arch_overrides), self.write(data))

    # -- success path -------------------------------------------------------------------
    def test_loads_in_eval_mode_without_gradients(self):
        payload, count = tiny_payload()
        backend = self.load(serialize(payload), count)
        self.assertFalse(backend.model.training)
        self.assertTrue(all(not p.requires_grad for p in backend.model.parameters()))
        logits, value = backend.infer(ObservationEncoder().encode(GameState.initial()))
        self.assertEqual(len(logits), ACTION_SPACE_SIZE)
        self.assertTrue(all(math.isfinite(x) for x in logits) and -1 <= value <= 1)

    def test_predictions_repeat_exactly_and_never_change_weights_or_mode(self):
        payload, count = tiny_payload()
        backend = self.load(serialize(payload), count)
        spec = spec_for(serialize(payload), count)
        adapter = PolicyValueAdapter(backend, fake_info(name="tiny", parameter_count=count))
        before = {k: v.clone() for k, v in backend.model.state_dict().items()}
        grad_mode = torch.is_grad_enabled()
        states = [GameState.initial(), black_to_move_initial(), golden_state("fixture/mate_in_one_red")]
        first = [adapter.predict(s) for s in states]
        for _ in range(5):
            for state, expected in zip(states, first):
                again = adapter.predict(state)
                self.assertEqual((dict(again.priors), again.value), (dict(expected.priors), expected.value))
        for name, tensor in backend.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], tensor), f"weights changed: {name}")
        self.assertFalse(backend.model.training)
        self.assertEqual(torch.is_grad_enabled(), grad_mode, "predict must not leak grad-mode changes")
        self.assertTrue(all(p.grad is None for p in backend.model.parameters()))
        for state, result in zip(states, first):
            assert_valid_priors(self, result.priors, ENGINE.legal_moves(state))
        self.assertEqual(spec.architecture.channels, 8)

    def test_training_mode_is_refused_at_inference_time(self):
        payload, count = tiny_payload()
        backend = self.load(serialize(payload), count)
        backend.model.train()
        with self.assertRaisesRegex(NeuralModelError, "eval"):
            PolicyValueAdapter(backend, fake_info()).predict(GameState.initial())

    # -- integrity / trust --------------------------------------------------------------
    def test_missing_checkpoint(self):
        payload, count = tiny_payload()
        spec = spec_for(serialize(payload), count)
        with self.assertRaises(CheckpointNotFoundError):
            load_torch_backend(spec, self.dir / "absent.pt")
        with self.assertRaises(CheckpointNotFoundError):
            load_torch_backend(spec, self.dir)

    def test_hash_and_size_mismatch_are_refused_before_deserialising(self):
        payload, count = tiny_payload()
        data = serialize(payload)
        spec = spec_for(data, count)
        flipped = bytearray(data)
        flipped[len(flipped) // 2] ^= 0x01
        with self.assertRaisesRegex(CheckpointIntegrityError, "SHA-256"):
            load_torch_backend(spec, self.write(bytes(flipped)))
        with self.assertRaisesRegex(CheckpointIntegrityError, "size"):
            load_torch_backend(spec, self.write(data + b"\0"))
        with self.assertRaisesRegex(CheckpointIntegrityError, "size"):
            load_torch_backend(spec, self.write(data[:-1]))

    def test_other_random_weights_with_the_right_shape_are_still_refused_by_hash(self):
        a, count = tiny_payload(seed=1)
        b, _ = tiny_payload(seed=2)
        with self.assertRaises(CheckpointIntegrityError):
            load_torch_backend(spec_for(serialize(a), count), self.write(serialize(b)))

    def test_pickled_code_is_never_executed(self):
        marker = self.dir / "pwned"

        class Evil:
            def __reduce__(self):
                import os
                return os.mkdir, (str(marker),)

        data = serialize({"model_state": Evil(), "model_config": dict(TINY)})
        with self.assertRaises(UnsupportedCheckpointError):
            self.load(data, 1)
        self.assertFalse(marker.exists(), "unsafe deserialisation executed code")

    # -- format / architecture ----------------------------------------------------------
    def test_garbage_and_wrongly_shaped_payloads(self):
        _, count = tiny_payload()
        for name, data in {"garbage": b"not a torch file" * 10, "tensor": serialize(torch.zeros(3)),
                           "empty dict": serialize({}), "state only": serialize({"model_state": {}}),
                           "list": serialize([1, 2, 3])}.items():
            with self.subTest(name), self.assertRaises(UnsupportedCheckpointError):
                self.load(data, count)

    def test_architecture_mismatch_between_payload_and_manifest(self):
        payload, count = tiny_payload()
        data = serialize(payload)
        for key, wrong in (("channels", 16), ("blocks", 2), ("policy_size", 2086), ("input_planes", 14)):
            with self.subTest(key), self.assertRaises(UnsupportedCheckpointError):
                self.load(data, count, **{key: wrong})
        bad_cfg = serialize({**payload, "model_config": {**TINY, "channels": 16}})
        with self.assertRaises(UnsupportedCheckpointError):
            self.load(bad_cfg, count)
        extra_cfg = serialize({**payload, "model_config": {**TINY, "extra": 1}})
        with self.assertRaises(UnsupportedCheckpointError):
            self.load(extra_cfg, count)

    def test_state_dict_missing_or_extra_keys_and_bad_shapes(self):
        payload, count = tiny_payload()
        state = dict(payload["model_state"])
        missing = {k: v for k, v in state.items() if k != "value_head.6.bias"}
        extra = {**state, "surprise.weight": torch.zeros(1)}
        reshaped = {**state, "stem.0.weight": torch.zeros(3, 3)}
        for name, bad in {"missing": missing, "extra": extra, "reshaped": reshaped}.items():
            with self.subTest(name), self.assertRaises(UnsupportedCheckpointError):
                self.load(serialize({**payload, "model_state": bad}), count)

    def test_parameter_count_must_match_the_manifest(self):
        payload, count = tiny_payload()
        with self.assertRaisesRegex(UnsupportedCheckpointError, "parameters"):
            self.load(serialize(payload), count + 1)

    def test_non_finite_weights_are_refused(self):
        payload, count = tiny_payload()
        for bad in (math.nan, math.inf):
            state = {k: v.clone() for k, v in payload["model_state"].items()}
            state["stem.0.weight"][0, 0, 0, 0] = bad
            with self.subTest(bad), self.assertRaisesRegex(UnsupportedCheckpointError, "non-finite"):
                self.load(serialize({**payload, "model_state": state}), count)

    # -- configuration through load_adapter ---------------------------------------------
    def tiny_manifest(self, data: bytes, count: int) -> Path:
        doc = json.loads(DEFAULT_MANIFEST_PATH.read_text())
        base = doc["models"]["distill-v4"]
        doc["models"] = {"tiny": {**base, "file": "tiny.pt", "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest(),
                                  "architecture": {**TINY, "parameter_count": count}}}
        doc["default_model"] = "tiny"
        path = self.dir / "manifest.json"
        path.write_text(json.dumps(doc))
        return path

    def test_load_adapter_end_to_end_and_error_paths(self):
        payload, count = tiny_payload()
        data = serialize(payload)
        manifest = self.tiny_manifest(data, count)
        settings = NeuralModelSettings(path=self.write(data), manifest_path=manifest)
        adapter = load_adapter(settings)
        self.assertEqual((adapter.info.name, adapter.info.device, adapter.info.parameter_count),
                         ("tiny", "cpu", count))
        assert_valid_priors(self, adapter.predict(GameState.initial()).priors,
                            ENGINE.legal_moves(GameState.initial()))
        with self.assertRaises(CheckpointNotFoundError):
            load_adapter(NeuralModelSettings(path=self.dir / "nope.pt", manifest_path=manifest))
        with self.assertRaises(NeuralConfigurationError):
            load_adapter(NeuralModelSettings(path=self.write(data), manifest_path=manifest, sha256="0" * 64))
        with self.assertRaises(NeuralConfigurationError):
            load_adapter(NeuralModelSettings(path=self.write(data), manifest_path=manifest, device="cuda"))
        self.write(data[:-1], "tiny.pt")
        with self.assertRaises(CheckpointIntegrityError):
            load_adapter(NeuralModelSettings(path=self.dir / "tiny.pt", manifest_path=manifest))

    def test_thread_setting_is_applied(self):
        payload, count = tiny_payload()
        data = serialize(payload)
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        load_torch_backend(spec_for(data, count), self.write(data), threads=1)
        self.assertEqual(torch.get_num_threads(), 1)


class _FixedOutputModel(nn.Module if HAVE_TORCH else object):
    def __init__(self, logits, value):
        super().__init__()
        self._logits, self._value = logits, value

    def forward(self, x):
        return self._logits, self._value


@unittest.skipUnless(HAVE_TORCH, SKIP_REASON)
class BackendOutputValidationTests(unittest.TestCase):
    """A misbehaving network must be reported, not smoothed over."""

    def run_model(self, logits, value):
        backend = TorchPolicyValueBackend(_FixedOutputModel(logits, value))
        return PolicyValueAdapter(backend, fake_info()).predict(GameState.initial())

    def good_logits(self):
        return torch.zeros(1, ACTION_SPACE_SIZE)

    def test_a_well_formed_output_passes(self):
        result = self.run_model(self.good_logits(), torch.tensor([0.5]))
        self.assertEqual(result.value, 0.5)

    def test_wrong_shapes_and_dtypes(self):
        cases = {"policy 2086": (torch.zeros(1, 2086), torch.tensor([0.0])),
                 "policy batch 2": (torch.zeros(2, ACTION_SPACE_SIZE), torch.tensor([0.0])),
                 "policy flat": (torch.zeros(ACTION_SPACE_SIZE), torch.tensor([0.0])),
                 "value batch 2": (self.good_logits(), torch.tensor([0.0, 0.0])),
                 "value (1,1)": (self.good_logits(), torch.tensor([[0.0]])),
                 "integer policy": (torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.int64), torch.tensor([0.0])),
                 "integer value": (self.good_logits(), torch.tensor([0]))}
        for name, (logits, value) in cases.items():
            with self.subTest(name), self.assertRaises(InvalidModelOutputError):
                self.run_model(logits, value)

    def test_non_finite_and_out_of_range(self):
        nan_logits = self.good_logits()
        nan_logits[0, 17] = math.nan
        inf_logits = self.good_logits()
        inf_logits[0, 4000] = math.inf
        for name, (logits, value) in {"nan logit": (nan_logits, torch.tensor([0.0])),
                                      "inf logit": (inf_logits, torch.tensor([0.0])),
                                      "nan value": (self.good_logits(), torch.tensor([math.nan])),
                                      "inf value": (self.good_logits(), torch.tensor([math.inf])),
                                      "value 1.2": (self.good_logits(), torch.tensor([1.2]))}.items():
            with self.subTest(name), self.assertRaises(InvalidModelOutputError):
                self.run_model(logits, value)


if __name__ == "__main__":
    unittest.main()
