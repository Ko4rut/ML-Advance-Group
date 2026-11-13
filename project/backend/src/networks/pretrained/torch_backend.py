"""PyTorch backend for the cc-lite policy-value checkpoints (imports torch).

Only imported lazily by :func:`load_adapter` / :func:`load_torch_backend`; the
rest of the backend never imports this module.

The network definition below is a re-implementation of the architecture of
``src/model/network.py`` in https://github.com/hskl18/cc-lite (Apache-2.0,
revision 7d305c0e5adbfae4e23db40997603d8dc86b078b). Layer layout and names are
kept identical so the pretrained ``state_dict`` loads with ``strict=True``;
equality of outputs with the upstream implementation is verified by
experiments/neural_mcts/phase3/test_real_checkpoint.py.
"""

from __future__ import annotations

from array import array
import io
from pathlib import Path

import torch
from torch import nn

from ...encoding import OBSERVATION_LENGTH, OBSERVATION_SHAPE
from .errors import InvalidModelOutputError, NeuralModelError, UnsupportedCheckpointError
from .manifest import ArchitectureSpec, ModelSpec, read_verified_bytes

_POLICY_SIZE = 8100


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class CCLitePolicyValueNet(nn.Module):
    """Residual tower with a policy head (8100 logits) and a tanh value head."""

    def __init__(self, architecture: ArchitectureSpec) -> None:
        super().__init__()
        c = architecture.channels
        self.stem = nn.Sequential(
            nn.Conv2d(architecture.input_planes, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(c) for _ in range(architecture.blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 10 * 9, architecture.policy_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(10 * 9, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.blocks(self.stem(x))
        return self.policy_head(trunk), self.value_head(trunk).squeeze(-1)


class TorchPolicyValueBackend:
    """Single-observation CPU inference on an eval-mode, gradient-free model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def infer(self, observation: array) -> tuple[list[float], float]:
        if len(observation) != OBSERVATION_LENGTH:
            raise InvalidModelOutputError(f"observation must have {OBSERVATION_LENGTH} values")
        if self.model.training:
            raise NeuralModelError("model is in training mode; inference requires eval()")
        batch = torch.frombuffer(observation, dtype=torch.float32).reshape(1, *OBSERVATION_SHAPE)
        with torch.inference_mode():
            logits, value = self.model(batch)
        if tuple(logits.shape) != (1, _POLICY_SIZE) or not logits.dtype.is_floating_point:
            raise InvalidModelOutputError(f"unexpected policy output {tuple(logits.shape)} {logits.dtype}")
        if tuple(value.shape) != (1,) or not value.dtype.is_floating_point:
            raise InvalidModelOutputError(f"unexpected value output {tuple(value.shape)} {value.dtype}")
        return logits[0].tolist(), float(value[0])


def load_torch_backend(spec: ModelSpec, path: Path, *, threads: int | None = None) -> TorchPolicyValueBackend:
    """Verify ``path`` against ``spec`` (size + SHA-256), then load it safely on CPU.

    The verified bytes are deserialized with ``weights_only=True`` (tensors and
    plain containers only, no pickled code), the payload's architecture must equal
    the manifest's, and the state dict is loaded with ``strict=True``.
    """
    data = read_verified_bytes(path, spec)  # raises CheckpointNotFound/IntegrityError
    if threads is not None:
        torch.set_num_threads(threads)  # process-global; documented in the model card
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:  # pickle/zip errors surface as many types
        raise UnsupportedCheckpointError(f"{path}: not a safely loadable torch payload: {exc}") from exc
    del data
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict) \
            or not isinstance(payload.get("model_config"), dict):
        raise UnsupportedCheckpointError(f"{path}: payload lacks model_state/model_config")
    architecture = spec.architecture
    expected = {"channels": architecture.channels, "blocks": architecture.blocks,
                "input_planes": architecture.input_planes, "policy_size": architecture.policy_size}
    if payload["model_config"] != expected:
        raise UnsupportedCheckpointError(
            f"{path}: architecture {payload['model_config']} != manifest {expected}")
    model = CCLitePolicyValueNet(architecture)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (RuntimeError, KeyError, TypeError) as exc:
        raise UnsupportedCheckpointError(f"{path}: state dict does not match the architecture: {exc}") from exc
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != architecture.parameter_count:
        raise UnsupportedCheckpointError(
            f"{path}: {parameter_count} parameters != manifest {architecture.parameter_count}")
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise UnsupportedCheckpointError(f"{path}: non-finite values in {name}")
    return TorchPolicyValueBackend(model)
