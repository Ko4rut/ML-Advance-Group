"""Pretrained policy-value inference (Phase 3). Importing this package never imports torch."""

from __future__ import annotations

from .adapter import InferenceBackend, ModelInfo, PolicyValueAdapter, PolicyValueResult
from .errors import (
    CheckpointIntegrityError,
    CheckpointNotFoundError,
    InvalidModelOutputError,
    NeuralConfigurationError,
    NeuralModelError,
    NeuralRuntimeUnavailableError,
    UnsupportedCheckpointError,
)
from .manifest import (
    DEFAULT_MANIFEST_PATH,
    Manifest,
    ModelSpec,
    NeuralModelSettings,
    ResolvedModel,
    load_manifest,
)


def load_adapter(settings: NeuralModelSettings | None = None) -> PolicyValueAdapter:
    """Build a ready adapter from settings (default: environment variables).

    Fails loudly (a ``NeuralModelError`` subclass) if the runtime, file, hash,
    format or architecture is wrong. Create it once and share it: every call
    loads the weights again.
    """
    settings = settings or NeuralModelSettings.from_env()
    resolved = settings.resolve()
    try:
        from .torch_backend import load_torch_backend
    except ImportError as exc:
        raise NeuralRuntimeUnavailableError(
            "PyTorch is required for neural inference; install the optional extra "
            "(see models/neural/README.md)") from exc
    backend = load_torch_backend(resolved.spec, resolved.path, threads=settings.threads)
    spec = resolved.spec
    info = ModelInfo(
        name=spec.name, sha256=spec.sha256, source_repository=spec.source_repository,
        source_revision=spec.source_revision, observation=spec.observation,
        action_vocabulary=spec.action_vocabulary, device=settings.device,
        parameter_count=spec.architecture.parameter_count)
    return PolicyValueAdapter(backend, info)


__all__ = [
    "CheckpointIntegrityError", "CheckpointNotFoundError", "DEFAULT_MANIFEST_PATH", "InferenceBackend",
    "InvalidModelOutputError", "Manifest", "ModelInfo", "ModelSpec", "NeuralConfigurationError",
    "NeuralModelError", "NeuralModelSettings", "NeuralRuntimeUnavailableError", "PolicyValueAdapter",
    "PolicyValueResult", "ResolvedModel", "UnsupportedCheckpointError", "load_adapter", "load_manifest",
]
