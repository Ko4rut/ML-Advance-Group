"""Versioned checkpoint manifest, environment settings and integrity checks.

Pure standard library. The manifest (``models/neural/manifest.json``) is the
only source of trust: a checkpoint is accepted iff its size and SHA-256 equal
the values pinned there. Environment variables select *which* manifest entry to
use and where its file lives; they can never widen trust (``..._SHA256`` is only
an extra assertion that must equal the pinned value).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import astuple, dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re

from ...encoding import CCLITE_VOCABULARY, OBSERVATION_ENCODING_NAME
from .errors import (
    CheckpointIntegrityError,
    CheckpointNotFoundError,
    NeuralConfigurationError,
    UnsupportedCheckpointError,
)

# project/backend, independent of the process working directory.
BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = BACKEND_ROOT / "models" / "neural"
DEFAULT_MANIFEST_PATH = DEFAULT_MODEL_DIR / "manifest.json"

ENV_MODEL = "XIANGQI_NEURAL_MODEL"
ENV_PATH = "XIANGQI_NEURAL_MODEL_PATH"
ENV_SHA256 = "XIANGQI_NEURAL_MODEL_SHA256"
ENV_DEVICE = "XIANGQI_NEURAL_DEVICE"
ENV_THREADS = "XIANGQI_NEURAL_THREADS"

SUPPORTED_DEVICES = ("cpu",)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    channels: int
    blocks: int
    input_planes: int
    policy_size: int
    parameter_count: int


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One trusted checkpoint entry of the manifest."""

    name: str
    file: str
    bytes: int
    sha256: str
    architecture: ArchitectureSpec
    format: str
    upstream_path: str
    source_repository: str
    source_revision: str
    license: str
    observation: str
    action_vocabulary: str


@dataclass(frozen=True, slots=True)
class Manifest:
    default_model: str
    models: Mapping[str, ModelSpec]

    def get(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError:
            raise NeuralConfigurationError(
                f"unknown neural model {name!r}; manifest defines {sorted(self.models)}"
            ) from None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NeuralConfigurationError(f"invalid neural manifest: {message}")


def parse_manifest(data: object) -> Manifest:
    """Validate a decoded manifest document and check it against the production contract."""
    _require(isinstance(data, dict), "top level must be an object")
    _require(data.get("schema_version") == 1, "unsupported schema_version")
    contract, source, models = data.get("contract"), data.get("source"), data.get("models")
    _require(isinstance(contract, dict) and isinstance(source, dict), "missing contract/source")
    _require(isinstance(models, dict) and models, "models must be a non-empty object")
    # The code in src/encoding implements exactly one contract; refuse any other.
    if contract.get("observation") != OBSERVATION_ENCODING_NAME:
        raise UnsupportedCheckpointError(
            f"manifest observation {contract.get('observation')!r} != {OBSERVATION_ENCODING_NAME!r}")
    if contract.get("action_vocabulary") != CCLITE_VOCABULARY.name:
        raise UnsupportedCheckpointError(
            f"manifest vocabulary {contract.get('action_vocabulary')!r} != {CCLITE_VOCABULARY.name!r}")
    _require(contract.get("policy_output") == "logits", "policy_output must be 'logits'")
    specs: dict[str, ModelSpec] = {}
    for name, entry in models.items():
        _require(isinstance(entry, dict), f"model {name!r} must be an object")
        sha = entry.get("sha256")
        _require(isinstance(sha, str) and _SHA256_RE.match(sha) is not None,
                 f"model {name!r}: sha256 must be 64 lowercase hex characters")
        size, file = entry.get("bytes"), entry.get("file")
        _require(isinstance(size, int) and not isinstance(size, bool) and size > 0,
                 f"model {name!r}: bytes must be a positive integer")
        _require(isinstance(file, str) and file and Path(file).name == file,
                 f"model {name!r}: file must be a plain file name")
        arch = entry.get("architecture")
        _require(isinstance(arch, dict), f"model {name!r}: missing architecture")
        try:
            architecture = ArchitectureSpec(**{k: arch[k] for k in (
                "channels", "blocks", "input_planes", "policy_size", "parameter_count")})
        except (KeyError, TypeError):
            raise NeuralConfigurationError(f"invalid neural manifest: model {name!r}: bad architecture") from None
        _require(all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                     for v in astuple(architecture)), f"model {name!r}: architecture values must be positive ints")
        specs[name] = ModelSpec(
            name=name, file=file, bytes=size, sha256=sha, architecture=architecture,
            format=str(entry.get("format", "")), upstream_path=str(entry.get("upstream_path", "")),
            source_repository=str(source.get("repository", "")),
            source_revision=str(source.get("revision", "")), license=str(source.get("license", "")),
            observation=contract["observation"], action_vocabulary=contract["action_vocabulary"])
    default = data.get("default_model")
    _require(default in specs, "default_model must name a model entry")
    return Manifest(default_model=default, models=specs)


def load_manifest(path: str | Path | None = None) -> Manifest:
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NeuralConfigurationError(f"neural manifest not found: {manifest_path}") from None
    except (OSError, ValueError) as exc:
        raise NeuralConfigurationError(f"neural manifest unreadable: {manifest_path}: {exc}") from None
    return parse_manifest(data)


def read_verified_bytes(path: Path, spec: ModelSpec) -> bytes:
    """Read the checkpoint once and verify size + SHA-256 on those exact bytes.

    The verified bytes are what get deserialized, so the file cannot be swapped
    between verification and loading.
    """
    if not path.is_file():
        raise CheckpointNotFoundError(
            f"checkpoint for {spec.name!r} not found: {path} "
            f"(see models/neural/README.md for how to provision it)")
    size = path.stat().st_size
    if size != spec.bytes:
        raise CheckpointIntegrityError(
            f"{path}: size {size} != pinned {spec.bytes} for {spec.name!r}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != spec.bytes or not hmac.compare_digest(digest, spec.sha256):
        raise CheckpointIntegrityError(
            f"{path}: SHA-256 {digest} != pinned {spec.sha256} for {spec.name!r}")
    return data


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    spec: ModelSpec
    path: Path


@dataclass(frozen=True, slots=True)
class NeuralModelSettings:
    """Where the checkpoint is and how to run it. Build with ``from_env`` or directly."""

    model: str | None = None            # manifest key; None -> manifest default
    path: Path | None = None            # None -> models/neural/<manifest file>
    sha256: str | None = None           # optional extra assertion; must equal the pinned hash
    device: str = "cpu"
    threads: int | None = None          # None -> leave torch's global thread setting alone
    manifest_path: Path = DEFAULT_MANIFEST_PATH

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "NeuralModelSettings":
        env = os.environ if env is None else env

        def get(name: str) -> str | None:
            value = env.get(name)
            return value.strip() or None if value is not None else None

        path = get(ENV_PATH)
        threads_text = get(ENV_THREADS)
        try:
            threads = int(threads_text) if threads_text else None
        except ValueError:
            raise NeuralConfigurationError(f"{ENV_THREADS} must be a positive integer") from None
        return cls(model=get(ENV_MODEL), path=Path(path).expanduser() if path else None,
                   sha256=get(ENV_SHA256), device=get(ENV_DEVICE) or "cpu", threads=threads)

    def validate_static(self) -> None:
        if self.device not in SUPPORTED_DEVICES:
            raise NeuralConfigurationError(
                f"device {self.device!r} is not supported; Phase 3 validated {SUPPORTED_DEVICES} only")
        if self.threads is not None and (isinstance(self.threads, bool) or self.threads < 1):
            raise NeuralConfigurationError("threads must be a positive integer")

    def resolve(self) -> ResolvedModel:
        """Select the manifest entry and locate its file. Does not read the file."""
        self.validate_static()
        manifest = load_manifest(self.manifest_path)
        spec = manifest.get(self.model or manifest.default_model)
        if self.sha256 is not None:
            asserted = self.sha256.strip().lower()
            if asserted != spec.sha256:
                raise NeuralConfigurationError(
                    f"{ENV_SHA256} {asserted!r} does not equal the SHA-256 pinned in the manifest "
                    f"for {spec.name!r}; unlisted checkpoints are not trusted")
        if self.path is None:
            path = DEFAULT_MODEL_DIR / spec.file
        else:
            path = self.path if self.path.is_absolute() else BACKEND_ROOT / self.path
        return ResolvedModel(spec=spec, path=path.resolve())
