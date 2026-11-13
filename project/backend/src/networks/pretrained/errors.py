"""Exception hierarchy for pretrained-model loading and inference.

Every failure mode of the neural stack raises one of these. Nothing in this
package catches them to substitute random weights, another checkpoint or a
heuristic value.
"""


class NeuralModelError(Exception):
    """Base class for all pretrained-model errors."""


class NeuralConfigurationError(NeuralModelError, ValueError):
    """Invalid or unsupported configuration (env vars, manifest, device)."""


class CheckpointNotFoundError(NeuralModelError, FileNotFoundError):
    """The configured checkpoint file does not exist."""


class CheckpointIntegrityError(NeuralModelError):
    """Size or SHA-256 differs from the value pinned in the manifest (untrusted/corrupt file)."""


class UnsupportedCheckpointError(NeuralModelError):
    """The file is not a checkpoint this adapter can safely load (format, architecture, contract)."""


class NeuralRuntimeUnavailableError(NeuralModelError):
    """The optional ML runtime (PyTorch) is not installed."""


class InvalidModelOutputError(NeuralModelError, ValueError):
    """The network produced an output that violates the inference contract."""
