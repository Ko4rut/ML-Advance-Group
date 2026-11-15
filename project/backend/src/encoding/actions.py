"""Move/action ID conversion and legal action masks.

Two different integer spaces exist and must never be conflated:

* The repository's **internal action ID** (``from_to_8100_v1``, used by the Gold
  pipeline and tabular Q): ``source * 90 + target`` in ``[0, 8100)``.
* A pretrained model's **native policy index** (here: the cc-lite checkpoints).
  For cc-lite this index happens to be numerically identical to the internal ID;
  that identity is *verified by golden tests against the upstream codec*, and it
  is kept behind a separate vocabulary type so a checkpoint with a different
  vocabulary (e.g. 2086 labels) can be added without touching the Gold contract.

Pure standard library; no numpy/torch import.
"""

from collections.abc import Iterable, Sequence
import math

from ..environment import Move

ACTION_ENCODING_NAME = "from_to_8100_v1"
BOARD_SQUARES = 90
ACTION_SPACE_SIZE = BOARD_SQUARES * BOARD_SQUARES


def _check_index(index: object, size: int, what: str) -> int:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < size:
        raise ValueError(f"{what} must be an integer in [0, {size})")
    return index


class ActionEncoder:
    """Internal Gold action IDs (``from_to_8100_v1``). Stateless."""

    encoding_name = ACTION_ENCODING_NAME
    size = ACTION_SPACE_SIZE

    def encode(self, move: Move) -> int:
        if not isinstance(move, Move):
            raise TypeError("move must be a Move")
        return move.source * BOARD_SQUARES + move.target

    def decode(self, action_id: int) -> Move:
        """Structural inverse of ``encode``. Does not check legality."""
        _check_index(action_id, ACTION_SPACE_SIZE, "action_id")
        return Move(action_id // BOARD_SQUARES, action_id % BOARD_SQUARES)

    def legal_action_ids(self, moves: Iterable[Move]) -> list[int]:
        """Sorted unique IDs of ``moves``; raises on duplicates."""
        ids = [self.encode(move) for move in moves]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate moves")
        return sorted(ids)

    def legal_mask(self, moves: Iterable[Move]) -> list[bool]:
        """Dense mask of length 8100 that is True exactly at the given legal moves."""
        mask = [False] * ACTION_SPACE_SIZE
        for action_id in self.legal_action_ids(moves):
            mask[action_id] = True
        return mask


class CCLiteVocabulary:
    """Native policy vocabulary of the cc-lite checkpoints (8100 logits).

    ``to_native``/``from_native`` are the only sanctioned way to cross between
    ``Move`` and a model policy index. Currently ``native == source*90+target``.
    """

    name = "cclite_from_to_8100"
    size = ACTION_SPACE_SIZE

    def to_native(self, move: Move) -> int:
        if not isinstance(move, Move):
            raise TypeError("move must be a Move")
        return move.source * BOARD_SQUARES + move.target

    def from_native(self, index: int) -> Move:
        _check_index(index, ACTION_SPACE_SIZE, "native policy index")
        return Move(index // BOARD_SQUARES, index % BOARD_SQUARES)


CCLITE_VOCABULARY = CCLiteVocabulary()


def legal_priors(
    scores: Sequence[float],
    moves: Sequence[Move],
    *,
    kind: str = "logits",
    vocabulary: CCLiteVocabulary = CCLITE_VOCABULARY,
) -> dict[Move, float]:
    """Mask a native policy to ``moves`` and normalize over them.

    ``scores`` holds one entry per native index (a list, ``array`` or ndarray).
    ``kind`` is ``"logits"`` (stable softmax over the legal entries) or
    ``"probabilities"`` (renormalized). Illegal entries never influence the
    result, but every entry must be finite so a corrupt output is rejected
    instead of silently masked. Raises ``ValueError`` (never falls back to a
    guess) on wrong length, non-finite scores, negative probabilities, an empty
    or duplicated legal-move list, or zero legal probability mass.

    The returned priors are positive/zero finite floats summing to 1 over
    exactly the supplied legal moves.
    """
    if kind not in ("logits", "probabilities"):
        raise ValueError("kind must be 'logits' or 'probabilities'")
    if len(scores) != vocabulary.size:
        raise ValueError(f"policy must contain {vocabulary.size} scores")
    values = scores.tolist() if hasattr(scores, "tolist") else [float(x) for x in scores]
    if not all(map(math.isfinite, values)):
        raise ValueError("policy contains non-finite scores")
    if not moves:
        raise ValueError("legal move list must not be empty")
    indices = [vocabulary.to_native(move) for move in moves]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate legal moves")
    weights = [values[index] for index in indices]
    if kind == "logits":
        peak = max(weights)
        weights = [math.exp(weight - peak) for weight in weights]
    elif min(values) < 0:
        raise ValueError("negative policy probability")
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("no finite positive legal policy mass")
    return {move: weight / total for move, weight in zip(moves, weights)}
