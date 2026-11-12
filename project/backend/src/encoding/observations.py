"""Convert game states to neural-network feature planes.

Pure standard library: importing this module never imports numpy or torch, so
classic search keeps working without any ML dependency. Observations are
``array.array('f')`` buffers (float32, C-contiguous), which numpy and torch can
wrap without copying (``numpy.frombuffer``); ``to_numpy`` is a lazy helper.

The contract below is the one verified for the cc-lite pretrained checkpoints
(docs/neural_mcts/02_CONTRACT.md). It is an *absolute* encoding: the board is
never rotated or color-swapped for the side to move.

Plane layout, shape (16, 10, 9), float32, plane-major then row-major (row 0 is
Black's back rank, the engine's own square order):

====== =========================================
0-6    Red   king, advisor, elephant, horse, rook, cannon, pawn
7-13   Black king, advisor, elephant, horse, rook, cannon, pawn
14     all ones when Red is to move, all zeros when Black is to move
15     all ones when the side to move is in check, else zeros
====== =========================================
"""

from array import array
from collections.abc import Iterable

from ..environment import GameState, Side
from ..environment.rules import RuleValidator

ENCODING_NAME = "absolute_planes16_v1"
NUM_PLANES = 16
BOARD_ROWS = 10
BOARD_COLS = 9
PLANE_SIZE = BOARD_ROWS * BOARD_COLS
OBSERVATION_SHAPE = (NUM_PLANES, BOARD_ROWS, BOARD_COLS)
OBSERVATION_LENGTH = NUM_PLANES * PLANE_SIZE
BLACK_PLANE_OFFSET = 7
PLANE_RED_TO_MOVE = 14
PLANE_SIDE_IN_CHECK = 15

# 'f' is float32 on every CPython platform; fail loudly rather than mis-size.
if array("f").itemsize != 4:  # pragma: no cover
    raise RuntimeError("array('f') is not 32-bit on this platform")

_ONES_PLANE = array("f", [1.0]) * PLANE_SIZE


def piece_plane(piece: int) -> int:
    """Plane index of a non-empty engine piece value (+Red, -Black, 1..7)."""
    if not isinstance(piece, int) or isinstance(piece, bool) or piece == 0 or abs(piece) > 7:
        raise ValueError("piece must be a non-zero engine piece value in [-7, 7]")
    return abs(piece) - 1 + (BLACK_PLANE_OFFSET if piece < 0 else 0)


class ObservationEncoder:
    """Encode ``GameState`` objects into the 16-plane absolute observation."""

    encoding_name = ENCODING_NAME
    shape = OBSERVATION_SHAPE
    length = OBSERVATION_LENGTH

    def __init__(self, validator: RuleValidator | None = None) -> None:
        self._validator = validator or RuleValidator()

    def encode(self, state: GameState, *, in_check: bool | None = None) -> array:
        """Return a fresh float32 buffer of ``OBSERVATION_LENGTH`` values.

        ``in_check`` may be supplied by a caller that already knows whether the
        side to move is in check; it must then agree with the rules engine. When
        omitted it is computed with the rule validator. Only the board and side
        to move are encoded: repetition and no-capture counters are *not* part
        of the observation.
        """
        if not isinstance(state, GameState):
            raise TypeError("state must be a GameState")
        planes = array("f", [0.0]) * OBSERVATION_LENGTH
        for square, piece in enumerate(state.board):
            if piece:
                planes[piece_plane(piece) * PLANE_SIZE + square] = 1.0
        if state.side_to_move is Side.RED:
            start = PLANE_RED_TO_MOVE * PLANE_SIZE
            planes[start:start + PLANE_SIZE] = _ONES_PLANE
        if in_check is None:
            in_check = self._validator.is_in_check(state, state.side_to_move)
        if in_check:
            start = PLANE_SIDE_IN_CHECK * PLANE_SIZE
            planes[start:start + PLANE_SIZE] = _ONES_PLANE
        return planes

    def encode_batch(self, states: Iterable[GameState]) -> array:
        """Concatenate observations: a flat buffer for shape (B, 16, 10, 9)."""
        batch = array("f")
        for state in states:
            batch.extend(self.encode(state))
        return batch


def to_numpy(observation: array):
    """Copy one observation into a writable float32 ndarray of shape (16, 10, 9)."""
    if len(observation) != OBSERVATION_LENGTH:
        raise ValueError(f"expected {OBSERVATION_LENGTH} values, got {len(observation)}")
    return batch_to_numpy(observation)[0]


def batch_to_numpy(observations: array):
    """Copy a flat batch buffer into a writable float32 ndarray (B, 16, 10, 9)."""
    if len(observations) % OBSERVATION_LENGTH:
        raise ValueError(f"length must be a multiple of {OBSERVATION_LENGTH}")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("numpy is required for to_numpy/batch_to_numpy") from exc
    return np.array(np.frombuffer(observations, dtype=np.float32)).reshape(-1, *OBSERVATION_SHAPE)
