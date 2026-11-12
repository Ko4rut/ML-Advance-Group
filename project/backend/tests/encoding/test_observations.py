import subprocess
import sys
import unittest
from array import array
from dataclasses import replace
from pathlib import Path

from src.encoding import (
    OBSERVATION_LENGTH, OBSERVATION_SHAPE, ObservationEncoder, batch_to_numpy, to_numpy,
)
from src.encoding.observations import piece_plane
from src.environment import GameEngine, GameState, Move, Side

BACKEND = Path(__file__).resolve().parents[2]

try:
    import numpy as np
except ImportError:  # numpy is an optional dependency of the backend
    np = None


def position(pieces, side=Side.RED):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side)


def color_swap(state):
    """Rotate 180 degrees and exchange colors and side to move."""
    return GameState(tuple(-p for p in reversed(state.board)), state.side_to_move.opponent)


def plane(obs, index):
    return obs[index * 90:(index + 1) * 90]


class ObservationContractTests(unittest.TestCase):
    def setUp(self):
        self.encoder = ObservationEncoder()
        self.engine = GameEngine()
        self.initial = GameState.initial()

    def test_type_length_and_shape_constants(self):
        obs = self.encoder.encode(self.initial)
        self.assertIsInstance(obs, array)
        self.assertEqual(obs.typecode, "f")
        self.assertEqual(obs.itemsize, 4)
        self.assertEqual(len(obs), OBSERVATION_LENGTH)
        self.assertEqual(OBSERVATION_SHAPE, (16, 10, 9))
        self.assertEqual(OBSERVATION_LENGTH, 16 * 10 * 9)

    def test_initial_position_exact_planes(self):
        obs = self.encoder.encode(self.initial)
        at = lambda channel, row, col: obs[channel * 90 + row * 9 + col]
        self.assertEqual(at(0, 9, 4), 1)    # Red king
        self.assertEqual(at(1, 9, 3), 1)    # Red advisor
        self.assertEqual(at(3, 9, 1), 1)    # Red horse
        self.assertEqual(at(4, 9, 0), 1)    # Red rook
        self.assertEqual(at(5, 7, 1), 1)    # Red cannon
        self.assertEqual(at(6, 6, 0), 1)    # Red pawn
        self.assertEqual(at(7, 0, 4), 1)    # Black king
        self.assertEqual(at(10, 0, 1), 1)   # Black horse
        self.assertEqual(at(13, 3, 8), 1)   # Black pawn
        self.assertEqual(sum(obs[:14 * 90]), 32)
        self.assertEqual(sum(plane(obs, 14)), 90)
        self.assertEqual(sum(plane(obs, 15)), 0)
        self.assertTrue(set(obs) <= {0.0, 1.0})

    def test_per_plane_piece_counts_match_board(self):
        obs = self.encoder.encode(self.initial)
        expected = [1, 2, 2, 2, 2, 2, 5]
        for offset in (0, 7):
            self.assertEqual([sum(plane(obs, offset + i)) for i in range(7)], expected)

    def test_black_to_move_keeps_absolute_board_and_clears_turn_plane(self):
        red = self.encoder.encode(self.initial)
        black = self.encoder.encode(GameState(self.initial.board, Side.BLACK))
        self.assertEqual(red[:14 * 90], black[:14 * 90])
        self.assertEqual(sum(plane(black, 14)), 0)

    def test_color_rotation_equals_channel_exchange(self):
        state = self.initial
        for _ in range(3):
            state = state.moved(self.engine.legal_moves(state)[0])
        own, swapped = self.encoder.encode(state), self.encoder.encode(color_swap(state))
        for i in range(7):
            for src, dst in ((i, 7 + i), (7 + i, i)):
                a = plane(own, src)
                self.assertEqual(list(a)[::-1], list(plane(swapped, dst)))
        self.assertEqual([1 - v for v in plane(own, 14)], list(plane(swapped, 14)))

    def test_check_plane_both_sides_and_override(self):
        red_in_check = position({(0, 3): -1, (9, 4): 1, (4, 4): -5}, Side.RED)
        black_in_check = color_swap(red_in_check)
        for state in (red_in_check, black_in_check):
            self.assertEqual(sum(plane(self.encoder.encode(state), 15)), 90)
        calm = self.initial
        self.assertEqual(sum(plane(self.encoder.encode(calm), 15)), 0)
        # An explicit caller-supplied flag is honored (caller's responsibility to be right).
        self.assertEqual(sum(plane(self.encoder.encode(calm, in_check=True), 15)), 90)
        self.assertEqual(sum(plane(self.encoder.encode(red_in_check, in_check=False), 15)), 0)

    def test_counters_and_history_are_not_encoded(self):
        base = self.initial
        variant = replace(base, no_capture_count=57, ply_count=33, history=(base.position_key(),) * 2)
        self.assertEqual(self.encoder.encode(base), self.encoder.encode(variant))

    def test_deterministic_and_fresh_buffers(self):
        a, b = self.encoder.encode(self.initial), self.encoder.encode(self.initial)
        self.assertEqual(a, b)
        self.assertIsNot(a, b)
        a[0] = 9.0  # mutating one result must not affect another
        self.assertEqual(self.encoder.encode(self.initial)[0], 0.0)

    def test_encoding_does_not_mutate_state(self):
        state = self.initial
        board = state.board
        self.encoder.encode(state)
        self.assertEqual(state.board, board)

    def test_batch_is_concatenation_and_empty_batch_is_empty(self):
        s1 = self.initial
        s2 = s1.moved(self.engine.legal_moves(s1)[0])
        batch = self.encoder.encode_batch([s1, s2])
        self.assertEqual(len(batch), 2 * OBSERVATION_LENGTH)
        self.assertEqual(batch[:OBSERVATION_LENGTH], self.encoder.encode(s1))
        self.assertEqual(batch[OBSERVATION_LENGTH:], self.encoder.encode(s2))
        self.assertEqual(len(self.encoder.encode_batch([])), 0)

    def test_rejects_non_state(self):
        for bad in (None, [0] * 90, "state"):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                self.encoder.encode(bad)

    def test_piece_plane_validation(self):
        self.assertEqual(piece_plane(1), 0)
        self.assertEqual(piece_plane(7), 6)
        self.assertEqual(piece_plane(-1), 7)
        self.assertEqual(piece_plane(-7), 13)
        for bad in (0, 8, -8, True, 1.0, "1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                piece_plane(bad)

    def test_importing_encoding_does_not_import_numpy_or_torch(self):
        code = ("import sys; import src.encoding; "
                "assert 'numpy' not in sys.modules and 'torch' not in sys.modules, "
                "sorted(m for m in sys.modules if m in ('numpy','torch'))")
        result = subprocess.run([sys.executable, "-c", code], cwd=BACKEND,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipIf(np is None, "numpy not installed; optional conversion helpers not exercised")
class NumpyConversionTests(unittest.TestCase):
    def test_single_and_batch_conversion(self):
        encoder = ObservationEncoder()
        state = GameState.initial()
        obs = encoder.encode(state)
        single = to_numpy(obs)
        self.assertEqual(single.shape, (16, 10, 9))
        self.assertEqual(single.dtype, np.float32)
        self.assertTrue(single.flags.writeable)
        self.assertEqual(single[0, 9, 4], 1.0)
        self.assertEqual(single.sum(), sum(obs))
        batch = batch_to_numpy(encoder.encode_batch([state, state]))
        self.assertEqual(batch.shape, (2, 16, 10, 9))
        np.testing.assert_array_equal(batch[0], single)
        single[0, 9, 4] = 5  # a copy: the source buffer is untouched
        self.assertEqual(obs[9 * 9 + 4], 1.0)

    def test_bad_lengths_rejected(self):
        with self.assertRaises(ValueError):
            to_numpy(array("f", [0.0] * 10))
        with self.assertRaises(ValueError):
            batch_to_numpy(array("f", [0.0] * (OBSERVATION_LENGTH + 1)))


if __name__ == "__main__":
    unittest.main()
