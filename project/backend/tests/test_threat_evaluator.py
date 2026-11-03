"""Regression tests for a restrained opening bonus and threat scoring."""

import unittest

from src.agents.position_evaluator import (
    DEVELOPMENT_BONUS,
    PIECE_VALUES,
    evaluate_position,
)
from src.agents.threat_evaluator import attacks_square, threat_score
from src.environment import GameState, Side


def board_from(pieces):
    board = [0] * 90
    for (row, column), piece in pieces.items():
        board[9 * row + column] = piece
    return tuple(board)


class ThreatEvaluatorTests(unittest.TestCase):
    def test_horse_bonus_is_small_and_piece_values_unchanged(self):
        self.assertEqual(DEVELOPMENT_BONUS[4], 4)
        self.assertEqual(PIECE_VALUES[4], 40)
        self.assertEqual(PIECE_VALUES[5], 90)
        self.assertEqual(PIECE_VALUES[6], 45)

    def test_horse_leg_blocked(self):
        board = board_from({(4, 4): -4, (5, 4): -7, (6, 5): 6})
        self.assertFalse(attacks_square(board, 4 * 9 + 4, 6 * 9 + 5))
        open_board = board_from({(4, 4): -4, (6, 5): 6})
        self.assertTrue(attacks_square(open_board, 4 * 9 + 4, 6 * 9 + 5))

    def test_cannon_requires_exactly_one_screen(self):
        target = 6 * 9 + 4
        source = 2 * 9 + 4
        no_screen = board_from({(2, 4): -6, (6, 4): 4})
        one_screen = board_from({(2, 4): -6, (4, 4): 7, (6, 4): 4})
        two_screens = board_from({
            (2, 4): -6, (3, 4): 7, (4, 4): -7, (6, 4): 4,
        })
        self.assertFalse(attacks_square(no_screen, source, target))
        self.assertTrue(attacks_square(one_screen, source, target))
        self.assertFalse(attacks_square(two_screens, source, target))

    def test_unprotected_cannon_is_more_at_risk_on_attackers_turn(self):
        board = board_from({
            (0, 4): -1, (9, 4): 1,
            (4, 0): -5, (4, 3): 6,
        })
        # The opposing rook has an open lane to the cannon.
        self.assertLess(threat_score(board, int(Side.BLACK)),
                        threat_score(board, int(Side.RED)))

    def test_defended_cannon_is_safer_against_expensive_rook(self):
        exposed = board_from({
            (0, 4): -1, (9, 4): 1,
            (4, 0): -5, (4, 3): 6,
        })
        protected = board_from({
            (0, 4): -1, (9, 4): 1,
            (4, 0): -5, (4, 3): 6, (5, 3): 5,
        })
        self.assertGreater(threat_score(protected, -1),
                           threat_score(exposed, -1))

    def test_low_value_attacker_still_penalizes_defended_rook(self):
        board = board_from({
            (0, 4): -1, (9, 4): 1,
            (4, 3): -4, (5, 3): 7, (6, 4): 5, (6, 5): 6,
        })
        # Horse at (4,3) attacks rook at (6,4), not blocked.
        self.assertTrue(attacks_square(board, 4 * 9 + 3, 6 * 9 + 4))
        self.assertLess(threat_score(board, -1), 0)

    def test_initial_position_remains_balanced(self):
        initial = GameState.initial()
        self.assertEqual(evaluate_position(initial, Side.RED), 0)
        self.assertEqual(evaluate_position(initial, Side.BLACK), 0)

    def test_color_and_turn_mirror_symmetry(self):
        original = board_from({
            (0, 4): -1, (9, 4): 1,
            (4, 0): -5, (4, 3): 6,
        })
        # Rotate board 180 degrees and swap sides.
        mirrored = tuple(-piece for piece in reversed(original))
        self.assertEqual(threat_score(original, -1),
                         -threat_score(mirrored, 1))


if __name__ == "__main__":
    unittest.main()
