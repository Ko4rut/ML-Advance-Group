"""Parity tests against pre-optimization Xiangqi rules and Alpha-Beta search."""

import unittest

from src.agents import AlphaBetaAgent
from src.environment import GameEngine, GameState, Side
from src.evaluation.compare_alphabeta import (
    OriginalAlphaBeta, OriginalGameEngine, sample_positions,
)


class SearchOptimizationParityTests(unittest.TestCase):
    def test_legal_moves_and_terminal_outcomes_match_reference(self):
        before = OriginalGameEngine()
        after = GameEngine()
        for state in sample_positions(8, seed=41, spacing=7):
            self.assertEqual(before.legal_moves(state), after.legal_moves(state))
            self.assertEqual(before.outcome(state), after.outcome(state))
            self.assertEqual(before.validator.is_in_check(state, Side.RED),
                             after.validator.is_in_check(state, Side.RED))
            self.assertEqual(before.validator.is_in_check(state, Side.BLACK),
                             after.validator.is_in_check(state, Side.BLACK))

    def test_alphabeta_depth2_selects_same_move(self):
        before = OriginalAlphaBeta(OriginalGameEngine(), 2)
        after = AlphaBetaAgent(GameEngine(), 2)
        for state in sample_positions(4, seed=42, spacing=5):
            original_move, _ = before.select_action(state)
            new_move = after.select_action(state).move
            self.assertEqual(original_move, new_move)

    def test_game_state_transition_stays_immutable(self):
        engine = GameEngine()
        state = GameState.initial()
        original_board = state.board
        move = engine.legal_moves(state)[0]
        updated = state.moved(move)
        self.assertEqual(state.board, original_board)
        self.assertEqual(len(updated.history), len(state.history) + 1)
        self.assertEqual(updated.history[-1], updated.position_key())
        self.assertEqual(updated.ply_count, state.ply_count + 1)


if __name__ == "__main__":
    unittest.main()
