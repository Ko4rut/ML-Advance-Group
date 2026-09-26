import copy
import tempfile
import unittest
from pathlib import Path

from xiangqi_ai.agents.random_agent import RandomAgent
from xiangqi_ai.encoding.actions import decode_move, encode_move, legal_action_mask
from xiangqi_ai.environment.engine import RulesConfig, apply_move, outcome
from xiangqi_ai.environment.env import XiangqiEnv
from xiangqi_ai.environment.movegen import legal_moves
from xiangqi_ai.environment.rules import is_in_check, targets
from xiangqi_ai.environment.state import GameState
from xiangqi_ai.environment.types import Move, Side
from xiangqi_ai.evaluation.match import play_match
from xiangqi_ai.storage.game_record import load_record, replay, save_record


def position(pieces, side=Side.RED, **kwargs):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side, **kwargs)


def geometry(pieces, square):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return set(targets(tuple(board), square[0] * 9 + square[1]))


class RulesTests(unittest.TestCase):
    def test_initial_position_and_move_count(self):
        state = GameState.initial()
        self.assertEqual(sum(p != 0 for p in state.board), 32)
        self.assertEqual(len(legal_moves(state)), 44)
        self.assertFalse(is_in_check(state.board, Side.RED))
        self.assertFalse(is_in_check(state.board, Side.BLACK))

    def test_horse_leg(self):
        moves = geometry({(7, 4): 4, (6, 4): 7}, (7, 4))
        self.assertNotIn(5 * 9 + 3, moves)
        self.assertNotIn(5 * 9 + 5, moves)
        self.assertIn(6 * 9 + 2, moves)

    def test_elephant_eye_and_river(self):
        moves = geometry({(5, 4): 3, (6, 5): 7}, (5, 4))
        self.assertEqual(moves, {7 * 9 + 2})
        black = geometry({(4, 4): -3, (3, 5): -7}, (4, 4))
        self.assertEqual(black, {2 * 9 + 2})

    def test_cannon_needs_exactly_one_screen(self):
        pieces = {(7, 1): 6, (5, 1): 7, (2, 1): -5, (0, 1): -4, (7, 4): -5}
        moves = geometry(pieces, (7, 1))
        self.assertIn(6 * 9 + 1, moves)
        self.assertIn(2 * 9 + 1, moves)
        for square in (5 * 9 + 1, 4 * 9 + 1, 1, 7 * 9 + 4):
            self.assertNotIn(square, moves)

    def test_rook_stops_at_first_piece(self):
        moves = geometry({(5, 4): 5, (5, 6): -7, (3, 4): 7}, (5, 4))
        self.assertIn(5 * 9 + 6, moves)
        self.assertNotIn(5 * 9 + 7, moves)
        self.assertNotIn(3 * 9 + 4, moves)
        self.assertNotIn(2 * 9 + 4, moves)

    def test_palace_bounds(self):
        self.assertEqual(geometry({(9, 3): 1}, (9, 3)), {8 * 9 + 3, 9 * 9 + 4})
        self.assertEqual(geometry({(7, 3): 2}, (7, 3)), {8 * 9 + 4})
        self.assertEqual(geometry({(0, 5): -1}, (0, 5)), {4, 14})

    def test_pawns_both_sides(self):
        self.assertEqual(geometry({(5, 4): 7}, (5, 4)), {4 * 9 + 4})
        self.assertEqual(geometry({(4, 4): 7}, (4, 4)), {3 * 9 + 4, 4 * 9 + 3, 4 * 9 + 5})
        self.assertEqual(geometry({(4, 4): -7}, (4, 4)), {5 * 9 + 4})
        self.assertEqual(geometry({(5, 4): -7}, (5, 4)), {6 * 9 + 4, 5 * 9 + 3, 5 * 9 + 5})

    def test_facing_kings_and_pin(self):
        state = position({(0, 4): -1, (9, 4): 1, (5, 4): 5})
        self.assertNotIn(Move(49, 48), legal_moves(state))
        self.assertIn(Move(49, 40), legal_moves(state))
        exposed = position({(0, 4): -1, (9, 4): 1})
        self.assertTrue(is_in_check(exposed.board, Side.RED))
        self.assertTrue(is_in_check(exposed.board, Side.BLACK))

    def test_check_requires_evasion(self):
        state = position({(0, 3): -1, (9, 4): 1, (5, 4): -5, (7, 0): 5})
        self.assertTrue(is_in_check(state.board, Side.RED))
        self.assertNotIn(Move(63, 64), legal_moves(state))
        self.assertIn(Move(63, 67), legal_moves(state))

    def test_mate_and_stalemate_are_losses(self):
        common = {(0, 4): -1, (9, 3): 1, (2, 3): 5, (2, 5): 5}
        mate = position({**common, (1, 4): 5}, Side.BLACK)
        stalemate = position({**common, (1, 0): 5}, Side.BLACK)
        self.assertEqual(outcome(mate).reason, "checkmate")
        self.assertEqual(outcome(stalemate).reason, "stalemate")
        self.assertEqual(outcome(stalemate).winner, Side.RED)
        with self.assertRaises(ValueError):
            apply_move(mate, Move(4, 13))


class StateAndIntegrationTests(unittest.TestCase):
    def test_capture_counters_and_immutability(self):
        state = position({(0, 3): -1, (9, 4): 1, (5, 0): 5, (4, 0): -7},
                         no_capture_count=10)
        child = apply_move(state, Move(45, 36))
        self.assertEqual(child.no_capture_count, 0)
        self.assertEqual(child.side_to_move, Side.BLACK)
        self.assertEqual(child.ply_count, 1)
        self.assertEqual(state.board[45], 5)
        self.assertEqual(child.board[45], 0)
        self.assertEqual(len(child.history), len(state.history) + 1)

    def test_clone_is_safe_and_illegal_move_does_not_mutate(self):
        env = XiangqiEnv()
        original = env.clone_state()
        with self.assertRaises(ValueError):
            env.step(Move(54, 55))
        self.assertEqual(env.state, original)
        env.step(Move(54, 45))
        self.assertEqual(original, GameState.initial())
        self.assertNotEqual(env.state, original)

    def test_repetition(self):
        state = GameState.initial()
        for move in [Move(82, 65), Move(1, 20), Move(65, 82), Move(20, 1)] * 2:
            state = apply_move(state, move)
        self.assertEqual(outcome(state).reason, "repetition")
        self.assertFalse(any(legal_action_mask(state)))

    def test_no_capture_rule_and_reward(self):
        env = XiangqiEnv(RulesConfig(no_capture_limit=1))
        result = env.step(Move(54, 45))
        self.assertTrue(result.terminated)
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.result.reason, "no_capture_limit")
        self.assertEqual(env.get_legal_actions(), [])
        env.reset(position({(0, 4): -1, (9, 3): 1, (2, 3): 5,
                            (2, 5): 5, (2, 4): 5}))
        result = env.step(Move(22, 13))
        self.assertEqual(result.reward, 1)
        self.assertEqual(result.result.winner, Side.RED)

    def test_encoding_mask(self):
        state = GameState.initial()
        moves = legal_moves(state)
        mask = legal_action_mask(state)
        self.assertEqual(len(mask), 8100)
        self.assertEqual(sum(mask), len(moves))
        for move in moves:
            self.assertEqual(decode_move(encode_move(move)), move)
            self.assertTrue(mask[encode_move(move)])
        for bad in (-1, 8100, 0):
            with self.assertRaises(ValueError):
                decode_move(bad)

    def test_match_seed_truncation_and_replay(self):
        def run():
            return play_match(RandomAgent(10), RandomAgent(11), max_plies=20, seed=10)
        record, again = run(), run()
        self.assertEqual([m["action"] for m in record["moves"]],
                         [m["action"] for m in again["moves"]])
        self.assertTrue(record["truncated"])
        self.assertFalse(record["terminated"])
        states = replay(record)
        for before, after in zip(states, states[1:]):
            self.assertFalse(is_in_check(after.board, before.side_to_move))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.json"
            save_record(record, path)
            self.assertEqual(replay(load_record(path)), states)
        corrupt = copy.deepcopy(record)
        corrupt["final_board"][0] = 0
        with self.assertRaises(ValueError):
            replay(corrupt)
        corrupt = copy.deepcopy(record)
        corrupt["winner"] = 1
        with self.assertRaises(ValueError):
            replay(corrupt)

    def test_completed_match_replay(self):
        record = play_match(RandomAgent(1), RandomAgent(2), max_plies=10,
                            rules=RulesConfig(no_capture_limit=1))
        self.assertTrue(record["terminated"])
        self.assertFalse(record["truncated"])
        self.assertEqual(len(replay(record)), 2)


if __name__ == "__main__":
    unittest.main()
