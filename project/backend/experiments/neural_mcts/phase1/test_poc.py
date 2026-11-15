"""Phase 1 experimental contract tests and explicit real-checkpoint checks."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from contracts import (action_id, color_swap, decode_action, fixtures, legal_priors,
                       observation, predict)
from fetch_artifacts import verify
from src.environment import GameEngine, GameState, Move, Side


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = GameEngine()
        self.state = GameState.initial()
        self.moves = self.engine.legal_moves(self.state)

    def test_initial_planes_and_absolute_piece_order(self):
        planes = observation(self.state, self.engine)
        self.assertEqual(planes.shape, (16, 10, 9))
        self.assertEqual(planes.dtype, np.float32)
        self.assertEqual(planes[0, 9, 4], 1)  # Red king
        self.assertEqual(planes[7, 0, 4], 1)  # Black king
        self.assertEqual(planes[4, 9, 0], 1)  # Red rook
        self.assertEqual(planes[10, 0, 1], 1)  # Black horse
        self.assertEqual(planes[:14].sum(), 32)
        self.assertTrue(planes[14].all())
        self.assertFalse(planes[15].any())

    def test_black_turn_plane_does_not_rotate_the_absolute_board(self):
        red = observation(self.state, self.engine)
        black = observation(GameState(self.state.board, Side.BLACK), self.engine)
        np.testing.assert_array_equal(red[:14], black[:14])
        self.assertFalse(black[14].any())

    def test_color_rotation_matches_channel_exchange(self):
        state = fixtures()["asymmetric_black"]
        own = observation(state, self.engine)
        swapped = observation(color_swap(state), self.engine)
        np.testing.assert_array_equal(own[:7, ::-1, ::-1], swapped[7:14])
        np.testing.assert_array_equal(own[7:14, ::-1, ::-1], swapped[:7])
        np.testing.assert_array_equal(1 - own[14], swapped[14])

    def test_check_plane_for_both_sides(self):
        for name in ("in_check_red", "in_check_black"):
            with self.subTest(name=name):
                self.assertTrue(observation(fixtures()[name], self.engine)[15].all())

    def test_all_native_indices_round_trip(self):
        for index in range(8100):
            self.assertEqual(action_id(decode_action(index)), index)

    def test_unknown_action_index_rejected(self):
        for index in (-1, 8100, True, 2.5, "3041"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                decode_action(index)

    def test_zero_logits_yield_uniform_legal_policy(self):
        priors = legal_priors(np.zeros(8100), self.moves)
        self.assertEqual(set(priors), set(self.moves))
        self.assertAlmostEqual(sum(priors.values()), 1)
        self.assertTrue(all(p == 1 / len(self.moves) for p in priors.values()))

    def test_illegal_large_logit_has_no_effect(self):
        scores = np.zeros(8100)
        scores[0] = 1e100  # 0 -> 0 is not a legal move.
        priors = legal_priors(scores, self.moves)
        self.assertNotIn(Move(0, 0), priors)
        self.assertEqual(priors, legal_priors(np.zeros(8100), self.moves))

    def test_extreme_legal_logits_are_stable(self):
        scores = np.full(8100, -1e6)
        scores[action_id(self.moves[0])] = 1e6
        priors = legal_priors(scores, self.moves)
        self.assertEqual(priors[self.moves[0]], 1)
        self.assertEqual(sum(priors.values()), 1)

    def test_nonfinite_policy_is_rejected(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            scores = np.zeros(8100)
            scores[0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                legal_priors(scores, self.moves)

    def test_zero_probability_mass_is_an_error_not_a_hidden_fallback(self):
        with self.assertRaises(ValueError):
            legal_priors(np.zeros(8100), self.moves, kind="probabilities")

    def test_probability_input_is_normalized_over_legal_moves(self):
        priors = legal_priors(np.ones(8100), self.moves, kind="probabilities")
        self.assertAlmostEqual(sum(priors.values()), 1)
        self.assertEqual(set(priors), set(self.moves))

    def test_negative_probabilities_and_bad_representations_rejected(self):
        with self.assertRaises(ValueError):
            legal_priors(-np.ones(8100), self.moves, kind="probabilities")
        with self.assertRaises(ValueError):
            legal_priors(np.ones(8100), self.moves, kind="unknown")

    def test_bad_shape_duplicate_or_empty_legal_moves_rejected(self):
        for scores, moves in ((np.zeros(2086), self.moves), (np.zeros(8100), []),
                              (np.zeros(8100), [self.moves[0]] * 2)):
            with self.subTest(size=len(scores), moves=len(moves)), self.assertRaises(ValueError):
                legal_priors(scores, moves)

    def test_bad_value_rejected(self):
        for value in (float("nan"), float("inf"), 1.01, -1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                predict(self.state, self.engine, lambda _: (np.zeros(8100), value))

    def test_side_to_move_value_is_passed_through_without_color_sign_flip(self):
        for side in (Side.RED, Side.BLACK):
            state = GameState(self.state.board, side)
            _, value = predict(state, self.engine, lambda _: (np.zeros(8100), 0.75))
            self.assertEqual(value, 0.75)

    def test_terminal_results_bypass_network_for_both_colors_and_draws(self):
        def forbidden(_):
            self.fail("Terminal state called the network")
        states = fixtures()
        for name, expected in (("checkmated_red", -1), ("checkmated_black", -1),
                               ("no_capture_draw", 0), ("repetition_draw", 0)):
            with self.subTest(name=name):
                self.assertEqual(predict(states[name], self.engine, forbidden), ({}, expected))

    def test_mate_in_one_exists_for_both_colors(self):
        for name in ("mate_in_one_red", "mate_in_one_black"):
            state = fixtures()[name]
            self.assertIsNone(self.engine.outcome(state))
            self.assertTrue(any((result := self.engine.outcome(state.moved(move))) is not None
                                and result.winner is state.side_to_move
                                for move in self.engine.legal_moves(state)))

    def test_artifact_corruption_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "artifact"
            path.write_bytes(b"verified")
            record = {"bytes": 8, "sha256": hashlib.sha256(b"verified").hexdigest()}
            verify(path, record)
            path.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                verify(path, record)
            path.write_bytes(b"short")
            with self.assertRaises(ValueError):
                verify(path, record)


@unittest.skipUnless(os.environ.get("XIANGQI_PHASE1_SOURCE") and
                     os.environ.get("XIANGQI_PHASE1_CHECKPOINT"),
                     "Real pretrained checkpoint and pinned source paths not provided")
class RealCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qualify import CheckpointProbe
        cls.probe = CheckpointProbe(Path(os.environ["XIANGQI_PHASE1_SOURCE"]),
                                    Path(os.environ["XIANGQI_PHASE1_CHECKPOINT"]))

    def test_actual_weights_match_manifest_and_are_loaded_on_cpu_in_eval_mode(self):
        self.assertFalse(self.probe.model.training)
        self.assertTrue(all(p.device.type == "cpu" for p in self.probe.model.parameters()))
        self.assertEqual(self.probe.model.config.input_planes, 16)
        self.assertEqual(self.probe.model.config.policy_size, 8100)
        self.assertIn("model_state", self.probe.payload)

    def test_all_fixtures_match_source_encoding_policy_value_and_terminal_contract(self):
        engine = GameEngine()
        for name, state in fixtures().items():
            with self.subTest(name=name):
                result = self.probe.check(state, engine)
                self.assertTrue(result["observation_parity"])
                self.assertTrue(result["legal_mapping_parity"])


if __name__ == "__main__":
    unittest.main()
