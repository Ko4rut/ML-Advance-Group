"""PolicyValueAdapter invariants with FAKE backends (mock tests; no network weights, no torch)."""

import math
import unittest

from src.encoding import ACTION_SPACE_SIZE, CCLITE_VOCABULARY, ObservationEncoder
from src.environment import GameEngine, GameState, Move, Side
from src.networks.pretrained import (
    InvalidModelOutputError,
    PolicyValueAdapter,
)

from .support import (
    ENGINE, FakeBackend, ForbiddenBackend, assert_valid_priors, black_to_move_initial, fake_info,
    golden_nonterminal_states, golden_state, logits_favoring, no_capture_draw_state, repetition_draw_state,
)


def adapter_for(backend):
    return PolicyValueAdapter(backend, fake_info())


class PriorTests(unittest.TestCase):
    def test_priors_cover_exactly_the_legal_moves_for_both_sides(self):
        for state in (GameState.initial(), black_to_move_initial()):
            backend = FakeBackend()
            result = adapter_for(backend).predict(state)
            moves = ENGINE.legal_moves(state)
            assert_valid_priors(self, result.priors, moves)
            self.assertFalse(result.is_terminal)
            self.assertEqual(backend.calls, 1)

    def test_all_zero_logits_give_an_explicit_uniform_distribution(self):
        state = GameState.initial()
        result = adapter_for(FakeBackend()).predict(state)
        n = len(ENGINE.legal_moves(state))
        for prior in result.priors.values():
            self.assertAlmostEqual(prior, 1 / n, places=12)

    def test_policy_bias_reaches_the_priors(self):
        state = GameState.initial()
        moves = ENGINE.legal_moves(state)
        favoured = moves[7]
        result = adapter_for(FakeBackend(logits_favoring(favoured))).predict(state)
        self.assertEqual(max(result.priors, key=result.priors.get), favoured)
        self.assertGreater(result.priors[favoured], 0.8)

    def test_huge_logit_on_an_illegal_action_is_masked(self):
        state = GameState.initial()
        moves = ENGINE.legal_moves(state)
        illegal = Move(0, 1)  # a rook cannot jump onto the horse's square
        self.assertNotIn(illegal, moves)
        logits = [0.0] * ACTION_SPACE_SIZE
        logits[CCLITE_VOCABULARY.to_native(illegal)] = 1e30
        result = adapter_for(FakeBackend(logits)).predict(state)
        self.assertNotIn(illegal, result.priors)
        assert_valid_priors(self, result.priors, moves)
        for prior in result.priors.values():
            self.assertAlmostEqual(prior, 1 / len(moves), places=12)

    def test_priors_follow_the_supplied_move_order_and_are_read_only(self):
        state = GameState.initial()
        moves = list(reversed(ENGINE.legal_moves(state)))
        result = adapter_for(FakeBackend()).predict(state, moves)
        self.assertEqual(list(result.priors), moves)
        with self.assertRaises(TypeError):
            result.priors[moves[0]] = 1.0

    def test_golden_states_all_map_and_normalize(self):
        adapter = adapter_for(FakeBackend([(-1) ** i * (i % 13) / 5 for i in range(ACTION_SPACE_SIZE)]))
        states = golden_nonterminal_states()
        checked = 0
        for _, state in states[::6]:
            moves = ENGINE.legal_moves(state)
            if ENGINE.outcome(state, legal_moves=moves) is None:
                assert_valid_priors(self, adapter.predict(state, moves).priors, moves)
                checked += 1
        self.assertGreater(checked, 50)


class ValueTests(unittest.TestCase):
    def test_value_is_passed_through_unchanged_for_both_sides(self):
        for value in (-1.0, -0.3, 0.0, 0.7, 1.0):
            for state in (GameState.initial(), black_to_move_initial()):
                result = adapter_for(FakeBackend(value=value)).predict(state)
                self.assertEqual(result.value, value, "no colour-dependent sign flip at the adapter")

    def test_the_encoder_tells_the_network_who_moves(self):
        backend = FakeBackend()
        adapter = adapter_for(backend)
        adapter.predict(GameState.initial())
        adapter.predict(black_to_move_initial())
        red_obs, black_obs = backend.observations
        plane14 = lambda obs: set(obs[14 * 90:15 * 90])
        self.assertEqual((plane14(red_obs), plane14(black_obs)), ({1.0}, {0.0}))
        self.assertEqual(red_obs, ObservationEncoder().encode(GameState.initial()))

    def test_integer_values_are_accepted_as_numbers(self):
        self.assertEqual(adapter_for(FakeBackend(value=1)).predict(GameState.initial()).value, 1.0)

    def test_invalid_values_are_rejected(self):
        for bad in (math.nan, math.inf, -math.inf, 1.0000001, -1.5, "0.5", None, True, [0.1]):
            with self.subTest(value=bad), self.assertRaises(InvalidModelOutputError):
                adapter_for(FakeBackend(value=bad)).predict(GameState.initial())


class InvalidPolicyOutputTests(unittest.TestCase):
    def predict_with(self, logits):
        return adapter_for(FakeBackend(logits)).predict(GameState.initial())

    def test_nan_and_infinity_anywhere_are_rejected_even_on_illegal_slots(self):
        for bad in (math.nan, math.inf, -math.inf):
            for index in (0, 8099, CCLITE_VOCABULARY.to_native(Move(64, 67))):
                logits = [0.0] * ACTION_SPACE_SIZE
                logits[index] = bad
                with self.subTest(value=bad, index=index), self.assertRaises(InvalidModelOutputError):
                    self.predict_with(logits)

    def test_all_negative_infinity_means_no_legal_mass_and_is_rejected(self):
        with self.assertRaises(InvalidModelOutputError):
            self.predict_with([-math.inf] * ACTION_SPACE_SIZE)

    def test_wrong_policy_length_is_rejected(self):
        for size in (0, 90, ACTION_SPACE_SIZE - 1, ACTION_SPACE_SIZE + 1):
            with self.subTest(size=size), self.assertRaises(InvalidModelOutputError):
                self.predict_with([0.0] * size)

    def test_a_failing_prediction_does_not_leave_a_fallback_result(self):
        adapter = adapter_for(FakeBackend([math.nan] * ACTION_SPACE_SIZE))
        with self.assertRaises(InvalidModelOutputError):
            adapter.predict(GameState.initial())


class InputValidationTests(unittest.TestCase):
    def test_non_state_and_bad_move_lists_raise(self):
        adapter = adapter_for(FakeBackend())
        state = GameState.initial()
        moves = ENGINE.legal_moves(state)
        with self.assertRaises(TypeError):
            adapter.predict("not a state")
        with self.assertRaises(TypeError):
            adapter.predict(state, [moves[0], (1, 2)])
        with self.assertRaises(ValueError):
            adapter.predict(state, [moves[0], moves[0]])
        self.assertEqual(adapter.backend.calls, 0)

    def test_verify_legal_moves_catches_a_wrong_list(self):
        adapter = adapter_for(FakeBackend())
        state = GameState.initial()
        moves = ENGINE.legal_moves(state)
        with self.assertRaises(ValueError):
            adapter.predict(state, moves[:-1], verify_legal_moves=True)
        with self.assertRaises(ValueError):
            adapter.predict(state, [Move(0, 1)], verify_legal_moves=True)
        with self.assertRaises(ValueError):
            adapter.predict(state, [], verify_legal_moves=True)
        self.assertEqual(adapter.predict(state, moves, verify_legal_moves=True).priors.keys(), set(moves))
        self.assertEqual(adapter.backend.calls, 1)

    def test_an_iterator_of_moves_is_accepted(self):
        state = GameState.initial()
        result = adapter_for(FakeBackend()).predict(state, iter(ENGINE.legal_moves(state)))
        assert_valid_priors(self, result.priors, ENGINE.legal_moves(state))

    def test_construction_rejects_a_model_with_a_different_contract(self):
        with self.assertRaises(InvalidModelOutputError):
            PolicyValueAdapter(FakeBackend(), fake_info(observation="other_planes_v9"))
        with self.assertRaises(InvalidModelOutputError):
            PolicyValueAdapter(FakeBackend(), fake_info(action_vocabulary="ccz_2086"))


class TerminalPrecedenceTests(unittest.TestCase):
    def terminal_states(self):
        return {
            "checkmated_red": golden_state("fixture/checkmated_red"),
            "checkmated_black": golden_state("fixture/checkmated_black"),
            "stalemate": golden_state("special/stalemate_black"),
            "repetition": repetition_draw_state(),
            "no_capture": no_capture_draw_state(),
        }

    def test_engine_outcome_wins_and_the_network_is_not_called(self):
        adapter = PolicyValueAdapter(ForbiddenBackend(), fake_info())
        for name, state in self.terminal_states().items():
            with self.subTest(name):
                expected = ENGINE.outcome(state)
                self.assertIsNotNone(expected)
                result = adapter.predict(state)
                self.assertTrue(result.is_terminal)
                self.assertEqual(result.terminal, expected)
                self.assertEqual(dict(result.priors), {}, "no policy is invented for a finished game")
                self.assertEqual(result.value, float(expected.reward(state.side_to_move)))
        self.assertEqual(adapter.inference_calls, 0)
        self.assertEqual(adapter.terminal_shortcuts, 5)

    def test_values_follow_the_side_to_move(self):
        adapter = PolicyValueAdapter(ForbiddenBackend(), fake_info())
        for name in ("fixture/checkmated_red", "fixture/checkmated_black", "special/stalemate_black"):
            state = golden_state(name)
            self.assertEqual(adapter.predict(state).value, -1.0, "the side with no legal move has lost")
        self.assertEqual(adapter.predict(repetition_draw_state()).value, 0.0)
        self.assertEqual(adapter.predict(no_capture_draw_state()).value, 0.0)

    def test_terminal_with_caller_supplied_moves(self):
        adapter = PolicyValueAdapter(ForbiddenBackend(), fake_info())
        mated = golden_state("fixture/checkmated_red")
        self.assertEqual(adapter.predict(mated, []).value, -1.0)
        draw = repetition_draw_state()  # has legal moves, but the engine ends the game
        result = adapter.predict(draw, ENGINE.legal_moves(draw))
        self.assertEqual((result.value, result.terminal.reason), (0.0, "repetition"))

    def test_an_absurd_network_value_cannot_override_a_terminal_state(self):
        adapter = adapter_for(FakeBackend(value=0.99))
        self.assertEqual(adapter.predict(golden_state("fixture/checkmated_black")).value, -1.0)
        self.assertEqual(adapter.backend.calls, 0)

    def test_a_mate_in_one_position_is_not_terminal_and_does_call_the_network(self):
        backend = FakeBackend()
        result = adapter_for(backend).predict(golden_state("fixture/mate_in_one_black"))
        self.assertFalse(result.is_terminal)
        self.assertEqual(backend.calls, 1)


class RepeatabilityTests(unittest.TestCase):
    def test_identical_inputs_give_identical_results_and_the_state_is_untouched(self):
        state = golden_state("fixture/mate_in_one_red")
        before = (state.board, state.side_to_move, state.history, state.no_capture_count)
        adapter = adapter_for(FakeBackend(logits_favoring(Move(64, 67), 3.0), value=-0.5))
        first, second = adapter.predict(state), adapter.predict(state)
        self.assertEqual((dict(first.priors), first.value), (dict(second.priors), second.value))
        self.assertEqual(before, (state.board, state.side_to_move, state.history, state.no_capture_count))
        self.assertEqual(adapter.inference_calls, 2)
