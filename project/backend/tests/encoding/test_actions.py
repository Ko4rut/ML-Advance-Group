import importlib
import math
import unittest
from array import array

from src.encoding import (
    ACTION_SPACE_SIZE, CCLITE_VOCABULARY, ActionEncoder, CCLiteVocabulary, legal_priors,
)
from src.environment import GameEngine, GameState, Move, Side

try:
    import numpy as np
except ImportError:
    np = None


class ActionEncoderTests(unittest.TestCase):
    def setUp(self):
        self.codec = ActionEncoder()
        self.engine = GameEngine()
        self.moves = self.engine.legal_moves(GameState.initial())

    def test_space_size_matches_gold_contract(self):
        gold = importlib.import_module("src.data.03_gold.models")
        self.assertEqual(ACTION_SPACE_SIZE, gold.ACTION_SPACE_SIZE)
        self.assertEqual(ACTION_SPACE_SIZE, 8100)
        self.assertEqual(self.codec.encoding_name, "from_to_8100_v1")

    def test_encoding_formula_is_source_times_90_plus_target(self):
        for move in self.moves:
            self.assertEqual(self.codec.encode(move), move.source * 90 + move.target)

    def test_every_index_round_trips(self):
        for index in range(ACTION_SPACE_SIZE):
            self.assertEqual(self.codec.encode(self.codec.decode(index)), index)

    def test_unknown_ids_and_wrong_types_rejected(self):
        for bad in (-1, 8100, 10 ** 9, True, 2.5, "3041", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.codec.decode(bad)
        for bad in ((1, 2), None, 41):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                self.codec.encode(bad)

    def test_legal_ids_sorted_unique_and_duplicates_rejected(self):
        ids = self.codec.legal_action_ids(self.moves)
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(self.moves))
        with self.assertRaises(ValueError):
            self.codec.legal_action_ids([self.moves[0], self.moves[0]])

    def test_mask_is_true_exactly_on_legal_moves(self):
        mask = self.codec.legal_mask(self.moves)
        self.assertEqual(len(mask), ACTION_SPACE_SIZE)
        self.assertEqual(sum(mask), len(self.moves))
        self.assertEqual({i for i, bit in enumerate(mask) if bit},
                         {self.codec.encode(m) for m in self.moves})
        self.assertFalse(mask[self.codec.encode(Move(0, 0))])  # impossible move
        self.assertEqual(sum(self.codec.legal_mask([])), 0)


class VocabularyTests(unittest.TestCase):
    def test_native_vocabulary_is_separate_but_verified_identical_to_internal_ids(self):
        codec, vocab = ActionEncoder(), CCLITE_VOCABULARY
        self.assertIsInstance(vocab, CCLiteVocabulary)
        self.assertEqual(vocab.size, 8100)
        for index in range(8100):
            move = vocab.from_native(index)
            self.assertEqual(move, codec.decode(index))
            self.assertEqual(vocab.to_native(move), index)

    def test_bad_native_indices_rejected(self):
        for bad in (-1, 8100, True, 1.5, "7"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                CCLITE_VOCABULARY.from_native(bad)
        with self.assertRaises(TypeError):
            CCLITE_VOCABULARY.to_native((3, 4))

    def test_transposed_mapping_would_be_detected(self):
        """target*90+source is not the contract: it picks different legal slots."""
        moves = GameEngine().legal_moves(GameState.initial())
        correct = {m.source * 90 + m.target for m in moves}
        transposed = {m.target * 90 + m.source for m in moves}
        self.assertNotEqual(correct, transposed)


class LegalPriorsTests(unittest.TestCase):
    def setUp(self):
        self.moves = GameEngine().legal_moves(GameState.initial())

    def test_zero_logits_give_uniform_distribution_over_legal_moves(self):
        priors = legal_priors([0.0] * 8100, self.moves)
        self.assertEqual(set(priors), set(self.moves))
        self.assertAlmostEqual(math.fsum(priors.values()), 1.0, places=12)
        self.assertTrue(all(abs(p - 1 / len(self.moves)) < 1e-15 for p in priors.values()))

    def test_priors_follow_softmax_of_legal_logits_only(self):
        scores = [0.0] * 8100
        a, b = self.moves[0], self.moves[1]
        scores[a.source * 90 + a.target] = math.log(3)
        scores[b.source * 90 + b.target] = math.log(1)
        priors = legal_priors(scores, self.moves)
        rest = len(self.moves) - 2
        total = 3 + 1 + rest
        self.assertAlmostEqual(priors[a], 3 / total, places=12)
        self.assertAlmostEqual(priors[b], 1 / total, places=12)

    def test_huge_logit_on_illegal_action_has_no_effect(self):
        scores = [0.0] * 8100
        scores[0] = 1e100  # Move(0, 0): not a legal move
        self.assertEqual(legal_priors(scores, self.moves), legal_priors([0.0] * 8100, self.moves))
        self.assertNotIn(Move(0, 0), legal_priors(scores, self.moves))

    def test_extreme_legal_logits_are_numerically_stable(self):
        scores = [-1e6] * 8100
        winner = self.moves[5]
        scores[winner.source * 90 + winner.target] = 1e6
        priors = legal_priors(scores, self.moves)
        self.assertEqual(priors[winner], 1.0)
        self.assertAlmostEqual(math.fsum(priors.values()), 1.0, places=12)

    def test_nonfinite_anywhere_is_rejected_even_on_illegal_slots(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            for slot in (0, self.moves[0].source * 90 + self.moves[0].target):
                scores = [0.0] * 8100
                scores[slot] = bad
                with self.subTest(bad=bad, slot=slot), self.assertRaises(ValueError):
                    legal_priors(scores, self.moves)

    def test_probabilities_are_renormalized_over_legal_moves(self):
        priors = legal_priors([1.0] * 8100, self.moves, kind="probabilities")
        self.assertAlmostEqual(math.fsum(priors.values()), 1.0, places=12)
        self.assertEqual(set(priors), set(self.moves))

    def test_zero_legal_probability_mass_is_an_error_not_a_fallback(self):
        with self.assertRaises(ValueError):
            legal_priors([0.0] * 8100, self.moves, kind="probabilities")
        scores = [0.0] * 8100
        scores[0] = 1.0  # all mass on an illegal slot
        with self.assertRaises(ValueError):
            legal_priors(scores, self.moves, kind="probabilities")

    def test_negative_probability_and_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            legal_priors([-1.0] * 8100, self.moves, kind="probabilities")
        with self.assertRaises(ValueError):
            legal_priors([0.0] * 8100, self.moves, kind="softmax")

    def test_bad_shape_empty_or_duplicate_moves_rejected(self):
        for scores, moves in (([0.0] * 2086, self.moves), ([0.0] * 8101, self.moves),
                              ([0.0] * 8100, []), ([0.0] * 8100, [self.moves[0]] * 2)):
            with self.subTest(n=len(scores), m=len(moves)), self.assertRaises(ValueError):
                legal_priors(scores, moves)

    def test_array_input_accepted(self):
        priors = legal_priors(array("f", [0.0]) * 8100, self.moves)
        self.assertAlmostEqual(math.fsum(priors.values()), 1.0, places=12)

    @unittest.skipIf(np is None, "numpy not installed")
    def test_numpy_input_matches_list_input(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=8100).astype(np.float32)
        from_np = legal_priors(scores, self.moves)
        from_list = legal_priors([float(x) for x in scores], self.moves)
        for move in self.moves:
            self.assertAlmostEqual(from_np[move], from_list[move], places=12)


if __name__ == "__main__":
    unittest.main()
