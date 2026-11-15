"""Parity with the *upstream* cc-lite encoder/codec, via a committed golden fixture.

Expected values were produced by experiments/neural_mcts/phase2/generate_golden.py
from the SHA-256-verified upstream source, never from this repo's production code.
These tests need neither torch nor the upstream source.
"""

import hashlib
import json
import math
import random
import unittest
from pathlib import Path

from src.encoding import ActionEncoder, CCLITE_VOCABULARY, ObservationEncoder, legal_priors
from src.environment import GameEngine, GameState, Move, Side

GOLDEN = Path(__file__).with_name("golden") / "cclite_v1.json"
DATA = json.loads(GOLDEN.read_text())
STATES = [(r["name"], r, GameState(tuple(r["board"]), Side[r["side"]])) for r in DATA["states"]]


class GoldenFixtureTests(unittest.TestCase):
    def test_fixture_provenance_and_coverage(self):
        self.assertEqual(DATA["upstream_revision"], "7d305c0e5adbfae4e23db40997603d8dc86b078b")
        self.assertEqual(DATA["engine_legal_set_mismatches"], [])
        self.assertGreaterEqual(len(STATES), 400)
        self.assertGreaterEqual(sum(len(r["upstream_legal_action_ids"]) for _, r, _ in STATES), 15000)
        sides = {state.side_to_move for _, _, state in STATES}
        self.assertEqual(sides, {Side.RED, Side.BLACK})
        names = {name for name, _, _ in STATES}
        for required in ("special/cannon_capture_over_screen", "special/horse_leg_blocked_north",
                         "special/elephant_eye_blocked", "special/flying_general_pinned_rook",
                         "special/stalemate_black", "special/pawn_after_river_black",
                         "fixture/checkmated_red", "fixture/checkmated_black",
                         "fixture/mate_in_one_red", "fixture/mate_in_one_black"):
            self.assertIn(required, names)
        self.assertTrue(any(not r["upstream_legal_action_ids"] for _, r, _ in STATES),
                        "fixture must include terminal states with no legal move")


class EncoderParityTests(unittest.TestCase):
    def test_observation_bytes_equal_upstream_for_every_state(self):
        encoder = ObservationEncoder()
        for name, record, state in STATES:
            with self.subTest(name=name):
                digest = hashlib.sha256(encoder.encode(state).tobytes()).hexdigest()
                self.assertEqual(digest, record["upstream_observation_sha256"])

    def test_check_plane_equals_upstream_check_flag(self):
        encoder = ObservationEncoder()
        for name, record, state in STATES:
            with self.subTest(name=name):
                plane = encoder.encode(state)[15 * 90:]
                self.assertEqual(plane[0] == 1.0, record["upstream_side_in_check"])

    def test_color_swapped_observation_differs_for_asymmetric_states(self):
        """Guard against a vacuous parity test: color matters in the golden hashes."""
        encoder = ObservationEncoder()
        by_name = {name: (r, s) for name, r, s in STATES}
        record, state = by_name["special/cannon_capture_over_screen"]
        swapped = by_name["special/cannon_capture_over_screen/color_swapped"][0]
        self.assertNotEqual(record["upstream_observation_sha256"], swapped["upstream_observation_sha256"])


class ActionParityTests(unittest.TestCase):
    def test_engine_legal_ids_equal_upstream_legal_ids_for_every_state(self):
        engine, codec = GameEngine(), ActionEncoder()
        for name, record, state in STATES:
            with self.subTest(name=name):
                moves = engine.legal_moves(state)
                self.assertEqual(codec.legal_action_ids(moves), record["upstream_legal_action_ids"])

    def test_every_upstream_legal_index_decodes_to_an_engine_legal_move(self):
        engine = GameEngine()
        mapped = 0
        for name, record, state in STATES:
            legal = set(engine.legal_moves(state))
            decoded = {CCLITE_VOCABULARY.from_native(i) for i in record["upstream_legal_action_ids"]}
            with self.subTest(name=name):
                self.assertEqual(decoded, legal)
            mapped += len(decoded)
        self.assertGreaterEqual(mapped, 15000)

    def test_mask_and_masked_priors_cover_exactly_the_upstream_legal_set(self):
        engine, codec = GameEngine(), ActionEncoder()
        rng = random.Random(0)
        scores = [rng.gauss(0, 3) for _ in range(8100)]
        for name, record, state in STATES:
            moves = engine.legal_moves(state)
            if not moves:
                continue
            with self.subTest(name=name):
                mask = codec.legal_mask(moves)
                self.assertEqual([i for i, bit in enumerate(mask) if bit],
                                 record["upstream_legal_action_ids"])
                priors = legal_priors(scores, moves)
                self.assertEqual({CCLITE_VOCABULARY.to_native(m) for m in priors},
                                 set(record["upstream_legal_action_ids"]))
                self.assertAlmostEqual(math.fsum(priors.values()), 1.0, places=12)
                self.assertTrue(all(p >= 0 and math.isfinite(p) for p in priors.values()))

    def test_transposed_vocabulary_fails_golden_parity(self):
        """Mutation check: a wrong (target*90+source) mapping must NOT match upstream."""
        engine = GameEngine()
        initial = next(r for n, r, _ in STATES if n == "fixture/initial_red")
        wrong = sorted(m.target * 90 + m.source for m in engine.legal_moves(GameState.initial()))
        self.assertNotEqual(wrong, initial["upstream_legal_action_ids"])


if __name__ == "__main__":
    unittest.main()
