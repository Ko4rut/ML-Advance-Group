"""API integration tests for per-match curated opening book."""

import unittest

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.training.opening_book import OPENING_LINES, sample_opening, BOOK_VERSION


class OpeningBookApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def create(self, red="alphabeta", black="alphabeta", **extras):
        body = {
            "red": {"type": "human"} if red == "human" else {"type": "agent", "agent_id": red},
            "black": {"type": "human"} if black == "human" else {"type": "agent", "agent_id": black},
        }
        body.update(extras)
        response = self.client.post("/api/v1/matches", json=body)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_defaults_keep_previous_api_behavior(self):
        match = self.create(red="human", black="human")
        self.assertEqual(match["opening_mode"], "none")
        self.assertIsNone(match["opening"])
        self.assertEqual(match["ply"], 0)

    def test_agent_matches_auto_randomize_opening_unless_disabled(self):
        auto = self.create(red="random", black="human")
        self.assertEqual(auto["opening_mode"], "book")
        self.assertIsNotNone(auto["opening"])
        self.assertTrue(auto["opening"]["active"])
        self.assertEqual(auto["opening"]["planned_plies"], 6)
        disabled = self.create(red="random", black="human", opening_mode="none")
        self.assertEqual(disabled["opening_mode"], "none")
        self.assertIsNone(disabled["opening"])

    def test_catalogue_metadata(self):
        response = self.client.get("/api/v1/openings")
        self.assertEqual(response.status_code, 200)
        rows = response.json()
        # The API serves the current default book (v2); counts follow it.
        self.assertEqual(len(rows), len(OPENING_LINES))
        self.assertEqual(
            len({row["family"] for row in rows}),
            len({line.family for line in OPENING_LINES}),
        )
        self.assertNotIn("moves", rows[0])

    def test_agent_plays_scripted_moves_then_hands_off(self):
        seed = 44
        prefix = sample_opening(seed, 2).moves
        created = self.create(opening_mode="book", opening_seed=seed, opening_plies=2)
        match_id = created["match_id"]
        self.assertEqual(created["opening"]["book_version"], BOOK_VERSION)
        self.assertTrue(created["opening"]["active"])
        self.assertEqual(created["opening"]["seed"], seed)
        self.assertEqual(created["opening"]["planned_plies"], 2)
        for expected_ply, expected in enumerate(prefix):
            response = self.client.post(
                f"/api/v1/matches/{match_id}/step",
                json={"expected_ply": expected_ply},
            )
            self.assertEqual(response.status_code, 200, response.text)
            current = response.json()
            self.assertEqual(current["last_move"], {
                "source": expected.source, "target": expected.target,
            })
        self.assertEqual(current["opening"]["completed_plies"], 2)
        self.assertFalse(current["opening"]["active"])
        # The next AI step is normal Alpha-Beta; it must stay legal.
        legal = current["legal_moves"]
        result = self.client.post(
            f"/api/v1/matches/{match_id}/step", json={"expected_ply": 2}
        )
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIn(result.json()["last_move"], legal)

    def test_human_deviation_disables_book(self):
        created = self.create(
            red="human", black="random",
            opening_mode="book", opening_seed=42, opening_plies=6,
        )
        match_id = created["match_id"]
        intended = sample_opening(42, 6).moves[0]
        alternative = next(
            move for move in created["legal_moves"]
            if (move["source"], move["target"]) != (intended.source, intended.target)
        )
        response = self.client.post(
            f"/api/v1/matches/{match_id}/moves",
            json={**alternative, "expected_ply": 0},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["opening"]["active"])
        followup = self.client.post(
            f"/api/v1/matches/{match_id}/step", json={"expected_ply": 1}
        )
        self.assertEqual(followup.status_code, 200, followup.text)
        self.assertEqual(followup.json()["ply"], 2)
        self.assertFalse(followup.json()["opening"]["active"])

    def test_matching_human_move_preserves_book(self):
        seed = 73
        first = sample_opening(seed, 6).moves[0]
        created = self.create(
            red="human", black="random",
            opening_mode="book", opening_seed=seed, opening_plies=6,
        )
        match_id = created["match_id"]
        moved = self.client.post(
            f"/api/v1/matches/{match_id}/moves",
            json={"source": first.source, "target": first.target, "expected_ply": 0},
        )
        self.assertEqual(moved.status_code, 200, moved.text)
        self.assertTrue(moved.json()["opening"]["active"])
        self.assertEqual(moved.json()["opening"]["completed_plies"], 1)
        reply = self.client.post(
            f"/api/v1/matches/{match_id}/step", json={"expected_ply": 1}
        )
        self.assertEqual(reply.status_code, 200, reply.text)
        second = sample_opening(seed, 6).moves[1]
        self.assertEqual(reply.json()["last_move"], {
            "source": second.source, "target": second.target,
        })

    def test_seeded_reset_replays_same_line(self):
        created = self.create(
            red="random", black="random",
            opening_mode="book", opening_seed=13, opening_plies=6,
        )
        match_id = created["match_id"]
        before = created["opening"]
        self.client.post(f"/api/v1/matches/{match_id}/step", json={"expected_ply": 0})
        reset = self.client.post(f"/api/v1/matches/{match_id}/reset")
        self.assertEqual(reset.status_code, 200, reset.text)
        state = reset.json()
        self.assertEqual(state["ply"], 0)
        self.assertEqual(state["opening"]["id"], before["id"])
        self.assertEqual(state["opening"]["seed"], before["seed"])
        self.assertTrue(state["opening"]["active"])

    def test_validation_for_mode_plies_and_seed(self):
        cases = [
            {"opening_mode": "invalid"},
            {"opening_mode": "book", "opening_plies": 0},
            {"opening_mode": "book", "opening_plies": 7},
            {"opening_mode": "book", "opening_seed": -1},
        ]
        for fields in cases:
            body = {
                "red": {"type": "agent", "agent_id": "random"},
                "black": {"type": "agent", "agent_id": "random"},
            }
            body.update(fields)
            response = self.client.post("/api/v1/matches", json=body)
            self.assertEqual(response.status_code, 422, response.text)


if __name__ == "__main__":
    unittest.main()
