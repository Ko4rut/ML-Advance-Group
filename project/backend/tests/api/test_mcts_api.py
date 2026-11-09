"""End-to-end API tests for Monte Carlo Tree Search matches."""

import unittest

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.services.matches import MatchService
from src.api.services.mcts import MCTSService


class MCTSApiTests(unittest.TestCase):
    def setUp(self) -> None:
        # A fixed, small budget keeps API tests deterministic and fast.
        self.mcts_service = MCTSService(config={
            "c": 1.4,
            "rollout_policy": "capture_first",
            "rollout_depth": 0,
            "eval_scale": 100,
            "seed": 17,
            "max_simulations": 8,
            "time_limit": 2.0,
        })
        self.match_service = MatchService(mcts_service=self.mcts_service)
        self.client = TestClient(create_app(self.match_service))

    def create_match(self, red: str, black: str, **options) -> dict:
        def participant(agent_id: str) -> dict:
            if agent_id == "human":
                return {"type": "human"}
            return {"type": "agent", "agent_id": agent_id}

        body = {
            "red": participant(red),
            "black": participant(black),
            "opening_mode": "none",
        }
        body.update(options)
        response = self.client.post("/api/v1/matches", json=body)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_catalog_lists_mcts_as_available(self) -> None:
        result = self.client.get("/api/v1/agents")
        self.assertEqual(result.status_code, 200)
        agents = {agent["id"]: agent for agent in result.json()}
        self.assertTrue(agents["mcts"]["available"])
        self.assertFalse(agents["neural"]["available"])
        self.assertFalse(agents["dqn"]["available"])

    def test_red_mcts_selects_legal_move_with_configured_budget(self) -> None:
        match = self.create_match("mcts", "human")
        match_id = match["match_id"]
        url = f"/api/v1/matches/{match_id}/step"

        response = self.client.post(url, json={"expected_ply": 0})
        self.assertEqual(response.status_code, 200, response.text)
        current = response.json()
        self.assertEqual(current["ply"], 1)
        self.assertEqual(current["side_to_move"], "black")
        self.assertIn(current["last_move"], match["legal_moves"])
        agent = self.match_service._matches[match_id].red_agent
        self.assertEqual(agent.search.last_simulations, 8)
        self.assertEqual(
            self.client.post(url, json={"expected_ply": 0}).status_code, 409
        )
        self.assertEqual(
            self.client.post(url, json={"expected_ply": 1}).status_code, 409
        )

    def test_black_mcts_replies_to_human_move(self) -> None:
        match = self.create_match("human", "mcts")
        match_id = match["match_id"]
        first = match["legal_moves"][0]
        response = self.client.post(
            f"/api/v1/matches/{match_id}/moves",
            json={**first, "expected_ply": 0},
        )
        self.assertEqual(response.status_code, 200, response.text)
        after_human = response.json()

        response = self.client.post(
            f"/api/v1/matches/{match_id}/step",
            json={"expected_ply": 1},
        )
        self.assertEqual(response.status_code, 200, response.text)
        current = response.json()
        self.assertIn(current["last_move"], after_human["legal_moves"])
        self.assertEqual(current["ply"], 2)
        self.assertEqual(current["side_to_move"], "red")

    def test_opening_book_and_reset_still_work_with_mcts(self) -> None:
        match = self.create_match(
            "mcts", "human", opening_mode="book",
            opening_seed=42, opening_plies=1,
        )
        match_id = match["match_id"]
        response = self.client.post(
            f"/api/v1/matches/{match_id}/step",
            json={"expected_ply": 0},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn(response.json()["last_move"], match["legal_moves"])
        self.assertEqual(response.json()["opening"]["completed_plies"], 1)

        response = self.client.post(f"/api/v1/matches/{match_id}/reset")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["ply"], 0)
        self.assertEqual(response.json()["opening"]["completed_plies"], 0)

    def test_default_config_file_is_loadable(self) -> None:
        from src.environment import GameEngine

        agent = MCTSService().make_agent(GameEngine())
        self.assertIsNotNone(agent.search)


if __name__ == "__main__":
    unittest.main()
