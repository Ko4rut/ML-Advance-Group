"""API coverage for Tabular Q checkpoint loading and greedy match play."""

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.agents.tabular_q_agent import TabularQAgent
from src.api.main import create_app
from src.api.services.matches import MatchService
from src.api.services.tabular_q import TabularQService
from src.environment import GameState
from src.training.tabular_qlearning import TabularQConfig, TabularQTable, state_key


def write_checkpoint(path: Path) -> int:
    board = GameState.initial()
    move = (54, 45)
    action_id = move[0] * 90 + move[1]
    table = TabularQTable()
    key = state_key(board.board, "red")
    table.observe(key, action_id)
    table.values[key][action_id] = 0.9
    table.save(path, TabularQConfig(), {"source": "api-test"})
    return action_id


class TabularQApiTests(unittest.TestCase):
    def test_missing_checkpoint_not_advertised_and_create_returns_503(self):
        service = MatchService(
            tabular_q_service=TabularQService("/path/that/does/not/exist/model.gz")
        )
        with TestClient(create_app(service)) as client:
            info = {row["id"]: row for row in client.get("/api/v1/agents").json()}
            self.assertIn("tabular_q", info)
            self.assertFalse(info["tabular_q"]["available"])
            result = client.post("/api/v1/matches", json={
                "red": {"type": "agent", "agent_id": "tabular_q"},
                "black": {"type": "human"},
                "opening_mode": "none",
            })
            self.assertEqual(result.status_code, 503)
            self.assertIn("checkpoint", result.json()["detail"].lower())

    def test_corrupt_checkpoint_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json.gz"
            path.write_bytes(b"not a gzip model")
            service = TabularQService(path)
            self.assertFalse(service.available)
            self.assertIn("Unable to load", service.unavailable_reason)

    def test_checkpoint_loaded_and_greedy_move_via_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q_table.json.gz"
            write_checkpoint(path)
            service = TabularQService(path)
            match_service = MatchService(tabular_q_service=service)
            with TestClient(create_app(match_service)) as client:
                info = {row["id"]: row for row in client.get("/api/v1/agents").json()}
                self.assertTrue(info["tabular_q"]["available"])
                response = client.post("/api/v1/matches", json={
                    "red": {"type": "agent", "agent_id": "tabular_q"},
                    "black": {"type": "human"},
                    "opening_mode": "none",
                })
                self.assertEqual(response.status_code, 201, response.text)
                match = response.json()
                agent = match_service._matches[match["match_id"]].red_agent
                self.assertIsInstance(agent, TabularQAgent)
                self.assertEqual(agent.epsilon, 0.0)
                result = client.post(
                    f"/api/v1/matches/{match['match_id']}/step",
                    json={"expected_ply": 0},
                )
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(result.json()["last_move"], {"source": 54, "target": 45})
                self.assertTrue(agent.last_action_supported)

    def test_unknown_state_fallback_is_rule_based_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q_table.json.gz"
            write_checkpoint(path)
            service = TabularQService(path)
            from src.environment import GameEngine
            agent = service.make_agent(GameEngine())
            initial = GameState.initial()
            known = agent.select_action(initial)
            self.assertTrue(agent.last_action_supported)
            self.assertEqual((known.move.source, known.move.target), (54, 45))
            # A fresh legal position outside the Q-table should use RuleBasedAgent.
            engine = GameEngine()
            next_state = engine.apply_move(initial, known.move)
            self.assertNotIn(state_key(next_state.board, "black"), agent.table.values)
            first = agent.select_action(next_state)
            second = agent.select_action(next_state)
            self.assertFalse(agent.last_action_supported)
            self.assertEqual(first.move, second.move)
            self.assertIn(first.move, engine.legal_moves(next_state))


if __name__ == "__main__":
    unittest.main()
