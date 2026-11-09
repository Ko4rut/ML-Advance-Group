"""Tabular Q status route integration tests with a real lightweight checkpoint."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.services.matches import MatchService
from src.api.services.tabular_q import TabularQService
from src.environment import GameState
from src.training.tabular_qlearning import (
    MODEL_VERSION,
    TabularQConfig,
    TabularQTable,
    state_key,
)


class TabularQModelStatusApiTests(unittest.TestCase):
    def test_missing_checkpoint_returns_safe_unavailable_status(self) -> None:
        service = MatchService(
            tabular_q_service=TabularQService(
                checkpoint_path="/missing/tabular-q/checkpoint.json.gz",
            ),
        )
        with TestClient(create_app(service)) as client:
            response = client.get("/api/v1/models/tabular-q/status")
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["agent_id"], "tabular_q")
            self.assertFalse(result["available"])
            self.assertEqual(result["source"], "local")
            self.assertIsNone(result["model_version"])
            self.assertEqual(result["known_states"], 0)
            self.assertEqual(result["known_state_action_pairs"], 0)
            self.assertEqual(result["epsilon"], 0.0)
            self.assertEqual(result["unknown_state_fallback"], "rule_based")
            self.assertNotIn("/missing", str(result))
            listed = {a["id"]: a for a in client.get("/api/v1/agents").json()}
            self.assertFalse(listed["tabular_q"]["available"])

    def test_loaded_model_matches_catalog_and_agent_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q_table.json.gz"
            table = TabularQTable()
            table.observe(state_key(GameState.initial().board, "red"), 54 * 90 + 45)
            table.values[state_key(GameState.initial().board, "red")][54 * 90 + 45] = 0.8
            table.save(path, TabularQConfig(), {"source": "status-test"})

            match_service = MatchService(
                tabular_q_service=TabularQService(checkpoint_path=path)
            )
            with TestClient(create_app(match_service)) as client:
                status_response = client.get("/api/v1/models/tabular-q/status")
                self.assertEqual(status_response.status_code, 200, status_response.text)
                status = status_response.json()
                self.assertTrue(status["available"])
                self.assertEqual(status["source"], "local")
                self.assertEqual(status["model_version"], MODEL_VERSION)
                self.assertEqual(status["known_states"], 1)
                self.assertEqual(status["known_state_action_pairs"], 1)
                self.assertEqual(status["epsilon"], 0.0)

                listed = {a["id"]: a for a in client.get("/api/v1/agents").json()}
                self.assertTrue(listed["tabular_q"]["available"])
                created = client.post("/api/v1/matches", json={
                    "red": {"type": "agent", "agent_id": "tabular_q"},
                    "black": {"type": "human"},
                    "opening_mode": "none",
                })
                self.assertEqual(created.status_code, 201, created.text)
                match = created.json()
                moved = client.post(
                    f"/api/v1/matches/{match['match_id']}/step",
                    json={"expected_ply": 0},
                )
                self.assertEqual(moved.status_code, 200, moved.text)
                self.assertEqual(moved.json()["last_move"], {"source": 54, "target": 45})

    def test_unconfigured_model_still_has_healthy_api(self) -> None:
        service = MatchService(tabular_q_service=TabularQService(
            checkpoint_path="/missing/model.json.gz",
        ))
        with TestClient(create_app(service)) as client:
            self.assertEqual(client.get("/api/v1/health").json(), {"status": "ok"})
            self.assertFalse(
                client.get("/api/v1/models/tabular-q/status").json()["available"]
            )


if __name__ == "__main__":
    unittest.main()
