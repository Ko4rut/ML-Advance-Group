"""Ensure local .env config is loaded before Tabular Q service initialization."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.config import BACKEND_ROOT, load_backend_env
from src.api.main import create_app
from src.api.services.tabular_q_checkpoint import TabularQCheckpointSource
from src.environment import GameState
from src.training.tabular_qlearning import TabularQConfig, TabularQTable, state_key


class BackendDotenvTests(unittest.TestCase):
    def test_relative_checkpoint_path_is_anchored_to_backend_root(self) -> None:
        source = TabularQCheckpointSource(
            checkpoint_path="models/tabular_q/q_table.json.gz"
        )
        expected = (BACKEND_ROOT / "models" / "tabular_q" / "q_table.json.gz").resolve()
        self.assertEqual(source.checkpoint_path, expected)

    def test_env_does_not_replace_explicit_process_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "XIANGQI_TABULAR_Q_CHECKPOINT=models/tabular_q/q_table.json.gz\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {
                "XIANGQI_TABULAR_Q_CHECKPOINT": "/explicit/model.json.gz",
            }):
                self.assertTrue(load_backend_env(env_path))
                self.assertEqual(
                    os.environ["XIANGQI_TABULAR_Q_CHECKPOINT"],
                    "/explicit/model.json.gz",
                )

    def test_fastapi_reads_dotenv_and_exposes_available_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "q_table.json.gz"
            table = TabularQTable()
            key = state_key(GameState.initial().board, "red")
            action = 54 * 90 + 45
            table.observe(key, action)
            table.values[key][action] = 0.5
            table.save(checkpoint, TabularQConfig(), {"test": "dotenv"})

            config_file = root / ".env"
            config_file.write_text(
                "XIANGQI_TABULAR_Q_CHECKPOINT="
                + checkpoint.as_posix()
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("XIANGQI_TABULAR_Q_CHECKPOINT", None)
                os.environ.pop("XIANGQI_TABULAR_Q_DRIVE_FILE_ID", None)
                with patch(
                    "src.api.main.load_backend_env",
                    side_effect=lambda: load_backend_env(config_file),
                ) as load_mock:
                    with TestClient(create_app()) as client:
                        load_mock.assert_called_once_with()
                        response = client.get("/api/v1/models/tabular-q/status")
                        self.assertEqual(response.status_code, 200, response.text)
                        status = response.json()
                        self.assertTrue(status["available"])
                        self.assertEqual(status["source"], "local")
                        self.assertEqual(status["known_states"], 1)

                        agents = {
                            agent["id"]: agent
                            for agent in client.get("/api/v1/agents").json()
                        }
                        self.assertTrue(agents["tabular_q"]["available"])


if __name__ == "__main__":
    unittest.main()
