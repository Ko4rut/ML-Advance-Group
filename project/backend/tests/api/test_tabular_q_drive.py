"""Drive checkpoint resolution and API readiness tests with no external I/O."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.services.matches import MatchService
from src.api.services.tabular_q import TabularQService
from src.api.services.tabular_q_checkpoint import TabularQCheckpointSource
from src.environment import GameState
from src.training.tabular_qlearning import TabularQConfig, TabularQTable, state_key


class DriveCheckpointServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source_file = root / "source_q.json.gz"
        table = TabularQTable()
        table.observe(state_key(GameState.initial().board, "red"), 54 * 90 + 45)
        table.save(self.source_file, TabularQConfig(), {"source": "test"})

    def test_download_only_once_and_use_cache_for_subsequent_service(self) -> None:
        calls: list[str] = []

        def download(file_id: str, destination: Path) -> None:
            calls.append(file_id)
            shutil.copyfile(self.source_file, destination)

        root = Path(self.tmp.name) / "cache"
        file_id = "1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c"
        service = TabularQService(
            drive_file_id=file_id, cache_dir=root, downloader=download
        )
        self.assertTrue(service.available)
        self.assertEqual(calls, [file_id])
        second = TabularQService(
            drive_file_id=file_id, cache_dir=root, downloader=download
        )
        self.assertTrue(second.available)
        self.assertEqual(calls, [file_id])

    def test_api_agent_available_after_drive_download(self) -> None:
        def download(file_id: str, destination: Path) -> None:
            shutil.copyfile(self.source_file, destination)

        service = MatchService(tabular_q_service=TabularQService(
            drive_file_id="1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c",
            cache_dir=Path(self.tmp.name) / "cache",
            downloader=download,
        ))
        with TestClient(create_app(service)) as client:
            listed = {item["id"]: item for item in client.get("/api/v1/agents").json()}
            self.assertTrue(listed["tabular_q"]["available"])
            response = client.post("/api/v1/matches", json={
                "red": {"type": "agent", "agent_id": "tabular_q"},
                "black": {"type": "human"},
                "opening_mode": "none",
            })
            self.assertEqual(response.status_code, 201, response.text)
            match = response.json()
            action = client.post(
                f"/api/v1/matches/{match['match_id']}/step",
                json={"expected_ply": 0},
            )
            self.assertEqual(action.status_code, 200, action.text)
            self.assertEqual(action.json()["last_move"], {"source": 54, "target": 45})

    def test_download_failure_never_advertises_agent_as_ready(self) -> None:
        def fail_download(file_id: str, destination: Path) -> None:
            raise RuntimeError("Permission denied for Drive")

        service = TabularQService(
            drive_file_id="1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c",
            cache_dir=Path(self.tmp.name) / "cache",
            downloader=fail_download,
        )
        self.assertFalse(service.available)
        self.assertIn("Permission denied", service.unavailable_reason)
        cache_dir = Path(self.tmp.name) / "cache"
        self.assertEqual(list(cache_dir.iterdir()), [])

    def test_invalid_download_does_not_activate_model(self) -> None:
        def bad_download(file_id: str, destination: Path) -> None:
            destination.write_bytes(b"not a gzip checkpoint")

        service = TabularQService(
            drive_file_id="1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c",
            cache_dir=Path(self.tmp.name) / "cache",
            downloader=bad_download,
        )
        self.assertFalse(service.available)
        self.assertIn("Unable to load", service.unavailable_reason)

    def test_local_path_takes_priority_and_has_no_download(self) -> None:
        def fail_download(file_id: str, destination: Path) -> None:
            raise AssertionError("Drive should not be used with a local checkpoint")

        service = TabularQService(
            checkpoint_path=self.source_file,
            drive_file_id="1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c",
            cache_dir=Path(self.tmp.name) / "cache",
            downloader=fail_download,
        )
        self.assertTrue(service.available)

    def test_invalid_id_rejected_without_download(self) -> None:
        source = TabularQCheckpointSource(
            drive_file_id="../../config", cache_dir=Path(self.tmp.name),
            downloader=lambda file_id, destination: self.fail("must not download"),
        )
        with self.assertRaises(ValueError):
            source.resolve()

    def test_environment_config_supports_drive_id(self) -> None:
        with patch.dict(os.environ, {
            "XIANGQI_TABULAR_Q_DRIVE_FILE_ID": "1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c",
            "XIANGQI_TABULAR_Q_CACHE_DIR": str(Path(self.tmp.name) / "cache"),
        }, clear=True):
            source = TabularQCheckpointSource()
            self.assertEqual(
                source.drive_file_id,
                "1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c",
            )


if __name__ == "__main__":
    unittest.main()
