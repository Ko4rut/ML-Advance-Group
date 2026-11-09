"""Tests for the standalone Tabular Q checkpoint import/download CLI."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.download_tabular_q_model import install_checkpoint
from src.environment import GameState
from src.training.tabular_qlearning import (
    MODEL_VERSION,
    TabularQConfig,
    TabularQTable,
    state_key,
)


class DownloadTabularQModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "downloaded.json.gz"
        self.target = self.root / "models" / "tabular_q" / "q_table.json.gz"

        table = TabularQTable()
        board = GameState.initial()
        key = state_key(board.board, "red")
        table.observe(key, 54 * 90 + 45)
        table.values[key][54 * 90 + 45] = 0.4
        table.save(self.source, TabularQConfig(), {"example": "test"})

    def test_import_manual_download_without_credentials(self) -> None:
        def no_google(file_id: str, path: Path) -> None:
            self.fail("Drive must not be contacted for local import")

        info = install_checkpoint(
            destination=self.target,
            source_file=self.source,
            downloader=no_google,
        )
        self.assertEqual(info["source"], "local_copy")
        self.assertEqual(info["known_states"], 1)
        self.assertEqual(info["known_state_action_pairs"], 1)
        self.assertEqual(info["model_version"], MODEL_VERSION)
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_drive_download_with_injected_fetcher(self) -> None:
        calls = []

        def fake_download(file_id: str, target: Path) -> None:
            calls.append(file_id)
            target.write_bytes(self.source.read_bytes())

        info = install_checkpoint(
            destination=self.target,
            drive_file_id="some-drive-id-for-testing",
            downloader=fake_download,
        )
        self.assertEqual(calls, ["some-drive-id-for-testing"])
        self.assertEqual(info["source"], "google_drive")
        self.assertTrue(self.target.is_file())

    def test_existing_model_not_overwritten_without_replace(self) -> None:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            install_checkpoint(destination=self.target, source_file=self.source)
        self.assertEqual(self.target.read_bytes(), b"existing")

    def test_replace_valid_model(self) -> None:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_bytes(b"old")
        info = install_checkpoint(
            destination=self.target,
            source_file=self.source,
            replace=True,
        )
        self.assertEqual(info["known_states"], 1)
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_invalid_replacement_keeps_old_model(self) -> None:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        original = self.source.read_bytes()
        self.target.write_bytes(original)
        invalid = self.root / "bad.gz"
        invalid.write_bytes(b"bad gzip")
        with self.assertRaises((OSError, ValueError, EOFError)):
            install_checkpoint(
                destination=self.target,
                source_file=invalid,
                replace=True,
            )
        self.assertEqual(self.target.read_bytes(), original)
        self.assertEqual(
            list(self.target.parent.glob(".q_table_*.downloading")),
            [],
        )

    def test_missing_source_fails_without_creating_model(self) -> None:
        with self.assertRaises(FileNotFoundError):
            install_checkpoint(
                destination=self.target,
                source_file=self.root / "missing.gz",
            )
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
