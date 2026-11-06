"""Bronze ingestion tests: immutable copy, idempotence, provenance and collisions."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.data.bronze.data_pipeline import BronzeIngester
from src.data.bronze.models import BronzeFileManifest


class BronzeIngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "datasets"
        self.source.mkdir()
        self.destination = self.root / "data" / "01_bronze"

    def sample(self, name="games.jsonl", content='{"game_id":"g1"}\n'):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def ingest(self, inputs=None, **kwargs):
        return BronzeIngester().ingest(
            inputs or [self.source], self.destination,
            source="alphabeta", source_version="batch-v1", **kwargs
        )

    def test_copy_preserves_bytes_and_metadata(self):
        raw = self.sample(content='{"game_id":"g1"}\r\n')
        report = self.ingest()
        target = self.destination / "alphabeta" / "games.jsonl"
        manifest_file = target.with_name(target.name + ".manifest.json")
        manifest = BronzeFileManifest.model_validate_json(manifest_file.read_text())
        self.assertEqual(target.read_bytes(), raw.read_bytes())
        self.assertEqual(report.copied_files, 1)
        self.assertEqual(manifest.file_sha256, hashlib.sha256(raw.read_bytes()).hexdigest())
        self.assertEqual(manifest.size_bytes, len(raw.read_bytes()))
        self.assertEqual(manifest.source_file, raw.name)

    def test_rerun_idempotent(self):
        self.sample()
        self.ingest()
        report = self.ingest()
        self.assertEqual((report.copied_files, report.unchanged_files), (0, 1))

    def test_different_bytes_never_overwrite(self):
        self.sample(content="original\n")
        self.ingest()
        self.sample(content="modified\n")
        with self.assertRaisesRegex(ValueError, "collision"):
            self.ingest()
        self.assertEqual(
            (self.destination / "alphabeta" / "games.jsonl").read_text(),
            "original\n"
        )

    def test_nested_paths_and_glob(self):
        self.sample("folder/one.jsonl")
        self.sample("folder/two.pgn", "abc")
        self.sample("other/three.jsonl")
        report = self.ingest()
        self.assertEqual(report.copied_files, 2)
        self.assertFalse((self.destination / "alphabeta" / "folder" / "two.pgn").exists())
        self.assertTrue((self.destination / "alphabeta" / "folder" / "one.jsonl").exists())

    def test_empty_file_and_checkpoint_are_not_ingested(self):
        self.sample("empty.jsonl", "")
        self.sample("games.jsonl.checkpoint.json", "{}")
        report = self.ingest()
        self.assertEqual(report.empty_files, 1)
        self.assertEqual(report.copied_files, 0)

    def test_explicit_external_format(self):
        self.sample("moves.pgn", "1. C2=5 H8+7")
        report = self.ingest(patterns=["*.pgn"])
        self.assertEqual(report.copied_files, 1)

    def test_manifest_conflict_detected(self):
        self.sample()
        self.ingest()
        manifest_file = self.destination / "alphabeta" / "games.jsonl.manifest.json"
        payload = json.loads(manifest_file.read_text())
        payload["file_sha256"] = "a" * 64
        manifest_file.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "conflict"):
            self.ingest()

    def test_reject_destination_nested_inside_input(self):
        self.sample()
        with self.assertRaisesRegex(ValueError, "inside input"):
            BronzeIngester().ingest(
                [self.root], self.root / "data" / "01_bronze",
                source="alphabeta", source_version="v1"
            )

    def test_duplicate_source_names_do_not_mix(self):
        self.sample()
        source2 = self.root / "other"
        source2.mkdir()
        (source2 / "games.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "same Bronze path"):
            self.ingest(inputs=[self.source / "games.jsonl", source2 / "games.jsonl"])


if __name__ == "__main__":
    unittest.main()
