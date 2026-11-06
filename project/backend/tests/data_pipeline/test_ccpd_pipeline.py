"""Integration tests: CCPD Bronze PGN to replay-validated Silver records."""

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.data.silver.data_pipeline.ccpd import CCPDPipeline
from src.data.silver.data_pipeline.ccpd.decoder import decode_pgn
from src.data.silver.data_pipeline.ccpd.pgn_parser import parse_fen
from src.data.silver.data_pipeline.ccpd.notation import resolve_move
from src.environment import GameState, Move, Side

FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
SAMPLE = f'''[Game "Chinese Chess"]
[Red "王琳娜"]
[Black "孔令儀"]
[Result "1-0"]
[FEN "{FEN}"]

1. 兵七進一 炮２平５
2. 馬八進七 馬２進３
1-0
'''


class CCPDSilverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bronze = self.root / "01_bronze" / "ccpd"
        self.bronze.mkdir(parents=True)

    def add_game(self, name="00000001.pgn", content=SAMPLE, *, manifest=True):
        pgn = self.bronze / name
        pgn.write_bytes(content.encode("big5"))
        if manifest:
            path = pgn.with_name(pgn.name + ".manifest.json")
            path.write_text(json.dumps({
                "schema_version": 1,
                "source": "ccpd",
                "source_version": "git-test",
                "source_file": name,
                "file_sha256": hashlib.sha256(pgn.read_bytes()).hexdigest(),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "license": "CC-BY-4.0",
            }), encoding="utf-8")
        return pgn

    def rows(self, root, name):
        return [
            json.loads(line)
            for line in (root / name).read_text(encoding="utf-8").splitlines()
        ]

    def test_big5_pgn_replay_and_recorded_match_result(self):
        file = self.add_game()
        dest = self.root / "silver"
        report = CCPDPipeline().run([file], dest)
        self.assertEqual((report["accepted_games"], report["transitions"]), (1, 4))
        game = self.rows(dest, "games.jsonl")[0]
        steps = self.rows(dest, "transitions.jsonl")
        self.assertEqual(game["outcome"]["winner"], "red")
        self.assertEqual(game["outcome"]["result_source"], "recorded")
        self.assertEqual(game["provenance"]["file_sha256"],
                         hashlib.sha256(file.read_bytes()).hexdigest())
        self.assertEqual(game["provenance"]["license"], "CC-BY-4.0")
        self.assertEqual(
            [(step["action"]["source"], step["action"]["target"]) for step in steps[:3]],
            [(56, 47), (19, 22), (82, 65)]
        )
        self.assertTrue(steps[-1]["recorded_game_end"])
        self.assertFalse(steps[-1]["terminated"])
        self.assertFalse(steps[-1]["truncated"])
        self.assertTrue(all(step["source_type"] == "human" for step in steps))

    def test_missing_manifest_quarantined(self):
        self.add_game(manifest=False)
        report = CCPDPipeline().run([self.bronze], self.root / "silver")
        self.assertEqual(report["quarantine_error_codes"], {"MISSING_MANIFEST": 1})

    def test_hash_conflict_quarantined(self):
        pgn = self.add_game()
        path = pgn.with_name(pgn.name + ".manifest.json")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["file_sha256"] = "f" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
        report = CCPDPipeline().run([self.bronze], self.root / "silver")
        self.assertEqual(report["quarantine_error_codes"], {"BRONZE_HASH_MISMATCH": 1})

    def test_duplicate_games_are_counted_but_not_repeated(self):
        self.add_game("first.pgn")
        self.add_game("second.pgn")
        report = CCPDPipeline().run([self.bronze], self.root / "silver")
        self.assertEqual((report["accepted_games"], report["duplicate_games"]), (1, 1))
        self.assertEqual(len(self.rows(self.root / "silver", "transitions.jsonl")), 4)

    def test_illegal_chinese_notation_quarantined(self):
        self.add_game(content=SAMPLE.replace("兵七進一", "兵七進九"))
        report = CCPDPipeline().run([self.bronze], self.root / "silver")
        self.assertEqual(report["quarantine_error_codes"], {"ILLEGAL_NOTATION": 1})

    def test_result_mismatch_quarantined(self):
        self.add_game(content=SAMPLE.replace("\n1-0\n", "\n0-1\n"))
        report = CCPDPipeline().run([self.bronze], self.root / "silver")
        self.assertEqual(report["quarantine_error_codes"], {"RESULT_MISMATCH": 1})

    def test_fen_matches_project_board(self):
        board, side = parse_fen(FEN)
        self.assertEqual(board, list(GameState.initial().board))
        self.assertEqual(side, "red")

    def test_front_back_rook_notation(self):
        board = list(GameState.initial().board)
        board[57] = board[75] = 5
        state = GameState(tuple(board), Side.RED)
        legal = [Move(57, 58), Move(75, 76)]
        self.assertEqual(resolve_move("前車平五", state, legal, 1), Move(57, 58))
        self.assertEqual(resolve_move("後車平五", state, legal, 1), Move(75, 76))

    def test_big5_decoder(self):
        text, codec = decode_pgn(SAMPLE.encode("big5"))
        self.assertIn("王琳娜", text)
        self.assertEqual(codec, "big5")

    def test_never_overwrite_silver(self):
        self.add_game()
        out = self.root / "silver"
        CCPDPipeline().run([self.bronze], out)
        with self.assertRaises(FileExistsError):
            CCPDPipeline().run([self.bronze], out)

    def test_empty_pgn_goes_to_quarantine(self):
        self.add_game(content="")
        report = CCPDPipeline().run([self.bronze], self.root / "silver")
        self.assertEqual(report["quarantined_records"], 1)


if __name__ == "__main__":
    unittest.main()
