"""summary/ generation: byte-stable CSV/report, partial runs, errors, optional plots, CLI."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import cli
from src.evaluation import report
from src.evaluation import tournament as t

CSV_NAMES = ("standings.csv", "pair_results.csv", "crosstable.csv", "color_breakdown.csv",
             "cost.csv", "openings.csv")


def tiny_raw(**overrides):
    raw = {
        "name": "unit_report",
        "master_seed": 99,
        "opening_book_version": "xiangqi-curated-v2",
        "openings": ["screen_horses_c57", "pawn_left"],
        "colors_per_opening": 2,
        "max_plies": 16,
        "agents": {"random": {}, "rules": {}},
        "workers": 1,
    }
    raw.update(overrides)
    return raw


def run_tiny(root, raw=None):
    config = t.resolve_config(raw or tiny_raw())
    with mock.patch("sys.stdout"):
        summary = t.run_tournament(config, output_root=root, workers=1)
    return Path(summary["run_dir"])


def csv_bytes(run_dir):
    summary = Path(run_dir) / "summary"
    return {name: (summary / name).read_bytes() for name in sorted(p.name for p in summary.glob("*.csv"))}


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.run_dir = run_tiny(cls._dir.name)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def setUp(self):
        self._copy = tempfile.TemporaryDirectory()
        self.addCleanup(self._copy.cleanup)
        self.run = Path(self._copy.name) / self.run_dir.name
        shutil.copytree(self.run_dir, self.run)

    def test_generates_all_files_from_a_real_tiny_run(self):
        result = report.generate_report(self.run)
        summary = self.run / "summary"
        for name in (*CSV_NAMES, "report.md"):
            self.assertTrue((summary / name).is_file(), name)
        self.assertEqual(result["completed"], result["total"])
        self.assertEqual(result["completed"], 4)
        for name in ("crosstable_heatmap.png", "elo_ci.png", "cost_per_move.png"):
            self.assertTrue((summary / "plots" / name).is_file(), name)
        text = (summary / "report.md").read_text(encoding="utf-8")
        for number in range(1, 10):
            self.assertIn(f"## {number}. ", text)
        self.assertIn("random", text)

    def test_regeneration_is_byte_identical_for_csv_and_report(self):
        report.generate_report(self.run, plots=False)
        first = csv_bytes(self.run)
        first_report = (self.run / "summary" / "report.md").read_bytes()
        shutil.rmtree(self.run / "summary")
        report.generate_report(self.run, plots=False)
        self.assertEqual(first, csv_bytes(self.run))
        self.assertEqual(first_report, (self.run / "summary" / "report.md").read_bytes())
        self.assertNotIn(b"\r", first_report)

    def test_csv_numbers_match_raw_log(self):
        report.generate_report(self.run, plots=False)
        lines = (self.run / "summary" / "pair_results.csv").read_text(encoding="utf-8").splitlines()
        header = lines[0].split(",")
        row = dict(zip(header, lines[1].split(",")))
        records = [json.loads(line) for line in (self.run / "games.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(int(row["games"]), len(records))
        wins = sum(1 for r in records if r["outcome"]["winner"] is not None
                   and (r["red_agent"] if r["outcome"]["winner"] == "red" else r["black_agent"]) == "random")
        self.assertEqual(int(row["W"]), wins)

    def test_incomplete_run_still_reports_and_says_so(self):
        path = self.run / "games.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
        result = report.generate_report(self.run, plots=False)
        self.assertEqual((result["completed"], result["total"]), (2, 4))
        text = (self.run / "summary" / "report.md").read_text(encoding="utf-8")
        self.assertIn("Chưa hoàn tất: 2/4 ván", text)

    def test_torn_last_line_is_skipped_without_modifying_the_log(self):
        path = self.run / "games.jsonl"
        original = path.read_bytes()
        path.write_bytes(original + b'{"schema_version":1,"game_in')
        result = report.generate_report(self.run, plots=False)
        self.assertEqual(result["completed"], 4)
        self.assertTrue(path.read_bytes().endswith(b'{"schema_version":1,"game_in'))
        self.assertIn("bị cắt dở", (self.run / "summary" / "report.md").read_text(encoding="utf-8"))

    def test_corrupt_middle_line_or_duplicate_is_an_error(self):
        path = self.run / "games.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join([lines[0], "{broken", *lines[1:]]) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            report.generate_report(self.run, plots=False)
        path.write_text("\n".join([lines[0], lines[0]]) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            report.generate_report(self.run, plots=False)

    def test_missing_manifest_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(ValueError):
            report.generate_report(empty)

    def test_error_games_are_listed_and_excluded(self):
        path = self.run / "games.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        records[0]["outcome"] = {"status": "error", "winner": None, "reason": "error"}
        records[0]["error"] = {"type": "RuntimeError", "message": "agent exploded", "ply": 11, "actor": "rules"}
        path.write_text("".join(t.dump_record(r).decode("utf-8") for r in records), encoding="utf-8")
        report.generate_report(self.run, plots=False)
        summary = self.run / "summary"
        errors = (summary / "errors.csv").read_text(encoding="utf-8")
        self.assertIn("agent exploded", errors)
        pair = (summary / "pair_results.csv").read_text(encoding="utf-8").splitlines()
        row = dict(zip(pair[0].split(","), pair[1].split(",")))
        self.assertEqual((row["games"], row["E"]), ("4", "1"))
        text = (summary / "report.md").read_text(encoding="utf-8")
        self.assertIn("ván lỗi", text)
        # fixing the log removes the stale errors.csv on the next run
        path.write_text("".join(t.dump_record(json.loads(l)).decode("utf-8")
                                for l in (self.run_dir / "games.jsonl").read_text(encoding="utf-8").splitlines()),
                        encoding="utf-8")
        report.generate_report(self.run, plots=False)
        self.assertFalse((summary / "errors.csv").exists())

    def test_missing_matplotlib_skips_plots_without_crashing(self):
        with mock.patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}), \
                mock.patch("sys.stdout"):
            result = report.generate_report(self.run, plots=True)
        self.assertEqual(result["plots"], [])
        self.assertFalse((self.run / "summary" / "plots").exists())
        text = (self.run / "summary" / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("![", text)
        self.assertTrue((self.run / "summary" / "standings.csv").is_file())

    def test_no_plots_flag(self):
        result = report.generate_report(self.run, plots=False)
        self.assertEqual(result["plots"], [])
        self.assertFalse((self.run / "summary" / "plots").exists())

    def test_warnings_reflect_manifest(self):
        manifest_path = self.run / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["excluded_agents"] = {"tabular_q": "checkpoint not found: x"}
        manifest["provenance"]["git"]["dirty"] = True
        manifest["provenance"]["git"]["status"] = [" M a.py"]
        manifest["resume_history"] = [{"code_changed": True}]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report.generate_report(self.run, plots=False)
        text = (self.run / "summary" / "report.md").read_text(encoding="utf-8")
        self.assertIn("Agent bị loại khỏi giải", text)
        self.assertIn("checkpoint not found", text)
        self.assertIn("Cây git bẩn", text)
        self.assertIn("resume 1 lần", text)
        self.assertIn("random", text.split("Agent không tất định")[1].split("\n")[0])

    def test_tabular_q_high_fallback_is_called_out(self):
        summary = {
            "totals": {"games": 1, "finished": 1, "draws": 0, "truncated": 0, "error": 0},
            "cost": [{"agent": "tabular_q", "fallback_rate": 0.96, "think_p95_s": None,
                      "simulations_mean": None}],
            "rank_differs": [], "pairs": [],
        }
        manifest = {"agents": {"tabular_q": {"is_deterministic": True}}, "workers": 1}
        warnings = report.build_warnings(manifest, summary, 1, 1, [])
        self.assertTrue(any("gần như là `rules`" in w and "96.0%" in w for w in warnings))

    def test_cli_command(self):
        with mock.patch("sys.stdout"):
            code = cli.main(["tournament-report", str(self.run), "--no-plots"])
        self.assertEqual(code, 0)
        self.assertTrue((self.run / "summary" / "report.md").is_file())
        with self.assertRaises(SystemExit):
            cli.main(["tournament-report", str(self.run / "nope")])


class WorkerIndependenceTests(unittest.TestCase):
    def test_one_and_two_workers_give_identical_deterministic_summaries(self):
        outputs = []
        for workers in (1, 2):
            with tempfile.TemporaryDirectory() as tmp:
                config = t.resolve_config(tiny_raw())
                with mock.patch("sys.stdout"):
                    summary = t.run_tournament(config, output_root=tmp, workers=workers)
                run_dir = Path(summary["run_dir"])
                report.generate_report(run_dir, plots=False)
                files = csv_bytes(run_dir)
                files.pop("cost.csv")  # wall-clock thinking time differs between runs by nature
                outputs.append(files)
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
