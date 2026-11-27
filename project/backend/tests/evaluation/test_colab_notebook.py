"""Exercise the notebook's actual workflow with real, small CLI tournaments.

No Colab account, network, Drive mount or neural checkpoint is needed here.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import yaml

from src.evaluation import tournament


BACKEND = tournament.BACKEND_ROOT
NOTEBOOK = BACKEND / "notebooks/xiangqi_colab_tournament.ipynb"


def notebook_workflow():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "colab_notebook_workflow",
        "hashlib": hashlib, "json": json, "os": os, "Path": Path,
        "shutil": shutil, "signal": signal, "subprocess": subprocess,
        "sys": sys, "tempfile": tempfile, "zipfile": zipfile,
    }
    for cell_id in ("prepare-tabular-q", "workflow-functions"):
        cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
    return namespace


class ColabNotebookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.session = self.root / "drive" / "equal_time_01"
        self.candidate = self.root / "candidate.yaml"
        self.raw = {
            "name": "colab_integration", "master_seed": 42,
            "opening_book_version": "xiangqi-curated-v2",
            "openings": ["screen_horses_c57", "pawn_left"],
            "colors_per_opening": 2, "max_plies": 16,
            "min_games_per_pair": 1,
            "budget_mode": "equal_time", "time_per_move": 0.01,
            "agents": {"random": {}, "rules": {}}, "workers": 1,
        }
        self.write_config()
        self.workflow = notebook_workflow()

    def write_config(self):
        self.candidate.write_text(yaml.safe_dump(self.raw), encoding="utf-8")

    def run_notebook(self, workers=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.workflow["run_and_export"](
                BACKEND, self.session, self.candidate, workers, "full",
            )

    def records(self, result):
        return (result["run_dir"] / "games.jsonl").read_bytes()

    def test_all_cells_compile_and_notebook_has_no_saved_outputs(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=cell["id"])
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_budget_selection_routes_config_and_drive_folder(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell = next(cell for cell in notebook["cells"] if cell["id"] == "settings")
        for budget_mode, mode, expected_folder in (
            ("equal_time", "full", "equal_time_01"),
            ("default", "full", "default_01"),
            ("default", "smoke", "default_smoke_01"),
        ):
            with self.subTest(budget_mode=budget_mode, mode=mode):
                tree = ast.parse("".join(cell["source"]))
                for statement in tree.body:
                    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                        name = getattr(statement.targets[0], "id", None)
                        if name in {"BUDGET_MODE", "MODE"}:
                            statement.value = ast.Constant(budget_mode if name == "BUDGET_MODE" else mode)
                namespace = {}
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(ast.fix_missing_locations(tree), "notebook-settings", "exec"), namespace)
                self.assertEqual(namespace["SESSION_NAME"], expected_folder)
                self.assertEqual(namespace["SESSION_DIR"], namespace["DRIVE_ROOT"] / expected_folder)
                selected = BACKEND / namespace["CONFIG_RELATIVE"]
                self.assertEqual(yaml.safe_load(selected.read_text())["budget_mode"], budget_mode)

    def test_default_workers_follow_runtime_cpus_and_affinity(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell = next(cell for cell in notebook["cells"] if cell["id"] == "settings")
        for cpu_count, affinity, expected in (
            (2, {0, 1}, 2),
            (8, {2, 3}, 2),
            (None, {0}, 1),
        ):
            with self.subTest(cpu_count=cpu_count, affinity=affinity):
                namespace = {}
                with mock.patch.object(os, "cpu_count", return_value=cpu_count), \
                        mock.patch.object(os, "sched_getaffinity", return_value=affinity, create=True), \
                        contextlib.redirect_stdout(io.StringIO()):
                    exec(compile("".join(cell["source"]), "notebook-settings", "exec"), namespace)
                self.assertEqual(namespace["WORKERS"], expected)

    def test_increasing_workers_resumes_existing_games(self):
        config = tournament.resolve_config(self.raw)
        self.session.mkdir(parents=True)
        frozen = self.session / "tournament.input.yaml"
        shutil.copyfile(self.candidate, frozen)
        (self.session / "experiment.json").write_text(json.dumps({
            "mode": "full", "config_hash": config.config_hash,
        }))
        execute = tournament._execute

        def interrupt_after_first(pending, data, run_id, config_hash, workers, on_result):
            def save_and_interrupt(record):
                on_result(record)
                raise KeyboardInterrupt
            execute(pending, data, run_id, config_hash, 1, save_and_interrupt)

        with mock.patch.object(tournament, "_execute", side_effect=interrupt_after_first), \
                contextlib.redirect_stdout(io.StringIO()):
            original = tournament.run_tournament(
                config, config_path=str(frozen), output_root=self.session / "runs", workers=1,
            )
        run_dir = Path(original["run_dir"])
        saved_record = (run_dir / "games.jsonl").read_bytes()
        result = self.run_notebook(workers=2)
        self.assertEqual(result["run_dir"], run_dir)
        self.assertTrue(self.records(result).startswith(saved_record))
        self.assertEqual(len(self.records(result).splitlines()), 4)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["workers"], 2)
        self.assertEqual(manifest["resume_history"][-1]["workers"], 2)
        self.assertEqual(manifest["config_hash"], config.config_hash)

    def test_default_budget_saves_logs_report_and_zip_in_separate_session(self):
        original_session = self.session
        original_session.mkdir(parents=True)
        sentinel = original_session / "existing-result.txt"
        sentinel.write_text("equal_time results stay here\n")
        self.session = self.root / "drive/default_01"
        self.raw["budget_mode"] = "default"
        self.raw.pop("time_per_move")
        self.write_config()
        result = self.run_notebook()
        self.assertEqual(result["status"], "complete")
        records = [json.loads(line) for line in self.records(result).splitlines()]
        self.assertEqual(len(records), 4)
        self.assertTrue(all(row["budget_mode"] == "default" for row in records))
        self.assertEqual(result["archive"], self.root / "drive/default_01.zip")
        with zipfile.ZipFile(result["archive"]) as archive:
            self.assertTrue(any(name.startswith("default_01/") and name.endswith("/summary/report.md")
                                for name in archive.namelist()))
        self.assertEqual(sentinel.read_text(), "equal_time results stay here\n")

    def test_empty_drive_path_loads_valid_checkpoint_from_repository(self):
        with contextlib.redirect_stdout(io.StringIO()):
            params = self.workflow["prepare_tabular_q"](BACKEND, "")
        self.assertEqual(Path(params["checkpoint"]), BACKEND / "models/tabular_q/q_table.json.gz")
        self.assertTrue(params["required"])

    def test_invalid_explicit_drive_path_does_not_fall_back_to_repo(self):
        with self.assertRaisesRegex(FileNotFoundError, "Drive path"):
            self.workflow["prepare_tabular_q"](BACKEND, str(self.root / "missing.json.gz"))
        invalid = self.root / "invalid.json.gz"
        invalid.write_bytes(b"not a gzip checkpoint")
        with self.assertRaises(OSError):
            self.workflow["prepare_tabular_q"](BACKEND, str(invalid))

    def test_missing_repository_checkpoint_requests_source_or_drive_path(self):
        with self.assertRaisesRegex(FileNotFoundError, "SOURCE_REF.*TABULAR_Q_DRIVE_PATH"):
            self.workflow["prepare_tabular_q"](self.root / "empty-backend", "")

    def test_fresh_checkout_and_reconnect_keep_saved_source_commit(self):
        origin = self.root / "origin"
        origin.mkdir()

        def git(*args):
            return subprocess.run(
                ["git", "-C", str(origin), *args], check=True, capture_output=True, text=True,
            ).stdout.strip()

        git("init", "-b", "main")
        marker = origin / "project/backend/src/evaluation/tournament.py"
        marker.parent.mkdir(parents=True)
        marker.write_text("# original\n")
        git("add", ".")
        git("-c", "user.name=Notebook Test", "-c", "user.email=test@example.invalid", "commit", "-m", "first")
        first_commit = git("rev-parse", "HEAD")
        checkout = self.root / "checkout"
        self.session.mkdir(parents=True)
        namespace = dict(self.workflow, REPO_URL=str(origin), REPO_DIR=checkout,
                         SOURCE_REF="main", SESSION_DIR=self.session,
                         BACKEND_DIR=checkout / "project/backend", INCLUDE_NEURAL_MCTS=False)
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source_cell = next(cell for cell in notebook["cells"] if cell["id"] == "source")
        compiled = compile("".join(source_cell["source"]), "notebook-source-cell", "exec")
        real_run = subprocess.run

        def offline_run(command, **kwargs):
            if command[:4] == [sys.executable, "-m", "pip", "install"]:
                return subprocess.CompletedProcess(command, 0)
            return real_run(command, **kwargs)

        with mock.patch.dict(os.environ), mock.patch.object(subprocess, "run", side_effect=offline_run), \
                mock.patch.object(sys, "path", sys.path.copy()):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compiled, namespace)
        self.assertEqual(json.loads((self.session / "source.json").read_text())["commit"], first_commit)
        marker.write_text("# changed upstream\n")
        git("add", ".")
        git("-c", "user.name=Notebook Test", "-c", "user.email=test@example.invalid", "commit", "-m", "second")
        self.assertNotEqual(git("rev-parse", "HEAD"), first_commit)
        # Recreate the VM's checkout, as Colab does after runtime reclamation.
        shutil.rmtree(checkout)
        with mock.patch.dict(os.environ), mock.patch.object(subprocess, "run", side_effect=offline_run), \
                mock.patch.object(sys, "path", sys.path.copy()):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compiled, namespace)
        restored_commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        self.assertEqual(restored_commit, first_commit)

    def test_run_resume_report_and_verified_zip_without_duplicate_games(self):
        result = self.run_notebook(workers=2)
        self.assertEqual(result["status"], "complete")
        raw_log = self.records(result)
        records = [json.loads(line) for line in raw_log.splitlines()]
        self.assertEqual(len(records), 4)
        self.assertEqual(len({row["game_index"] for row in records}), 4)
        self.assertTrue(all(row["budget_mode"] == "equal_time" for row in records))
        self.assertTrue((result["run_dir"] / "summary/report.md").is_file())
        self.assertTrue((result["run_dir"] / "summary/standings.csv").is_file())
        self.assertTrue(list((result["run_dir"] / "summary/plots").glob("*.png")))
        self.assertEqual(json.loads((self.session / "export.json").read_text())["status"], "complete")
        with zipfile.ZipFile(result["archive"]) as archive:
            name = f"{self.session.name}/runs/{result['run_dir'].name}/games.jsonl"
            self.assertEqual(archive.read(name), raw_log)
            self.assertIsNone(archive.testzip())
        digest = hashlib.sha256(result["archive"].read_bytes()).hexdigest()
        self.assertTrue(result["archive"].with_suffix(".zip.sha256").read_text().startswith(digest))
        resumed = self.run_notebook(workers=1)
        self.assertEqual(resumed["run_dir"], result["run_dir"])
        self.assertEqual(self.records(resumed), raw_log)

    def test_changed_budget_is_rejected_before_resuming(self):
        result = self.run_notebook()
        raw_log = self.records(result)
        self.raw["time_per_move"] = 0.02
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Config/model/mode"):
            self.run_notebook()
        self.assertEqual(self.records(result), raw_log)

    def test_interrupted_run_exports_partial_report_then_resumes(self):
        # Interrupt only after a real game is safely written by the real runner.
        config = tournament.resolve_config(self.raw)
        self.session.mkdir(parents=True)
        frozen = self.session / "tournament.input.yaml"
        shutil.copyfile(self.candidate, frozen)
        (self.session / "experiment.json").write_text(json.dumps({
            "mode": "full", "config_hash": config.config_hash,
        }))
        execute = tournament._execute

        def interrupt_after_first(pending, data, run_id, config_hash, workers, on_result):
            def save_and_interrupt(record):
                on_result(record)
                raise KeyboardInterrupt
            execute(pending, data, run_id, config_hash, 1, save_and_interrupt)

        with mock.patch.object(tournament, "_execute", side_effect=interrupt_after_first):
            with contextlib.redirect_stdout(io.StringIO()):
                tournament.run_tournament(
                    config, config_path=str(frozen), output_root=self.session / "runs", workers=1,
                )
        with contextlib.redirect_stdout(io.StringIO()):
            partial = self.workflow["export_session"](BACKEND, self.session)
        self.assertEqual(partial["status"], "interrupted")
        first = json.loads(self.records(partial).splitlines()[0])
        self.assertTrue((partial["run_dir"] / "summary/report.md").is_file())
        resumed = self.run_notebook()
        records = [json.loads(line) for line in self.records(resumed).splitlines()]
        self.assertEqual(len(records), 4)
        self.assertIn(first, records)
        self.assertEqual(resumed["status"], "complete")

    def test_report_failure_still_archives_raw_logs_and_records_failure(self):
        result = self.run_notebook()
        error = subprocess.CalledProcessError(1, ["tournament-report"])
        with mock.patch.dict(self.workflow, {"checked_cli": mock.Mock(side_effect=error)}):
            with contextlib.redirect_stdout(io.StringIO()):
                exported = self.workflow["export_session"](BACKEND, self.session)
        self.assertIsNotNone(exported["report_error"])
        with zipfile.ZipFile(exported["archive"]) as archive:
            exported_metadata = json.loads(archive.read(f"{self.session.name}/export.json"))
            self.assertIsNotNone(exported_metadata["report_error"])
            self.assertTrue(any(name.endswith("/games.jsonl") for name in archive.namelist()))
        self.assertEqual(self.records(exported), self.records(result))


if __name__ == "__main__":
    unittest.main()
