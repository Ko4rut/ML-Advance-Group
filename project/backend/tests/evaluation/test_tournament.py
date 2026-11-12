"""Round-robin tournament: schedule, seeds, game play, resume, errors, workers.

Fast by design: only ``random`` / ``rules`` (plus fake agents) with tiny max_plies.
"""

import hashlib
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.base import Agent, AgentDecision, SearchBudget
from src.environment import GameState, XiangqiEnv
from src.evaluation import agent_registry
from src.evaluation import tournament as t
from src.training.opening_book import get_opening

SIX = ("random", "rules", "alphabeta", "heuristic", "mcts", "tabular_q")


def small_raw(**overrides):
    raw = {
        "name": "unit",
        "master_seed": 123,
        "opening_book_version": "xiangqi-curated-v2",
        "openings": ["screen_horses_c57", "pawn_left"],
        "colors_per_opening": 2,
        "max_plies": 16,
        "agents": {"random": {}, "rules": {}},
        "workers": 1,
    }
    raw.update(overrides)
    return raw


def quiet_run(config, directory, **kwargs):
    """run_tournament without leaving logging handlers or stdout noise behind."""
    with mock.patch("sys.stdout"):
        summary = t.run_tournament(config, output_root=directory, workers=1, **kwargs)
    return summary


def read_records(run_dir):
    path = Path(run_dir) / "games.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ScheduleTests(unittest.TestCase):
    def data(self, names=SIX, openings=None):
        openings = openings or [f"o{i}" for i in range(25)]
        return {"master_seed": 20261110, "openings": openings,
                "colors_per_opening": 2, "agents": {name: {} for name in names}}

    def test_six_agents_give_15_pairs_750_games_balanced_colors(self):
        specs = t.build_schedule(self.data())
        self.assertEqual(len(specs), 750)
        self.assertEqual([s.game_index for s in specs], list(range(750)))
        pairs = {}
        for spec in specs:
            pairs.setdefault((spec.agent_a, spec.agent_b), []).append(spec)
        self.assertEqual(len(pairs), 15)
        for (a, b), games in pairs.items():
            self.assertEqual(len(games), 50)
            self.assertLess(SIX.index(a), SIX.index(b))  # combinations order
            by_opening = {}
            for game in games:
                by_opening.setdefault(game.opening_id, []).append(game.a_color)
            self.assertEqual(len(by_opening), 25)
            for colors in by_opening.values():
                self.assertEqual(sorted(colors), ["black", "red"])
            self.assertEqual(sum(g.a_color == "red" for g in games), 25)

    def test_schedule_is_deterministic_and_keys_unique(self):
        self.assertEqual(t.build_schedule(self.data()), t.build_schedule(self.data()))
        keys = [s.game_key for s in t.build_schedule(self.data())]
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(keys[0], "random|rules|o0|red")
        self.assertEqual(keys[1], "random|rules|o0|black")

    def test_five_agents_give_500_games(self):
        self.assertEqual(len(t.build_schedule(self.data(names=SIX[:5]))), 500)

    def test_red_and_black_agents_follow_a_color(self):
        spec_red, spec_black = t.build_schedule(self.data())[:2]
        self.assertEqual((spec_red.red_agent, spec_red.black_agent), ("random", "rules"))
        self.assertEqual((spec_black.red_agent, spec_black.black_agent), ("rules", "random"))

    def test_seeds_are_pinned_sha256_values(self):
        key = "random|rules|screen_horses_c57|red"
        self.assertEqual(t.derive_seed(20261110, key), 14371474737487284533)
        self.assertEqual(t.derive_seed(20261110, key, "red"), 14135918741382286364)
        self.assertEqual(t.derive_seed(20261110, key, "black"), 11612156349797399117)
        expected = int(hashlib.sha256(f"20261110|{key}".encode()).hexdigest()[:16], 16)
        self.assertEqual(t.derive_seed(20261110, key), expected)
        spec = t.build_schedule(self.data(openings=["screen_horses_c57"]))[0]
        self.assertEqual(spec.seed, expected)
        self.assertNotEqual(spec.seed_red, spec.seed_black)

    def test_tournament_class_schedule_matches_function(self):
        tour = t.Tournament(None, small_raw())
        self.assertEqual(len(tour.schedule()), 4)  # 1 pair x 2 openings x 2 colors


class ConfigTests(unittest.TestCase):
    def test_games_per_pair_is_derived(self):
        config = t.resolve_config(small_raw())
        self.assertEqual(config.games_per_pair, 4)

    def test_shipped_config_is_valid_with_25_openings(self):
        raw = t.load_raw_config(t.BACKEND_ROOT / "configs" / "tournament.yaml")
        self.assertEqual(len(raw["openings"]), 25)
        with tempfile.TemporaryDirectory() as empty:
            # Point the optional checkpoint at nothing: agent is excluded, config still valid.
            raw["agents"]["tabular_q"]["checkpoint"] = str(Path(empty) / "missing.json.gz")
            config = t.resolve_config(raw, env={})
        self.assertEqual(config.games_per_pair, 50)
        self.assertIn("tabular_q", config.excluded)
        self.assertEqual(len(t.build_schedule(config.data)), 500)  # 5 agents, 10 pairs

    def test_same_final_position_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same final position"):
            t.resolve_config(small_raw(openings=["elephant_left", "horse_left"]))

    def test_other_validation_errors(self):
        with self.assertRaisesRegex(ValueError, "unknown opening"):
            t.resolve_config(small_raw(openings=["nope"]))
        with self.assertRaisesRegex(ValueError, "must exceed the longest opening"):
            t.resolve_config(small_raw(max_plies=10))
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            t.resolve_config(small_raw(agents={"random": {}, "ghost": {}}))
        with self.assertRaisesRegex(ValueError, "at least two agents"):
            t.resolve_config(small_raw(agents={"random": {}}))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            t.resolve_config(small_raw(openings=["pawn_left", "pawn_left"]))

    def test_hash_ignores_workers_but_not_content(self):
        base = t.resolve_config(small_raw()).config_hash
        self.assertEqual(base, t.resolve_config(small_raw(workers=8)).config_hash)
        self.assertNotEqual(base, t.resolve_config(small_raw(master_seed=124)).config_hash)
        self.assertNotEqual(base, t.resolve_config(small_raw(max_plies=17)).config_hash)

    def test_missing_tabular_q_checkpoint_is_excluded_with_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            agents = {"random": {}, "rules": {},
                      "tabular_q": {"checkpoint": str(Path(directory) / "none.json.gz")}}
            config = t.resolve_config(small_raw(agents=agents), env={})
            self.assertEqual(list(config.data["agents"]), ["random", "rules"])
            self.assertIn("checkpoint not found", config.excluded["tabular_q"])
            agents["tabular_q"]["required"] = True
            with self.assertRaisesRegex(ValueError, "required"):
                t.resolve_config(small_raw(agents=agents), env={})

    def test_pilot_and_smoke_overrides(self):
        raw = small_raw(openings=["screen_horses_c57", "pawn_left", "lie_cannon_p7"], max_plies=300)
        pilot = t.resolve_config(raw, mode="pilot")
        self.assertEqual(pilot.games_per_pair, 1)
        self.assertEqual(pilot.full_games_per_pair, 6)
        smoke = t.resolve_config(raw, mode="smoke")
        self.assertEqual(smoke.data["max_plies"], 40)
        self.assertEqual(len(smoke.data["openings"]), 2)


class PlayGameTests(unittest.TestCase):
    def setUp(self):
        self.config = t.resolve_config(small_raw())
        self.specs = t.build_schedule(self.config.data)

    def play(self, spec, **overrides):
        data = {**self.config.data, **overrides}
        return t.play_tournament_game(spec, data, run_id="r", config_hash="h")

    def test_opening_is_applied_and_excluded_from_agent_costs(self):
        spec = self.specs[0]
        record = self.play(spec)
        opening = get_opening(spec.opening_id)
        env = XiangqiEnv()
        for move in opening.moves:
            env.step(move)
        replay = GameState.initial()
        for row in record["moves"][: len(opening.moves)]:
            replay = replay.moved(type(opening.moves[0])(row["source"], row["target"]))
        self.assertEqual(replay.board, env.state.board)
        self.assertEqual([m["actor"] for m in record["moves"][:10]], ["opening"] * 10)
        self.assertTrue(all(m["think_seconds"] == 0 and m["nodes"] == 0
                            for m in record["moves"][:10]))
        self.assertEqual(record["opening"]["plies"], 10)
        agent_moves = [m for m in record["moves"] if m["actor"] != "opening"]
        self.assertEqual(sum(record["agent_plies"].values()), len(agent_moves))
        self.assertEqual(record["plies"], len(record["moves"]))
        self.assertEqual(record["moves"][10]["side"], "red")  # 10 plies -> Red to move
        self.assertEqual(record["moves"][10]["actor"], spec.red_agent)

    def test_truncated_is_not_a_draw(self):
        record = self.play(self.specs[0], max_plies=12)
        self.assertEqual(record["outcome"],
                         {"status": "truncated", "winner": None, "reason": "max_plies"})
        self.assertEqual(record["plies"], 12)

    def test_finished_game_reports_engine_result(self):
        # random vs rules at the real cap usually ends in checkmate/draw, never truncated here
        spec = next(s for s in t.build_schedule({**self.config.data, "max_plies": 300})
                    if s.opening_id == "pawn_left" and s.a_color == "black")
        record = self.play(spec, max_plies=300)
        self.assertEqual(record["outcome"]["status"], "finished")
        self.assertIn(record["outcome"]["winner"], ("red", "black", None))
        self.assertTrue(record["outcome"]["reason"])

    def test_same_spec_same_fingerprint_for_deterministic_agents(self):
        data = t.resolve_config(small_raw(agents={"rules": {}, "alphabeta": {"max_depth": 1}},
                                          max_plies=24)).data
        spec = t.build_schedule(data)[0]
        a = t.play_tournament_game(spec, data)
        b = t.play_tournament_game(spec, data)
        self.assertEqual(a["fingerprint"], b["fingerprint"])
        self.assertEqual([(m["source"], m["target"]) for m in a["moves"]],
                         [(m["source"], m["target"]) for m in b["moves"]])

    def test_random_agent_is_reproducible_from_its_seed(self):
        a = self.play(self.specs[0])
        b = self.play(self.specs[0])
        self.assertEqual(a["fingerprint"], b["fingerprint"])
        self.assertNotEqual(a["fingerprint"], self.play(self.specs[1])["fingerprint"])

    def test_record_schema_fields(self):
        record = self.play(self.specs[0])
        for key in ("schema_version", "run_id", "config_hash", "game_index", "game_key", "pair",
                    "red_agent", "black_agent", "a_color", "opening", "seeds", "budgets", "moves",
                    "outcome", "plies", "agent_plies", "fingerprint", "elapsed_seconds",
                    "worker_pid", "finished_at"):
            self.assertIn(key, record)
        self.assertNotIn("error", record)
        json.dumps(record)  # serialisable


class RecordingAgent(Agent):
    """Picks the first legal move and remembers every budget it was given."""

    def __init__(self, engine, log):
        self.engine, self.log = engine, log

    def select_action(self, state, budget=None):
        self.log.append(budget)
        return AgentDecision(self.engine.legal_moves(state)[0])


class ExplodingAgent(Agent):
    def __init__(self, engine):
        self.engine = engine

    def select_action(self, state, budget=None):
        raise RuntimeError("boom")


def fake_spec(name, factory, budget=None):
    return agent_registry.AgentSpec(
        name=name, class_name=name, module="tests", deterministic=True,
        resolve=lambda params, ctx: dict(params), factory=factory,
        budget=budget or (lambda params: None),
    )


class FakeAgentTests(unittest.TestCase):
    def test_each_side_receives_its_own_budget(self):
        log_a, log_m = [], []
        fakes = {
            "alphabeta": fake_spec("alphabeta", lambda e, p, s: RecordingAgent(e, log_a)),
            "mcts": fake_spec("mcts", lambda e, p, s: RecordingAgent(e, log_m),
                              budget=lambda p: SearchBudget(time_limit=1.0)),
        }
        with mock.patch.dict(agent_registry.REGISTRY, fakes):
            data = t.resolve_config(small_raw(agents={"alphabeta": {}, "mcts": {}})).data
            spec = t.build_schedule(data)[0]
            record = t.play_tournament_game(spec, data)
        self.assertTrue(log_a and log_m)
        self.assertTrue(all(b is None for b in log_a))  # alphabeta: no time limit at all
        self.assertTrue(all(b == SearchBudget(time_limit=1.0) for b in log_m))
        self.assertEqual(record["budgets"], {"red": None, "black": {"time_limit": 1.0}})

    def test_real_registry_budgets_follow_the_plan(self):
        ab = agent_registry.budget_for("alphabeta", {"max_depth": 2})
        self.assertIsNone(ab)
        mcts = agent_registry.budget_for("mcts", {"max_simulations": None, "time_limit": 1.0})
        self.assertEqual((mcts.time_limit, mcts.max_simulations), (1.0, None))
        exact = agent_registry.budget_for("mcts", {"max_simulations": 50, "time_limit": 1.0})
        self.assertEqual((exact.time_limit, exact.max_simulations), (None, 50))
        # heuristic: no budget passed; the agent uses its own constructor limits
        self.assertIsNone(agent_registry.budget_for("heuristic", {"max_depth": 6}))

    def test_empty_params_resolve_to_each_agents_own_code_defaults(self):
        """No override in the config => exactly what the code ships with."""
        import inspect

        from src.agents.alphabeta_agent import AlphaBetaAgent
        from src.agents.heuristic import HeuristicAgent

        resolved, _ = agent_registry.resolve_agents(
            {"alphabeta": {}, "heuristic": {}, "mcts": {}, "tabular_q": {"required": False}},
            env={})
        ab = inspect.signature(AlphaBetaAgent.__init__).parameters
        self.assertEqual(resolved["alphabeta"], {"max_depth": ab["max_depth"].default})
        hp = inspect.signature(HeuristicAgent.__init__).parameters
        self.assertEqual(resolved["heuristic"], {
            "max_depth": hp["max_depth"].default, "max_nodes": hp["max_nodes"].default,
            "time_limit": hp["time_limit"].default})
        import yaml
        mcts_yaml = yaml.safe_load((t.BACKEND_ROOT / "configs" / "mcts.yaml").read_text())
        self.assertEqual(resolved["mcts"]["time_limit"], mcts_yaml["time_limit"])
        self.assertIsNone(resolved["mcts"]["max_simulations"])
        self.assertEqual(resolved["tabular_q"]["epsilon"], 0.0)

    def test_shipped_config_has_no_agent_overrides_except_wiring(self):
        raw = t.load_raw_config(t.BACKEND_ROOT / "configs" / "tournament.yaml")
        self.assertEqual(raw["agents"]["alphabeta"], {})
        self.assertEqual(raw["agents"]["heuristic"], {})
        self.assertEqual(raw["agents"]["rules"], {})

    def test_agent_error_becomes_error_game_and_tournament_continues(self):
        fakes = {"boom": fake_spec("boom", lambda e, p, s: ExplodingAgent(e))}
        with mock.patch.dict(agent_registry.REGISTRY, fakes), \
                tempfile.TemporaryDirectory() as directory:
            config = t.resolve_config(small_raw(agents={"random": {}, "boom": {}, "rules": {}}))
            summary = quiet_run(config, directory)
            records = read_records(summary["run_dir"])
        self.assertEqual(summary["completed"], 12)  # 3 pairs x 4 games, nothing aborted
        errors = [r for r in records if r["outcome"]["status"] == "error"]
        self.assertEqual(len(errors), 8)  # every game involving boom
        self.assertEqual(summary["errors"], 8)
        self.assertEqual(errors[0]["error"]["type"], "RuntimeError")
        self.assertEqual(errors[0]["error"]["actor"], "boom")
        ok = [r for r in records if r["outcome"]["status"] != "error"]
        self.assertEqual(len(ok), 4)  # random vs rules unaffected


class RunAndResumeTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = self._dir.name
        self.config = t.resolve_config(small_raw())

    def test_full_run_writes_all_files(self):
        summary = quiet_run(self.config, self.root)
        run_dir = Path(summary["run_dir"])
        self.assertEqual(summary["status"], "complete")
        for name in ("manifest.json", "config.resolved.yaml", "games.jsonl", "progress.json",
                     "run.log"):
            self.assertTrue((run_dir / name).is_file(), name)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["config_hash"], self.config.config_hash)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["total_games"], 4)
        self.assertIn("git", manifest["provenance"])
        self.assertTrue(manifest["agents"]["random"]["is_deterministic"] is False)
        self.assertTrue(manifest["agents"]["rules"]["is_deterministic"])
        self.assertEqual(len(manifest["openings"]), 2)
        progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual((progress["completed"], progress["total"]), (4, 4))
        records = read_records(run_dir)
        self.assertEqual([r["game_index"] for r in records], [0, 1, 2, 3])
        self.assertTrue(all(r["config_hash"] == self.config.config_hash for r in records))

    def test_resume_after_torn_last_line_runs_each_game_exactly_once(self):
        summary = quiet_run(self.config, self.root)
        run_dir = Path(summary["run_dir"])
        full = read_records(run_dir)
        path = run_dir / "games.jsonl"
        lines = path.read_bytes().split(b"\n")
        # keep 2 complete games + half of the third (simulated crash mid-write)
        path.write_bytes(b"\n".join(lines[:2]) + b"\n" + lines[2][: len(lines[2]) // 2])
        resumed = quiet_run(self.config, self.root, resume_dir=run_dir)
        self.assertEqual(resumed["new_games"], 2)  # games 2 and 3 only
        again = read_records(run_dir)
        self.assertEqual([r["game_index"] for r in again], [0, 1, 2, 3])
        self.assertEqual([r["fingerprint"] for r in again], [r["fingerprint"] for r in full])
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["resume_history"]), 1)
        # resuming a finished run is a no-op
        self.assertEqual(quiet_run(self.config, self.root, resume_dir=run_dir)["new_games"], 0)

    def test_resume_rejects_changed_config_and_lists_keys(self):
        run_dir = Path(quiet_run(self.config, self.root)["run_dir"])
        other = t.resolve_config(small_raw(max_plies=18))
        with self.assertRaisesRegex(ValueError, r"max_plies: 16 -> 18"):
            quiet_run(other, self.root, resume_dir=run_dir)

    def test_resume_requires_flag_when_git_commit_changed(self):
        run_dir = Path(quiet_run(self.config, self.root)["run_dir"])
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["provenance"]["git"]["commit"] = "0" * 40
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with mock.patch.object(t.provenance, "git_info",
                               return_value={"available": True, "commit": "1" * 40,
                                             "dirty": False, "status": [], "untracked": []}):
            with self.assertRaisesRegex(ValueError, "allow-code-change"):
                quiet_run(self.config, self.root, resume_dir=run_dir)
            quiet_run(self.config, self.root, resume_dir=run_dir, allow_code_change=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["resume_history"][-1]["code_changed"])

    def test_duplicate_game_index_is_an_error(self):
        run_dir = Path(quiet_run(self.config, self.root)["run_dir"])
        path = run_dir / "games.jsonl"
        first = path.read_bytes().split(b"\n")[0]
        path.write_bytes(path.read_bytes() + first + b"\n")
        with self.assertRaisesRegex(ValueError, "duplicate game_index"):
            quiet_run(self.config, self.root, resume_dir=run_dir)

    def test_corrupt_middle_line_is_an_error_not_silently_dropped(self):
        run_dir = Path(quiet_run(self.config, self.root)["run_dir"])
        path = run_dir / "games.jsonl"
        lines = path.read_bytes().split(b"\n")
        lines[1] = b"{not json"
        path.write_bytes(b"\n".join(lines))
        with self.assertRaisesRegex(ValueError, "corrupt record"):
            quiet_run(self.config, self.root, resume_dir=run_dir)

    def test_retry_errors_reruns_only_failed_games(self):
        fakes = {"boom": fake_spec("boom", lambda e, p, s: ExplodingAgent(e))}
        with mock.patch.dict(agent_registry.REGISTRY, fakes):
            config = t.resolve_config(small_raw(agents={"random": {}, "boom": {}}))
            run_dir = Path(quiet_run(config, self.root)["run_dir"])
            self.assertEqual(quiet_run(config, self.root, resume_dir=run_dir)["new_games"], 0)
            retried = quiet_run(config, self.root, resume_dir=run_dir, retry_errors=True)
        self.assertEqual(retried["new_games"], 4)
        self.assertTrue((run_dir / "games.retried.jsonl").is_file())  # audit trail kept

    def test_one_and_two_workers_play_identical_games(self):
        config = t.resolve_config(small_raw(agents={"random": {}, "rules": {}, "alphabeta": {"max_depth": 1}}))
        with mock.patch("sys.stdout"):
            one = t.run_tournament(config, output_root=self.root, workers=1)
            two = t.run_tournament(config, output_root=Path(self.root) / "w2", workers=2)
        fp = lambda run: {r["game_index"]: r["fingerprint"] for r in read_records(run["run_dir"])}
        self.assertEqual(fp(one), fp(two))
        self.assertEqual(len(fp(two)), 12)
        logging.getLogger(t.LOGGER_NAME).handlers.clear()


if __name__ == "__main__":
    unittest.main()
