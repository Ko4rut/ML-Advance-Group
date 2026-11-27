"""budget_mode (default / equal_time), min_games_per_pair, new agents, resume and report.

Fast by design: real search agents only get tiny budgets; the interruption/resume tests use
``random``, ``rules`` and a deterministic fake searcher so results can be compared exactly.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from src.agents.base import Agent, AgentDecision, SearchBudget
from src.environment import XiangqiEnv
from src.evaluation import agent_registry as registry
from src.evaluation import compare, report
from src.evaluation import tournament as t

BACKEND = t.BACKEND_ROOT
OLD_RUN = BACKEND / "results" / "tournaments" / "20261112-091526_501370b5"


def raw_config(**overrides):
    raw = {
        "name": "unit_budget",
        "master_seed": 7,
        "opening_book_version": "xiangqi-curated-v3",
        "openings": ["central_cannon_screen_horses", "pawn_left", "pawn_center"],
        "colors_per_opening": 2,
        "max_plies": 16,
        "agents": {"random": {}, "rules": {}},
        "workers": 1,
        "min_games_per_pair": 1,
    }
    raw.update(overrides)
    return raw


def quiet_run(config, root, **kwargs):
    with mock.patch("sys.stdout"):
        return t.run_tournament(config, output_root=root, workers=1, **kwargs)


def read_records(run_dir):
    path = Path(run_dir) / "games.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FirstMoveSearcher(Agent):
    """Deterministic stand-in for a searching agent; remembers the budgets it received."""

    def __init__(self, engine, log):
        self.engine, self.log = engine, log

    def select_action(self, state, budget=None):
        self.log.append(budget)
        return AgentDecision(self.engine.legal_moves(state)[0], nodes=7)


def fake_fixed(name="rules"):
    """Instant non-searching stand-in (the real rule-based agent costs ~60 ms per move)."""
    return {name: registry.AgentSpec(
        name=name, class_name="FirstMoveFixed", module="tests", deterministic=True,
        resolve=lambda params, ctx: {}, factory=lambda engine, params, seed: FirstMoveSearcher(engine, []),
        budget_applicable=False,
    )}


def fast_registry(log=None):
    """`rules` (no search) and `alphabeta` (searching) replaced by instant deterministic fakes."""
    fakes = fake_searcher([] if log is None else log, "alphabeta")
    fakes.update(fake_fixed("rules"))
    return fakes


def fake_searcher(log, name="alphabeta"):
    def resolve(params, ctx):
        return {"time_budget": ctx.time_per_move} if ctx.equal_time else {}

    return {name: registry.AgentSpec(
        name=name, class_name="FirstMoveSearcher", module="tests", deterministic=True,
        resolve=resolve, factory=lambda engine, params, seed: FirstMoveSearcher(engine, log),
        budget=registry._time_budget, budget_applicable=True,
    )}


# --------------------------------------------------------------------------
# Config: budget_mode, time_per_move, hash, min_games_per_pair
# --------------------------------------------------------------------------


class BudgetModeConfigTests(unittest.TestCase):
    def test_default_mode_is_implicit_and_keeps_pre_existing_hashes(self):
        implicit = t.resolve_config(raw_config())
        explicit = t.resolve_config(raw_config(budget_mode="default"))
        self.assertEqual(implicit.budget_mode, "default")
        self.assertEqual(implicit.config_hash, explicit.config_hash)
        # The resolved config records the mode, but the *hash view* of a default run has no new
        # key, so runs written before budget_mode existed still resume (hash unchanged).
        self.assertEqual(implicit.data["budget_mode"], "default")
        self.assertNotIn("budget_mode", implicit.hash_view)
        self.assertNotIn("time_per_move", implicit.hash_view)

    @unittest.skipUnless(OLD_RUN.is_dir(), "old run directory not present (results/ is git-ignored)")
    def test_legacy_config_still_hashes_to_the_old_run(self):
        manifest = json.loads((OLD_RUN / "manifest.json").read_text(encoding="utf-8"))
        config = t.resolve_config(t.load_raw_config(BACKEND / "configs" / "tournament.yaml"),
                                  enforce_min_games=False)
        if manifest["agents"]["tabular_q"]["params"]["checkpoint_sha256"] != \
                config.data["agents"].get("tabular_q", {}).get("checkpoint_sha256"):
            self.skipTest("local Tabular Q checkpoint differs from the one used by the old run")
        self.assertEqual(config.config_hash, manifest["config_hash"])

    def test_equal_time_is_hashed_with_its_time(self):
        default = t.resolve_config(raw_config())
        equal = t.resolve_config(raw_config(budget_mode="equal_time"))
        longer = t.resolve_config(raw_config(budget_mode="equal_time", time_per_move=2.0))
        self.assertEqual(equal.data["time_per_move"], 1.0)  # default T
        self.assertEqual(equal.hash_view["budget_mode"], "equal_time")
        self.assertEqual(equal.hash_view["time_per_move"], 1.0)
        self.assertEqual(len({default.config_hash, equal.config_hash, longer.config_hash}), 3)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "budget_mode must be one of"):
            t.resolve_config(raw_config(budget_mode="fast"))
        with self.assertRaisesRegex(ValueError, "time_per_move only applies"):
            t.resolve_config(raw_config(time_per_move=1.0))
        for bad in (0, -1.0, "1", None, True, float("inf")):
            if bad is None:
                continue  # None means "use the default T"
            with self.subTest(time_per_move=bad), self.assertRaisesRegex(ValueError, "time_per_move"):
                t.resolve_config(raw_config(budget_mode="equal_time", time_per_move=bad))

    def test_equal_time_rejects_per_agent_budget_overrides(self):
        for agent, params in (("alphabeta", {"max_depth": 3}), ("minimax", {"max_depth": 3}),
                              ("heuristic", {"max_nodes": 100}), ("heuristic", {"time_limit": 2.0}),
                              ("mcts", {"time_limit": 2.0}), ("mcts", {"max_simulations": 10})):
            with self.subTest(agent=agent, params=params), \
                    self.assertRaisesRegex(ValueError, "cannot be set when budget_mode is 'equal_time'"):
                t.resolve_config(raw_config(budget_mode="equal_time", agents={"random": {}, agent: params}))
        # ... while default mode still accepts them.
        t.resolve_config(raw_config(agents={"random": {}, "alphabeta": {"max_depth": 3}}))


class MinGamesPerPairTests(unittest.TestCase):
    def test_default_minimum_is_100_and_rejects_fewer(self):
        raw = raw_config()
        del raw["min_games_per_pair"]
        with self.assertRaisesRegex(ValueError, r"6 games per pair .* below min_games_per_pair=100"):
            t.resolve_config(raw)

    def test_shipped_configs_reach_100_games_per_pair(self):
        for name in ("tournament_default.yaml", "tournament_equal_time.yaml"):
            raw = t.load_raw_config(BACKEND / "configs" / name)
            raw["agents"]["tabular_q"]["required"] = False
            raw["agents"]["neural_mcts"]["required"] = False
            config = t.resolve_config(raw)  # would raise below 100
            self.assertGreaterEqual(config.games_per_pair, 100, name)

    def test_repeating_an_opening_does_not_fake_the_count(self):
        raw = raw_config(openings=["pawn_left"] * 60)
        del raw["min_games_per_pair"]
        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            t.resolve_config(raw)

    def test_smoke_and_pilot_are_exempt_and_the_value_is_configurable(self):
        raw = raw_config()
        del raw["min_games_per_pair"]
        self.assertEqual(t.resolve_config(raw, mode="smoke").mode, "smoke")
        self.assertEqual(t.resolve_config(raw, mode="pilot").games_per_pair, 1)
        raw["min_games_per_pair"] = 6
        self.assertEqual(t.resolve_config(raw).games_per_pair, 6)
        raw["min_games_per_pair"] = 7
        with self.assertRaisesRegex(ValueError, "below min_games_per_pair=7"):
            t.resolve_config(raw)
        for bad in (0, -5, 1.5, "100", True):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "min_games_per_pair must be"):
                t.resolve_config(raw_config(min_games_per_pair=bad))

    def test_resuming_an_existing_run_is_not_blocked_by_the_guard(self):
        with mock.patch.dict(registry.REGISTRY, fast_registry()), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.yaml"
            raw = raw_config()
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            config = t.resolve_config(raw)
            run_dir = quiet_run(config, directory, config_path=str(path))["run_dir"]
            del raw["min_games_per_pair"]  # now the default guard (100) would reject 6 games/pair
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                t.resolve_config(raw)
            resumed, _ = t.load_resume_config(run_dir)
            self.assertEqual(resumed.config_hash, config.config_hash)


# --------------------------------------------------------------------------
# Registry: agents and budgets per mode
# --------------------------------------------------------------------------


class RegistryTests(unittest.TestCase):
    def test_minimax_is_registered_with_the_constructor_default_depth(self):
        from src.agents.minimax_agent import MinimaxAgent
        import inspect

        resolved, excluded = registry.resolve_agents({"minimax": {}}, env={})
        self.assertEqual(resolved["minimax"],
                         {"max_depth": inspect.signature(MinimaxAgent.__init__).parameters["max_depth"].default})
        self.assertIsNone(registry.budget_for("minimax", resolved["minimax"]))
        agent = registry.make_agent("minimax", resolved["minimax"], XiangqiEnv().engine, 1)
        self.assertIsInstance(agent, MinimaxAgent)
        self.assertTrue(registry.REGISTRY["minimax"].deterministic)

    def test_dqn_is_not_a_tournament_agent(self):
        self.assertNotIn("dqn", registry.REGISTRY)
        with self.assertRaisesRegex(ValueError, "unknown agent 'dqn'"):
            registry.resolve_agents({"dqn": {}}, env={})

    def test_default_budgets_are_unchanged(self):
        resolved, _ = registry.resolve_agents(
            {"random": {}, "rules": {}, "alphabeta": {}, "minimax": {}, "heuristic": {}, "mcts": {}}, env={})
        for name in ("random", "rules", "alphabeta", "minimax", "heuristic"):
            self.assertIsNone(registry.budget_for(name, resolved[name]), name)
        self.assertEqual(registry.budget_for("mcts", resolved["mcts"]), SearchBudget(time_limit=1.0))

    def test_equal_time_gives_every_searching_agent_the_same_time_only_budget(self):
        agents = {"random": {}, "rules": {}, "alphabeta": {}, "minimax": {}, "heuristic": {}, "mcts": {}}
        resolved, _ = registry.resolve_agents(agents, env={}, budget_mode="equal_time", time_per_move=0.75)
        budget = SearchBudget(time_limit=0.75)
        for name in ("alphabeta", "minimax", "heuristic", "mcts"):
            self.assertEqual(registry.budget_for(name, resolved[name]), budget, name)
            self.assertTrue(registry.REGISTRY[name].budget_applicable)
        for name in ("random", "rules"):
            self.assertIsNone(registry.budget_for(name, resolved[name]), name)
            self.assertFalse(registry.REGISTRY[name].budget_applicable)
        # Time is the only limit: the depth/node caps are lifted, mcts has no simulation cap.
        self.assertEqual(resolved["alphabeta"]["max_depth"], registry.UNBOUNDED_DEPTH)
        self.assertEqual(resolved["minimax"]["max_depth"], registry.UNBOUNDED_DEPTH)
        self.assertEqual(resolved["heuristic"]["max_depth"], registry.UNBOUNDED_DEPTH)
        self.assertEqual(resolved["heuristic"]["max_nodes"], registry.UNBOUNDED_NODES)
        self.assertEqual(resolved["heuristic"]["time_limit"], 0.75)
        self.assertIsNone(resolved["mcts"]["max_simulations"])
        self.assertEqual(resolved["mcts"]["time_limit"], 0.75)
        for name in ("alphabeta", "minimax", "heuristic"):
            self.assertIsNone(registry.budget_for(name, resolved[name]).max_depth)

    def test_equal_time_agents_are_built_with_the_lifted_caps(self):
        resolved, _ = registry.resolve_agents(
            {"alphabeta": {}, "minimax": {}, "heuristic": {}}, env={}, budget_mode="equal_time")
        engine = XiangqiEnv().engine
        self.assertEqual(registry.make_agent("alphabeta", resolved["alphabeta"], engine, 1).max_depth, 64)
        self.assertEqual(registry.make_agent("minimax", resolved["minimax"], engine, 1).max_depth, 64)
        heuristic = registry.make_agent("heuristic", resolved["heuristic"], engine, 1)
        self.assertEqual((heuristic.max_depth, heuristic.max_nodes, heuristic.time_limit),
                         (64, registry.UNBOUNDED_NODES, 1.0))

    def test_describe_marks_budget_applicability_and_determinism(self):
        resolved, _ = registry.resolve_agents({"random": {}, "rules": {}, "alphabeta": {}}, env={},
                                              budget_mode="equal_time")
        info = {n: registry.describe(n, p, "equal_time") for n, p in resolved.items()}
        self.assertFalse(info["random"]["budget_applicable"])
        self.assertFalse(info["rules"]["budget_applicable"])
        self.assertTrue(info["alphabeta"]["budget_applicable"])
        self.assertEqual(info["alphabeta"]["budget"], {"time_limit": 1.0})
        self.assertTrue(info["rules"]["is_deterministic"])
        self.assertFalse(info["alphabeta"]["is_deterministic"])  # the clock decides its depth now
        self.assertTrue(registry.describe("alphabeta", {"max_depth": 2}, "default")["is_deterministic"])

    def test_equal_time_alphabeta_searches_deeper_than_the_default_depth(self):
        """The time budget, not max_depth=2, must decide how deep iterative deepening goes."""
        engine = XiangqiEnv().engine
        state = XiangqiEnv().state
        default, _ = registry.resolve_agents({"alphabeta": {}}, env={})
        timed, _ = registry.resolve_agents({"alphabeta": {}}, env={}, budget_mode="equal_time",
                                           time_per_move=1.0)
        shallow = registry.make_agent("alphabeta", default["alphabeta"], engine, 1).select_action(state)
        deep = registry.make_agent("alphabeta", timed["alphabeta"], engine, 1).select_action(
            state, registry.budget_for("alphabeta", timed["alphabeta"]))
        self.assertGreater(deep.nodes, 2 * shallow.nodes)


# --------------------------------------------------------------------------
# neural_mcts: availability handling (mocked) and the real thing (when provisioned)
# --------------------------------------------------------------------------


class NeuralAvailabilityTests(unittest.TestCase):
    def resolve(self, params, **kwargs):
        return registry.resolve_agents({"neural_mcts": params}, env=kwargs.pop("env", {}), **kwargs)

    def test_missing_torch_excludes_an_optional_agent_with_a_clear_reason(self):
        with mock.patch.object(registry, "_torch_available", return_value=False):
            resolved, excluded = self.resolve({"required": False})
            self.assertEqual(resolved, {})
            self.assertIn("PyTorch is not installed", excluded["neural_mcts"])
            with self.assertRaisesRegex(ValueError, "required but PyTorch is not installed"):
                self.resolve({"required": True})

    def test_missing_checkpoint_excludes_or_stops(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(registry, "_torch_available", return_value=True):
            missing = str(Path(directory) / "nope.pt")
            _, excluded = self.resolve({"checkpoint": missing})
            self.assertIn("not found", excluded["neural_mcts"])
            self.assertIn("nope.pt", excluded["neural_mcts"])
            with self.assertRaisesRegex(ValueError, "required but checkpoint .* not found"):
                self.resolve({"checkpoint": missing, "required": True})

    def test_corrupt_checkpoint_fails_the_manifest_integrity_check(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(registry, "_torch_available", return_value=True):
            fake = Path(directory) / "distill-v4-large.pt"
            fake.write_bytes(b"not a checkpoint")
            _, excluded = self.resolve({"checkpoint": str(fake)})
            self.assertIn("integrity check", excluded["neural_mcts"])
            with self.assertRaisesRegex(ValueError, "required but .*integrity"):
                self.resolve({"checkpoint": str(fake), "required": True})

    def test_unknown_model_and_disabled_flag(self):
        with mock.patch.object(registry, "_torch_available", return_value=True):
            _, excluded = self.resolve({"model": "no-such-model"})
            self.assertIn("neural model configuration is invalid", excluded["neural_mcts"])
            _, excluded = self.resolve({}, env={"XIANGQI_NEURAL_ENABLED": "false"})
            self.assertIn("disabled", excluded["neural_mcts"])

    def test_tournament_excluding_neural_keeps_running_and_says_so(self):
        raw = raw_config(agents={"random": {}, "rules": {}, "neural_mcts": {"threads": 1}})
        with mock.patch.object(registry, "_torch_available", return_value=False), \
                mock.patch.dict(registry.REGISTRY, fast_registry()), tempfile.TemporaryDirectory() as directory:
            config = t.resolve_config(raw, env={})
            self.assertEqual(list(config.data["agents"]), ["random", "rules"])
            run_dir = Path(quiet_run(config, directory)["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("PyTorch", manifest["excluded_agents"]["neural_mcts"])
            self.assertIn("EXCLUDED", (run_dir / "run.log").read_text(encoding="utf-8"))
            report.generate_report(run_dir, plots=False)
            self.assertIn("Agent bị loại khỏi giải",
                          (run_dir / "summary" / "report.md").read_text(encoding="utf-8"))


def _neural_available() -> bool:
    try:
        resolved, _ = registry.resolve_agents({"neural_mcts": {}}, env={})
    except ValueError:
        return False
    return "neural_mcts" in resolved


@unittest.skipUnless(_neural_available(), "torch and/or the pinned neural checkpoint are not available")
class RealNeuralMCTSTests(unittest.TestCase):
    def test_resolve_records_model_hash_threads_and_defaults(self):
        resolved, _ = registry.resolve_agents({"neural_mcts": {}}, env={})
        params = resolved["neural_mcts"]
        from src.networks.pretrained import load_manifest

        spec = load_manifest().get(params["model"])
        self.assertEqual(params["checkpoint_sha256"], spec.sha256)
        self.assertEqual(params["threads"], 1)
        self.assertEqual((params["c_puct"], params["time_limit"], params["max_simulations"]), (1.5, 3.0, None))
        self.assertEqual(registry.budget_for("neural_mcts", params), SearchBudget(time_limit=3.0))
        info = registry.describe("neural_mcts", params)
        self.assertFalse(info["is_deterministic"])
        self.assertEqual(info["params"]["model"], spec.name)
        # the machine-specific path is kept out of the config hash, the content hash is in it
        view = registry.hash_view({"neural_mcts": params})["neural_mcts"]
        self.assertNotIn("checkpoint", view)
        self.assertIn("checkpoint_sha256", view)

    def test_equal_time_and_exact_replay_budgets(self):
        timed, _ = registry.resolve_agents({"neural_mcts": {}}, env={}, budget_mode="equal_time",
                                           time_per_move=0.5)
        self.assertEqual(registry.budget_for("neural_mcts", timed["neural_mcts"]), SearchBudget(time_limit=0.5))
        exact, _ = registry.resolve_agents({"neural_mcts": {"max_simulations": 8}}, env={})
        self.assertEqual(registry.budget_for("neural_mcts", exact["neural_mcts"]),
                         SearchBudget(max_simulations=8))
        with self.assertRaisesRegex(ValueError, "cannot be set when budget_mode is 'equal_time'"):
            registry.resolve_agents({"neural_mcts": {"max_simulations": 8}}, env={}, budget_mode="equal_time")

    def test_adapter_is_loaded_once_per_process_but_agents_are_fresh(self):
        import torch

        resolved, _ = registry.resolve_agents({"neural_mcts": {"max_simulations": 4}}, env={})
        params = resolved["neural_mcts"]
        registry._ADAPTER_CACHE.clear()
        with mock.patch("src.networks.pretrained.load_adapter", wraps=registry._load_adapter.__globals__[
                "importlib"].import_module("src.networks.pretrained").load_adapter) as load:
            engine = XiangqiEnv().engine
            first = registry.make_agent("neural_mcts", params, engine, 1)
            second = registry.make_agent("neural_mcts", params, engine, 2)
            self.assertEqual(load.call_count, 1)  # heavy weights: once
        self.assertIsNot(first, second)  # agents: one per game
        self.assertIs(first.search.evaluator.adapter, second.search.evaluator.adapter)
        self.assertEqual(torch.get_num_threads(), 1)  # pinned to params["threads"]
        self.assertEqual(first.model_info.sha256, params["checkpoint_sha256"])
        decision = first.select_action(XiangqiEnv().state, registry.budget_for("neural_mcts", params))
        self.assertIn(decision.move, engine.legal_moves(XiangqiEnv().state))
        self.assertGreater(decision.simulations, 0)

    def test_neural_plays_a_real_short_game(self):
        raw = raw_config(agents={"random": {}, "neural_mcts": {"max_simulations": 4}}, max_plies=14)
        config = t.resolve_config(raw)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(quiet_run(config, directory)["run_dir"])
            records = read_records(run_dir)
        self.assertEqual(len(records), 6)
        neural_moves = [m for r in records for m in r["moves"] if m["actor"] == "neural_mcts"]
        self.assertTrue(neural_moves)
        self.assertTrue(all(m["simulations"] > 0 for m in neural_moves))
        self.assertEqual({r["outcome"]["status"] for r in records} & {"error"}, set())


# --------------------------------------------------------------------------
# Games, manifest, schema
# --------------------------------------------------------------------------


class RecordAndManifestTests(unittest.TestCase):
    def test_records_and_manifest_carry_budget_mode_and_budgets(self):
        log = []
        with mock.patch.dict(registry.REGISTRY, fast_registry(log)), tempfile.TemporaryDirectory() as directory:
            config = t.resolve_config(raw_config(
                budget_mode="equal_time", time_per_move=0.25, agents={"random": {}, "alphabeta": {}}))
            summary = quiet_run(config, directory)
            run_dir = Path(summary["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            records = read_records(run_dir)
        self.assertEqual(summary["budget_mode"], "equal_time")
        self.assertEqual((manifest["budget_mode"], manifest["time_per_move"]), ("equal_time", 0.25))
        self.assertEqual(manifest["schema_version"], t.SCHEMA_VERSION)
        self.assertEqual(manifest["config"]["budget_mode"], "equal_time")
        self.assertFalse(manifest["agents"]["random"]["budget_applicable"])
        self.assertTrue(manifest["agents"]["alphabeta"]["budget_applicable"])
        self.assertEqual(manifest["agents"]["alphabeta"]["budget"], {"time_limit": 0.25})
        self.assertTrue(records)
        for record in records:
            self.assertEqual(record["schema_version"], 2)
            self.assertEqual(record["budget_mode"], "equal_time")
            budgets = record["budgets"]
            side_of_search = "red" if record["red_agent"] == "alphabeta" else "black"
            other = "black" if side_of_search == "red" else "red"
            self.assertEqual(budgets[side_of_search], {"time_limit": 0.25})
            self.assertIsNone(budgets[other])  # random does not search: no budget, no sleeping
        self.assertTrue(log and all(b == SearchBudget(time_limit=0.25) for b in log))

    def test_default_mode_records_say_default_and_pass_no_budget_to_fixed_depth_agents(self):
        log = []
        with mock.patch.dict(registry.REGISTRY, fast_registry(log)), tempfile.TemporaryDirectory() as directory:
            config = t.resolve_config(raw_config(agents={"random": {}, "alphabeta": {}}))
            run_dir = Path(quiet_run(config, directory)["run_dir"])
            records = read_records(run_dir)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(all(r["budget_mode"] == "default" for r in records))
        self.assertTrue(all(b is None for b in log))
        self.assertIsNone(manifest["time_per_move"])

    def test_old_schema_1_records_are_still_read_resumed_and_reported(self):
        with mock.patch.dict(registry.REGISTRY, fast_registry()), tempfile.TemporaryDirectory() as directory:
            config = t.resolve_config(raw_config())
            run_dir = Path(quiet_run(config, directory)["run_dir"])
            full = read_records(run_dir)
            # Rewrite the first two games the way schema 1 stored them (no budget_mode) ...
            legacy = []
            for record in full[:2]:
                record = dict(record, schema_version=1)
                del record["budget_mode"]
                legacy.append(record)
            (run_dir / "games.jsonl").write_bytes(b"".join(t.dump_record(r) for r in legacy))
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in ("budget_mode", "time_per_move"):
                del manifest[key]
            for info in manifest["agents"].values():
                del info["budget_applicable"]
            manifest["schema_version"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            # ... the report of such a run still works and calls it "default" ...
            report.generate_report(run_dir, plots=False)
            text = (run_dir / "summary" / "report.md").read_text(encoding="utf-8")
            self.assertIn("`default`", text)
            # ... and it resumes (mixed schema 1 + 2 lines) to the same games as one straight run.
            resumed = quiet_run(config, directory, resume_dir=run_dir)
            self.assertEqual(resumed["new_games"], len(full) - 2)
            again = read_records(run_dir)
        self.assertEqual([r["schema_version"] for r in again], [1, 1] + [2] * (len(full) - 2))
        self.assertEqual([r["fingerprint"] for r in again], [r["fingerprint"] for r in full])


# --------------------------------------------------------------------------
# Resume in both modes: interrupt, resume, compare with a straight run
# --------------------------------------------------------------------------


class ResumeBothModesTests(unittest.TestCase):
    def interrupted_then_resumed(self, raw, after):
        """Run until the (after+1)-th game is about to start, simulate Ctrl+C, then resume."""
        real = t.play_tournament_game
        calls = {"played": 0}

        def interrupting(*args, **kwargs):
            if calls["played"] == after:
                raise KeyboardInterrupt
            calls["played"] += 1
            return real(*args, **kwargs)

        config = t.resolve_config(raw)
        with tempfile.TemporaryDirectory() as straight_root, tempfile.TemporaryDirectory() as root:
            straight = read_records(quiet_run(config, straight_root)["run_dir"])
            with mock.patch.object(t, "play_tournament_game", interrupting):
                first = quiet_run(config, root)
            self.assertTrue(first["interrupted"])
            self.assertEqual((first["status"], first["completed"]), ("interrupted", after))
            run_dir = Path(first["run_dir"])
            progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "interrupted")
            self.assertEqual(len(read_records(run_dir)), after)
            resumed = quiet_run(config, root, resume_dir=run_dir)
            records = read_records(run_dir)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        return config, straight, records, resumed, manifest

    def check(self, raw, after):
        config, straight, records, resumed, manifest = self.interrupted_then_resumed(raw, after)
        total = len(t.build_schedule(config.data))
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["new_games"], total - after)  # nothing replayed, nothing missing
        indices = [r["game_index"] for r in records]
        self.assertEqual(indices, list(range(total)))  # all present, none duplicated
        self.assertEqual(len(set(indices)), total)
        # identical to the uninterrupted run (deterministic agents): same games, same results
        self.assertEqual([r["fingerprint"] for r in records], [r["fingerprint"] for r in straight])
        self.assertEqual([r["outcome"] for r in records], [r["outcome"] for r in straight])
        self.assertEqual({r["budget_mode"] for r in records}, {config.budget_mode})
        self.assertEqual(len(manifest["resume_history"]), 1)
        self.assertEqual(manifest["status"], "complete")
        return records

    def test_default_mode_resume_after_interrupt(self):
        with mock.patch.dict(registry.REGISTRY, fast_registry()):
            raw = raw_config(agents={"random": {}, "rules": {}, "alphabeta": {}})
            self.check(raw, after=7)  # 3 pairs x 6 games = 18 games

    def test_equal_time_mode_resume_after_interrupt(self):
        log = []
        with mock.patch.dict(registry.REGISTRY, fast_registry(log)):
            raw = raw_config(budget_mode="equal_time", time_per_move=0.2,
                             agents={"random": {}, "rules": {}, "alphabeta": {}})
            records = self.check(raw, after=11)
        self.assertTrue(all(b == SearchBudget(time_limit=0.2) for b in log))  # before and after the resume
        self.assertTrue(any(r["budgets"]["red"] or r["budgets"]["black"] for r in records))

    def test_resume_with_the_other_mode_is_refused(self):
        with mock.patch.dict(registry.REGISTRY, fast_registry()), tempfile.TemporaryDirectory() as root:
            default = t.resolve_config(raw_config())
            run_dir = quiet_run(default, root)["run_dir"]
            equal = t.resolve_config(raw_config(budget_mode="equal_time"))
            with self.assertRaisesRegex(ValueError, "config_hash differs"):
                quiet_run(equal, root, resume_dir=run_dir)

    def test_real_agents_in_equal_time_lose_and_duplicate_no_game(self):
        """Real (clock-bound, so not exactly replayable) agents: still every game exactly once."""
        raw = raw_config(budget_mode="equal_time", time_per_move=0.05, max_plies=14,
                         openings=["pawn_left", "pawn_center"],
                         agents={"random": {}, "alphabeta": {}, "minimax": {}, "heuristic": {}})
        config = t.resolve_config(raw)
        real = t.play_tournament_game
        calls = {"n": 0}

        def interrupting(*args, **kwargs):
            if calls["n"] == 9:
                raise KeyboardInterrupt
            calls["n"] += 1
            return real(*args, **kwargs)

        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(t, "play_tournament_game", interrupting):
                first = quiet_run(config, root)
            resumed = quiet_run(config, root, resume_dir=first["run_dir"])
            records = read_records(first["run_dir"])
        self.assertEqual((first["completed"], resumed["completed"]), (9, 24))  # 6 pairs x 4 games
        self.assertEqual([r["game_index"] for r in records], list(range(24)))
        self.assertEqual({r["outcome"]["status"] for r in records} & {"error"}, set())


# --------------------------------------------------------------------------
# Report, pilot estimate, compare
# --------------------------------------------------------------------------


class ReportBudgetTests(unittest.TestCase):
    def equal_time_run(self, directory):
        log = []
        with mock.patch.dict(registry.REGISTRY, fast_registry(log)):
            config = t.resolve_config(raw_config(
                budget_mode="equal_time", time_per_move=0.5, agents={"random": {}, "rules": {}, "alphabeta": {}}))
            return Path(quiet_run(config, directory)["run_dir"])

    def test_report_states_mode_applicability_and_the_think_time_table(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.equal_time_run(directory)
            report.generate_report(run_dir, plots=False)
            text = (run_dir / "summary" / "report.md").read_text(encoding="utf-8")
            cost_csv = (run_dir / "summary" / "cost.csv").read_text(encoding="utf-8").splitlines()
        self.assertIn("`equal_time`", text)
        self.assertIn("T = 0.50 giây mỗi nước", text)
        self.assertIn("Ngân sách chung áp dụng?", text)
        self.assertIn("không (không tìm kiếm)", text)  # random / rules
        self.assertIn("Ngân sách thời gian không áp dụng cho:** `random`, `rules`", text)
        self.assertIn("Thời gian suy nghĩ thực tế so với ngân sách", text)
        self.assertIn("Lớn nhất (giây)", text)
        self.assertIn("think_max_s", cost_csv[0])
        # the fake searcher answers instantly, so it is well within the 0.5 s budget
        row = next(line for line in text.splitlines() if line.startswith("| `alphabeta` | 0.500"))
        self.assertIn("đạt", row)
        self.assertNotIn("vượt ngân sách", text)

    def test_p95_above_110_percent_of_the_limit_is_flagged(self):
        manifest = {
            "budget_mode": "equal_time", "time_per_move": 1.0, "workers": 1,
            "agents": {
                "alphabeta": {"budget_applicable": True, "is_deterministic": False,
                              "budget": {"time_limit": 1.0}, "params": {}},
                "neural_mcts": {"budget_applicable": True, "is_deterministic": False,
                                "budget": {"time_limit": 1.0},
                                "params": {"model": "m", "checkpoint_sha256": "ab" * 32, "threads": 1}},
                "rules": {"budget_applicable": False, "is_deterministic": True, "params": {}},
            },
        }
        cost = [
            {"agent": "alphabeta", "moves": 10, "think_mean_s": 1.0, "think_p95_s": 1.10, "think_max_s": 1.2,
             "simulations_mean": None},
            {"agent": "neural_mcts", "moves": 10, "think_mean_s": 1.0, "think_p95_s": 1.101, "think_max_s": 1.4,
             "simulations_mean": 80.0},
            {"agent": "rules", "moves": 10, "think_mean_s": 0.05, "think_p95_s": 9.0, "think_max_s": 9.0,
             "simulations_mean": None},
        ]
        rows = {r["agent"]: r for r in report.time_budget_rows(manifest, cost)}
        self.assertFalse(rows["alphabeta"]["over"])  # exactly 1.10 x T is tolerated
        self.assertTrue(rows["neural_mcts"]["over"])
        self.assertIsNone(rows["rules"]["limit"])  # no search, no budget: never "over"
        self.assertFalse(rows["rules"]["over"])
        summary = {"totals": {"games": 1, "finished": 1, "draws": 0, "truncated": 0, "error": 0},
                   "cost": cost, "rank_differs": [], "pairs": []}
        warnings = report.build_warnings(manifest, summary, 1, 1, [])
        flagged = [w for w in warnings if "vượt ngân sách thời gian" in w]
        self.assertEqual(len(flagged), 1)
        self.assertIn("neural_mcts", flagged[0])
        self.assertIn("giới hạn thời gian \"mềm\"", flagged[0])
        self.assertTrue(any("equal_time" in w and "không còn trần" in w for w in warnings))

    def test_default_mode_limits_are_each_agents_own(self):
        manifest = {"budget_mode": "default", "agents": {
            "alphabeta": {"budget": None, "params": {"max_depth": 2}},
            "heuristic": {"budget": None, "params": {"time_limit": 1.0}},
            "mcts": {"budget": {"time_limit": 1.0}, "params": {"time_limit": 1.0}},
            "mcts_exact": {"budget": {"max_simulations": 50}, "params": {"time_limit": 1.0}},
        }}
        limits = {n: report.time_limit_of(manifest, n) for n in manifest["agents"]}
        self.assertEqual(limits, {"alphabeta": None, "heuristic": 1.0, "mcts": 1.0, "mcts_exact": None})

    @unittest.skipUnless(OLD_RUN.is_dir(), "old run directory not present (results/ is git-ignored)")
    def test_old_run_report_can_still_be_regenerated(self):
        """On a *copy*: results/tournaments/* must never be modified."""
        import shutil

        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / OLD_RUN.name
            shutil.copytree(OLD_RUN, copy, ignore=shutil.ignore_patterns("summary"))
            result = report.generate_report(copy, plots=False)
            text = (copy / "summary" / "report.md").read_text(encoding="utf-8")
        self.assertEqual((result["completed"], result["total"]), (750, 750))
        self.assertIn("`default`", text)
        self.assertIn("Thời gian suy nghĩ thực tế so với ngân sách", text)


class PilotEstimateTests(unittest.TestCase):
    def test_pilot_estimate_covers_every_pair_and_names_the_mode(self):
        with mock.patch.dict(registry.REGISTRY, fast_registry()), tempfile.TemporaryDirectory() as directory:
            raw = raw_config(budget_mode="equal_time", time_per_move=0.5,
                             agents={"random": {}, "rules": {}, "alphabeta": {}})
            del raw["min_games_per_pair"]
            config = t.resolve_config(raw, mode="pilot")  # exempt from min_games_per_pair
            self.assertEqual((config.games_per_pair, config.full_games_per_pair), (1, 6))
            run_dir = quiet_run(config, directory)["run_dir"]
            estimate = t.pilot_estimate(run_dir, 100, workers=5)
            self.assertEqual(len(estimate["pairs"]), 3)  # 3 agents -> 3 pairs, all in the pilot
            self.assertEqual(estimate["budget_mode"], "equal_time")
            self.assertEqual(estimate["time_per_move"], 0.5)
            self.assertEqual(set(estimate["think_mean_s"]), {"random", "rules", "alphabeta"})
            total = sum(row["seconds"] for row in estimate["pairs"])
            self.assertAlmostEqual(estimate["sequential_seconds"], total * 100)
            self.assertAlmostEqual(estimate["parallel_seconds"], total * 100 / 5)
            with mock.patch("builtins.print") as printed:
                t.print_pilot_estimate(run_dir, 100, 5)
            output = "\n".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertIn("equal_time, 0.5 s per move", output)
        self.assertIn("3 pairs x 100 games/pair = 300 games", output)
        self.assertIn("5 worker(s)", output)


class CompareTests(unittest.TestCase):
    def test_compare_two_runs_lists_rank_and_elo_per_agent(self):
        with mock.patch.dict(registry.REGISTRY, fast_registry()), tempfile.TemporaryDirectory() as directory:
            agents = {"random": {}, "rules": {}, "alphabeta": {}}
            a = quiet_run(t.resolve_config(raw_config(agents=agents)), directory)["run_dir"]
            b = quiet_run(t.resolve_config(raw_config(budget_mode="equal_time", agents=agents)),
                          directory)["run_dir"]
            comparison = compare.compare_runs(a, b)
            text = compare.format_comparison(comparison)
            self.assertEqual({r["agent"] for r in comparison["rows"]}, {"random", "rules", "alphabeta"})
            self.assertEqual(comparison["a"]["budget_mode"], "default")
            self.assertEqual(comparison["b"]["budget_mode"], "equal_time")
            self.assertIn("[default]", text)
            self.assertIn("[equal_time (T=1s)]", text)
            for agent in ("random", "rules", "alphabeta"):
                self.assertIn(agent, text)
            # the same deterministic games in both modes -> no change at all
            self.assertTrue(all(r["delta_rank"] == 0 for r in comparison["rows"]))

    def test_cli_command(self):
        from src import cli

        with mock.patch.dict(registry.REGISTRY, fast_registry()), tempfile.TemporaryDirectory() as directory:
            run = quiet_run(t.resolve_config(raw_config(agents={"random": {}, "rules": {}})), directory)["run_dir"]
            with mock.patch("builtins.print") as printed:
                self.assertEqual(cli.main(["tournament-compare", run, run]), 0)
            self.assertIn("rank A", str(printed.call_args_list[0].args[0]))
            with self.assertRaises(SystemExit):
                cli.main(["tournament-compare", run, str(Path(directory) / "nope")])


if __name__ == "__main__":
    unittest.main()
