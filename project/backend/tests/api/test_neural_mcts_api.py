"""Isolated application integration: fake inference, real adapter/search/HTTP flow."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.services.matches import MatchService
from src.api.services.mcts import MCTSService
from src.api.services.neural_mcts import NeuralMCTSService
from src.environment import GameEngine, GameState
from src.networks.pretrained import (
    CheckpointIntegrityError, NeuralModelError, NeuralRuntimeUnavailableError,
)
from tests.search.neural_support import fake_adapter, MATE_IN_ONE_RED, MATE_IN_ONE_BLACK


def participant(name):
    return {"type": "human"} if name == "human" else {"type": "agent", "agent_id": name}


def create(client, red="neural", black="human", **options):
    return client.post("/api/v1/matches", json={
        "red": participant(red), "black": participant(black), **options,
    })


class NeuralApiTests(unittest.TestCase):
    def setUp(self):
        self.adapter = fake_adapter()
        self.loader = Mock(return_value=self.adapter)
        p = patch("src.api.services.neural_mcts.load_adapter", self.loader)
        p.start()
        self.addCleanup(p.stop)
        self.neural = NeuralMCTSService(config={"max_simulations": 3, "time_limit": 3.0}, env={})
        self.service = MatchService(neural_service=self.neural, mcts_service=MCTSService(config={
            "time_limit": 1.0, "max_simulations": 1, "rollout_depth": 0,
        }))
        self.client = TestClient(create_app(self.service))

    def step(self, state):
        return self.client.post(f"/api/v1/matches/{state['match_id']}/step",
                                json={"expected_ply": state["ply"]})

    def test_startup_health_and_classic_creation_do_not_load_model(self):
        self.assertEqual(self.client.get("/api/v1/health").status_code, 200)
        self.assertEqual(create(self.client, "random").status_code, 201)
        self.loader.assert_not_called()

    def test_catalog_validates_forward_and_caches_readiness(self):
        for _ in range(2):
            entries = {a["id"]: a for a in self.client.get("/api/v1/agents").json()}
            self.assertTrue(entries["neural"]["available"])
            self.assertIsNone(entries["neural"]["unavailable_reason"])
        self.loader.assert_called_once()
        self.assertEqual(self.adapter.inference_calls, 1)

    def test_red_neural_move_uses_neural_budget_and_correct_turns(self):
        state = create(self.client).json()
        self.assertEqual(state["opening_mode"], "none")
        result = self.step(state)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIn(result.json()["last_move"], state["legal_moves"])
        self.assertEqual(result.json()["side_to_move"], "black")
        agent = self.service._matches[state["match_id"]].red_agent
        self.assertEqual(agent.search.last_simulations, 3)  # classic service would use 1
        self.assertEqual(self.adapter.inference_calls, 4)  # readiness + search
        self.assertEqual(self.step(state).status_code, 409)
        self.assertEqual(self.step(result.json()).status_code, 409)  # human turn

    def test_black_neural_replies_to_human(self):
        state = create(self.client, "human", "neural").json()
        self.assertEqual(self.step(state).status_code, 409)
        moved = self.client.post(f"/api/v1/matches/{state['match_id']}/moves", json={
            **state["legal_moves"][0], "expected_ply": 0,
        }).json()
        result = self.step(moved)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIn(result.json()["last_move"], moved["legal_moves"])
        self.assertEqual(result.json()["ply"], 2)
        self.assertEqual(result.json()["side_to_move"], "red")

    def test_reuse_across_moves_participants_matches_and_reset(self):
        state = create(self.client, "neural", "neural").json()
        for _ in range(4):
            state = self.step(state).json()
        match = self.service._matches[state["match_id"]]
        self.assertIsNot(match.red_agent, match.black_agent)
        self.assertIs(match.red_agent.search.evaluator.adapter, match.black_agent.search.evaluator.adapter)
        self.assertEqual(self.adapter.inference_calls, 13)
        reset = self.client.post(f"/api/v1/matches/{state['match_id']}/reset").json()
        self.assertEqual((reset["ply"], reset["last_move"]), (0, None))
        self.assertEqual(self.step(reset).status_code, 200)
        self.assertEqual(create(self.client).status_code, 201)
        self.loader.assert_called_once()

    def test_neural_and_classic_both_work_in_same_match(self):
        for red, black in (("neural", "mcts"), ("mcts", "neural")):
            state = create(self.client, red, black).json()
            for _ in range(2):
                result = self.step(state)
                self.assertEqual(result.status_code, 200, result.text)
                self.assertIn(result.json()["last_move"], state["legal_moves"])
                state = result.json()

    def test_explicit_book_then_neural_and_reset(self):
        state = create(self.client, "neural", "neural", opening_mode="book",
                       opening_seed=42, opening_plies=1).json()
        state = self.step(state).json()
        self.assertEqual(state["opening"]["completed_plies"], 1)
        self.assertEqual(self.adapter.inference_calls, 1)  # just readiness
        self.assertEqual(self.step(state).status_code, 200)
        self.assertEqual(self.adapter.inference_calls, 4)
        reset = self.client.post(f"/api/v1/matches/{state['match_id']}/reset").json()
        self.assertEqual(reset["opening"]["completed_plies"], 0)

    def test_mate_and_terminal_authority_both_colors(self):
        for position in (MATE_IN_ONE_RED, MATE_IN_ONE_BLACK):
            state = create(self.client, "neural", "neural").json()
            self.service._matches[state["match_id"]].env.state = position
            before = self.adapter.inference_calls
            finished = self.step(state).json()
            self.assertEqual(finished["status"], "finished")
            self.assertEqual(finished["reason"], "checkmate")
            self.assertEqual(finished["winner"], "red" if int(position.side_to_move) == 1 else "black")
            self.assertEqual(self.adapter.inference_calls, before)
            self.assertEqual(self.step(finished).status_code, 409)

    def test_draw_is_terminal_without_network(self):
        state = create(self.client).json()
        match = self.service._matches[state["match_id"]]
        match.env.state = replace(GameState.initial(), no_capture_count=120)
        before = self.adapter.inference_calls
        self.assertEqual(self.step(state).status_code, 409)
        snapshot = self.client.get(f"/api/v1/matches/{state['match_id']}").json()
        self.assertEqual((snapshot["status"], snapshot["winner"]), ("finished", None))
        self.assertEqual(self.adapter.inference_calls, before)

    def test_runtime_failure_is_explicit_preserves_board_and_disables_agent(self):
        state = create(self.client).json()
        self.adapter.backend.value = float("nan")
        result = self.step(state)
        self.assertEqual(result.status_code, 503)
        self.assertIn("inference failed", result.json()["detail"])
        current = self.client.get(f"/api/v1/matches/{state['match_id']}").json()
        self.assertEqual(current["board"], state["board"])
        self.assertEqual(current["ply"], 0)
        self.assertFalse(self.neural.available)
        self.assertEqual(create(self.client, "random").status_code, 201)
        self.loader.assert_called_once()

    def test_invalid_participant_and_unknown_agent_are_422(self):
        self.assertEqual(create(self.client, "not-an-agent").status_code, 422)
        self.assertEqual(self.client.post("/api/v1/matches", json={
            "red": {"type": "human", "agent_id": "neural"}, "black": participant("human"),
        }).status_code, 422)

    def test_concurrent_first_requests_only_load_once(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertTrue(all(pool.map(lambda _: self.neural.available, range(8))))
        self.loader.assert_called_once()
        self.assertEqual(self.adapter.inference_calls, 1)


class NeuralFailureTests(unittest.TestCase):
    def check_unavailable(self, service, reason):
        client = TestClient(create_app(MatchService(neural_service=service)))
        self.assertEqual(client.get("/api/v1/health").status_code, 200)
        entry = next(a for a in client.get("/api/v1/agents").json() if a["id"] == "neural")
        self.assertFalse(entry["available"])
        self.assertIn(reason, entry["unavailable_reason"])
        result = create(client)
        self.assertEqual(result.status_code, 503, result.text)
        self.assertIn(reason, result.json()["detail"])
        self.assertEqual(create(client, "random").status_code, 201)

    def test_loader_failure_cached_for_missing_runtime_and_corrupt_weights(self):
        for error in (NeuralRuntimeUnavailableError("PyTorch missing"),
                      CheckpointIntegrityError("SHA-256 mismatch"), OSError("runtime library missing")):
            with self.subTest(error=error), patch("src.api.services.neural_mcts.load_adapter", side_effect=error) as loader:
                self.check_unavailable(NeuralMCTSService(config={}, env={}), str(error))
                loader.assert_called_once()

    def test_missing_checkpoint_real_loader_error(self):
        # If BOTH dependencies and weights are absent, the loader reports runtime first.
        with tempfile.TemporaryDirectory() as tmp:
            self.check_unavailable(NeuralMCTSService(config={"path": str(Path(tmp) / "missing.pt")}, env={}),
                                   "not found" if importlib.util.find_spec("torch") else "PyTorch")

    def test_invalid_configuration_never_loads_weights(self):
        cases = [({"c_puct": float("nan")}, "c_puct"), ({"time_limit": 0}, "time_limit"),
                 ({"max_simulations": True}, "max_simulations"), ({"max_simulations": 1.5}, "max_simulations"),
                 ({"seed": -1}, "seed"), ({"typo": 3}, "unknown"),
                 ({"model": "unknown"}, "unknown neural model"), ({"device": "cuda"}, "device"),
                 ({"threads": "bad"}, "THREADS"), ({"sha256": "bad"}, "SHA-256"),
                 ({"enabled": "yes"}, "enabled")]
        with patch("src.api.services.neural_mcts.load_adapter") as loader:
            for config, reason in cases:
                with self.subTest(config=config):
                    self.check_unavailable(NeuralMCTSService(config=config, env={}), reason)
            loader.assert_not_called()

    def test_feature_flag_disables_without_loading(self):
        with patch("src.api.services.neural_mcts.load_adapter") as loader:
            self.check_unavailable(NeuralMCTSService(config={}, env={"XIANGQI_NEURAL_ENABLED": "false"}), "disabled")
            self.check_unavailable(NeuralMCTSService(config={}, env={"XIANGQI_NEURAL_ENABLED": "maybe"}), "true/false")
            loader.assert_not_called()

    def test_missing_malformed_and_non_mapping_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            self.check_unavailable(NeuralMCTSService(config_path=path, env={}), "No such file")
            path.write_text("[invalid")
            self.check_unavailable(NeuralMCTSService(config_path=path, env={}), "unavailable")
            path.write_text("- list")
            self.check_unavailable(NeuralMCTSService(config_path=path, env={}), "mapping")

    def test_both_models_and_environment_precedence(self):
        for name in ("distill-v4", "distill-v4-large"):
            with patch("src.api.services.neural_mcts.load_adapter", return_value=fake_adapter()) as loader:
                service = NeuralMCTSService(config={"model": "distill-v4-large"}, env={"XIANGQI_NEURAL_MODEL": name})
                self.assertTrue(service.available)
                self.assertEqual(loader.call_args.args[0].resolve().spec.name, name)

    def test_default_config_has_approved_model_and_budget(self):
        config, model = NeuralMCTSService(env={})._settings()
        self.assertEqual(model.resolve().spec.name, "distill-v4-large")
        self.assertEqual(config["time_limit"], 3.0)
        self.assertEqual(config["c_puct"], 1.5)
        self.assertIsNone(config["max_simulations"])

    def test_failed_readiness_forward_never_reports_available(self):
        adapter = fake_adapter(float("inf"))
        with patch("src.api.services.neural_mcts.load_adapter", return_value=adapter):
            self.check_unavailable(NeuralMCTSService(config={}, env={}), "not finite")
