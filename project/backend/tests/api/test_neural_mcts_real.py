"""Real checkpoint -> HTTP -> service -> search -> network -> legal engine move.

Provision BOTH models as documented in models/neural/README.md. REQUIRE_REAL=1
fails instead of skipping when the optional validation environment is absent.
"""

import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.services.matches import MatchService
from src.api.services.neural_mcts import NeuralMCTSService, load_adapter
from src.networks.pretrained import NeuralModelSettings
from tests.api.test_neural_mcts_api import create


class RealNeuralApplicationCases:
    variant: str

    @classmethod
    def setUpClass(cls):
        path = NeuralModelSettings(model=cls.variant).resolve().path
        if importlib.util.find_spec("torch") is None or not path.is_file():
            reason = f"real API verification needs torch and {path}"
            if os.environ.get("XIANGQI_NEURAL_REQUIRE_REAL") == "1":
                raise AssertionError(reason)
            raise unittest.SkipTest(reason)

    def setUp(self):
        # Fixed small budget is plumbing evidence, not a playing-strength test.
        self.loader_patch = patch("src.api.services.neural_mcts.load_adapter", wraps=load_adapter)
        self.loader = self.loader_patch.start()
        self.addCleanup(self.loader_patch.stop)
        self.neural = NeuralMCTSService(config={"model": self.variant, "max_simulations": 4,
                                                 "time_limit": 3.0}, env={})
        self.service = MatchService(neural_service=self.neural)
        self.client = TestClient(create_app(self.service))

    def test_real_human_matches_both_colors_model_reused_after_reset(self):
        for red, black in (("human", "neural"), ("neural", "human")):
            state = create(self.client, red, black).json()
            self.assertEqual(state["opening_mode"], "none")
            mid = state["match_id"]
            for _ in range(4):
                human = state["red"] if state["side_to_move"] == "red" else state["black"]
                before = self.neural._adapter.inference_calls
                started = time.monotonic()
                if human["type"] == "human":
                    response = self.client.post(f"/api/v1/matches/{mid}/moves", json={
                        **state["legal_moves"][0], "expected_ply": state["ply"],
                    })
                else:
                    response = self.client.post(f"/api/v1/matches/{mid}/step", json={"expected_ply": state["ply"]})
                    self.assertGreater(self.neural._adapter.inference_calls, before)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn(response.json()["last_move"], state["legal_moves"])
                state = response.json()
                if human["type"] == "agent":
                    print(f"[real API {self.variant}] {state['ply']=} latency={time.monotonic()-started:.4f}s "
                          f"network_calls={self.neural._adapter.inference_calls-before}")
            self.assertEqual(self.client.post(f"/api/v1/matches/{mid}/reset").json()["ply"], 0)
            agent = self.service._matches[mid].black_agent if black == "neural" else self.service._matches[mid].red_agent
            self.assertEqual(agent.model_info.name, self.variant)
        self.loader.assert_called_once()
        self.assertEqual(self.neural._adapter.inference_calls, 17)  # readiness + 4 searches * 4
        self.assertTrue(next(a for a in self.client.get("/api/v1/agents").json() if a["id"] == "neural")["available"])

    def test_real_neural_self_play_calls_network_for_both_participants(self):
        response = create(self.client, "neural", "neural")
        self.assertEqual(response.status_code, 201, response.text)
        state = response.json()
        mid = state["match_id"]
        for _ in range(6):
            response = self.client.post(f"/api/v1/matches/{mid}/step", json={"expected_ply": state["ply"]})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn(response.json()["last_move"], state["legal_moves"])
            state = response.json()
        match = self.service._matches[mid]
        self.assertIs(match.red_agent.search.evaluator.adapter, match.black_agent.search.evaluator.adapter)
        self.assertEqual(self.neural._adapter.inference_calls, 25)
        self.loader.assert_called_once()

    def test_actual_corrupt_checkpoint_does_not_break_classic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupt.pt"
            path.write_bytes(b"invalid checkpoint")
            service = MatchService(neural_service=NeuralMCTSService(
                config={"model": self.variant, "path": str(path)}, env={}))
            client = TestClient(create_app(service))
            entry = next(a for a in client.get("/api/v1/agents").json() if a["id"] == "neural")
            self.assertFalse(entry["available"])
            self.assertIn("size", entry["unavailable_reason"])
            self.assertEqual(create(client).status_code, 503)
            self.assertEqual(create(client, "random").status_code, 201)


class RealNeuralApplicationLargeTests(RealNeuralApplicationCases, unittest.TestCase):
    variant = "distill-v4-large"


class RealNeuralApplicationV4Tests(RealNeuralApplicationCases, unittest.TestCase):
    variant = "distill-v4"
