"""Bounded real-model HTTP plumbing smoke; no strength benchmark.

Run from project/backend with the API/test/neural extras and provisioned weights:
  python experiments/neural_mcts/phase5/application_smoke.py --model distill-v4-large --mode game --output /tmp/game.json
  python experiments/neural_mcts/phase5/application_smoke.py --model distill-v4 --mode budget --output /tmp/budget.json
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api.services.matches import MatchService
from src.api.services.mcts import MCTSService
from src.api.services.neural_mcts import NeuralMCTSService, load_adapter


def run(model, mode):
    config = {"model": model, "time_limit": 3.0}
    if mode == "game":
        config["max_simulations"] = 100
    neural = NeuralMCTSService(config=config, env={})
    service = MatchService(neural_service=neural, mcts_service=MCTSService(config={
        "time_limit": 3.0, "max_simulations": 100, "rollout_depth": 0, "seed": 42,
    }))
    client = TestClient(create_app(service))
    with patch("src.api.services.neural_mcts.load_adapter", wraps=load_adapter) as loader:
        started = time.monotonic()
        catalog = client.get("/api/v1/agents").json()
        assert next(a for a in catalog if a["id"] == "neural")["available"], catalog
        cold_readiness = time.monotonic() - started
        reply = client.post("/api/v1/matches", json={
            "red": {"type": "agent", "agent_id": "neural"},
            "black": {"type": "agent", "agent_id": "mcts" if mode == "game" else "neural"},
            "opening_mode": "none",
        })
        assert reply.status_code == 201, reply.text
        state = reply.json()
        mid = state["match_id"]
        rows = []
        for _ in range(240 if mode == "game" else 6):
            if state["status"] == "finished":
                break
            side = state["side_to_move"]
            calls = neural._adapter.inference_calls
            started = time.monotonic()
            reply = client.post(f"/api/v1/matches/{mid}/step", json={"expected_ply": state["ply"]})
            elapsed = time.monotonic() - started
            assert reply.status_code == 200, reply.text
            updated = reply.json()
            assert updated["last_move"] in state["legal_moves"]
            agent = service._matches[mid].agent(service._matches[mid].env.state.side_to_move.opponent)
            rows.append({"ply": updated["ply"], "side": side, "move": updated["last_move"],
                         "http_seconds": elapsed, "simulations": agent.search.last_simulations,
                         "network_calls": neural._adapter.inference_calls - calls})
            state = updated
        if mode == "game":
            assert state["status"] == "finished", "bounded smoke was truncated; not a completed game"
            assert client.post(f"/api/v1/matches/{mid}/step", json={"expected_ply": state["ply"]}).status_code == 409
        assert client.post(f"/api/v1/matches/{mid}/reset").json()["ply"] == 0
        assert loader.call_count == 1
        assert neural._adapter.inference_calls > 1
        import torch
        return {"model": asdict(neural._adapter.info), "mode": mode, "config": config,
                "python": platform.python_version(), "torch": torch.__version__,
                "threads": torch.get_num_threads(), "cpu": platform.processor(),
                "cold_catalog_seconds": cold_readiness, "load_count": loader.call_count,
                "inference_calls": neural._adapter.inference_calls, "moves": rows,
                "result": {k: state[k] for k in ("status", "winner", "reason", "ply")},
                "all_moves_legal": True, "reset_passed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("distill-v4-large", "distill-v4"), required=True)
    parser.add_argument("--mode", choices=("game", "budget"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "moves"}, indent=2))
