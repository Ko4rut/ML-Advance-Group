"""Bounded CPU qualification of the pinned pretrained checkpoint, without MCTS.

Imports original Apache-2.0 source from a verified external directory. It never
initializes a fallback predictor, updates weights, or registers a backend agent.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time

import numpy as np

from contracts import action_id, fen, fixtures, observation, predict
from fetch_artifacts import MANIFEST, verify
from src.environment import GameEngine, GameState


class CheckpointProbe:
    def __init__(self, source: Path, checkpoint: Path, variant: str = "distill-v4"):
        self.manifest = json.loads(MANIFEST.read_text())
        self.variant = variant
        checkpoint_record = (self.manifest["checkpoint"] if variant == "distill-v4"
                  else self.manifest["alternate_checkpoints"][variant])
        for relative, record in self.manifest["files"].items():
            verify(source / relative, record)
        verify(checkpoint, checkpoint_record)
        # Paths are explicit CLI inputs; no artifact is copied into production.
        sys.path.insert(0, str(source / "src"))
        import torch

        self.torch = torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        native_loader = importlib.import_module("model.checkpoint")
        native_board = importlib.import_module("xiangqi.board")
        native_encoding = importlib.import_module("xiangqi.encoding")
        native_search = importlib.import_module("mcts.search")
        # Explicitly verify that imports resolved to the pinned downloaded files.
        for module in (native_loader, native_board, native_encoding, native_search,
                       importlib.import_module("model.network")):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(source.resolve()):
                raise ValueError(f"Unexpected source import: {path}")
        started = time.perf_counter()
        self.model, self.payload = native_loader.load_checkpoint(checkpoint, map_location="cpu")
        self.load_seconds = time.perf_counter() - started
        self.model.eval()
        self.reference = native_search.TorchEvaluator(self.model, "cpu")
        self.Board = native_board.Board
        self.native_encoding = native_encoding
        self.forward_calls = 0
        self.first_forward_seconds = None

    def forward(self, planes):
        torch = self.torch
        started = time.perf_counter()
        with torch.inference_mode():
            logits, value = self.model(torch.from_numpy(planes).unsqueeze(0))
        if self.first_forward_seconds is None:
            self.first_forward_seconds = time.perf_counter() - started
        if tuple(logits.shape) != (1, 8100) or tuple(value.shape) != (1,):
            raise ValueError("Unexpected actual checkpoint output shapes")
        self.forward_calls += 1
        return logits[0].numpy().copy(), float(value[0])

    def check(self, state: GameState, engine: GameEngine) -> dict:
        native = self.Board.from_fen(fen(state))
        np.testing.assert_array_equal(observation(state, engine),
                                      self.native_encoding.encode_board(native))
        moves = engine.legal_moves(state)
        own_ids = {action_id(move) for move in moves}
        native_ids = set(self.native_encoding.MoveCodec.legal_indices(native))
        # The source FEN intentionally has no repo history. Compare geometric
        # legal moves, not its distinct repetition/no-progress adjudication.
        if own_ids != native_ids:
            raise AssertionError(f"Legal mapping mismatch: {own_ids ^ native_ids}")
        calls = self.forward_calls
        priors, value = predict(state, engine, self.forward)
        terminal = engine.outcome(state, legal_moves=moves)
        row = {"side": state.side_to_move.name, "fen": fen(state),
               "engine_legal_count": len(moves), "value": value,
               "terminal": None if terminal is None else terminal.reason,
               "inference_called": self.forward_calls != calls,
               "observation_parity": True, "legal_mapping_parity": True}
        if terminal is not None:
            assert priors == {} and self.forward_calls == calls
            assert value == terminal.reward(state.side_to_move)
            row["policy_sum"] = 0.0
            return row
        expected, expected_value = self.reference.evaluate(native)
        reference_priors = {self.native_encoding.MoveCodec.encode(move): prior
                            for move, prior in expected.items()}
        error = max(abs(prior - reference_priors[action_id(move)])
                    for move, prior in priors.items())
        if error > 1e-6 or abs(value - expected_value) > 1e-6:
            raise AssertionError("Source inference-wrapper parity failed")
        assert set(priors) == set(moves)
        assert abs(sum(priors.values()) - 1) < 1e-12
        scores, repeated_value = self.forward(observation(state, engine))
        repeated_scores, third_value = self.forward(observation(state, engine))
        np.testing.assert_array_equal(scores, repeated_scores)
        assert value == repeated_value == third_value
        winning = [move for move in moves
                   if (result := engine.outcome(state.moved(move))) is not None
                   and result.winner is state.side_to_move]
        best = max(priors, key=priors.get)
        row.update(policy_sum=sum(priors.values()),
                   policy_source_max_abs_error=error,
                   value_source_abs_error=abs(value - expected_value),
                   repeated_forward_exact=True,
                   raw_logit_min=float(scores.min()), raw_logit_max=float(scores.max()),
                   top_move=[best.source, best.target], top_prior=priors[best],
                   winning_moves=[[m.source, m.target] for m in winning],
                   winning_prior_mass=sum(priors[m] for m in winning),
                   prior_min=min(priors.values()), prior_max=max(priors.values()))
        return row


def qualify(source: Path, checkpoint: Path, variant: str = "distill-v4") -> dict:
    engine = GameEngine()
    probe = CheckpointProbe(source, checkpoint, variant)
    before = {name: tensor.detach().clone() for name, tensor in probe.model.state_dict().items()}
    results = {name: probe.check(state, engine) for name, state in fixtures().items()}
    rng, state, sampled_moves, sampled_states = random.Random(0), GameState.initial(), 0, 0
    for _ in range(40):
        moves = engine.legal_moves(state)
        if engine.outcome(state, legal_moves=moves) is not None:
            break
        probe.check(state, engine)
        sampled_states += 1
        sampled_moves += len(moves)
        state = state.moved(rng.choice(moves))
    planes = observation(GameState.initial(), engine)
    durations = []
    for _ in range(20):
        started = time.perf_counter()
        probe.forward(planes)
        durations.append(time.perf_counter() - started)
    for name, tensor in probe.model.state_dict().items():
        assert probe.torch.equal(before[name], tensor), f"Model state changed: {name}"
    return {
        "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "candidate": f"cc-lite {variant}", "source_revision": probe.manifest["revision"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "model_config": probe.model.config.__dict__,
        "payload_keys": sorted(probe.payload), "checkpoint_step": probe.payload.get("step"),
        "checkpoint_metadata": probe.payload.get("metadata"),
        "parameter_count": sum(p.numel() for p in probe.model.parameters()),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": probe.torch.__version__, "numpy": np.__version__,
                        "device": "cpu", "threads": probe.torch.get_num_threads()},
        "measurements": {"load_seconds_excluding_imports": probe.load_seconds,
                         "first_forward_seconds_excluding_imports": probe.first_forward_seconds,
                         "warm_forward_count": len(durations),
                         "warm_forward_p50_seconds": float(np.percentile(durations, 50)),
                         "warm_forward_p95_seconds": float(np.percentile(durations, 95)),
                         "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "rss_note": "whole POC includes imports, parity and state snapshots; not model-only RSS"},
        "fixtures": results,
        "sampled_trajectory": {"seed": 0, "states": sampled_states,
                               "legal_moves_mapped": sampled_moves,
                               "all_observation_action_and_inference_parity": True},
        "model_state_unchanged": True, "forward_calls": probe.forward_calls,
        "mcts_executed": False, "training_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="distill-v4",
                        help="distill-v4 (default) or an alternate_checkpoints key")
    args = parser.parse_args()
    result = qualify(args.source, args.checkpoint, args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": result["checkpoint_sha256"],
                      "fixtures": len(result["fixtures"]),
                      "sampled_states": result["sampled_trajectory"]["states"],
                      "measurements": result["measurements"]}, indent=2))


if __name__ == "__main__":
    main()
