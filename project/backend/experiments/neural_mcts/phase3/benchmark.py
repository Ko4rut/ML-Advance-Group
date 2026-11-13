"""CPU latency / memory / load-time measurement of the production inference adapter.

One process measures one (model, threads) setting because torch's thread count is
process-global:

    python experiments/neural_mcts/phase3/benchmark.py --model distill-v4-large --threads 1 \
        --output docs/neural_mcts/evidence/phase3/benchmark-distill-v4-large-t1.json

Method: 200 non-terminal states from seeded random walks (fixed seed), 10 warm-up
predictions, then each state predicted once with the legal-move list supplied (as MCTS
will). Reported per call: whole ``predict`` (encode + forward + mask/softmax + checks) and,
timed in the same loop on the same observation, the raw ``backend.infer`` forward; adapter
overhead = predict - forward. Memory is VmRSS read from /proc (Linux) at each stage;
VmHWM is the process peak. Not a strength or MCTS throughput measurement.
"""

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND))


def rss_mib(field="VmRSS"):
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarize(seconds):
    ms = [s * 1000 for s in seconds]
    return {"n": len(ms), "mean_ms": statistics.fmean(ms), "p50_ms": percentile(ms, 0.5),
            "p95_ms": percentile(ms, 0.95), "max_ms": max(ms)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="distill-v4-large")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's own)")
    parser.add_argument("--states", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    stages = {"python_start": rss_mib()}
    from src.environment import GameEngine, GameState
    from src.networks.pretrained import NeuralModelSettings, load_adapter
    t0 = time.perf_counter()
    import torch
    import_seconds = time.perf_counter() - t0
    stages["after_import_torch"] = rss_mib()

    t0 = time.perf_counter()
    adapter = load_adapter(NeuralModelSettings(model=args.model, threads=args.threads))
    load_seconds = time.perf_counter() - t0  # SHA-256 + deserialize + build + validate
    stages["after_load"] = rss_mib()

    engine, rng, states = GameEngine(), random.Random(0), []
    while len(states) < args.states:
        state = GameState.initial()
        for _ in range(rng.randint(0, 60)):
            moves = engine.legal_moves(state)
            if engine.outcome(state, legal_moves=moves) is not None:
                break
            state = state.moved(rng.choice(moves))
        moves = engine.legal_moves(state)
        if engine.outcome(state, legal_moves=moves) is None:
            states.append((state, moves))

    for state, moves in states[:10]:  # warm-up
        adapter.predict(state, moves)
    predict_s, forward_s = [], []
    encoder = adapter._encoder
    for state, moves in states:  # interleaved so CPU frequency / core migration affects both equally
        observation = encoder.encode(state)
        t = time.perf_counter()
        adapter.predict(state, moves)
        predict_s.append(time.perf_counter() - t)
        t = time.perf_counter()
        adapter.backend.infer(observation)
        forward_s.append(time.perf_counter() - t)
    stages["after_predictions"] = rss_mib()
    stages["peak_VmHWM"] = rss_mib("VmHWM")

    result = {
        "model": adapter.info.name, "sha256": adapter.info.sha256,
        "parameter_count": adapter.info.parameter_count,
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "torch": torch.__version__, "cpu_threads_used": torch.get_num_threads(),
                    "cpu_model": next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                                       if l.startswith("model name")), "unknown")},
        "method": {"states": len(states), "seed": 0, "warmup": 10, "legal_moves_supplied": True},
        "import_torch_seconds": import_seconds,
        "load_seconds_incl_sha256_deserialize_validate": load_seconds,
        "predict": summarize(predict_s), "raw_forward": summarize(forward_s),
        "adapter_overhead_p50_ms": statistics.median(p - f for p, f in zip(predict_s, forward_s)) * 1000,
        "rss_mib": stages,
        "model_only_rss_delta_mib": stages["after_load"] - stages["after_import_torch"],
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
