"""Per-node cost estimate for a 3 s/move budget (no MCTS executed, no training).

Measures, on CPU: network forward at several batch sizes/thread counts, and the
engine-side cost one MCTS node would pay (legal_moves + outcome + moved + encoding).
Positions come from a fixed-seed legal random walk. Results are estimates of
throughput, not a search benchmark.
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np

from contracts import observation
from qualify import CheckpointProbe
from src.environment import GameEngine, GameState


def trajectory(engine, n=60, seed=0):
    rng, state, states = random.Random(seed), GameState.initial(), []
    while len(states) < n:
        moves = engine.legal_moves(state)
        if engine.outcome(state, legal_moves=moves) is not None:
            break
        states.append(state)
        state = state.moved(rng.choice(moves))
    return states


def timed(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", default="distill-v4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = GameEngine()
    probe = CheckpointProbe(args.source, args.checkpoint, args.variant)
    torch = probe.torch
    states = trajectory(engine)

    def node_cost():  # engine work for one expanded node, excluding the network
        for state in states:
            moves = engine.legal_moves(state)
            engine.outcome(state, legal_moves=moves)
            state.moved(moves[0])
            observation(state, engine)
    engine_per_node = timed(node_cost, 5) / len(states)

    batch_rows = {}
    for threads in (1, 4):
        torch.set_num_threads(threads)
        for batch in (1, 8, 32):
            planes = torch.from_numpy(np.stack([observation(states[i % len(states)], engine)
                                                for i in range(batch)]))

            def run():
                with torch.inference_mode():
                    probe.model(planes)
            run()  # warm-up
            seconds = timed(run, 15)
            batch_rows[f"threads={threads},batch={batch}"] = {
                "seconds_per_call": seconds, "ms_per_position": 1000 * seconds / batch,
                "positions_per_second": batch / seconds}
    result = {"variant": args.variant, "positions_sampled": len(states),
              "engine_ms_per_node_excl_network": 1000 * engine_per_node,
              "forward": batch_rows, "budget_seconds": 3.0,
              "estimated_nodes_in_budget": {
                  key: int(3.0 / (row["seconds_per_call"] / int(key.split("batch=")[1]) + engine_per_node))
                  for key, row in batch_rows.items()},
              "note": "estimate = 3 s / (network ms per position + engine ms per node); ignores tree overhead, batching latency and the engine cost of expanding every child"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
