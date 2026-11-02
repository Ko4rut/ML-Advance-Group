"""CPU-only batch match generation; one complete game record per JSONL line.

This is a data collector, not a trainer. In particular, an Alpha-Beta move is
not an MCTS policy target, and truncated games have no winner/outcome label.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..agents import (
    Agent,
    AlphaBetaAgent,
    HeuristicAgent,
    MinimaxAgent,
    RandomAgent,
    RuleBasedAgent,
)
from ..environment import Side, XiangqiEnv

AGENT_NAMES = ("random", "rules", "minimax", "alphabeta", "astar")


def _side_name(side: Side | None) -> str | None:
    if side is None:
        return None
    return "red" if side is Side.RED else "black"


def _make_agent(name: str, env: XiangqiEnv, seed: int, depth: int) -> Agent:
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "rules":
        return RuleBasedAgent(env.engine)
    if name == "minimax":
        return MinimaxAgent(env.engine, max_depth=depth)
    if name == "alphabeta":
        return AlphaBetaAgent(env.engine, max_depth=depth)
    if name == "astar":
        return HeuristicAgent(env.engine)
    raise ValueError(f"Unsupported agent: {name}")


def play_game(
    red: str,
    black: str,
    *,
    game_index: int = 0,
    seed: int = 42,
    red_depth: int = 2,
    black_depth: int = 2,
    max_plies: int = 160,
    opening_plies: int = 0,
    no_progress_limit: int = 0,
) -> dict[str, Any]:
    """Run one match without UI/API, preserving transitions for later training.

    Opening moves are sampled uniformly from legal moves and identified as
    `opening_random`, never incorrectly labelled as expert demonstrations.
    """
    if red not in AGENT_NAMES or black not in AGENT_NAMES:
        raise ValueError(f"Agents must be chosen from: {', '.join(AGENT_NAMES)}")
    if red_depth < 1 or black_depth < 1 or max_plies < 1:
        raise ValueError("depth and max_plies must be positive")
    if opening_plies < 0 or no_progress_limit < 0 or game_index < 0:
        raise ValueError("opening_plies, no_progress_limit and game_index must be nonnegative")

    env = XiangqiEnv()
    game_seed = seed + game_index * 1009
    opening_rng = random.Random(game_seed + 2)
    agents = {
        Side.RED: _make_agent(red, env, game_seed, red_depth),
        Side.BLACK: _make_agent(black, env, game_seed + 1, black_depth),
    }
    names = {Side.RED: red, Side.BLACK: black}
    initial_board = list(env.state.board)
    moves: list[dict[str, Any]] = []
    started = time.perf_counter()

    while env.state.ply_count < max_plies:
        state = env.state
        if env.engine.outcome(state) is not None:
            break
        if no_progress_limit and state.no_capture_count >= no_progress_limit:
            break

        side = state.side_to_move
        decision = None
        actor = names[side]
        move_started = time.perf_counter()
        if state.ply_count < opening_plies:
            actor = "opening_random"
            move = opening_rng.choice(env.engine.legal_moves(state))
        else:
            decision = agents[side].select_action(state)
            move = decision.move

        think_seconds = time.perf_counter() - move_started
        step = env.step(move)
        row: dict[str, Any] = {
            "ply": step.state.ply_count,
            "board": list(state.board),
            "side": _side_name(side),
            "no_capture_count": state.no_capture_count,
            "actor": actor,
            "action": {"source": move.source, "target": move.target},
            "next_board": list(step.state.board),
            "next_no_capture_count": step.state.no_capture_count,
            "reward": step.reward,
            "terminated": step.terminated,
            "nodes": decision.nodes if decision else 0,
            "simulations": decision.simulations if decision else 0,
            "think_seconds": round(think_seconds, 6),
        }
        if decision is not None and decision.policy is not None:
            row["policy"] = [
                {"source": action.source, "target": action.target, "probability": float(probability)}
                for action, probability in sorted(
                    decision.policy.items(),
                    key=lambda item: (item[0].source, item[0].target),
                )
            ]
        moves.append(row)
        if step.terminated:
            break

    actual_result = env.engine.outcome(env.state)
    if actual_result is not None:
        outcome = {
            "status": "finished",
            "winner": _side_name(actual_result.winner),
            "reason": actual_result.reason,
        }
    else:
        reason = "max_plies"
        if no_progress_limit and env.state.no_capture_count >= no_progress_limit:
            reason = "no_progress_limit"
        outcome = {"status": "truncated", "winner": None, "reason": reason}

    move_keys = [(entry["action"]["source"], entry["action"]["target"]) for entry in moves]
    fingerprint = hashlib.sha256(
        json.dumps(move_keys, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": 1,
        "game_id": uuid4().hex,
        "game_index": game_index,
        "seed": game_seed,
        "red_agent": red,
        "black_agent": black,
        "red_depth": red_depth if red in {"minimax", "alphabeta"} else None,
        "black_depth": black_depth if black in {"minimax", "alphabeta"} else None,
        "opening_plies": opening_plies,
        "max_plies": max_plies,
        "no_progress_limit": no_progress_limit,
        "initial_board": initial_board,
        "initial_side": "red",
        "moves": moves,
        "outcome": outcome,
        "final_board": list(env.state.board),
        "ply_count": env.state.ply_count,
        "fingerprint": fingerprint,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def generate_batch(
    red: str,
    black: str,
    *,
    games: int,
    output: str | Path,
    seed: int = 42,
    red_depth: int = 2,
    black_depth: int = 2,
    max_plies: int = 160,
    opening_plies: int = 0,
    no_progress_limit: int = 0,
    progress_every: int = 10,
    append: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Write JSONL incrementally so completed games survive interruptions."""
    if games < 1 or progress_every < 1:
        raise ValueError("games and progress_every must be positive")
    if red not in AGENT_NAMES or black not in AGENT_NAMES:
        raise ValueError(f"Agents must be chosen from: {', '.join(AGENT_NAMES)}")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    fingerprints: set[str] = set()
    total_plies = 0
    started = time.perf_counter()

    # Refuse accidental overwrites. Use --append explicitly to resume/add games.
    with path.open("a" if append else "x", encoding="utf-8") as stream:
        for index in range(games):
            record = play_game(
                red,
                black,
                game_index=index,
                seed=seed,
                red_depth=red_depth,
                black_depth=black_depth,
                max_plies=max_plies,
                opening_plies=opening_plies,
                no_progress_limit=no_progress_limit,
            )
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            total_plies += record["ply_count"]
            outcome = record["outcome"]
            counters[outcome["status"]] += 1
            if outcome["status"] == "finished":
                winner = outcome["winner"] or "draw"
                counters[winner] += 1
            if record["fingerprint"] in fingerprints:
                counters["duplicate_in_batch"] += 1
            fingerprints.add(record["fingerprint"])

            if not quiet and ((index + 1) % progress_every == 0 or index + 1 == games):
                print(f"[{index + 1}/{games}] saved to {path}")

    return {
        "output": str(path),
        "games": games,
        "finished": counters["finished"],
        "truncated": counters["truncated"],
        "red_wins": counters["red"],
        "black_wins": counters["black"],
        "draws": counters["draw"],
        "unique_games_in_batch": len(fingerprints),
        "duplicate_games_in_batch": counters["duplicate_in_batch"],
        "average_plies": round(total_plies / games, 2),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
