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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from .batch_resume import configuration, load_completed, write_checkpoint, checkpoint_path_for
from .opening_book import BOOK_VERSION, sample_opening, validate_opening_book
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
    opening_mode: str = "random",
    no_progress_limit: int = 0,
    ply_progress_every: int = 0,
) -> dict[str, Any]:
    """Run one match without UI/API, preserving transitions for later training.

    Opening moves are sampled uniformly from legal moves and identified as
    `opening_random`, never incorrectly labelled as expert demonstrations.
    """
    if red not in AGENT_NAMES or black not in AGENT_NAMES:
        raise ValueError(f"Agents must be chosen from: {', '.join(AGENT_NAMES)}")
    if red_depth < 1 or black_depth < 1 or max_plies < 1:
        raise ValueError("depth and max_plies must be positive")
    if opening_mode not in {"none", "random", "book"}:
        raise ValueError("opening_mode must be none, random or book")
    if opening_mode == "book" and opening_plies < 1:
        raise ValueError("book mode requires opening_plies >= 1")
    if opening_plies < 0 or no_progress_limit < 0 or game_index < 0 or ply_progress_every < 0:
        raise ValueError("opening_plies, no_progress_limit and game_index must be nonnegative")

    env = XiangqiEnv()
    game_seed = seed + game_index * 1009
    opening_rng = random.Random(game_seed + 2)
    selection = sample_opening(game_seed, opening_plies) if opening_mode == "book" else None
    opening_moves = selection.moves if selection is not None else ()
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
        if state.ply_count < len(opening_moves):
            actor = "opening_book"
            move = opening_moves[state.ply_count]
            if move not in env.engine.legal_moves(state):
                raise ValueError(
                    f"Invalid book move at game={game_index} ply={state.ply_count}: "
                    f"{selection.line.opening_id}"
                )
        elif opening_mode == "random" and state.ply_count < opening_plies:
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
        if ply_progress_every and step.state.ply_count % ply_progress_every == 0:
            print(
                f"[game {game_index}] ply={step.state.ply_count}/{max_plies} "
                f"side={_side_name(step.state.side_to_move)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
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
        "opening_mode": opening_mode,
        "opening_book_version": BOOK_VERSION if selection is not None else None,
        "opening_id": selection.line.opening_id if selection is not None else None,
        "opening_family": selection.line.family if selection is not None else None,
        "opening_name": selection.line.name if selection is not None else None,
        "opening_applied_plies": sum(row["actor"] in {"opening_book", "opening_random"} for row in moves),
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



def _play_game_job(arguments: tuple[Any, ...]) -> dict[str, Any]:
    """Top-level callable for process workers; workers never write to Drive."""
    red, black, index, params = arguments
    return play_game(red, black, game_index=index, **params)


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
    opening_mode: str = "random",
    no_progress_limit: int = 0,
    progress_every: int = 10,
    ply_progress_every: int = 0,
    workers: int = 1,
    resume: bool = False,
    append: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Generate a fixed-size batch with validated resumability.

    The main process is the sole JSONL/checkpoint writer. With workers>1,
    independent games run in parallel and records are saved as they finish.
    Resuming skips completed global game_index values without resetting seeds.
    """
    if games < 1 or progress_every < 1 or workers < 1 or ply_progress_every < 0:
        raise ValueError("games, progress_every, workers must be positive")
    if red not in AGENT_NAMES or black not in AGENT_NAMES:
        raise ValueError(f"Agents must be chosen from: {', '.join(AGENT_NAMES)}")
    if append and not resume:
        raise ValueError("--append is unsafe; use --resume with a fixed output instead")
    if opening_mode not in {"none", "random", "book"}:
        raise ValueError("opening_mode must be none, random or book")
    if opening_mode == "book":
        if opening_plies < 1:
            raise ValueError("book mode requires opening_plies >= 1")
        validate_opening_book()

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = configuration(
        red, black, seed, red_depth, black_depth,
        max_plies, opening_plies, no_progress_limit,
        opening_mode=opening_mode,
        opening_book_version=BOOK_VERSION if opening_mode == "book" else None,
    )
    completed = load_completed(path, config, games, resume=resume)
    pending = [index for index in range(games) if index not in completed]
    if not quiet:
        print(
            f"[batch] total={games} saved={len(completed)} pending={len(pending)} "
            f"workers={workers} dataset={path}",
            flush=True,
        )
    if completed:
        # Checkpoint may be stale after a crash; rebuild from authoritative JSONL.
        write_checkpoint(path, config, games, completed)

    params = {
        "seed": seed,
        "red_depth": red_depth,
        "black_depth": black_depth,
        "max_plies": max_plies,
        "opening_plies": opening_plies,
        "opening_mode": opening_mode,
        "no_progress_limit": no_progress_limit,
        "ply_progress_every": ply_progress_every if workers == 1 else 0,
    }
    started = time.perf_counter()
    added = 0

    # Every record is durably flushed before updating the checkpoint. A crash
    # during checkpoint writing cannot cause lost indices on the next resume.
    def save_record(stream, record: dict[str, Any]) -> None:
        nonlocal added
        index = record["game_index"]
        if index in completed:
            raise ValueError(f"Duplicate game_index {index}")
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        completed[index] = record
        added += 1
        write_checkpoint(path, config, games, completed)
        if not quiet and (
            added % progress_every == 0 or len(completed) == games
        ):
            outcome = record["outcome"]
            opening_text = record.get("opening_id") or record.get("opening_mode", "none")
            elapsed = time.perf_counter() - started
            remaining = games - len(completed)
            eta = elapsed / added * remaining if added else 0.0
            print(
                f"[batch {len(completed)}/{games}] game={index} "
                f"opening={opening_text} "
                f"ply={record['ply_count']} {outcome['status']}:"
                f"{outcome['reason']} winner={outcome['winner']} "
                f"game_seconds={record['elapsed_seconds']:.1f} "
                f"batch_elapsed={elapsed:.1f}s eta={eta:.0f}s "
                f"saved={path}",
                flush=True,
            )

    if pending:
        with path.open("a" if resume else "x", encoding="utf-8") as stream:
            if workers == 1:
                for index in pending:
                    if not quiet:
                        print(f"[game {index}] started ({len(completed) + 1}/{games})", flush=True)
                    save_record(stream, play_game(
                        red, black, game_index=index, **params,
                    ))
            else:
                # Bound in-flight jobs to avoid using excessive memory for
                # a very large batch; preserve single-writer disk semantics.
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    waiting = iter(pending)
                    active = {}

                    def submit_next() -> bool:
                        try:
                            index = next(waiting)
                        except StopIteration:
                            return False
                        future = pool.submit(_play_game_job, (red, black, index, params))
                        active[future] = index
                        if not quiet:
                            print(f"[game {index}] dispatched", flush=True)
                        return True

                    for _ in range(min(workers * 2, len(pending))):
                        submit_next()
                    while active:
                        done, _ = wait(active, return_when=FIRST_COMPLETED)
                        for future in done:
                            del active[future]
                            save_record(stream, future.result())
                            submit_next()

    counters: Counter[str] = Counter()
    fingerprints: set[str] = set()
    duplicates = 0
    total_plies = 0
    for record in completed.values():
        total_plies += record["ply_count"]
        outcome = record["outcome"]
        counters[outcome["status"]] += 1
        if outcome["status"] == "finished":
            counters[outcome["winner"] or "draw"] += 1
        if record["fingerprint"] in fingerprints:
            duplicates += 1
        fingerprints.add(record["fingerprint"])

    if not quiet:
        print(
            f"[done] saved={len(completed)}/{games} new={added} "
            f"finished={counters['finished']} truncated={counters['truncated']} "
            f"checkpoint={checkpoint_path_for(path)}",
            flush=True,
        )
    return {
        "output": str(path),
        "checkpoint": str(checkpoint_path_for(path)),
        "games": len(completed),
        "new_games": added,
        "finished": counters["finished"],
        "truncated": counters["truncated"],
        "red_wins": counters["red"],
        "black_wins": counters["black"],
        "draws": counters["draw"],
        "unique_games_in_batch": len(fingerprints),
        "duplicate_games_in_batch": duplicates,
        "average_plies": round(total_plies / len(completed), 2) if completed else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
