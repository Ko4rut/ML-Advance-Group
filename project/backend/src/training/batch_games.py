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
from .topk_exploration import sample_top_k
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
    retry_attempt: int = 0,
    deduplicate_games: bool = False,
    max_duplicate_retries: int = 8,
    exploration_top_k: int = 1,
    exploration_score_margin: int = 12,
    exploration_temperature: float = 8.0,
    exploration_rate: float = 0.35,
    exploration_window: int = 20,
    exploration_max_moves: int = 2,
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

    if (
        retry_attempt < 0 or max_duplicate_retries < 0 or exploration_top_k < 1
        or exploration_score_margin < 0 or exploration_temperature <= 0
        or not 0 <= exploration_rate <= 1
        or exploration_window < 0 or exploration_max_moves < 0
    ):
        raise ValueError("invalid exploration or retry parameters")

    env = XiangqiEnv()
    game_seed = seed + game_index * 1009
    # Keep dataset's global seed stable; retries only change the per-game
    # sampling stream. Replay is determined by (game_index, retry_attempt).
    generation_seed = game_seed + retry_attempt * 1_000_003
    opening_rng = random.Random(generation_seed + 2)
    exploration_rng = random.Random(generation_seed + 17_593)
    selection = sample_opening(generation_seed, opening_plies) if opening_mode == "book" else None
    opening_moves = selection.moves if selection is not None else ()
    opening_end = len(opening_moves)
    if opening_mode == "random":
        opening_end = opening_plies
    agents = {
        Side.RED: _make_agent(red, env, game_seed, red_depth),
        Side.BLACK: _make_agent(black, env, game_seed + 1, black_depth),
    }
    names = {Side.RED: red, Side.BLACK: black}
    initial_board = list(env.state.board)
    moves: list[dict[str, Any]] = []
    started = time.perf_counter()
    exploration_used = 0

    while env.state.ply_count < max_plies:
        state = env.state
        if env.engine.outcome(state) is not None:
            break
        if no_progress_limit and state.no_capture_count >= no_progress_limit:
            break

        side = state.side_to_move
        decision = None
        search_nodes = 0
        exploration_details = None
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
            candidate_agent = agents[side]
            explore = (
                exploration_top_k > 1
                and isinstance(candidate_agent, AlphaBetaAgent)
                and exploration_used < exploration_max_moves
                and state.ply_count >= opening_end
                and state.ply_count < opening_end + exploration_window
                and exploration_rng.random() < exploration_rate
            )
            if explore:
                ranked, search_nodes = candidate_agent.rank_actions(state)
                move, selected_score, distribution = sample_top_k(
                    ranked, exploration_rng,
                    top_k=exploration_top_k,
                    score_margin=exploration_score_margin,
                    temperature=exploration_temperature,
                )
                exploration_used += 1
                actor = "alphabeta_exploration"
                exploration_details = {
                    "best_score": ranked[0][1],
                    "chosen_score": selected_score,
                    "candidates": [
                        {
                            "source": action.source,
                            "target": action.target,
                            "score": score,
                            "probability": round(probability, 8),
                        }
                        for action, score, probability in distribution
                    ],
                }
            else:
                decision = candidate_agent.select_action(state)
                move = decision.move
                search_nodes = decision.nodes

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
            "nodes": search_nodes,
            "simulations": decision.simulations if decision else 0,
            "think_seconds": round(think_seconds, 6),
        }
        if exploration_details is not None:
            row["exploration"] = exploration_details
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
        "retry_attempt": retry_attempt,
        "generation_seed": generation_seed,
        "deduplicate_games": deduplicate_games,
        "max_duplicate_retries": max_duplicate_retries,
        "exploration_top_k": exploration_top_k,
        "exploration_score_margin": exploration_score_margin,
        "exploration_temperature": exploration_temperature,
        "exploration_rate": exploration_rate,
        "exploration_window": exploration_window,
        "exploration_max_moves": exploration_max_moves,
        "exploration_applied_moves": exploration_used,
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
    deduplicate_games: bool = False,
    max_duplicate_retries: int = 8,
    exploration_top_k: int = 1,
    exploration_score_margin: int = 12,
    exploration_temperature: float = 8.0,
    exploration_rate: float = 0.35,
    exploration_window: int = 20,
    exploration_max_moves: int = 2,
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
    if (
        max_duplicate_retries < 0 or exploration_top_k < 1
        or exploration_score_margin < 0 or exploration_temperature <= 0
        or not 0 <= exploration_rate <= 1
        or exploration_window < 0 or exploration_max_moves < 0
    ):
        raise ValueError("invalid deduplication or exploration parameters")
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
        deduplicate_games=deduplicate_games,
        max_duplicate_retries=max_duplicate_retries,
        exploration_top_k=exploration_top_k,
        exploration_score_margin=exploration_score_margin,
        exploration_temperature=exploration_temperature,
        exploration_rate=exploration_rate,
        exploration_window=exploration_window,
        exploration_max_moves=exploration_max_moves,
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
        "deduplicate_games": deduplicate_games,
        "max_duplicate_retries": max_duplicate_retries,
        "ply_progress_every": ply_progress_every if workers == 1 else 0,
        "exploration_top_k": exploration_top_k,
        "exploration_score_margin": exploration_score_margin,
        "exploration_temperature": exploration_temperature,
        "exploration_rate": exploration_rate,
        "exploration_window": exploration_window,
        "exploration_max_moves": exploration_max_moves,
    }
    started = time.perf_counter()
    added = 0
    duplicate_rejections = 0
    seen_fingerprints = {record["fingerprint"] for record in completed.values()}
    if deduplicate_games and len(seen_fingerprints) != len(completed):
        raise ValueError("Existing dataset contains duplicate games: cannot resume dedup mode")

    # Every record is durably flushed before updating the checkpoint. A crash
    # during checkpoint writing cannot cause lost indices on the next resume.
    def save_record(stream, record: dict[str, Any]) -> bool:
        nonlocal added, duplicate_rejections
        index = record["game_index"]
        if index in completed:
            raise ValueError(f"Duplicate game_index {index}")
        if deduplicate_games and record["fingerprint"] in seen_fingerprints:
            duplicate_rejections += 1
            return False
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        completed[index] = record
        seen_fingerprints.add(record["fingerprint"])
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
        return True

    if pending:
        with path.open("a" if resume else "x", encoding="utf-8") as stream:
            if workers == 1:
                for index in pending:
                    if not quiet:
                        print(f"[game {index}] started ({len(completed) + 1}/{games})", flush=True)
                    for attempt in range(max_duplicate_retries + 1):
                        record = play_game(
                            red, black, game_index=index,
                            retry_attempt=attempt, **params,
                        )
                        if save_record(stream, record):
                            break
                    else:
                        raise RuntimeError(
                            f"Could not generate unique game_index={index} after "
                            f"{max_duplicate_retries + 1} attempts. "
                            "Increase exploration or choose more opening lines."
                        )
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
                        future = pool.submit(
                            _play_game_job,
                            (red, black, index, {**params, "retry_attempt": 0}),
                        )
                        active[future] = (index, 0)
                        if not quiet:
                            print(f"[game {index}] dispatched", flush=True)
                        return True

                    for _ in range(min(workers * 2, len(pending))):
                        submit_next()
                    while active:
                        done, _ = wait(active, return_when=FIRST_COMPLETED)
                        for future in done:
                            index, attempt = active.pop(future)
                            if save_record(stream, future.result()):
                                submit_next()
                            elif attempt < max_duplicate_retries:
                                next_attempt = attempt + 1
                                retry = pool.submit(
                                    _play_game_job,
                                    (
                                        red, black, index,
                                        {**params, "retry_attempt": next_attempt},
                                    ),
                                )
                                active[retry] = (index, next_attempt)
                            else:
                                raise RuntimeError(
                                    f"Could not generate unique game_index={index} "
                                    f"after {attempt + 1} attempts. "
                                    "Increase exploration or choose more opening lines."
                                )

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
        "duplicate_rejections_this_run": duplicate_rejections,
        "average_plies": round(total_plies / len(completed), 2) if completed else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
