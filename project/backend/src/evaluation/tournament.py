"""Balanced-color round-robin tournament between Xiangqi agents.

Design (see TOURNAMENT.md):

* Every pair plays ``len(openings) x colors_per_opening`` games: each fixed
  opening is played with A as Red and again with A as Black.
* ``games.jsonl`` (one game per line) is the single source of truth; all
  summaries are regenerated from it.
* Only the main process writes files (append + flush + fsync per game), so a
  crash or Ctrl+C never corrupts the log and ``--resume`` loses at most the
  games that were in flight.
* Results do not depend on worker count or completion order.

This module deliberately does not use ``MatchRunner`` (it applies one budget to
both sides and always starts from the initial position).
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import os
import signal
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..environment import GameState, XiangqiEnv
from ..training.opening_book import get_opening
from . import agent_registry, provenance
from .agent_registry import BACKEND_ROOT

SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = BACKEND_ROOT / "results" / "tournaments"
SMOKE_MAX_PLIES = 40
SMOKE_OPENINGS = 2
LOGGER_NAME = "tournament"

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfig:
    """Validated config. ``data`` is hashed; ``excluded``/``mode`` are not."""

    data: dict[str, Any]
    excluded: dict[str, str] = field(default_factory=dict)
    mode: str = "full"  # full | smoke | pilot
    workers: Any = "auto"  # not hashed: results never depend on it
    full_games_per_pair: int = 0  # games/pair of the un-overridden config (pilot estimate)

    @property
    def hash_view(self) -> dict[str, Any]:
        view = copy.deepcopy(self.data)
        view["agents"] = agent_registry.hash_view(self.data["agents"])
        return view

    @property
    def config_hash(self) -> str:
        return provenance.sha256_text(provenance.canonical_json(self.hash_view))

    @property
    def games_per_pair(self) -> int:
        return len(self.data["openings"]) * self.data["colors_per_opening"]


def load_raw_config(path: str | Path) -> dict[str, Any]:
    import yaml

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"tournament config {path} must contain a mapping")
    return loaded


def _final_position(opening) -> Any:
    state = GameState.initial()
    for move in opening.moves:
        state = state.moved(move)
    return state.position_key()


def resolve_config(
    raw: Mapping[str, Any],
    *,
    mode: str = "full",
    backend_root: Path = BACKEND_ROOT,
    env: Mapping[str, str] | None = None,
) -> ResolvedConfig:
    """Validate ``raw`` and apply the smoke/pilot overrides."""
    if mode not in {"full", "smoke", "pilot"}:
        raise ValueError(f"unknown mode {mode!r}")
    required = {"name", "master_seed", "opening_book_version", "openings", "max_plies", "agents"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"tournament config is missing: {missing}")
    master_seed = raw["master_seed"]
    if type(master_seed) is not int:
        raise ValueError("master_seed must be an integer")
    colors = raw.get("colors_per_opening", 2)
    if colors not in (1, 2):
        raise ValueError("colors_per_opening must be 1 or 2")
    max_plies = raw["max_plies"]
    if type(max_plies) is not int or max_plies < 1:
        raise ValueError("max_plies must be a positive integer")
    openings = list(raw["openings"])
    full_games_per_pair = len(openings) * colors
    if not openings:
        raise ValueError("openings must not be empty")

    # Validate the *full* opening list first; smoke/pilot only subset it.
    version = raw["opening_book_version"]
    if len(set(openings)) != len(openings):
        raise ValueError("openings contains duplicate ids")
    finals: dict[Any, str] = {}
    longest = 0
    for opening_id in openings:
        try:
            line = get_opening(opening_id, version)
        except KeyError as exc:
            raise ValueError(f"unknown opening {opening_id!r} in book {version}") from exc
        longest = max(longest, len(line.moves))
        key = _final_position(line)
        if key in finals:
            raise ValueError(
                f"openings {finals[key]!r} and {opening_id!r} reach the same final position"
            )
        finals[key] = opening_id

    if mode == "smoke":
        step = max(1, len(openings) // SMOKE_OPENINGS)
        openings = openings[::step][:SMOKE_OPENINGS]
        max_plies = min(max_plies, SMOKE_MAX_PLIES)
        longest = max(len(get_opening(i, version).moves) for i in openings)
    elif mode == "pilot":
        openings = openings[:1]
        colors = 1
        longest = len(get_opening(openings[0], version).moves)
    if max_plies <= longest:
        raise ValueError(
            f"max_plies ({max_plies}) must exceed the longest opening ({longest} plies)"
        )

    agents, excluded = agent_registry.resolve_agents(raw["agents"], backend_root, env)
    if len(agents) < 2:
        raise ValueError(f"need at least two agents, have {sorted(agents)}; excluded={excluded}")
    data = {
        "name": str(raw["name"]),
        "master_seed": master_seed,
        "opening_book_version": version,
        "openings": openings,
        "colors_per_opening": colors,
        "max_plies": max_plies,
        "agents": agents,
    }
    return ResolvedConfig(
        data=data, excluded=excluded, mode=mode,
        workers=raw.get("workers", "auto"), full_games_per_pair=full_games_per_pair,
    )


def resolve_workers(value: Any) -> int:
    if value in (None, "auto"):
        return max(1, (os.cpu_count() or 2) - 1)
    if type(value) is not int or value < 1:
        raise ValueError("workers must be 'auto' or a positive integer")
    return value


# --------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------


def derive_seed(master_seed: int, game_key: str, suffix: str = "") -> int:
    """Deterministic 64-bit seed; never uses Python's randomized ``hash()``."""
    text = f"{master_seed}|{game_key}" + (f"|{suffix}" if suffix else "")
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


@dataclass(frozen=True)
class GameSpec:
    game_index: int
    game_key: str
    agent_a: str
    agent_b: str
    opening_id: str
    a_color: str  # "red" | "black"
    red_agent: str
    black_agent: str
    seed: int
    seed_red: int
    seed_black: int


def build_schedule(data: Mapping[str, Any]) -> list[GameSpec]:
    """Deterministic schedule: pairs (config order) x openings x colors."""
    a_colors = ("red", "black")[: data["colors_per_opening"]]
    specs: list[GameSpec] = []
    for agent_a, agent_b in itertools.combinations(list(data["agents"]), 2):
        for opening_id in data["openings"]:
            for a_color in a_colors:
                red, black = (agent_a, agent_b) if a_color == "red" else (agent_b, agent_a)
                key = f"{agent_a}|{agent_b}|{opening_id}|{a_color}"
                master = data["master_seed"]
                specs.append(GameSpec(
                    game_index=len(specs), game_key=key,
                    agent_a=agent_a, agent_b=agent_b, opening_id=opening_id, a_color=a_color,
                    red_agent=red, black_agent=black,
                    seed=derive_seed(master, key),
                    seed_red=derive_seed(master, key, "red"),
                    seed_black=derive_seed(master, key, "black"),
                ))
    return specs


# --------------------------------------------------------------------------
# One game
# --------------------------------------------------------------------------


def _side_name(side) -> str:
    return "red" if int(side) == 1 else "black"


def _fingerprint(move_keys: list[list[int]]) -> str:
    """Same scheme as training/batch_games.py."""
    return hashlib.sha256(
        json.dumps(move_keys, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def play_tournament_game(
    spec: GameSpec,
    data: Mapping[str, Any],
    *,
    run_id: str = "",
    config_hash: str = "",
) -> dict[str, Any]:
    """Play one game and return its ``games.jsonl`` record.

    Never raises for agent failures: those become ``status="error"``.
    Opening plies are applied with ``env.step`` (so repetition history is
    right) and are excluded from every cost statistic.
    """
    started = time.perf_counter()
    agents_params = data["agents"]
    names = {"red": spec.red_agent, "black": spec.black_agent}
    seeds = {"red": spec.seed_red, "black": spec.seed_black}
    opening = get_opening(spec.opening_id, data["opening_book_version"])
    budgets = {
        side: agent_registry.budget_for(name, agents_params[name]) for side, name in names.items()
    }

    moves: list[dict[str, Any]] = []
    move_keys: list[list[int]] = []
    agent_plies = {"red": 0, "black": 0}
    error: dict[str, Any] | None = None
    env = XiangqiEnv()
    env.reset()

    try:
        for action in opening.moves:
            state = env.state
            if action not in env.engine.legal_moves(state):
                raise ValueError(f"illegal opening move in {spec.opening_id}")
            side = _side_name(state.side_to_move)
            step = env.step(action)
            moves.append({
                "ply": step.state.ply_count, "side": side, "actor": "opening",
                "source": action.source, "target": action.target,
                "think_seconds": 0.0, "nodes": 0, "simulations": 0,
            })
            move_keys.append([action.source, action.target])

        agents = {
            side: agent_registry.make_agent(names[side], agents_params[names[side]], env.engine, seeds[side])
            for side in ("red", "black")
        }
        while env.state.ply_count < data["max_plies"] and not env.is_terminal():
            state = env.state
            side = _side_name(state.side_to_move)
            agent = agents[side]
            began = time.perf_counter()
            try:
                decision = agent.select_action(state, budgets[side])
                think = time.perf_counter() - began
                if decision.move not in env.engine.legal_moves(state):
                    raise ValueError(f"illegal move {decision.move!r}")
                step = env.step(decision.move)
            except Exception as exc:  # noqa: BLE001 - agent failures must not stop the tournament
                error = {
                    "type": type(exc).__name__, "message": str(exc)[:500],
                    "traceback": traceback.format_exc(limit=-6)[-2000:],
                    "ply": state.ply_count + 1, "actor": names[side], "side": side,
                }
                break
            row = {
                "ply": step.state.ply_count, "side": side, "actor": names[side],
                "source": decision.move.source, "target": decision.move.target,
                "think_seconds": round(think, 6), "nodes": decision.nodes,
                "simulations": decision.simulations,
            }
            if hasattr(agent, "last_action_supported"):
                row["supported"] = bool(agent.last_action_supported)
            moves.append(row)
            move_keys.append([decision.move.source, decision.move.target])
            agent_plies[side] += 1
    except Exception as exc:  # noqa: BLE001 - setup failures (bad checkpoint, ...) too
        error = {
            "type": type(exc).__name__, "message": str(exc)[:500],
            "traceback": traceback.format_exc(limit=-6)[-2000:],
            "ply": env.state.ply_count, "actor": None, "side": None,
        }

    result = env.engine.outcome(env.state)
    if error is not None:
        outcome = {"status": "error", "winner": None, "reason": "error"}
    elif result is not None:
        winner = None if result.winner is None else _side_name(result.winner)
        outcome = {"status": "finished", "winner": winner, "reason": result.reason}
    else:
        outcome = {"status": "truncated", "winner": None, "reason": "max_plies"}

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "game_index": spec.game_index,
        "game_key": spec.game_key,
        "pair": [spec.agent_a, spec.agent_b],
        "red_agent": spec.red_agent,
        "black_agent": spec.black_agent,
        "a_color": spec.a_color,
        "opening": {
            "id": opening.opening_id, "family": opening.family, "name": opening.name,
            "book_version": data["opening_book_version"], "plies": len(opening.moves),
        },
        "seeds": {"game": spec.seed, "red": spec.seed_red, "black": spec.seed_black},
        "budgets": {side: agent_registry.budget_to_dict(budgets[side]) for side in ("red", "black")},
        "moves": moves,
        "outcome": outcome,
        "plies": env.state.ply_count,
        "agent_plies": agent_plies,
        "fingerprint": _fingerprint(move_keys),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "worker_pid": os.getpid(),
        "finished_at": provenance.utc_now_iso(),
    }
    if error is not None:
        record["error"] = error
    return record


def _worker_init() -> None:
    # Only the main process reacts to Ctrl+C; it then stops the workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _play_job(spec: GameSpec, data: dict[str, Any], run_id: str, config_hash: str) -> dict[str, Any]:
    """Top-level (picklable) job function for spawn-based process pools."""
    return play_tournament_game(spec, data, run_id=run_id, config_hash=config_hash)


# --------------------------------------------------------------------------
# games.jsonl: append-only writer and tolerant reader
# --------------------------------------------------------------------------


def dump_record(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class GamesLog:
    """Append one fully written, fsynced line per finished game."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = None

    def __enter__(self) -> "GamesLog":
        self._stream = self.path.open("ab")
        return self

    def append(self, record: Mapping[str, Any]) -> None:
        assert self._stream is not None
        self._stream.write(dump_record(record))
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def __exit__(self, *exc_info) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


@dataclass
class LogEntry:
    summary: dict[str, Any]
    raw: bytes


def read_games_log(
    path: Path,
    expected_keys: Mapping[int, str],
    config_hash: str,
    logger: logging.Logger | None = None,
) -> dict[int, LogEntry]:
    """Read ``games.jsonl``, repairing a torn final line.

    A final line cut off by a crash is dropped (with a warning) and the file is
    rewritten clean. Any other malformed line, a duplicate ``game_index``, or a
    record that does not belong to this schedule/config raises ``ValueError``.
    """
    if not path.exists():
        return {}
    payload = path.read_bytes()
    pieces = payload.split(b"\n")
    tail = pieces.pop()  # b"" when the file ends with a newline
    lines = [(number, piece) for number, piece in enumerate(pieces, start=1) if piece.strip()]
    repaired = False
    if tail.strip():
        number = len(pieces) + 1
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            if logger:
                logger.warning("games.jsonl line %d is truncated (crash?); discarding it", number)
            repaired = True
        else:
            lines.append((number, tail))
            repaired = True  # valid but missing its newline

    entries: dict[int, LogEntry] = {}
    for number, piece in lines:
        try:
            record = json.loads(piece)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"{path.name}: corrupt record on line {number}") from exc
        if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path.name}: unsupported record on line {number}")
        index = record.get("game_index")
        if type(index) is not int or index not in expected_keys:
            raise ValueError(f"{path.name}: line {number} has game_index {index!r} outside the schedule")
        if index in entries:
            raise ValueError(f"{path.name}: duplicate game_index {index} (line {number})")
        if record.get("game_key") != expected_keys[index]:
            raise ValueError(f"{path.name}: line {number} game_key does not match the schedule")
        if record.get("config_hash") != config_hash:
            raise ValueError(f"{path.name}: line {number} was written with a different config_hash")
        outcome = record.get("outcome") or {}
        entries[index] = LogEntry(
            summary={
                "game_index": index, "game_key": record["game_key"],
                "status": outcome.get("status"), "winner": outcome.get("winner"),
                "reason": outcome.get("reason"),
            },
            raw=piece,
        )
    if repaired:
        write_bytes_atomic(path, b"".join(entry.raw + b"\n" for _, entry in sorted(entries.items())))
    return entries


# --------------------------------------------------------------------------
# Run orchestration
# --------------------------------------------------------------------------


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "?"
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def describe_result(record: Mapping[str, Any]) -> str:
    outcome = record["outcome"]
    if outcome["status"] == "error":
        error = record.get("error") or {}
        return f"ERROR {error.get('type')}: {error.get('message', '')[:60]}"
    if outcome["status"] == "truncated":
        return f"truncated ({outcome['reason']})"
    if outcome["winner"] is None:
        return f"draw ({outcome['reason']})"
    winner = record["red_agent"] if outcome["winner"] == "red" else record["black_agent"]
    return f"{winner} wins ({outcome['reason']})"


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, child in value.items():
            flat.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value}


def config_differences(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    a, b = _flatten(dict(old)), _flatten(dict(new))
    return [
        f"{key}: {a.get(key, '<absent>')!r} -> {b.get(key, '<absent>')!r}"
        for key in sorted(set(a) | set(b)) if a.get(key, object()) != b.get(key, object())
    ]


def _new_run_id(config: ResolvedConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prefix = "" if config.mode == "full" else f"{config.mode}_"
    return f"{prefix}{stamp}_{config.config_hash[:8]}"


def _execute(
    pending: list[GameSpec],
    data: dict[str, Any],
    run_id: str,
    config_hash: str,
    workers: int,
    on_result: Callable[[dict[str, Any]], None],
) -> None:
    """Run ``pending`` games; ``workers == 1`` runs inline (no process pool)."""
    if workers <= 1 or len(pending) <= 1:
        for spec in pending:
            on_result(play_tournament_game(spec, data, run_id=run_id, config_hash=config_hash))
        return
    pool = ProcessPoolExecutor(max_workers=min(workers, len(pending)), initializer=_worker_init)
    inflight: set = set()
    queue = iter(pending)

    def submit_next() -> None:
        spec = next(queue, None)
        if spec is not None:
            inflight.add(pool.submit(_play_job, spec, data, run_id, config_hash))

    ok = False
    try:
        for _ in range(workers * 2):  # small window keeps cancellation cheap
            submit_next()
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                inflight.discard(future)
                on_result(future.result())
                submit_next()
        ok = True
    finally:
        if not ok:
            for future in inflight:
                future.cancel()
            for process in list(getattr(pool, "_processes", {}).values()):
                try:
                    process.terminate()  # stop long in-flight games right away
                except Exception:  # noqa: BLE001
                    pass
        pool.shutdown(wait=ok, cancel_futures=True)


def run_tournament(
    config: ResolvedConfig,
    *,
    config_path: str | None = None,
    output_root: str | Path | None = None,
    workers: Any = "auto",
    resume_dir: str | Path | None = None,
    allow_code_change: bool = False,
    retry_errors: bool = False,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Run (or resume) a tournament and return a summary of this invocation."""
    import yaml

    data = config.data
    chash = config.config_hash
    specs = build_schedule(data)
    expected_keys = {spec.game_index: spec.game_key for spec in specs}
    total = len(specs)
    n_workers = resolve_workers(workers)
    git_now = provenance.git_info(BACKEND_ROOT)
    session_started = time.time()

    if resume_dir is not None:
        run_dir = Path(resume_dir)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"not a tournament run directory (no manifest.json): {run_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != chash:
            diffs = config_differences(manifest.get("config", {}), config.hash_view)
            raise ValueError(
                "config_hash differs from the run being resumed "
                f"({manifest.get('config_hash', '?')[:12]} vs {chash[:12]}); changed keys:\n  "
                + "\n  ".join(diffs or ["(none visible; excluded/included agents changed?)"])
            )
        old_commit = (manifest.get("provenance", {}).get("git") or {}).get("commit")
        code_changed = bool(old_commit and git_now.get("commit") and old_commit != git_now["commit"])
        if code_changed and not allow_code_change:
            raise ValueError(
                f"git commit changed since the run started ({old_commit[:10]} -> "
                f"{git_now['commit'][:10]}); pass --allow-code-change to continue anyway"
            )
        run_id = manifest["run_id"]
        manifest.setdefault("resume_history", []).append({
            "at": provenance.utc_now_iso(), "argv": list(argv or sys.argv),
            "git_commit": git_now.get("commit"), "git_dirty": git_now.get("dirty"),
            "code_changed": code_changed, "workers": n_workers, "retry_errors": retry_errors,
        })
    else:
        run_id = _new_run_id(config)
        run_dir = Path(output_root or DEFAULT_OUTPUT_ROOT) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(config.data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        info = provenance.collect_provenance(BACKEND_ROOT, argv)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "name": data["name"],
            "mode": config.mode,
            "config_path": config_path,
            "config_hash": chash,
            "config": config.hash_view,
            "games_per_pair": config.games_per_pair,
            "total_games": total,
            "agents": {name: agent_registry.describe(name, params)
                       for name, params in data["agents"].items()},
            "excluded_agents": config.excluded,
            "openings": [
                {"id": oid, "family": get_opening(oid, data["opening_book_version"]).family,
                 "moves_sha256": provenance.sha256_text(provenance.canonical_json(
                     [[m.source, m.target] for m in get_opening(oid, data["opening_book_version"]).moves]))}
                for oid in data["openings"]
            ],
            "provenance": info,
            "started_at": provenance.utc_now_iso(),
            "workers": n_workers,
            "resume_history": [],
        }
        if info["git"].get("dirty"):
            (run_dir / "git.diff").write_text(provenance.git_diff(BACKEND_ROOT), encoding="utf-8")
    manifest.update({"status": "running", "workers": n_workers, "finished_at": None})
    write_json_atomic(run_dir / "manifest.json", manifest)

    logger = _setup_logger(run_dir / "run.log")
    log_path = run_dir / "games.jsonl"
    interrupted = False
    try:
        logger.info("Run %s (%s) -> %s", run_id, config.mode, run_dir)
        if config.excluded:
            for name, reason in config.excluded.items():
                logger.warning("WARNING: agent %r EXCLUDED from this tournament: %s", name, reason)
        if git_now.get("dirty"):
            logger.warning(
                "WARNING: git working tree is DIRTY (%d change(s)); results are not tied to a "
                "clean commit. See git.diff in the run directory.", len(git_now["status"]))
        logger.info("Agents: %s | games/pair: %d | total games: %d | workers: %d",
                    ", ".join(data["agents"]), config.games_per_pair, total, n_workers)

        entries = read_games_log(log_path, expected_keys, chash, logger)
        if retry_errors:
            retried = [e for e in entries.values() if e.summary["status"] == "error"]
            if retried:
                with (run_dir / "games.retried.jsonl").open("ab") as audit:
                    for entry in retried:
                        audit.write(entry.raw + b"\n")
                entries = {i: e for i, e in entries.items() if e.summary["status"] != "error"}
                write_bytes_atomic(log_path, b"".join(
                    e.raw + b"\n" for _, e in sorted(entries.items())))
                logger.info("Re-running %d game(s) that ended in error", len(retried))
        summaries = {index: entry.summary for index, entry in entries.items()}
        pending = [spec for spec in specs if spec.game_index not in summaries]
        logger.info("Already done: %d/%d | pending: %d", len(summaries), total, len(pending))

        done_this_session = 0

        def write_progress(status: str, eta: float | None) -> None:
            counts = {"error": 0, "truncated": 0}
            for summary in summaries.values():
                if summary["status"] in counts:
                    counts[summary["status"]] += 1
            write_json_atomic(run_dir / "progress.json", {
                "run_id": run_id, "status": status, "completed": len(summaries),
                "total": total, "errors": counts["error"], "truncated": counts["truncated"],
                "eta_seconds": eta, "updated_at": provenance.utc_now_iso(),
            })

        write_progress("running", None)
        with GamesLog(log_path) as games_log:
            def on_result(record: dict[str, Any]) -> None:
                nonlocal done_this_session
                games_log.append(record)
                outcome = record["outcome"]
                summaries[record["game_index"]] = {
                    "game_index": record["game_index"], "game_key": record["game_key"],
                    "status": outcome["status"], "winner": outcome["winner"],
                    "reason": outcome["reason"],
                }
                done_this_session += 1
                remaining = total - len(summaries)
                elapsed = time.time() - session_started
                eta = elapsed / done_this_session * remaining if done_this_session else None
                write_progress("running", eta)
                logger.info(
                    "[%d/%d] %s vs %s | opening=%s | A=%s | %s | %d plies | %.1fs | ETA %s",
                    len(summaries), total, record["pair"][0], record["pair"][1],
                    record["opening"]["id"], record["a_color"], describe_result(record),
                    record["plies"], record["elapsed_seconds"], format_duration(eta))

            try:
                _execute(pending, data, run_id, chash, n_workers, on_result)
            except KeyboardInterrupt:
                interrupted = True

        finished = len(summaries) == total
        status = "complete" if finished else ("interrupted" if interrupted else "incomplete")
        write_progress(status, 0.0 if finished else None)
        if interrupted:
            logger.warning("Interrupted. %d/%d games are safely saved. Resume with:\n"
                           "  python -m src tournament --resume %s", len(summaries), total, run_dir)
        elif finished:
            logger.info("Tournament complete: %d games in %s", total, run_dir)
        manifest["status"] = status
        manifest["finished_at"] = provenance.utc_now_iso()
        manifest.setdefault("session_seconds", []).append(round(time.time() - session_started, 3))
        manifest["total_seconds"] = round(sum(manifest["session_seconds"]), 3)
        write_json_atomic(run_dir / "manifest.json", manifest)
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_at"] = provenance.utc_now_iso()
        write_json_atomic(run_dir / "manifest.json", manifest)
        raise
    finally:
        _close_logger(logger)

    return {
        "run_id": run_id, "run_dir": str(run_dir), "status": status,
        "completed": len(summaries), "total": total, "interrupted": interrupted,
        "errors": sum(1 for s in summaries.values() if s["status"] == "error"),
        "truncated": sum(1 for s in summaries.values() if s["status"] == "truncated"),
        "new_games": done_this_session,
    }


def print_pilot_estimate(run_dir: str | Path, full_games_per_pair: int, workers: int) -> None:
    """Extrapolate total tournament time from the one-game-per-pair pilot."""
    per_pair: list[tuple[str, str, float, int]] = []
    for line in (Path(run_dir) / "games.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["outcome"]["status"] == "error":
            continue
        per_pair.append((*record["pair"], record["elapsed_seconds"], record["plies"]))
    print("\nPilot: one game per pair (first opening, A=red)")
    for a, b, seconds, plies in per_pair:
        print(f"  {a:>10} vs {b:<10} {seconds:8.1f}s  {plies:3d} plies")
    sequential = sum(seconds for _, _, seconds, _ in per_pair) * full_games_per_pair
    print(f"\nEstimate for {len(per_pair)} pairs x {full_games_per_pair} games/pair:")
    print(f"  sequential : {format_duration(sequential)}")
    print(f"  {workers} worker(s): {format_duration(sequential / workers)}"
          "  (assumes linear speed-up; one sample per pair is rough)")


def load_resume_config(
    run_dir: str | Path,
    config_override: str | None = None,
    **resolve_kwargs: Any,
) -> tuple[ResolvedConfig, str | None]:
    """Re-resolve the *current* config for a run being resumed (hash is checked later)."""
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"not a tournament run directory (no manifest.json): {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = config_override or manifest.get("config_path")
    if not config_path:
        raise ValueError("manifest has no config_path; pass --config")
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = BACKEND_ROOT / path
    raw = load_raw_config(path)
    return resolve_config(raw, mode=manifest.get("mode", "full"), **resolve_kwargs), config_path


# --------------------------------------------------------------------------
# Public class (names from the original skeleton are preserved)
# --------------------------------------------------------------------------


class Tournament:
    """Thin object API over the functions above.

    ``match_runner`` is accepted for backward compatibility and unused: the
    tournament plays games with :func:`play_tournament_game` instead.
    """

    def __init__(self, match_runner, config) -> None:
        del match_runner
        if isinstance(config, ResolvedConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = resolve_config(load_raw_config(config))
        else:
            self.config = resolve_config(config)

    def schedule(self, agents: Iterable[str] | None = None) -> list[GameSpec]:
        data = dict(self.config.data)
        if agents is not None:
            wanted = list(agents)
            unknown = [name for name in wanted if name not in data["agents"]]
            if unknown:
                raise ValueError(f"agents not in this tournament: {unknown}")
            data["agents"] = {name: data["agents"][name] for name in wanted}
        return build_schedule(data)

    def run(self, agents: Iterable[str] | None = None, **options: Any) -> dict[str, Any]:
        if agents is not None:
            raise NotImplementedError("run() always uses the configured agents; edit the config")
        return run_tournament(self.config, **options)

    def export(self, results: Iterable[Mapping[str, Any]], path: str | Path) -> None:
        """Write game records as JSONL sorted by ``game_index``."""
        rows = sorted(results, key=lambda record: record["game_index"])
        write_bytes_atomic(Path(path), b"".join(dump_record(row) for row in rows))
