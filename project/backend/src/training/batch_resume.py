"""Restart-safe JSONL batch bookkeeping.

The dataset itself is authoritative. Sidecar checkpoints are advisory, rebuilt
from valid completed records after each restart. Never reinterpret truncation
as a draw.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def configuration(
    red: str,
    black: str,
    seed: int,
    red_depth: int,
    black_depth: int,
    max_plies: int,
    opening_plies: int,
    no_progress_limit: int,
) -> dict[str, Any]:
    return {
        "red_agent": red,
        "black_agent": black,
        "seed": seed,
        "red_depth": red_depth if red in {"minimax", "alphabeta"} else None,
        "black_depth": black_depth if black in {"minimax", "alphabeta"} else None,
        "max_plies": max_plies,
        "opening_plies": opening_plies,
        "no_progress_limit": no_progress_limit,
    }


def checkpoint_path_for(output: Path) -> Path:
    return output.with_name(output.name + ".checkpoint.json")


def load_completed(
    output: Path,
    config: dict[str, Any],
    target_games: int,
    *,
    resume: bool,
) -> dict[int, dict[str, Any]]:
    if not resume:
        if output.exists() or checkpoint_path_for(output).exists():
            raise FileExistsError(
                f"Batch already exists: {output}; use --resume or choose a new output"
            )
        return {}

    if not output.exists():
        if checkpoint_path_for(output).exists():
            raise ValueError("Checkpoint exists without JSONL dataset; cannot resume safely")
        return {}

    # Repair only the torn tail. Stream the rest line-by-line so even a
    # multi-GB training dataset can resume without reading it into RAM.
    with output.open("rb+") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        if size:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\\n":
                # Seek backwards to the previous newline in bounded chunks.
                cursor = size
                line_start = 0
                while cursor > 0:
                    step = min(65536, cursor)
                    cursor -= step
                    stream.seek(cursor)
                    chunk = stream.read(step)
                    found = chunk.rfind(b"\\n")
                    if found >= 0:
                        line_start = cursor + found + 1
                        break
                stream.seek(line_start)
                tail = stream.read()
                try:
                    json.loads(tail)
                except (ValueError, UnicodeDecodeError):
                    stream.truncate(line_start)
                    print("[resume] Discarded incomplete JSONL tail", flush=True)
                else:
                    stream.seek(0, os.SEEK_END)
                    stream.write(b"\\n")
                    stream.flush()

        completed: dict[int, dict[str, Any]] = {}
        stream.seek(0)
        for line_no, line in enumerate(stream, start=1):
            try:
                game = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Corrupt record on line {line_no}") from exc
            if not isinstance(game, dict) or game.get("schema_version") != 1:
                raise ValueError(f"Unsupported record schema on line {line_no}")
            index = game.get("game_index")
            if type(index) is not int or not 0 <= index < target_games:
                raise ValueError(f"Invalid game_index on line {line_no}: {index!r}")
            if index in completed:
                raise ValueError(f"Duplicate game_index: {index}")
            for field, expected in config.items():
                if field == "seed":
                    expected = config["seed"] + index * 1009
                if game.get(field) != expected:
                    raise ValueError(
                        f"Game {index}: incompatible {field}; "
                        f"stored={game.get(field)!r}, expected={expected!r}"
                    )
            if not isinstance(game.get("moves"), list):
                raise ValueError(f"Game {index}: invalid moves")
            if len(game["moves"]) != game.get("ply_count"):
                raise ValueError(f"Game {index}: inconsistent ply_count")
            if not isinstance(game.get("fingerprint"), str):
                raise ValueError(f"Game {index}: missing fingerprint")
            if not isinstance(game.get("outcome"), dict):
                raise ValueError(f"Game {index}: invalid outcome")
            completed[index] = {
                "game_index": index,
                "ply_count": game["ply_count"],
                "fingerprint": game["fingerprint"],
                "outcome": game["outcome"],
            }
    return completed

def write_checkpoint(
    output: Path,
    config: dict[str, Any],
    target_games: int,
    completed: dict[int, dict[str, Any]],
) -> Path:
    checkpoint = checkpoint_path_for(output)
    temporary = checkpoint.with_name(checkpoint.name + ".tmp")
    payload = {
        "checkpoint_version": 1,
        "dataset": output.name,
        "target_games": target_games,
        "configuration": config,
        "completed_indices": sorted(completed),
        "completed_games": len(completed),
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
    os.replace(temporary, checkpoint)
    return checkpoint
