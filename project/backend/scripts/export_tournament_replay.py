"""Export one tournament game to the JSON format supported by /replay.

Run from project/backend:
    python -m scripts.export_tournament_replay --input path/to/games.jsonl \
        --red alphabeta --black mcts --output replay-game.json
The original tournament log is not modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def _read_games(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at line {line_number}")
            yield line_number, value


def _checked_action(row: Any, ply: int) -> dict[str, int]:
    if not isinstance(row, dict):
        raise ValueError(f"move {ply} is not an object")
    if row.get("ply") != ply:
        raise ValueError(f"move {ply} has an invalid ply index")
    source = row.get("source")
    target = row.get("target")
    for key, value in (("source", source), ("target", target)):
        if type(value) is not int or not 0 <= value < 90:
            raise ValueError(f"move {ply} has invalid {key} (expected integer 0..89)")
    if source == target:
        raise ValueError(f"move {ply} has identical source and target")
    return {"source": source, "target": target}


def normalize_tournament_game(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a tournament record to the existing frontend replay contract."""
    raw_moves = record.get("moves")
    if not isinstance(raw_moves, list):
        raise ValueError("tournament game is missing a moves array")
    if len(raw_moves) > 1000:
        raise ValueError("game has more than 1000 moves; the frontend cannot load it")
    moves: list[dict[str, Any]] = []
    for ply, row in enumerate(raw_moves, start=1):
        action = _checked_action(row, ply)
        normalized = {key: value for key, value in row.items() if key not in {"source", "target"}}
        normalized["action"] = action
        moves.append(normalized)

    run_id = record.get("run_id") or "unknown-run"
    game_key = record.get("game_key")
    if not isinstance(game_key, str) or not game_key:
        game_key = str(record.get("game_index", "unknown-index"))
    return {
        "schema_version": 1,
        "game_id": f"tournament:{run_id}:{game_key}",
        "initial_side": "red",
        "red_agent": record.get("red_agent"),
        "black_agent": record.get("black_agent"),
        "opening": record.get("opening"),
        "moves": moves,
        "outcome": record.get("outcome"),
    }


def export_tournament_replay(
    source: Path,
    destination: Path,
    *,
    game_index: int | None = None,
    red: str | None = None,
    black: str | None = None,
) -> dict[str, Any]:
    """Stream a JSONL log and write just the selected game as JSON."""
    if not source.is_file():
        raise FileNotFoundError(f"tournament log does not exist: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different files")
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    if game_index is not None and game_index < 0:
        raise ValueError("game_index must be nonnegative")
    if game_index is not None and (red is not None or black is not None):
        raise ValueError("specify --game-index OR --red/--black filters")

    for line_number, record in _read_games(source):
        if game_index is not None and record.get("game_index") != game_index:
            continue
        if red is not None and record.get("red_agent") != red:
            continue
        if black is not None and record.get("black_agent") != black:
            continue
        outcome = record.get("outcome")
        if isinstance(outcome, dict) and outcome.get("status") == "error":
            if game_index is not None:
                raise ValueError(f"selected game_index {game_index} is an errored game")
            continue
        result = normalize_tournament_game(record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return result

    filters = f"game_index={game_index}, red={red}, black={black}"
    raise ValueError(f"no matching non-error game in {source} ({filters})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="tournament games.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="new single-game JSON file")
    parser.add_argument("--game-index", type=int, default=None)
    parser.add_argument("--red", type=str, default=None, help="filter by Red agent")
    parser.add_argument("--black", type=str, default=None, help="filter by Black agent")
    args = parser.parse_args()
    try:
        game = export_tournament_replay(
            args.input, args.output, game_index=args.game_index, red=args.red, black=args.black,
        )
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Exported {game['game_id']} ({len(game['moves'])} plies) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
