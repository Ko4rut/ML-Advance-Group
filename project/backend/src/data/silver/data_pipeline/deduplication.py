"""Content-based game deduplication; ignore source-supplied fingerprints."""

import hashlib
import json

from src.data.models import MoveModel
from src.data.silver.models import SilverGame


def canonical_game_hash(initial_board: list[int], initial_side: str,
                        moves: list[MoveModel]) -> str:
    payload = {
        "board": initial_board,
        "side": initial_side,
        "actions": [[move.source, move.target] for move in moves],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class GameDeduplicator:
    def __init__(self) -> None:
        self._original_ids: dict[str, str] = {}

    def register(self, game: SilverGame) -> str | None:
        """Return existing game ID on duplicate, otherwise register it."""
        previous = self._original_ids.get(game.canonical_game_sha256)
        if previous is not None:
            return previous
        self._original_ids[game.canonical_game_sha256] = game.game_id
        return None
