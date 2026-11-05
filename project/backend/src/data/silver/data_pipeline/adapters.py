"""Source-specific conversion; future CCPD adapters share the same contract."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from src.data.bronze.models import AlphaBetaRawGame
from src.data.models import MoveModel, Provenance
from src.data.silver.models import GameMove, GameOutcome, SilverGame, SilverTransition

from .deduplication import canonical_game_hash
from .errors import SilverValidationError

RawT = TypeVar("RawT")

ACTOR_TYPES = {
    "alphabeta": "alphabeta",
    "alphabeta_exploration": "alphabeta",
    "opening_book": "opening_book",
    "opening_random": "random",
    "random": "random",
    "human": "human",
    "computer": "computer",
    "minimax": "computer",
    "rules": "computer",
    "astar": "computer",
    "dqn": "computer",
    "mcts": "computer",
    "heuristic": "computer",
}


class SourceAdapter(ABC, Generic[RawT]):
    @abstractmethod
    def convert(self, record: RawT, provenance: Provenance) -> SilverGame:
        """Convert a raw source record into the common Silver schema."""

    def audit(self, record: RawT, transitions: list[SilverTransition]) -> None:
        """Check source-specific snapshots against replay, when provided."""


class AlphaBetaAdapter(SourceAdapter[AlphaBetaRawGame]):
    def convert(self, record: AlphaBetaRawGame, provenance: Provenance) -> SilverGame:
        initial_side = getattr(record, "initial_side", "red")
        if initial_side not in ("red", "black"):
            raise SilverValidationError("INVALID_SIDE", f"invalid initial_side: {initial_side!r}")

        moves: list[GameMove] = []
        for number, row in enumerate(record.moves, start=1):
            try:
                action = MoveModel.model_validate(row.get("action"))
            except (ValueError, TypeError) as exc:
                raise SilverValidationError("INVALID_ACTION", str(exc), number) from exc
            expected_side = initial_side if number % 2 == 1 else (
                "black" if initial_side == "red" else "red"
            )
            actor = row.get("actor")
            if actor is None:
                actor = getattr(record, f"{expected_side}_agent", None)
            if actor is not None and not isinstance(actor, str):
                raise SilverValidationError("INVALID_ACTOR", "actor must be a string", number)
            actor = actor.lower() if actor is not None else None
            moves.append(GameMove(
                ply=number,
                action=action,
                source_type=ACTOR_TYPES.get(actor, "unknown"),
                actor=actor,
            ))

        if not moves:
            raise SilverValidationError("EMPTY_GAME", "game contains no moves")

        if record.outcome.get("status") not in ("finished", "truncated"):
            raise SilverValidationError("INVALID_OUTCOME", "Alpha-Beta game requires finished/truncated status")
        try:
            outcome = GameOutcome.model_validate(record.outcome)
        except ValueError as exc:
            raise SilverValidationError("INVALID_OUTCOME", str(exc)) from exc

        digest = canonical_game_hash(record.initial_board, initial_side, [move.action for move in moves])
        return SilverGame(
            game_id=f"{provenance.source}:{digest[:32]}",
            provenance=provenance,
            initial_board=record.initial_board,
            initial_side=initial_side,
            moves=moves,
            outcome=outcome,
            canonical_game_sha256=digest,
        )

    def audit(self, record: AlphaBetaRawGame, transitions: list[SilverTransition]) -> None:
        if len(record.moves) != len(transitions):
            raise SilverValidationError("MOVE_COUNT_MISMATCH", "transition count differs from input")
        for transition, raw_move in zip(transitions, record.moves):
            number = transition.ply
            checks = (
                ("ply", number),
                ("board", transition.board),
                ("next_board", transition.next_board),
                ("side", transition.side),
                ("terminated", transition.terminated),
            )
            for field, expected in checks:
                if field in raw_move and raw_move[field] != expected:
                    raise SilverValidationError(
                        "SOURCE_MISMATCH",
                        f"input {field} does not match replay at ply {number}",
                        number,
                    )
        final_board = getattr(record, "final_board", None)
        if final_board is not None and final_board != transitions[-1].next_board:
            raise SilverValidationError("SOURCE_MISMATCH", "final_board differs from replay")
        ply_count = getattr(record, "ply_count", None)
        if ply_count is not None and ply_count != len(transitions):
            raise SilverValidationError("SOURCE_MISMATCH", "ply_count differs from replay")
