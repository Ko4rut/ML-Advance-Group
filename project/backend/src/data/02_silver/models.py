"""Source-neutral Silver records. Engine replay remains an ETL responsibility."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..models import Board, MoveModel, Provenance, SideName, SourceType, StrictModel, VersionedRecord


class GameOutcome(StrictModel):
    status: Literal["finished", "truncated", "unknown"]
    winner: SideName | None = None
    reason: str | None = None
    # Engine terminal verdict and PGN result are different kinds of evidence.
    result_source: Literal["engine", "recorded"] = "engine"

    @model_validator(mode="after")
    def no_winner_for_unfinished(self) -> "GameOutcome":
        if self.status != "finished" and self.winner is not None:
            raise ValueError("unfinished games cannot declare a winner")
        return self


class GameMove(StrictModel):
    ply: int = Field(ge=1)
    action: MoveModel
    source_type: SourceType
    actor: str | None = None


class SilverGame(VersionedRecord):
    game_id: str = Field(min_length=1)
    provenance: Provenance
    initial_board: Board
    initial_side: SideName
    moves: list[GameMove]
    outcome: GameOutcome
    canonical_game_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def contiguous_plies(self) -> "SilverGame":
        if [move.ply for move in self.moves] != list(range(1, len(self.moves) + 1)):
            raise ValueError("game moves must have contiguous one-based ply numbers")
        return self


class SilverTransition(VersionedRecord):
    game_id: str = Field(min_length=1)
    ply: int = Field(ge=1)
    board: Board
    side: SideName
    action: MoveModel
    next_board: Board
    next_side: SideName
    source_type: SourceType
    actor: str | None = None
    recorded_reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    # PGN ends here, independently of engine checkmate/stalemate.
    recorded_game_end: bool = False

    @model_validator(mode="after")
    def consistent_turn(self) -> "SilverTransition":
        if self.side == self.next_side:
            raise ValueError("each move must alternate sides")
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated are mutually exclusive")
        return self
