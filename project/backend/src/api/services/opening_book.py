"""Application-level opening book orchestration for live API matches.

Each match samples exactly one curated legal line. Only AI turns may be
played from the book; humans always control their own moves. Any divergence
deactivates the chosen line, so an AI cannot force an incompatible reply.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from ...environment import GameState, Move
from ...training.opening_book import (
    BOOK_VERSION,
    OPENING_LINES,
    OpeningSelection,
    sample_opening,
)


@dataclass(slots=True)
class OpeningSession:
    selection: OpeningSelection
    seed: int
    completed_plies: int = 0
    active: bool = True


class OpeningBookService:
    """Isolate opening selection, progress and divergence from MatchService."""

    def start(
        self,
        mode: str,
        plies: int,
        seed: int | None = None,
    ) -> OpeningSession | None:
        if mode == "none":
            return None
        if mode != "book":
            raise ValueError(f"Unsupported API opening mode: {mode}")
        if seed is None:
            seed = secrets.randbits(63)
        return OpeningSession(
            selection=sample_opening(seed, plies),
            seed=seed,
        )

    def next_move(
        self,
        session: OpeningSession | None,
        state: GameState,
        legal_moves: list[Move],
    ) -> Move | None:
        """Get next scripted AI move only while the prefix still matches."""
        if session is None or not session.active:
            return None
        if state.ply_count != session.completed_plies:
            session.active = False
            return None
        moves = session.selection.moves
        if state.ply_count >= len(moves):
            session.active = False
            return None
        action = moves[state.ply_count]
        if action not in legal_moves:
            session.active = False
            return None
        return action

    def record_move(
        self,
        session: OpeningSession | None,
        state_before: GameState,
        action: Move,
    ) -> None:
        """Track human or AI moves without treating human moves as book labels."""
        if session is None or not session.active:
            return
        if state_before.ply_count != session.completed_plies:
            session.active = False
            return
        moves = session.selection.moves
        if state_before.ply_count >= len(moves):
            session.active = False
            return
        if action != moves[state_before.ply_count]:
            session.active = False
            return
        session.completed_plies += 1
        if session.completed_plies >= len(moves):
            session.active = False

    def list_openings(self) -> list[dict[str, str]]:
        """Metadata only: do not expose scripted AI future moves to callers."""
        return [
            {
                "id": line.opening_id,
                "family": line.family,
                "name": line.name,
            }
            for line in OPENING_LINES
        ]

    @staticmethod
    def describe(session: OpeningSession | None) -> dict | None:
        if session is None:
            return None
        line = session.selection.line
        return {
            "id": line.opening_id,
            "family": line.family,
            "name": line.name,
            "book_version": BOOK_VERSION,
            "seed": session.seed,
            "planned_plies": len(session.selection.moves),
            "completed_plies": session.completed_plies,
            "active": session.active,
        }
