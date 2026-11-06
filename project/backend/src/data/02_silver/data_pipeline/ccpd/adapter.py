"""Convert parsed CCPD games to common Silver schema using legal-move matching."""

from src.data.models import MoveModel, Provenance
from src.data.silver.models import GameMove, GameOutcome, SilverGame
from src.environment import GameEngine, GameState, Side

from ..deduplication import canonical_game_hash
from ..errors import SilverValidationError
from .notation import resolve_move
from .pgn_parser import CCPDRecord


class CCPDAdapter:
    """Decode notation with real engine legality; never infer an ambiguous action."""

    def __init__(self, engine: GameEngine | None = None) -> None:
        self.engine = engine or GameEngine()

    def convert(self, record: CCPDRecord, provenance: Provenance) -> SilverGame:
        side = Side.RED if record.initial_side == "red" else Side.BLACK
        state = GameState(tuple(record.initial_board), side)
        moves: list[GameMove] = []
        for ply, notation in enumerate(record.move_tokens, start=1):
            if self.engine.outcome(state) is not None:
                raise SilverValidationError(
                    "ENGINE_TERMINAL_TOO_EARLY", "more PGN moves after engine terminal position", ply
                )
            matched = resolve_move(notation, state, self.engine.legal_moves(state), ply)
            moves.append(GameMove(
                ply=ply,
                action=MoveModel(source=matched.source, target=matched.target),
                source_type="human", actor="human",
            ))
            # Matched is an engine-legal move; replay stage rechecks independently.
            state = state.moved(matched)

        if record.result == "1-0":
            outcome = GameOutcome(status="finished", winner="red",
                                  reason="pgn_result", result_source="recorded")
        elif record.result == "0-1":
            outcome = GameOutcome(status="finished", winner="black",
                                  reason="pgn_result", result_source="recorded")
        elif record.result == "1/2-1/2":
            outcome = GameOutcome(status="finished", winner=None,
                                  reason="pgn_result", result_source="recorded")
        else:
            outcome = GameOutcome(status="unknown", winner=None,
                                  reason=None, result_source="recorded")

        fingerprint = canonical_game_hash(record.initial_board, record.initial_side,
                                          [move.action for move in moves])
        return SilverGame(
            game_id=f"ccpd:{fingerprint[:32]}",
            provenance=provenance,
            initial_board=record.initial_board,
            initial_side=record.initial_side,
            moves=moves,
            outcome=outcome,
            canonical_game_sha256=fingerprint,
        )
