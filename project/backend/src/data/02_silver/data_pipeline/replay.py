"""Replay common Silver games with the existing Xiangqi GameEngine."""

from src.data.silver.models import SilverGame, SilverTransition
from src.environment import GameEngine, GameState, Move, Side

from .deduplication import canonical_game_hash
from .errors import SilverValidationError


def _side_name(side: Side | None) -> str | None:
    if side is None:
        return None
    return "red" if side is Side.RED else "black"


class SilverReplayValidator:
    def __init__(self, engine: GameEngine | None = None) -> None:
        self.engine = engine or GameEngine()

    def replay(self, game: SilverGame) -> list[SilverTransition]:
        digest = canonical_game_hash(
            game.initial_board, game.initial_side, [row.action for row in game.moves]
        )
        if digest != game.canonical_game_sha256:
            raise SilverValidationError("HASH_MISMATCH", "canonical game hash does not match moves")

        side = Side.RED if game.initial_side == "red" else Side.BLACK
        state = GameState(tuple(game.initial_board), side)
        transitions: list[SilverTransition] = []

        for game_move in game.moves:
            move = Move(game_move.action.source, game_move.action.target)
            try:
                next_state = self.engine.apply_move(state, move)
            except ValueError as exc:
                raise SilverValidationError("ILLEGAL_MOVE", str(exc), game_move.ply) from exc
            result = self.engine.outcome(next_state)
            transitions.append(SilverTransition(
                game_id=game.game_id,
                ply=game_move.ply,
                board=list(state.board),
                side=_side_name(state.side_to_move),
                action=game_move.action,
                next_board=list(next_state.board),
                next_side=_side_name(next_state.side_to_move),
                source_type=game_move.source_type,
                actor=game_move.actor,
                recorded_reward=None,
                terminated=result is not None,
                truncated=False,
            ))
            state = next_state

        actual = self.engine.outcome(state)
        declared = game.outcome
        if declared.result_source == "recorded":
            # CCPD result is the match record, not necessarily checkmate.
            # A game can end by resignation, adjudication or the clock.
            # Still reject a recorded winner contradicted by a verified terminal state.
            if actual is not None:
                if declared.status != "finished" or _side_name(actual.winner) != declared.winner:
                    raise SilverValidationError(
                        "OUTCOME_MISMATCH", "PGN result conflicts with engine terminal state"
                    )
            if transitions and declared.status == "finished":
                transitions[-1] = transitions[-1].model_copy(
                    update={"recorded_game_end": True}
                )
            return transitions

        # Preserve strict outcome checks for existing Alpha-Beta self-play.
        if actual is None:
            if declared.status == "finished":
                raise SilverValidationError(
                    "OUTCOME_MISMATCH", "source claims finished but board is non-terminal"
                )
        else:
            if declared.status != "finished":
                raise SilverValidationError(
                    "OUTCOME_MISMATCH", "board is terminal but source is not finished"
                )
            if _side_name(actual.winner) != declared.winner:
                raise SilverValidationError("OUTCOME_MISMATCH", "winner differs from GameEngine")
            if declared.reason is not None and declared.reason != actual.reason:
                raise SilverValidationError(
                    "OUTCOME_MISMATCH", "termination reason differs from GameEngine"
                )

        if transitions and actual is None and declared.status == "truncated":
            transitions[-1] = transitions[-1].model_copy(update={"truncated": True})
        return transitions
