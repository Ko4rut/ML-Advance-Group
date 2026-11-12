"""Experimental cc-lite codec and validation probes; not a production adapter."""

from dataclasses import replace
import math

import numpy as np

from src.environment import GameEngine, GameState, Move, Side


POLICY_SIZE = 8100
PIECE_SYMBOLS = " KAEHRCP"


def action_id(move: Move) -> int:
    return move.source * 90 + move.target


def decode_action(index: int) -> Move:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < POLICY_SIZE:
        raise ValueError("Unknown native policy index")
    return Move(index // 90, index % 90)


def observation(state: GameState, engine: GameEngine) -> np.ndarray:
    planes = np.zeros((16, 10, 9), dtype=np.float32)
    for square, piece in enumerate(state.board):
        if piece:
            channel = abs(piece) - 1 + (7 if piece < 0 else 0)
            planes[channel, square // 9, square % 9] = 1
    planes[14] = float(state.side_to_move is Side.RED)
    planes[15] = float(engine.validator.is_in_check(state, state.side_to_move))
    return planes


def fen(state: GameState) -> str:
    ranks = []
    for row in range(10):
        rank, empty = "", 0
        for piece in state.board[row * 9:(row + 1) * 9]:
            if not piece:
                empty += 1
                continue
            if empty:
                rank += str(empty)
                empty = 0
            symbol = PIECE_SYMBOLS[abs(piece)]
            rank += symbol if piece > 0 else symbol.lower()
        if empty:
            rank += str(empty)
        ranks.append(rank)
    return "/".join(ranks) + (" r" if state.side_to_move is Side.RED else " b")


def legal_priors(scores, moves: list[Move], *, kind: str = "logits") -> dict[Move, float]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (POLICY_SIZE,) or not np.isfinite(scores).all():
        raise ValueError("Policy must contain 8100 finite scores")
    if not moves or len(set(moves)) != len(moves):
        raise ValueError("Expected a nonempty unique legal-move list")
    indices = [action_id(move) for move in moves]
    weights = scores[indices]
    if kind == "logits":
        weights = np.exp(weights - np.max(weights))
    elif kind == "probabilities":
        if (scores < 0).any():
            raise ValueError("Negative policy probability")
    else:
        raise ValueError("Unknown policy representation")
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("No finite positive legal policy mass")
    return {move: float(weight / total) for move, weight in zip(moves, weights, strict=True)}


def predict(state: GameState, engine: GameEngine, forward) -> tuple[dict[Move, float], float]:
    moves = engine.legal_moves(state)
    result = engine.outcome(state, legal_moves=moves)
    if result is not None:
        return {}, float(result.reward(state.side_to_move))
    scores, raw_value = forward(observation(state, engine))
    value = float(raw_value)
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("Value must be finite and within [-1, 1]")
    return legal_priors(scores, moves), value


def color_swap(state: GameState) -> GameState:
    """Rotate 180 degrees and exchange colors; do not reuse old rule history."""
    return GameState(tuple(-p for p in reversed(state.board)), state.side_to_move.opponent)


def fixtures() -> dict[str, GameState]:
    engine = GameEngine()
    initial = GameState.initial()
    asymmetric = initial
    # Deterministic legal trajectory; no random model or trained weights involved.
    for _ in range(3):
        asymmetric = asymmetric.moved(engine.legal_moves(asymmetric)[0])
    board = [0] * 90
    for square, piece in {4: -1, 85: 1, 9: 5, 10: 5, 22: 7}.items():
        board[square] = piece
    red_mate = GameState(tuple(board), Side.RED)
    states = {
        "initial_red": initial,
        "initial_black": GameState(initial.board, Side.BLACK),
        "asymmetric_black": asymmetric,
        "asymmetric_red": color_swap(asymmetric),
        "mate_in_one_red": red_mate,
        "mate_in_one_black": color_swap(red_mate),
    }
    for name in ("mate_in_one_red", "mate_in_one_black"):
        state = states[name]
        winning = next(move for move in engine.legal_moves(state)
                       if (result := engine.outcome(state.moved(move))) is not None
                       and result.winner is state.side_to_move)
        states[f"checkmated_{state.side_to_move.opponent.name.lower()}"] = state.moved(winning)
    check = [0] * 90
    for square, piece in {4: -1, 84: 1, 13: 5}.items():
        check[square] = piece
    states["in_check_black"] = GameState(tuple(check), Side.BLACK)
    states["in_check_red"] = color_swap(states["in_check_black"])
    states["no_capture_draw"] = replace(initial, no_capture_count=120)
    states["repetition_draw"] = replace(initial, history=(initial.position_key(),) * 3)
    return states
