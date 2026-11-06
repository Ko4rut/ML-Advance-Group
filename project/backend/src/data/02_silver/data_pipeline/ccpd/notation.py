"""Resolve traditional Chinese Xiangqi notation against GameEngine legal moves.

Never infer an action from incomplete notation: a token must match exactly one
legal move. Unhandled or ambiguous forms are quarantined, not guessed.
"""

from __future__ import annotations

import unicodedata

from src.environment import GameState, Move, Side

from ..errors import SilverValidationError

_NUMBERS = {char: pos for pos, char in enumerate("一二三四五六七八九", start=1)}
_NUMBERS.update({str(n): n for n in range(1, 10)})
_NUMBERS.update({char: pos for pos, char in enumerate("壹貳參肆伍陸柒捌玖", start=1)})
_PIECES = {
    "帥": 1, "帅": 1, "將": 1, "将": 1,
    "仕": 2, "士": 2,
    "相": 3, "象": 3,
    "馬": 4, "马": 4, "傌": 4,
    "車": 5, "车": 5, "俥": 5,
    "炮": 6, "砲": 6,
    "兵": 7, "卒": 7,
}
_DIRECTION = {"進": "forward", "进": "forward", "退": "back", "平": "level"}
_PREFIX = {"前": "front", "後": "back", "后": "back", "中": "middle"}


def _file(col: int, side: Side) -> int:
    return 9 - col if side is Side.RED else col + 1


def _number(char: str, ply: int) -> int:
    norm = unicodedata.normalize("NFKC", char)
    result = _NUMBERS.get(norm)
    if result is None:
        raise SilverValidationError("INVALID_NOTATION", f"invalid Chinese numeral: {char!r}", ply)
    return result


def _front_to_back(positions: list[int], side: Side) -> list[int]:
    return sorted(positions, reverse=(side is Side.BLACK))


def _origin_matches(board: tuple[int, ...], side: Side, src: int, source_symbol: str,
                    descriptor: str, ply: int) -> bool:
    kind = _PIECES.get(source_symbol)
    if kind is None or board[src] != kind * int(side):
        return False
    src_file = _file(src % 9, side)
    if descriptor not in _PREFIX:
        return src_file == _number(descriptor, ply)
    same_file = [index for index, piece in enumerate(board)
                 if piece == kind * int(side) and index % 9 == src % 9]
    if len(same_file) < 2 or len(same_file) > 3:
        return False
    sorted_places = _front_to_back(same_file, side)
    desired = _PREFIX[descriptor]
    if desired == "front":
        return src == sorted_places[0]
    if desired == "back":
        return src == sorted_places[-1]
    return len(sorted_places) == 3 and src == sorted_places[1]


def resolve_move(token: str, state: GameState, legal: list[Move], ply: int) -> Move:
    token = unicodedata.normalize("NFKC", token.strip())
    if len(token) != 4:
        raise SilverValidationError("INVALID_NOTATION", f"unsupported notation: {token!r}", ply)
    if token[0] in _PREFIX:
        descriptor, piece_symbol = token[0], token[1]
    else:
        piece_symbol, descriptor = token[0], token[1]
    if piece_symbol not in _PIECES:
        raise SilverValidationError("INVALID_NOTATION", f"unknown piece: {piece_symbol!r}", ply)
    direction = _DIRECTION.get(token[2])
    if direction is None:
        raise SilverValidationError("INVALID_NOTATION", f"unknown direction: {token[2]!r}", ply)
    number = _number(token[3], ply)
    side = state.side_to_move
    board = state.board
    candidates: list[Move] = []
    for move in legal:
        if not _origin_matches(board, side, move.source, piece_symbol, descriptor, ply):
            continue
        src_row, src_col = divmod(move.source, 9)
        dst_row, dst_col = divmod(move.target, 9)
        if direction == "level":
            if dst_row != src_row or _file(dst_col, side) != number:
                continue
        else:
            advancing = dst_row < src_row if side is Side.RED else dst_row > src_row
            if direction == "forward" and not advancing:
                continue
            if direction == "back" and (advancing or dst_row == src_row):
                continue
            kind = abs(board[move.source])
            measure = _file(dst_col, side) if kind in (2, 3, 4) else abs(dst_row - src_row)
            if measure != number:
                continue
        candidates.append(move)
    if not candidates:
        raise SilverValidationError("ILLEGAL_NOTATION", f"no legal move matches {token!r}", ply)
    if len(candidates) != 1:
        raise SilverValidationError("AMBIGUOUS_NOTATION", f"multiple legal moves match {token!r}", ply)
    return candidates[0]
