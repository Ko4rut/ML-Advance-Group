"""A conservative parser for CCPD Chinese-notation PGN records."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.environment import GameState

from ..errors import SilverValidationError

_TAG = re.compile(r'^\[([a-zA-Z][a-zA-Z0-9_]*)\s+"((?:[^"\\]|\\.)*)"\]\s*$')
_FEN_CODES = {"k": 1, "a": 2, "b": 3, "e": 3, "n": 4, "h": 4, "r": 5, "c": 6, "p": 7}
_RESULTS = {"1-0": "red", "0-1": "black", "1/2-1/2": None, "*": None}
_MOVE_NO = re.compile(r"^\d+\.(?:\.\.)?")


@dataclass(frozen=True, slots=True)
class CCPDRecord:
    headers: dict[str, str]
    initial_board: list[int]
    initial_side: str
    move_tokens: list[str]
    result: str


def parse_fen(fen: str) -> tuple[list[int], str]:
    parts = fen.strip().split()
    if len(parts) < 2 or parts[1] not in ("w", "b"):
        raise SilverValidationError("INVALID_FEN", "FEN must declare board and side 'w' or 'b'")
    rows = parts[0].split("/")
    if len(rows) != 10:
        raise SilverValidationError("INVALID_FEN", "FEN must have exactly 10 ranks")
    board: list[int] = []
    for rank in rows:
        rank_pieces: list[int] = []
        for token in rank:
            if token.isascii() and token in "123456789":
                rank_pieces.extend([0] * int(token))
            elif token.lower() in _FEN_CODES:
                kind = _FEN_CODES[token.lower()]
                rank_pieces.append(kind if token.isupper() else -kind)
            else:
                raise SilverValidationError("INVALID_FEN", f"unsupported FEN piece: {token!r}")
        if len(rank_pieces) != 9:
            raise SilverValidationError("INVALID_FEN", "each FEN rank must have 9 columns")
        board.extend(rank_pieces)
    if board.count(1) != 1 or board.count(-1) != 1:
        raise SilverValidationError("INVALID_FEN", "FEN must contain exactly two generals")
    return board, "red" if parts[1] == "w" else "black"


def _strip_annotations(text: str) -> str:
    """Discard comments and side variations without dropping mainline moves."""
    out: list[str] = []
    brace = 0
    variation = 0
    semicolon = False
    for char in text:
        if semicolon:
            if char == "\n":
                semicolon = False
                out.append(" ")
            continue
        if brace:
            if char == "{":
                brace += 1
            elif char == "}":
                brace -= 1
            continue
        if char == "{":
            brace = 1
            continue
        if char == ";":
            semicolon = True
            continue
        if char == "(":
            variation += 1
            continue
        if char == ")":
            if variation == 0:
                raise SilverValidationError("INVALID_PGN", "unbalanced PGN variation")
            variation -= 1
            continue
        if not variation:
            out.append(char)
    if brace or variation:
        raise SilverValidationError("INVALID_PGN", "unclosed PGN comment/variation")
    return "".join(out)


def parse_pgn(text: str) -> CCPDRecord:
    headers: dict[str, str] = {}
    movelines: list[str] = []
    in_movetext = False
    for line in text.lstrip("\ufeff").splitlines():
        stripped = line.strip()
        if not in_movetext and stripped.startswith("["):
            found = _TAG.fullmatch(stripped)
            if not found:
                raise SilverValidationError("INVALID_PGN", f"malformed PGN tag: {stripped[:70]}")
            key, val = found.groups()
            headers[key] = val.replace(r'\"', '"').replace(r"\\", "\\")
        elif stripped:
            in_movetext = True
            movelines.append(stripped)

    if not movelines:
        raise SilverValidationError("EMPTY_GAME", "PGN has no move text")
    if "FEN" in headers:
        board, side = parse_fen(headers["FEN"])
    else:
        starting = GameState.initial()
        board, side = list(starting.board), "red"

    tokens = _strip_annotations("\n".join(movelines)).split()
    moves: list[str] = []
    trailing_result: str | None = None
    for element in tokens:
        while True:
            numbered = _MOVE_NO.match(element)
            if numbered is None:
                break
            element = element[numbered.end():]
        if not element or element.startswith("$") or element == "...":
            continue
        if element in _RESULTS:
            trailing_result = element
            continue
        if trailing_result is not None:
            raise SilverValidationError("INVALID_PGN", "move appears after result marker")
        token = element.rstrip("!?+#")
        if not token or len(token) not in (4, 5):
            raise SilverValidationError("INVALID_NOTATION", f"unrecognized move token: {element!r}", len(moves) + 1)
        moves.append(token)
    if not moves:
        raise SilverValidationError("EMPTY_GAME", "PGN contains no moves")
    result = headers.get("Result", trailing_result or "*")
    if result not in _RESULTS:
        raise SilverValidationError("INVALID_RESULT", f"invalid PGN result: {result!r}")
    if trailing_result is not None and trailing_result != result:
        raise SilverValidationError("RESULT_MISMATCH", "PGN header and movetext results disagree")
    return CCPDRecord(headers=headers, initial_board=board, initial_side=side, move_tokens=moves, result=result)
