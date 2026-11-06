"""Strict decoding of raw CCPD PGN without changing Bronze bytes."""

from ..errors import SilverValidationError


def decode_pgn(payload: bytes) -> tuple[str, str]:
    """Return Unicode PGN and codec. CCPD master records commonly use Big5."""
    for encoding in ("utf-8-sig", "big5", "cp950"):
        try:
            return payload.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise SilverValidationError("INVALID_ENCODING", "CCPD PGN cannot be decoded as UTF-8/Big5/CP950")
