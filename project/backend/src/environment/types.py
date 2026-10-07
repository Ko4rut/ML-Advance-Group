'''Shared game data types for the Xiangqi environment.'''

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Side(IntEnum):
    '''Player side. Red moves first.'''

    BLACK = -1
    RED = 1

    @property
    def opponent(self) -> "Side":
        return Side.RED if self is Side.BLACK else Side.BLACK


class Piece(IntEnum):
    '''Piece kind independent of side.'''

    KING = 1
    ADVISOR = 2
    ELEPHANT = 3
    HORSE = 4
    ROOK = 5
    CANNON = 6
    PAWN = 7


@dataclass(frozen=True, slots=True)
class Move:
    '''A move represented by source and target square indices.'''

    source: int
    target: int

    def __post_init__(self) -> None:
        for name, square in (("source", self.source), ("target", self.target)):
            if not isinstance(square, int) or isinstance(square, bool) or not 0 <= square < 90:
                raise ValueError(f"{name} must be an integer square in [0, 90)")


@dataclass(frozen=True, slots=True)
class GameResult:
    '''Terminal game outcome.'''

    winner: Optional[Side]
    reason: str

    def reward(self, side: Side) -> int:
        '''Return reward from the perspective of the supplied side.'''
        side = Side(side)
        if self.winner is None:
            return 0
        return 1 if self.winner is side else -1
