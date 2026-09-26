"""Shared game data types."""


class Side:
    pass

class Piece:
    pass

class Move:
    def __init__(self, source, target):
        pass

class GameResult:
    def __init__(self, winner, reason):
        pass

    def reward(self, side):
        pass
