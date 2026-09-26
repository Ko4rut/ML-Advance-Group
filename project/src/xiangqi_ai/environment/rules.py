"""Piece movement and check detection."""


class RulesConfig:
    def __init__(self, repetition_count=3, no_capture_limit=120):
        pass

class RuleValidator:
    def in_palace(self, row, col, side):
        pass

    def piece_targets(self, state, source):
        pass

    def is_in_check(self, state, side):
        pass
