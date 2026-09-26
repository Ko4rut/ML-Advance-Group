"""Board, side to move and rule-related history."""


class GameState:
    def __init__(self, board, side_to_move, ply_count=0, no_capture_count=0, history=None):
        pass

    @classmethod
    def initial(cls):
        pass

    def clone(self):
        pass

    def position_key(self):
        pass
