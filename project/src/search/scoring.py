"""Shared static evaluation for heuristic and tree search agents."""


class PositionEvaluator:
    def evaluate(self, state, perspective):
        pass

    def material(self, state, perspective):
        pass

    def mobility(self, state, perspective):
        pass

    def king_safety(self, state, perspective):
        pass
