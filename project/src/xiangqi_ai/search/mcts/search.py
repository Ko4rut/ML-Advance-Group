"""Reusable MCTS selection, expansion, evaluation and backup."""


class MCTSSearch:
    def __init__(self, engine, selection, evaluator):
        pass

    def search(self, state, budget=None):
        pass

    def select(self, node):
        pass

    def expand(self, node):
        pass

    def evaluate(self, node):
        pass

    def backpropagate(self, path, value):
        pass

    def visit_policy(self, root, temperature=1.0):
        pass
