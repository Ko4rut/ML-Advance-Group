"""Interchangeable UCT and PUCT selection.

``child.mean_value()`` is already stored from the perspective of the player who
chooses among the children (see node.py), so both rules take the maximum score
without flipping any sign.
"""

import math

from .node import MCTSNode


class UCTSelection:
    """Classic UCB1 applied to trees: Q + c * sqrt(ln(N_parent) / N_child)."""

    def __init__(self, exploration_constant: float) -> None:
        self.exploration_constant = exploration_constant

    def select_child(self, node: MCTSNode) -> MCTSNode:
        if not node.children:
            raise ValueError("cannot select a child from a node without children")
        log_parent = math.log(max(node.N, 1))
        best_child: MCTSNode | None = None
        best_score = -math.inf
        # Strict ">" keeps the first child on ties (dict insertion order).
        for child in node.children.values():
            if child.N == 0:
                score = math.inf
            else:
                score = child.mean_value() + self.exploration_constant * math.sqrt(
                    log_parent / child.N
                )
            if score > best_score:
                best_child = child
                best_score = score
        return best_child


class PUCTSelection:
    """AlphaZero-style rule: Q + c * P * sqrt(N_parent) / (1 + N_child)."""

    def __init__(self, exploration_constant: float) -> None:
        self.exploration_constant = exploration_constant

    def select_child(self, node: MCTSNode) -> MCTSNode:
        if not node.children:
            raise ValueError("cannot select a child from a node without children")
        sqrt_parent = math.sqrt(node.N)
        best_child: MCTSNode | None = None
        best_score = -math.inf
        for child in node.children.values():
            score = child.mean_value() + (
                self.exploration_constant * child.prior * sqrt_parent / (1 + child.N)
            )
            if score > best_score:
                best_child = child
                best_score = score
        return best_child
