"""Heuristic Alpha-Beta search with configurable cutoff for STANDARD Connect 4.

Ported from Task 4 of assignment_connect4.ipynb on branch lab05/heuristic_cutoff
(PR #59). This module deliberately uses connect4_core's gravity-based rules;
Mean Connect 4 is a separate assignment with different transition rules.

All utility values are from the agent's perspective: -1 loss, 0 draw, +1 win.
Nonterminal heuristic scores are bounded strictly within (-1, +1).
"""

import math
import numpy as np
from connect4_core import actions, result, terminal, utility

_WIN_LENGTH = 4
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def heuristic_evaluation(board, player=1):
    """Score a board from player's perspective, always returning a value in [-1, 1]."""
    outcome = utility(board, player)
    if outcome is not None:
        return float(outcome)

    rows, columns = board.shape
    weights = {1: 1.0, 2: 6.0, 3: 30.0}
    score = 0.0

    for row in range(rows):
        for column in range(columns):
            for row_step, column_step in _DIRECTIONS:
                end_row = row + (_WIN_LENGTH - 1) * row_step
                end_column = column + (_WIN_LENGTH - 1) * column_step
                if not (0 <= end_row < rows and 0 <= end_column < columns):
                    continue

                window = [int(board[row + offset * row_step,
                                    column + offset * column_step])
                          for offset in range(_WIN_LENGTH)]
                own_count = window.count(player)
                opponent_count = window.count(-player)
                if opponent_count == 0 and own_count:
                    score += weights[own_count]
                elif own_count == 0 and opponent_count:
                    score -= weights[opponent_count]

    center = (columns - 1) / 2
    for column in range(columns):
        center_weight = 1.0 - abs(column - center) / (center + 1.0)
        score += 0.2 * center_weight * int(np.sum(board[:, column] == player))
        score -= 0.2 * center_weight * int(np.sum(board[:, column] == -player))

    return math.tanh(score / max(12.0, rows * columns * 3.0))

from numbers import Integral
from time import perf_counter


class HeuristicAlphaBetaAgent:
    """Depth-limited alpha-beta agent; call as agent(board, player)."""

    def __init__(self, depth=3):
        if isinstance(depth, bool) or not isinstance(depth, Integral) or depth < 1:
            raise ValueError("depth must be a positive integer")
        self.depth = int(depth)
        self.nodes = 0
        self.last_elapsed = 0.0

    def __call__(self, board, player=1):
        utility(board, player)  # Validate the board and player before searching.
        legal_actions = actions(board)
        if not legal_actions:
            raise ValueError("agent cannot move from a terminal state")

        self.nodes = 1
        started = perf_counter()
        center = (board.shape[1] - 1) / 2

        def ordered_children(state, to_move):
            ordered_columns = sorted(actions(state), key=lambda col: (abs(col - center), col))
            children = []
            for column in ordered_columns:
                child = result(state, to_move, column)
                children.append((heuristic_evaluation(child, player), column, child))
            maximizing = to_move == player
            children.sort(key=lambda item: item[0], reverse=maximizing)
            return children

        def search(state, to_move, remaining_depth, alpha, beta):
            self.nodes += 1
            outcome = utility(state, player)
            if outcome is not None:
                return float(outcome)
            if remaining_depth == 0:
                return heuristic_evaluation(state, player)

            maximizing = to_move == player
            best_value = -math.inf if maximizing else math.inf
            for _, _, child in ordered_children(state, to_move):
                value = search(child, -to_move, remaining_depth - 1, alpha, beta)
                if maximizing:
                    best_value = max(best_value, value)
                    alpha = max(alpha, best_value)
                else:
                    best_value = min(best_value, value)
                    beta = min(beta, best_value)
                if alpha >= beta:
                    break
            return best_value

        best_action = None
        best_value = -math.inf
        alpha, beta = -math.inf, math.inf
        for _, column, child in ordered_children(board, player):
            value = search(child, -player, self.depth - 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = column
            alpha = max(alpha, best_value)

        self.last_elapsed = perf_counter() - started
        return best_action

def play_heuristic_match(first_agent, second_agent, shape=(4, 5)):
    board = np.zeros(shape, dtype=int)
    player = 1
    plies = 0
    while not terminal(board):
        agent = first_agent if player == 1 else second_agent
        action = agent(board, player)
        board = result(board, player, action)
        plies += 1
        outcome = utility(board, player)
        if outcome is not None:
            winner = player if outcome == 1 else (-player if outcome == -1 else None)
            return {"winner": winner, "plies": plies, "board": board}
        player = -player
    return {"winner": None, "plies": plies, "board": board}
