"""Minimax search with alpha-beta pruning and move ordering for the Connect 4 assignment.

The structure follows ``tictactoe_alpha_beta_tree_search.ipynb``. The only differences are
that players are ``1`` / ``-1`` (the opponent of ``player`` is ``-player``) and that the
game functions come from ``connect4_core.py``.
"""

from __future__ import annotations

import math
import time

import numpy as np

from connect4_core import actions, empty_board, random_player, result, terminal, utility

# global variables
DEBUG = 1  # 0 ... silent, 1 ... count nodes, 2 ... debug each node (same meaning as tic-tac-toe notebook)
COUNT = 0


class SearchLimitReached(Exception):
    """raised when a search visits more than node_limit nodes."""


# ---- move ordering ----
# A move order function gets the state, the player who moves in that state and the list
# of legal columns. It returns the same columns in the order they should be searched.

def no_ordering(board, player, moves):
    """keep the order of actions(): left to right."""
    return list(moves)


def center_first(board, player, moves):
    """columns closest to the middle first; equal distance keeps the left-to-right order."""
    middle = (board.shape[1] - 1) / 2
    return sorted(moves, key=lambda column: abs(column - middle))


def win_block_center(board, player, moves):
    """winning moves first, then moves that block an immediate win of the opponent, then center first."""
    middle = (board.shape[1] - 1) / 2

    def rank(column):
        if utility(result(board, player, column), player) == 1:
            return 0  # wins immediately
        if utility(result(board, -player, column), -player) == 1:
            return 1  # the opponent would win here, so block it
        return 2

    return sorted(moves, key=lambda column: (rank(column), abs(column - middle)))


# ---- minimax search ----

def minimax_search(board, player=1, node_limit=None):
    """start the search.

    Returns {"move", "value", "nodes"}. The number of nodes is returned in the dict because
    ``from connect4_minimax import COUNT`` would only give a copy that is never updated.
    """
    global DEBUG, COUNT
    COUNT = 0

    value, move = max_value(board, player, node_limit)
    nodes = COUNT

    if DEBUG >= 1: print(f"Number of nodes searched: {nodes}")

    return {"move": move, "value": value, "nodes": nodes}


def max_value(state, player, node_limit=None):
    """player's best move."""
    global DEBUG, COUNT
    COUNT += 1
    if node_limit is not None and COUNT > node_limit:
        raise SearchLimitReached(f"more than {node_limit} nodes")

    # return utility of state if it is a terminal state
    v = utility(state, player)
    if DEBUG >= 2: print("max in: " + str(state.tolist()) + str([v]))
    if v is not None: return v, None

    v, move = -math.inf, None

    # check all possible actions in the state, return move with the largest value
    for a in actions(state):
        v2, a2 = min_value(result(state, player, a), player, node_limit)
        if v2 > v:
            v, move = v2, a

    if DEBUG >= 2: print("max out: " + str(state.tolist()) + str([v, move]))
    return v, move


def min_value(state, player, node_limit=None):
    """opponent's best response."""
    global DEBUG, COUNT
    COUNT += 1
    if node_limit is not None and COUNT > node_limit:
        raise SearchLimitReached(f"more than {node_limit} nodes")

    # return utility of state if it is a terminal state
    v = utility(state, player)
    if DEBUG >= 2: print("min in: " + str(state.tolist()) + str([v]))
    if v is not None: return v, None

    v, move = +math.inf, None

    # check all possible actions in the state, return move with the smallest value
    for a in actions(state):
        v2, a2 = max_value(result(state, -player, a), player, node_limit)
        if v2 < v:
            v, move = v2, a

    if DEBUG >= 2: print("min out: " + str(state.tolist()) + str([v, move]))
    return v, move


# ---- minimax search with alpha-beta pruning ----

def alpha_beta_search(board, player=1, move_order=no_ordering, node_limit=None):
    """start the search. move_order decides in which order the columns are tried.

    Returns {"move", "value", "nodes"}; see minimax_search for why "nodes" is in the dict.
    """
    global DEBUG, COUNT
    COUNT = 0

    value, move = max_value_ab(board, player, -math.inf, +math.inf, move_order, node_limit)
    nodes = COUNT

    if DEBUG >= 1: print(f"Number of nodes searched: {nodes}")

    return {"move": move, "value": value, "nodes": nodes}


def max_value_ab(state, player, alpha, beta, move_order=no_ordering, node_limit=None):
    """player's best move."""
    global DEBUG, COUNT
    COUNT += 1
    if node_limit is not None and COUNT > node_limit:
        raise SearchLimitReached(f"more than {node_limit} nodes")

    # return utility of state is a terminal state
    v = utility(state, player)
    if DEBUG >= 2: print(f"max: {state.tolist()} [alpha,beta]=[{alpha},{beta}] v={v}")
    if v is not None:
        if DEBUG >= 2: print("     found terminal state. backtracking.")
        return v, None

    v, move = -math.inf, None

    # check all possible actions in the state, update alpha and return move with the largest value
    for a in move_order(state, player, actions(state)):
        v2, a2 = min_value_ab(result(state, player, a), player, alpha, beta, move_order, node_limit)
        if DEBUG >= 2: print(f"max: {state.tolist()} (backtracked) [alpha,beta]=[{alpha},{beta}] v={v2}")

        if v2 > v:
            v, move = v2, a
            alpha = max(alpha, v)
        if v >= beta:
            if DEBUG >= 2: print(f"     v>=beta ({v}>={beta}): pruning remaining subtree (actions). backtracking.")
            return v, move

    return v, move


def min_value_ab(state, player, alpha, beta, move_order=no_ordering, node_limit=None):
    """opponent's best response."""
    global DEBUG, COUNT
    COUNT += 1
    if node_limit is not None and COUNT > node_limit:
        raise SearchLimitReached(f"more than {node_limit} nodes")

    # return utility of state is a terminal state
    v = utility(state, player)
    if DEBUG >= 2: print(f"min: {state.tolist()} [alpha,beta]=[{alpha},{beta}] v={v}")
    if v is not None:
        if DEBUG >= 2: print("     found terminal state. backtracking.")
        return v, None

    v, move = +math.inf, None

    # check all possible actions in the state, update beta and return move with the smallest value
    # (the opponent is the one who moves here, so the move order gets -player)
    for a in move_order(state, -player, actions(state)):
        v2, a2 = max_value_ab(result(state, -player, a), player, alpha, beta, move_order, node_limit)
        if DEBUG >= 2: print(f"min: {state.tolist()} (backtracked) [alpha,beta]=[{alpha},{beta}] v={v2}")

        if v2 < v:
            v, move = v2, a
            beta = min(beta, v)
        if v <= alpha:
            if DEBUG >= 2: print(f"     v<=alpha ({v}<={alpha}): pruning remaining subtree (actions). backtracking.")
            return v, move

    return v, move


# ---- helpers ----

def board_from_moves(shape, moves, first_player=1):
    """build a legal board by dropping discs into the listed columns in turn (uses result()).

    Returns (board, player_to_move)."""
    board = empty_board(shape)
    player = first_player
    for column in moves:
        board = result(board, player, column)
        player = -player
    return board, player


def timed_search(search, board, player=1, **kwargs):
    """run a search function and add "time_ms" to its result.

    The search runs silently. If it visits more than node_limit nodes, the result has
    move=None, value=None, nodes=node_limit and finished=False."""
    global DEBUG
    old_debug, DEBUG = DEBUG, 0
    start = time.perf_counter()
    try:
        info = search(board, player, **kwargs)
        info["finished"] = True
    except SearchLimitReached:
        info = {"move": None, "value": None, "nodes": kwargs.get("node_limit"), "finished": False}
    finally:
        DEBUG = old_debug
    info["time_ms"] = (time.perf_counter() - start) * 1000
    return info


# ---- agents ----

class AlphaBetaAgent:
    """agent that plays the move found by alpha-beta search; it keeps information about its searches."""

    def __init__(self, move_order=no_ordering, remember_moves=True, name="Alpha-Beta"):
        self.move_order = move_order
        self.remember_moves = remember_moves
        self.name = name
        self.known_moves = {}  # (board bytes, board shape, player) -> move
        self.move_times = []   # seconds for every search that was really done
        self.nodes = []        # number of nodes of every search that was really done

    def act(self, board, player=1):
        """return the best column for player."""
        key = (board.tobytes(), board.shape, player)
        # the move chosen at the root only depends on the board, so remembering it is exact
        if self.remember_moves and key in self.known_moves:
            return self.known_moves[key]

        start = time.perf_counter()
        info = timed_search(alpha_beta_search, board, player, move_order=self.move_order)
        self.move_times.append(time.perf_counter() - start)
        self.nodes.append(info["nodes"])

        move = int(info["move"])
        self.known_moves[key] = move
        return move


def alpha_beta_player(board, player=1):
    """simple agent function like alpha_beta_player in the tic-tac-toe notebook."""
    global DEBUG
    old_debug, DEBUG = DEBUG, 0
    try:
        return int(alpha_beta_search(board, player)["move"])
    finally:
        DEBUG = old_debug


def play(first, second, N=100, shape=(4, 4)):
    """let two agents play N games. first moves first and plays 1, second plays -1.

    Returns {"first": wins of first, "second": wins of second, "draw": draws}. This function
    is here because connect4_core.py has no play(); the group can move it there if they want."""
    results = {"first": 0, "second": 0, "draw": 0}

    for i in range(N):
        board = empty_board(shape)

        while not terminal(board):
            # first moves
            board = result(board, 1, first(board, 1))
            if terminal(board):
                break
            # second moves
            board = result(board, -1, second(board, -1))

        assert terminal(board)
        outcome = utility(board, 1)
        if outcome == 1: results["first"] += 1
        elif outcome == -1: results["second"] += 1
        else: results["draw"] += 1

    return results
