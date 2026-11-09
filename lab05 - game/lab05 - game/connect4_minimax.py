"""Minimax search with alpha-beta pruning and move ordering for the Connect 4 assignment.

The idea follows ``tictactoe_alpha_beta_tree_search.ipynb``. The differences are that
players are ``1`` / ``-1`` (the opponent of ``player`` is ``-player``), that the game
comes from ``connect4_core.py``, and that the max and min steps of the notebook are one
function (``Search.visit``) because they only differ in which side moves.
"""

from __future__ import annotations

import math
import time

from connect4_core import Position, empty_board, result, terminal, utility


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


# ---- search ----

class Search:
    """One search run for ``player``: counts the nodes, enforces the node limit and prints the trace.

    All values are from ``player``'s view (+1 win, 0 draw, -1 loss). With ``prune=False`` it
    is plain minimax; with ``prune=True`` it is minimax with alpha-beta pruning.
    verbose: 0 ... silent, 1 ... print the number of nodes, 2 ... trace each node.
    """

    def __init__(self, player, move_order=no_ordering, node_limit=None, prune=True, verbose=0):
        self.player = player
        self.move_order = move_order
        self.node_limit = node_limit
        self.prune = prune
        self.verbose = verbose
        self.nodes = 0

    def run(self, board):
        """search from board with ``player`` to move. Returns {"move", "value", "nodes"}."""
        value, move = self.visit(Position.from_board(board), self.player, -math.inf, +math.inf)
        if self.verbose >= 1: print(f"Number of nodes searched: {self.nodes}")
        return {"move": move, "value": value, "nodes": self.nodes}

    def visit(self, position, mover, alpha, beta):
        """value and best column of a state where ``mover`` moves. ``player`` maximizes, the opponent minimizes."""
        self.nodes += 1
        if self.node_limit is not None and self.nodes > self.node_limit:
            raise SearchLimitReached(f"more than {self.node_limit} nodes")

        # return utility of state if it is a terminal state
        v = position.utility(self.player)
        if self.verbose >= 2: print(f"node: {position.board.tolist()} mover={mover} [alpha,beta]=[{alpha},{beta}] v={v}")
        if v is not None: return v, None

        maximizing = mover == self.player
        v, move = (-math.inf if maximizing else +math.inf), None

        # try all actions, keep the move with the largest (max) or smallest (min) value, update alpha / beta
        for a in self.move_order(position.board, mover, position.legal_columns()):
            v2, _ = self.visit(position.drop(mover, a), -mover, alpha, beta)
            if maximizing and v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            elif not maximizing and v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            cutoff = v >= beta if maximizing else v <= alpha
            if self.prune and cutoff:
                if self.verbose >= 2: print(f"     cutoff (v={v}, alpha={alpha}, beta={beta}): pruning remaining actions.")
                break

        return v, move


def minimax_search(board, player=1, node_limit=None, verbose=0):
    """minimax search without pruning.

    Returns {"move", "value", "nodes"}. The number of nodes is part of the result, so no
    shared counter is needed.
    """
    return Search(player, node_limit=node_limit, prune=False, verbose=verbose).run(board)


def alpha_beta_search(board, player=1, move_order=no_ordering, node_limit=None, verbose=0):
    """minimax search with alpha-beta pruning. move_order decides in which order the columns are tried.

    Returns {"move", "value", "nodes"}.
    """
    return Search(player, move_order, node_limit, prune=True, verbose=verbose).run(board)


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
    start = time.perf_counter()
    try:
        info = search(board, player, verbose=0, **kwargs)
        info["finished"] = True
    except SearchLimitReached:
        info = {"move": None, "value": None, "nodes": kwargs.get("node_limit"), "finished": False}
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

        info = timed_search(alpha_beta_search, board, player, move_order=self.move_order)
        self.move_times.append(info["time_ms"] / 1000)
        self.nodes.append(info["nodes"])

        move = int(info["move"])
        self.known_moves[key] = move
        return move


def alpha_beta_player(board, player=1):
    """simple agent function like alpha_beta_player in the tic-tac-toe notebook."""
    return int(alpha_beta_search(board, player)["move"])


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
