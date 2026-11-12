"""Pure Monte Carlo Search for the Connect 4 assignment (Task 5).

The idea follows ``tictactoe_pure_monte_carlo_search.ipynb``: for every legal column,
play a number of random games to the end (a "playout") and pick the column with the
largest average utility. No evaluation function is needed, only the rules of the game.

The differences to the notebook are the same as in ``connect4_minimax.py``: players are
``1`` / ``-1`` (the opponent of ``player`` is ``-player``) and the game comes from
``connect4_core.py``. The playout uses ``Position`` instead of ``result`` / ``utility``,
because ``Position`` is checked once and then every ``drop`` is cheap.
"""

from __future__ import annotations

import time

import numpy as np

from connect4_core import Position, actions


def playout(state, action, player = 1):
    """Perform a random playout starting with the given action on the given board
    and return the utility of the finished game (+1 win, 0 draw, -1 loss for player).

    The input board is not changed. The random policy picks uniformly among the columns
    that are not full, which is what random_player does too; Position is used because
    random_player / result check the whole board again at every move and are much slower.
    """
    position = Position.from_board(state).drop(player, action)
    current_player = -player

    while True:
        # reached terminal state?
        u = position.utility(player)
        if u is not None: return u

        # we use a random playout policy
        a = np.random.choice(position.legal_columns())
        position = position.drop(current_player, a)

        # switch between players
        current_player = -current_player


def playouts(board, action, player = 1, N = 100):
    """Perform N playouts following the given action for the given board."""
    return [ playout(board, action, player) for i in range(N) ]


def pmcs_values(board, N = 100, player = 1, verbose = 0):
    """Average utility of every legal action. The N playouts are evenly divided between the actions.

    Different from the tic-tac-toe notebook: every action gets at least one playout
    (n = max(1, N // number of actions)), because N < number of actions would give n = 0 and
    the mean of an empty list. On a 6x7 board with 7 columns, N = 10 gives n = 1.
    verbose: 0 ... silent, 1 ... print the playouts per action and the average utility of each action.
    """
    acts = actions(board)
    if not acts:
        raise ValueError("pmcs cannot move from a terminal state")

    n = max(1, N // len(acts))
    ps = { i : float(np.mean(playouts(board, i, player, N = n))) for i in acts }

    if verbose >= 1:
        print(f"Actions: {acts} ({N} total playouts = {n} playouts per action)")
        print({ i : round(v, 3) for i, v in ps.items() })
    return ps


def pmcs(board, N = 100, player = 1, verbose = 0):
    """Pure Monte Carlo Search. Returns the action that has the largest average utility.
    The N playouts are evenly divided between the possible actions (at least one per action).
    Equal averages: the column on the left wins, like max(ps, key = ps.get) in the notebook."""
    ps = pmcs_values(board, N, player, verbose)
    return max(ps, key=ps.get)


class PMCSAgent:
    """agent that plays the column found by pure Monte Carlo search; it keeps the time of its moves."""

    def __init__(self, N = 100, name = None, verbose = 0):
        self.N = N
        self.name = name or f"PMCS ({N})"
        self.verbose = verbose
        self.move_times = []   # seconds for every move (so len(move_times) is the number of moves)

    def act(self, board, player = 1):
        """return the best column for player."""
        start = time.perf_counter()
        move = int(pmcs(board, N = self.N, player = player, verbose = self.verbose))
        self.move_times.append(time.perf_counter() - start)
        return move
