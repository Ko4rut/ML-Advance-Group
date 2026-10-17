from math import exp, isfinite
from annealing_schedules import temperature_linear, temperature_exponential
import time
from helper import conflicts
import numpy as np


def simulated_annealing(
    n,
    board=None,
    initial_temperature=0.4,
    max_iterations=1000,
    min_temperature=1e-8,
    eta=None,
    alpha=None,
    mode=0,
    seed=None,
):
    """Minimize queen conflicts with one random move per iteration.

    mode=0 uses linear cooling; mode=1 uses exponential cooling.
    Return the best board and the current/best score after every proposal.
    A supplied seed uses a local RNG; None uses NumPy's shared RNG.
    Stop at a solution, the temperature floor, or the iteration budget.
    """

    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n phải lớn hơn hoặc bằng 2.")

    if not isinstance(max_iterations, (int, np.integer)) or max_iterations <= 0:
        raise ValueError("max_iterations phải là một số nguyên dương.")

    if (not isfinite(initial_temperature) or not isfinite(min_temperature)
            or initial_temperature <= min_temperature or min_temperature <= 0):
        raise ValueError("Cần initial_temperature > min_temperature > 0.")

    if mode not in (0, 1):
        raise ValueError("mode must be 0 (linear) or 1 (exponential)")
    if mode == 0 and eta is not None and (not isfinite(eta) or eta <= 0):
        raise ValueError("eta must be positive and finite")
    if mode == 1 and alpha is not None and (not isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random if seed is None else np.random.RandomState(seed)
    start_timer = time.perf_counter()
    if board is None:
        board = rng.randint(0, n, size=n)
    else:
        board = np.asarray(board)
        if (board.shape != (n,) or not np.issubdtype(board.dtype, np.integer)
                or np.any(board < 0) or np.any(board >= n)):
            raise ValueError("board must contain n integer row positions in [0, n)")
        board = board.astype(int, copy=True)

    initial_board = board.copy()

    if mode == 0:
        schedule = "linear"

        if eta is None:
            eta = (initial_temperature - min_temperature) / max_iterations
        alpha = None

    elif mode == 1:
        schedule = "exponential"

        if alpha is None:
            alpha = (min_temperature / initial_temperature) ** (1 / max_iterations)
        eta = None
    

    def generating_neighboring_board(current_board):
        column = int(rng.randint(0, n))
        old_row = int(current_board[column])
        new_row = int(rng.randint(0, n - 1))

        # Ánh xạ [0, n-2] sang một hàng khác old_row mà không cần while.
        if new_row >= old_row:
            new_row += 1

        neighboring_board = current_board.copy()
        neighboring_board[column] = new_row
        return neighboring_board

    history = []
    current_conflicts = conflicts(board)
    best_board = board.copy()
    best_conflicts = current_conflicts
    temperature = initial_temperature
    iterations_completed = 0

    history.append(
        {
            "iteration": 0,
            "conflicts": current_conflicts,
            "best_conflicts": best_conflicts,
            "temperature": temperature,
            "accepted": None,
        }
    )

    stopped_reason = "max_iterations"
    for iteration in range(1, max_iterations + 1):
        if best_conflicts == 0:
            break
        if temperature <= min_temperature:
            stopped_reason = "min_temperature"
            break

        accepted = False
        neighboring_board = generating_neighboring_board(board)
        neighboring_conflicts = conflicts(neighboring_board)
        
        delta = neighboring_conflicts - current_conflicts

        if delta <= 0:
            accepted = True
        else:
            acceptance_probability = exp(-delta / temperature)
            accepted = rng.random() < acceptance_probability

        if accepted:
            board = neighboring_board
            current_conflicts = neighboring_conflicts

            if current_conflicts < best_conflicts:
                best_board = board.copy()
                best_conflicts = current_conflicts

        history.append(
            {
                "iteration": iteration,
                "conflicts": current_conflicts,
                "best_conflicts": best_conflicts,
                "temperature": temperature,
                "accepted": accepted,
            }
        )

        iterations_completed = iteration
        temperature = max(
            min_temperature,
            temperature_linear(iteration, initial_temperature, eta)
            if mode == 0 else temperature_exponential(iteration, initial_temperature, alpha),
        )

    end_timer = time.perf_counter()
    runtime_ms = (end_timer - start_timer) * 1000

    result = {
        "board_size": n,
        "initial_board": initial_board.tolist(),
        "best_board": best_board.tolist(),
        "best_conflicts": best_conflicts,
        "iterations": iterations_completed,
        "history": history,
        "runtime_ms": runtime_ms,
        "seed": seed,
        "stopped_reason": "solution" if best_conflicts == 0 else stopped_reason,
        "schedule": schedule,
        "initial_temperature": initial_temperature,
        "min_temperature": min_temperature,
        "eta": eta,
        "alpha": alpha,
        "max_iterations": max_iterations,
    }

    return result