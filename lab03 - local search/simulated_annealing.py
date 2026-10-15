from math import exp
from annealing_schedules import temperature_linear, temperature_exponential
import time
from helper import random_board, comb2, conflicts, show_board, _safe_file_part, create_experiment_id, _build_experiment_summary, save_log_result, summarize_experiment, RANDOM_SEED
import numpy as np


def simulated_annealing(
    n,
    board=None,
    initial_temperature=0.4,
    max_iterations=1000,
    min_temperature=1e-8,
    eta=None,
    alpha=None,
    mode=0
):
    """Run Simulated Annealing and return data without printing or saving files."""

    if n < 2:
        raise ValueError("n phải lớn hơn hoặc bằng 2.")

    if max_iterations <= 0:
        raise ValueError("max_iterations phải là một số nguyên dương.")

    if initial_temperature <= min_temperature or min_temperature <= 0:
        raise ValueError("Cần initial_temperature > min_temperature > 0.")

    start_timer = time.perf_counter()
    board = random_board(n)

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
        column = int(np.random.randint(0, n))
        old_row = int(current_board[column])
        new_row = int(np.random.randint(0, n - 1))

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

    for iteration in range(1, max_iterations + 1):
        if best_conflicts == 0:
            break

        accepted = False
        neighboring_board = generating_neighboring_board(board)
        neighboring_conflicts = conflicts(neighboring_board)
        
        delta = neighboring_conflicts - current_conflicts

        if delta <= 0:
            accepted = True
        else:
            acceptance_probability = exp(-delta / temperature)
            accepted = np.random.random() < acceptance_probability

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
        temperature = temperature_linear(iteration, initial_temperature, eta) if mode == 0 else temperature_exponential(iteration, initial_temperature, alpha)

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
        "seed": RANDOM_SEED,
        "schedule": schedule,
        "initial_temperature": initial_temperature,
        "min_temperature": min_temperature,
        "eta": eta,
        "alpha": alpha,
        "max_iterations": max_iterations,
    }

    return result