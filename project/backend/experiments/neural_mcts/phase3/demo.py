"""Smoke test / demo of the pretrained inference adapter (no MCTS).

    python experiments/neural_mcts/phase3/demo.py                 # env-configured model
    python experiments/neural_mcts/phase3/demo.py --model distill-v4 --plies 30

Needs torch (pip install '.[neural]') and the provisioned checkpoint. Prints the model
identity, top priors and value for a few states, shows engine-terminal precedence, then
plays ``--plies`` moves of argmax-policy self-play (a legality/plumbing check, NOT a
strength measurement). Exits non-zero on any failure.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

BACKEND = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND))

from src.environment import GameEngine, GameState, Move, Side  # noqa: E402
from src.networks.pretrained import NeuralModelError, NeuralModelSettings, load_adapter  # noqa: E402


def ucci(move: Move) -> str:
    """UCCI-style coordinates: files a-i left to right, ranks 0-9 from Red's back rank."""
    def square(index):
        return f"{'abcdefghi'[index % 9]}{9 - index // 9}"
    return square(move.source) + square(move.target)


def show(title, adapter, state):
    result = adapter.predict(state)
    if result.is_terminal:
        print(f"{title}: TERMINAL ({result.terminal.reason}) value={result.value:+.0f} "
              f"priors={len(result.priors)} (network not called)")
        return result
    top = sorted(result.priors.items(), key=lambda item: -item[1])[:5]
    moves = ", ".join(f"{ucci(m)} {p:.3f}" for m, p in top)
    print(f"{title}: {state.side_to_move.name} to move, {len(result.priors)} legal, "
          f"value={result.value:+.4f}, top: {moves}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="manifest key; default from env / manifest")
    parser.add_argument("--plies", type=int, default=20, help="argmax self-play plies (0 to skip)")
    args = parser.parse_args()
    settings = NeuralModelSettings.from_env()
    if args.model:
        settings = replace(settings, model=args.model)
    try:
        started = time.perf_counter()
        adapter = load_adapter(settings)
        print(f"loaded in {time.perf_counter() - started:.2f} s: {adapter.info.name} "
              f"sha256={adapter.info.sha256[:16]}… params={adapter.info.parameter_count} device={adapter.info.device}")
    except NeuralModelError as exc:
        print(f"CANNOT LOAD ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2

    engine = GameEngine()
    initial = GameState.initial()
    show("initial", adapter, initial)
    show("initial, Black to move (artificial)", adapter, GameState(initial.board, Side.BLACK))
    board = [0] * 90
    for square, piece in {4: -1, 85: 1, 9: 5, 10: 5, 22: 7}.items():
        board[square] = piece
    mate = GameState(tuple(board), Side.RED)
    show("mate-in-one (Red)", adapter, mate)
    winning = next(m for m in engine.legal_moves(mate) if (r := engine.outcome(mate.moved(m))) and r.winner is Side.RED)
    show(f"after {ucci(winning)} (checkmate)", adapter, mate.moved(winning))

    state, calls = initial, adapter.inference_calls
    for ply in range(args.plies):
        result = adapter.predict(state)
        if result.is_terminal:
            print(f"self-play ended at ply {ply}: {result.terminal.reason}")
            break
        move = max(result.priors, key=result.priors.get)
        assert move in set(engine.legal_moves(state)), "adapter produced an illegal move"
        state = state.moved(move)
    else:
        if args.plies:
            print(f"self-play: {args.plies} plies, all moves legal, {adapter.inference_calls - calls} network calls")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
