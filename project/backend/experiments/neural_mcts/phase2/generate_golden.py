"""Generate golden encoding/legal-move fixtures from the *upstream* cc-lite code.

The expected values stored in the fixture come from the pinned, SHA-256-verified
upstream encoder (`encode_board`) and codec (`MoveCodec.legal_indices`), not from
this repository's production encoder. CI tests then compare production code to
that independent record without needing torch or the upstream source.

Run from project/backend (Phase 1 runtime/source layout, see phase1/README.md):

    PYTHONPATH="$W/deps:$PWD:$PWD/experiments/neural_mcts/phase1" \
      python experiments/neural_mcts/phase2/generate_golden.py \
        --source "$W/source" --output tests/encoding/golden/cclite_v1.json
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np

from contracts import color_swap, fen, fixtures
from fetch_artifacts import MANIFEST, verify
from src.environment import GameEngine, GameState, Side

# Piece values: Red K1 A2 E3 H4 R5 C6 P7, Black negative. Coordinates are (row, col),
# row 0 = Black's back rank.
RED_K, RED_A, RED_E, RED_H, RED_R, RED_C, RED_P = 1, 2, 3, 4, 5, 6, 7


def position(pieces, side=Side.RED):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side)


def special_positions() -> dict[str, GameState]:
    bk, rk = {(0, 3): -1}, {(9, 5): 1}  # kings on different files (no facing generals)
    cases = {
        "cannon_capture_over_screen": {**bk, **rk, (7, 1): RED_C, (4, 1): RED_P, (1, 1): -RED_R},
        "cannon_no_screen_cannot_capture": {**bk, **rk, (5, 4): RED_C, (2, 4): -RED_R},
        "cannon_two_screens_blocks": {**bk, **rk, (7, 1): RED_C, (5, 1): RED_P, (4, 1): -RED_P, (1, 1): -RED_R},
        "horse_leg_blocked_north": {**bk, **rk, (5, 4): RED_H, (4, 4): RED_P},
        "horse_all_eight_free": {**bk, **rk, (5, 4): RED_H},
        "horse_leg_blocked_by_enemy": {**bk, **rk, (5, 4): RED_H, (5, 5): -RED_P, (6, 4): -RED_P},
        "elephant_eye_blocked": {**bk, **rk, (7, 2): RED_E, (6, 3): RED_P, (8, 3): RED_P},
        "elephant_cannot_cross_river": {**bk, **rk, (5, 2): RED_E},
        "advisor_and_king_palace": {(0, 5): -1, (8, 4): RED_A, (9, 3): RED_K},  # kings NOT facing: a facing-generals position is unreachable and upstream differs there
        "flying_general_pinned_rook": {(0, 4): -1, (9, 4): 1, (5, 4): RED_R, (5, 0): -RED_R},
        "flying_general_king_step": {(0, 3): -1, (9, 4): 1},
        "pawn_before_river_red": {**bk, **rk, (6, 4): RED_P},
        "pawn_after_river_red": {**bk, **rk, (4, 4): RED_P},
        "pawn_last_rank": {**bk, **rk, (0, 6): RED_P},
        "pawn_before_river_black": {(0, 5): -1, (9, 3): 1, (3, 4): -RED_P},
        "pawn_after_river_black": {(0, 5): -1, (9, 3): 1, (5, 4): -RED_P},
        "rook_own_and_enemy_blockers": {**bk, **rk, (5, 4): RED_R, (5, 7): RED_P, (5, 1): -RED_P, (2, 4): -RED_P},
        "check_by_cannon_black_to_move": {(0, 4): -1, (9, 3): 1, (7, 4): RED_C, (3, 4): -RED_P},
        "check_by_horse_leg_free": {(0, 4): -1, (9, 3): 1, (2, 3): RED_H, (0, 3): 0},
        "check_by_rook_two_replies": {(0, 4): -1, (9, 3): 1, (0, 8): RED_R, (1, 3): -RED_A, (1, 5): -RED_A},
        "stalemate_black": {(0, 3): -1, (9, 5): 1, (1, 4): RED_R},
        "advisor_capture_and_blocked": {**bk, **rk, (8, 4): RED_A, (7, 3): -RED_P, (9, 3): RED_P},
        "king_captures_adjacent": {**bk, (9, 4): 1, (8, 4): -RED_P},
        "cannon_cannot_capture_adjacent": {**bk, **rk, (5, 4): RED_C, (4, 4): -RED_P},
    }
    states = {}
    for name, pieces in cases.items():
        side = Side.BLACK if "black_to_move" in name or name.endswith("_black") else Side.RED
        state = position(pieces, side)
        states[f"special/{name}"] = state
        states[f"special/{name}/color_swapped"] = color_swap(state)
    return states


def random_walks(engine, seeds=range(1, 7), plies=80):
    states = {}
    for seed in seeds:
        rng, state = random.Random(seed), GameState.initial()
        for ply in range(plies):
            states.setdefault(f"walk{seed}/{ply}", state)
            moves = engine.legal_moves(state)
            if engine.outcome(state, legal_moves=moves) is not None:
                break
            state = state.moved(rng.choice(moves))
    return states


def build_states(engine):
    states = {name: s for name, s in fixtures().items() if name not in ("no_capture_draw", "repetition_draw")}
    states = {f"fixture/{name}": s for name, s in states.items()}
    states.update(special_positions())
    states.update(random_walks(engine))
    unique, seen = {}, set()
    for name, state in states.items():
        key = state.position_key()
        if key not in seen:
            seen.add(key)
            unique[name] = state
    return unique


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    for relative, record in manifest["files"].items():
        verify(args.source / relative, record)
    sys.path.insert(0, str(args.source / "src"))
    native_board = importlib.import_module("xiangqi.board")
    native_encoding = importlib.import_module("xiangqi.encoding")
    engine, records, mismatches = GameEngine(), [], []
    for name, state in build_states(engine).items():
        native = native_board.Board.from_fen(fen(state))
        planes = np.ascontiguousarray(native_encoding.encode_board(native), dtype=np.float32)
        upstream_ids = sorted(int(i) for i in native_encoding.MoveCodec.legal_indices(native))
        engine_ids = sorted(m.source * 90 + m.target for m in engine.legal_moves(state))
        if upstream_ids != engine_ids:
            mismatches.append({"name": name, "fen": fen(state),
                               "only_upstream": sorted(set(upstream_ids) - set(engine_ids)),
                               "only_engine": sorted(set(engine_ids) - set(upstream_ids))})
        records.append({
            "name": name, "side": state.side_to_move.name, "board": list(state.board),
            "upstream_observation_sha256": hashlib.sha256(planes.tobytes()).hexdigest(),
            "upstream_side_in_check": bool(planes[15, 0, 0]),
            "upstream_legal_action_ids": upstream_ids,
        })
    print(f"{len(records)} states; {sum(len(r['upstream_legal_action_ids']) for r in records)} "
          f"upstream legal moves; engine/upstream legal-set mismatches: {len(mismatches)}")
    for item in mismatches:
        print(json.dumps(item))
    if args.report_only:
        return
    document = {
        "description": "Expected values come from the pinned upstream cc-lite encoder/codec, not from this repo.",
        "upstream_repository": manifest["repository"], "upstream_revision": manifest["revision"],
        "upstream_file_sha256": {k: v["sha256"] for k, v in manifest["files"].items()
                                 if k.startswith("src/")},
        "observation_layout": "float32 (16,10,9) C-order bytes, native byte order",
        "engine_legal_set_mismatches": mismatches,
        "states": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
