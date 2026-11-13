"""Mutation checks for Phase 4: each injected bug must make tests/search fail.

Copies src/, tests/, scripts/, experiments/ and models/ (checkpoints symlinked) to a temp dir,
applies ONE bug at a time and runs ``tests/search`` (fake-evaluator, classic-regression and
real-weight tests, the latter REQUIRED via XIANGQI_NEURAL_REQUIRE_REAL=1). Nothing in the repo
is edited. Includes bugs in the CLASSIC code to show the classic regression tests bite too.

    PYTHONPATH=<torch deps> python docs/neural_mcts/evidence/phase4/mutation_checks.py
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

BACKEND = Path(__file__).resolve().parents[4]
NS = "src/search/mcts/neural_search.py"
TIMEOUT_SECONDS = 240  # a clean run takes ~45 s
START = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # resume: skip the first START mutations

MUTATIONS = [
    ("uniform priors instead of network priors", NS, "len(leaf.expand(priors))",
     "len(leaf.expand({m: 1.0 / len(legal_moves) for m in legal_moves}))"),
    ("leaf value sign flipped", NS, "return result.value, len(leaf.expand(priors))", "return -result.value, len(leaf.expand(priors))"),
    ("terminal override removed (network sees finished games)", NS,
     "if leaf.terminal_result is not None:\n            self.last_terminal_leaves += 1", "if False:\n            self.last_terminal_leaves += 1"),
    ("only one child created per expansion", NS, "len(leaf.expand(priors))",
     "len(leaf.expand(dict(list(priors.items())[:1])))"),
    ("evaluator handed an incomplete legal list", NS, "legal_moves = list(leaf.untried_moves)  # the engine's exact legal set",
     "legal_moves = list(leaf.untried_moves)[:-1]"),
    ("prior coverage check removed", NS, "if set(priors) != set(legal_moves):", "if False:"),
    ("prior normalization check removed", NS, "or not math.isclose(math.fsum(priors.values()), 1.0, abs_tol=PRIOR_SUM_TOLERANCE)", ""),
    ("value range check removed", NS, "or not math.isfinite(value) or not -1.0 <= value <= 1.0:", ":"),
    ("evaluator 'terminal' claim ignored", NS, 'if getattr(result, "is_terminal", False):', "if False:"),
    ("UCT accepted for neural search", NS, "if not isinstance(selection, PUCTSelection):", "if False:"),
    ("time limit ignored", NS, "deadline = time.monotonic() + time_limit if time_limit is not None else None", "deadline = None"),
    ("root win shortcut disabled", NS, "if self.root_win_shortcut:", "if False:"),
    ("best child ignores the prior tie-break", NS, "key=lambda c: (c.N, c.mean_value(), c.prior)", "key=lambda c: (c.N, c.mean_value())"),
    ("simulation cap ignored", NS, "if max_simulations is not None and simulations >= max_simulations:", "if False:"),
    ("agent bypasses search.best_child", "src/agents/neural_mcts_agent.py", "best = self.search.best_child(root)",
     "best = max(root.children.values(), key=lambda c: (c.N, c.mean_value()))"),
    ("adapter not actually called by NeuralEvaluator", "src/search/mcts/evaluators.py",
     "return self.adapter.predict(state, legal_moves)", "return self.adapter.predict(state, legal_moves) if False else __import__('types').SimpleNamespace(priors={m: 1.0 / len(legal_moves) for m in legal_moves}, value=0.0, is_terminal=False)"),
    # classic code must stay protected by the regression tests
    ("CLASSIC: uniform prior constant changed", "src/search/mcts/search.py",
     "return node.expand({move: 1.0 / node.num_legal})[0]", "return node.expand({move: 0.5 / node.num_legal})[0]"),
    ("CLASSIC: UCT exploration term changed", "src/search/mcts/selection.py",
     "log_parent / child.N", "2 * log_parent / child.N"),
    ("CLASSIC: backprop sign broken", "src/search/mcts/search.py", "            value = -value\n            node.update(value)",
     "            node.update(value)\n            value = -value"),
]


def run(root: Path) -> tuple[int, str]:
    env = dict(os.environ, XIANGQI_NEURAL_REQUIRE_REAL="1", PYTHONDONTWRITEBYTECODE="1")
    try:
        done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests/search", "-t", "."],
                              cwd=root, env=env, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:  # a mutation that makes the search run forever is detected by the hang
        return 1, f"TIMEOUT after {TIMEOUT_SECONDS}s (hang)"
    lines = done.stderr.strip().splitlines()
    return done.returncode, " / ".join(lines[-2:]) if lines else ""


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("src", "tests", "scripts", "experiments"):
            shutil.copytree(BACKEND / name, Path(tmp) / name, ignore=shutil.ignore_patterns("__pycache__"))
        (Path(tmp) / "models").mkdir()
        shutil.copytree(BACKEND / "models" / "neural", Path(tmp) / "models" / "neural", symlinks=True,
                        copy_function=lambda s, d: os.symlink(Path(s).resolve(), d) if s.endswith(".pt") else shutil.copy2(s, d))
        code, tail = run(Path(tmp))
        print(f"baseline: {'PASS' if code == 0 else 'FAIL'} ({tail})", flush=True)
        ok &= code == 0
        for label, relative, old, new in MUTATIONS[START:]:
            path = Path(tmp) / relative
            pristine = path.read_text()
            if old not in pristine:
                print(f"!! pattern not found for {label!r}", flush=True)
                ok = False
                continue
            path.write_text(pristine.replace(old, new, 1))
            try:
                code, tail = run(Path(tmp))
            finally:
                path.write_text(pristine)
            ok &= code != 0
            print(f"{'DETECTED ' if code else 'MISSED   '} {label}  [{tail}]", flush=True)
    print("ALL MUTATIONS DETECTED" if ok else "SOME CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
