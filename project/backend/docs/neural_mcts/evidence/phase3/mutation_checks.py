"""Mutation checks: prove the Phase 3 tests can fail.

Copies src/, tests/, scripts/, experiments/ into a temp dir (checkpoints symlinked, nothing in the repo is
edited), applies ONE deliberate bug at a time and runs tests/networks with torch and the real
weights REQUIRED. Every mutation must make at least one test fail; the baseline must pass.

    PYTHONPATH=<torch deps> python docs/neural_mcts/evidence/phase3/mutation_checks.py
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

BACKEND = Path(__file__).resolve().parents[4]

MUTATIONS = [
    ("unsafe pickle loading", "src/networks/pretrained/torch_backend.py", "map_location=\"cpu\", weights_only=True)", "map_location=\"cpu\", weights_only=False)"),
    ("SHA-256 comparison disabled", "src/networks/pretrained/manifest.py",
     "not hmac.compare_digest(digest, spec.sha256)", "False"),
    ("size check disabled", "src/networks/pretrained/manifest.py", "if size != spec.bytes:", "if False:"),
    ("value sign flipped for Black", "src/networks/pretrained/adapter.py",
     "return PolicyValueResult(MappingProxyType(priors), value)",
     "return PolicyValueResult(MappingProxyType(priors), -value if state.side_to_move < 0 else value)"),
    ("terminal shortcut removed", "src/networks/pretrained/adapter.py", "if result is not None:  # engine", "if False:  # engine"),
    ("uniform-prior fallback on bad output", "src/networks/pretrained/adapter.py",
     'raise InvalidModelOutputError(f"invalid policy output: {exc}") from exc',
     "priors = {m: 1.0 / len(moves) for m in moves}"),
    ("value range check removed", "src/networks/pretrained/adapter.py", "or not -1.0 <= value <= 1.0", ""),
    ("model left in training mode", "src/networks/pretrained/torch_backend.py", "        self.model.eval()\n", ""),
    ("non-strict state-dict load", "src/networks/pretrained/torch_backend.py", 'payload["model_state"], strict=True)', 'payload["model_state"], strict=False)'),
    ("architecture check removed", "src/networks/pretrained/torch_backend.py",
     'if payload["model_config"] != expected:', "if False:"),
    ("SHA env assertion ignored", "src/networks/pretrained/manifest.py", "if asserted != spec.sha256:", "if False:"),
    ("non-finite weights accepted", "src/networks/pretrained/torch_backend.py",
     "not bool(torch.isfinite(tensor).all())", "False"),
    ("device restriction lifted", "src/networks/pretrained/manifest.py", 'SUPPORTED_DEVICES = ("cpu",)',
     'SUPPORTED_DEVICES = ("cpu", "cuda", "cuda:0", "mps", "gpu")'),
    ("illegal moves not masked (priors over all indices of legal list reversed)",
     "src/networks/pretrained/adapter.py", "legal_priors(logits, moves,", "legal_priors(logits, list(reversed(moves)),"),
]


def run(root: Path) -> tuple[int, str]:
    env = dict(os.environ, XIANGQI_NEURAL_REQUIRE_REAL="1", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests/networks", "-t", "."],
                          cwd=root, env=env, capture_output=True, text=True)
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
        print(f"baseline: {'PASS' if code == 0 else 'FAIL'} ({tail})")
        ok &= code == 0
        for label, relative, old, new in MUTATIONS:
            path = Path(tmp) / relative
            pristine = path.read_text()
            if old not in pristine:
                print(f"!! pattern not found for {label!r}")
                ok = False
                continue
            path.write_text(pristine.replace(old, new, 1))
            try:
                code, tail = run(Path(tmp))
            finally:
                path.write_text(pristine)
            detected = code != 0
            ok &= detected
            print(f"{'DETECTED ' if detected else 'MISSED   '} {label}  [{tail}]")
    print("ALL MUTATIONS DETECTED" if ok else "SOME CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
