# Phase 0 — Baseline test evidence

Date: 2026-11-12. Tested HEAD: `d5ad26067aa29b72fe9a0337e7cf4500593b4619`. Production source/tests/configs were unchanged during the initial runs documented below. **Follow-up:** the user authorized the test-helper repair; [the repaired working tree passes 270/270](00_BASELINE_REPAIR.md). Original failing logs are preserved.

## Actual runs

| Run | Result | Evidence |
| --- | --- | --- |
| Initial targeted run, existing `.venv`, backend cwd | **FAIL:** 62 reported tests, 4 errors, 21.417 s | [raw output](evidence/targeted-initial.log) |
| Corrected targeted environment | **PASS:** 78 tests, no failures/errors/skips, 23.352 s | [raw output](evidence/targeted-corrected.log) |
| Full existing backend discovery | **FAIL:** 270 tests, 269 passed, 1 error, no skips, 39.299 s | [raw output](evidence/full-suite.log) |
| Inline architecture probes | Assertions passed; classic moves legal for both sides at 1/2/100 simulations; uniform-prior PUCT wiring observed | [raw output](evidence/search-probes.log) |

The targeted and full runs overlap; do not add their counts. Timing is this host's observation, not a performance guarantee. The initial full suite was **not green**; the separately documented follow-up is green.

## Commands and environment

Commands ran from `project/backend` unless stated otherwise. Variables below abbreviate the actual absolute paths; they do not change dependencies:

```sh
REPO=/home/thien408/Projects/2026-2027/ai_advanced/ML-Advance-Group
PY="$REPO/.venv/bin/python"
cd "$REPO/project/backend"
```

Initial invocation (exit 1):

```sh
PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest \
  tests.search.test_mcts tests.environment.test_environment tests.test_foundation \
  tests.data_pipeline.test_gold_pipeline tests.data_pipeline.test_gold_rewards \
  tests.api.test_mcts_api tests.evaluation.test_api_parity \
  tests.evaluation.test_tournament -v
```

Errors were: environment/foundation imports of `project.backend...` could not resolve `project`; API test import could not resolve `fastapi`; tournament worker comparison could not bind the multiprocessing forkserver's local socket in the sandbox (`PermissionError: [Errno 1] Operation not permitted`). They were recorded before correcting the environment.

The existing `.venv` already had Pydantic, PyYAML, HTTPX, and Uvicorn. FastAPI and python-dotenv are already declared in `pyproject.toml`'s API extra. Only those missing requirements and their resolver-selected dependencies were installed to a temporary target, using network-approved execution:

```sh
"$PY" -m pip install --target /tmp/neural-mcts-phase0/deps \
  'fastapi>=0.115,<1' 'python-dotenv>=1.0,<2'
```

Selected versions: Python 3.14.9; FastAPI 0.162.5, Starlette 1.10.0, python-dotenv 1.2.4, Pydantic 2.14.3, PyYAML 6.0.3, HTTPX 0.28.1. No production dependency constraints were changed and the shared `.venv` was not modified. The temporary install also resolved anyio 4.16.0, annotated-doc 0.0.6, annotated-types 0.8.0, idna 3.20, opentelemetry-api 1.45.2, pydantic-core 2.50.4, typing-extensions 4.16.0, and typing-inspection 0.4.5.

Both corrected runs used repository root on `PYTHONPATH` for the existing mixed import styles, plus temporary dependency target first. They ran with sandbox escalation approved to enable existing local multiprocessing tests:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="/tmp/neural-mcts-phase0/deps:$REPO" \
"$PY" -m unittest \
  tests.search.test_mcts tests.environment.test_environment tests.test_foundation \
  tests.data_pipeline.test_gold_pipeline tests.data_pipeline.test_gold_rewards \
  tests.api.test_mcts_api tests.evaluation.test_api_parity \
  tests.evaluation.test_tournament -v

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="/tmp/neural-mcts-phase0/deps:$REPO" \
"$PY" -m unittest discover -s tests -v
```

The broader run was performed once after targeted success. No tests were altered, filtered to hide errors, or replaced with mocks to claim a green suite.

## Initial regression error (resolved in follow-up)

```text
ERROR: test_sample_limit_preserves_full_bronze_provenance
  (data_pipeline.test_silver_pipeline.SilverPipelineTests)
tests/data_pipeline/test_silver_pipeline.py:66
  self.read_jsonl(root / "silver", "games.jsonl")
TypeError: SilverPipelineTests.read_jsonl() takes 2 positional arguments but 3 were given
```

**VERIFIED at initial audit:** helper at line 41 accepts `self,path`, call at line 66 supplied two arguments after self. Both were in unchanged HEAD code. The historical `TOURNAMENT_HANDOVER.md` also mentions this same error. It was initially left untouched under the documentation-only scope. The user subsequently authorized its repair; the original Silver assertion now executes successfully, and the full suite passes. See [repair evidence](00_BASELINE_REPAIR.md).

## Existing coverage and gaps

| Area | Existing protection actually exercised | Missing evidence for neural integration |
| --- | --- | --- |
| MCTS | Legal initial move, visit totals, red mate-in-one, unprotected capture, favorable Q for both sides, seeded rollout repeatability, immutability, terminal rejection | Fake neural policy/value invocation, PUCT policy influence throughout tree, both-color neural mate fixtures |
| Signs | Three-node backup sign alternation; static eval side-to-move; checkmated side = -1 | Checkpoint output perspective, two-/three-ply neural parity across both colors |
| Policy | Root visit policy normalization and greedy choice | Native vocabulary mapping, illegal mask, NaN/zero logits/probs, prior normalization |
| Engine | Initial 44 moves, horse/elephant blockers, cannon, pawn, facing kings, mate/stalemate, history/rewards | Cross-model orientation and legal-move mapping parity |
| API | MCTS available/neural unavailable, both colors, fixed eight-simulation budget, opening/reset, catalog | Real missing/corrupt checkpoint states, lazy inference loading, explicit neural service budget |
| Tournament | Per-side budgets, config/provenance, paired colors, seeds, truncation, resume, one/two workers | Neural registration and model identity/hash; equal-time/equal-simulation comparison |
| Gold | Selected-action IDs, legal masks, game splits, duplicates, terminal/truncated distinction, shaped reward conventions | Gold is not a pretrained codec or MCTS visit-policy target |

No existing test references to `PUCTSelection`, `NeuralEvaluator`, `ActionEncoder`, or `ObservationEncoder` were found. The additional inline diagnostic exercised PUCT manually but does not become an enduring regression test.

## Reproducing the small expansion diagnostic

Run with the corrected environment above; this uses existing code and static evaluation, without a network:

```python
from src.agents import SearchBudget
from src.environment import GameEngine, GameState
from src.search.mcts import MCTSSearch, PUCTSelection, RolloutEvaluator

engine = GameEngine()
for count in (2, 44, 45):
    search = MCTSSearch(engine, PUCTSelection(1.4),
                        RolloutEvaluator(engine, max_depth=0), seed=0)
    root = search.search(GameState.initial(), SearchBudget(max_simulations=count))
    assert root.num_legal == 44
    assert all(c.prior == 1 / 44 for c in root.children.values())
    print(count, len(root.children), len(root.untried_moves),
          sorted(c.N for c in root.children.values()))
```

The full [probe log](evidence/search-probes.log) also records both-color classic decisions and a soft deadline observation. These are diagnostic smoke checks, not new production tests or a benchmark tournament.

## Explicitly not executed

Checkpoint download/hash computation/load/forward; cross-framework parity; neural game; training/fine-tuning; strength or latency benchmark; browser/UI test; full external-project tests; CI. Source-file hashes and upstream manifest hashes in the shortlist must not be mistaken for locally computed checkpoint SHA-256 values.
