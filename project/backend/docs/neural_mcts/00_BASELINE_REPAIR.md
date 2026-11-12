# Phase 0 follow-up — Baseline test repair

The user separately authorized fixing the outstanding baseline error before any later phase. **Current result: all 270 backend tests pass. Phase 1 has not started.** The original audit logs remain preserved as historical evidence.

## Change

In [test_silver_pipeline.py](../../tests/data_pipeline/test_silver_pipeline.py), line 66, `read_jsonl` expects one `Path` argument, but the test supplied a directory and filename separately. The call now joins the complete path:

```diff
- self.read_jsonl(root / "silver", "games.jsonl")
+ self.read_jsonl(root / "silver" / "games.jsonl")
```

This restores execution of the existing assertion that sampled Silver rows preserve the SHA-256 of the complete Bronze source. No assertion was removed, no test was skipped, and no production code, MCTS behavior, dependency declaration, or neural implementation was changed. No new test was necessary: the existing test directly reproduces this defect.

## Verification

| Check | Actual result | Evidence |
| --- | --- | --- |
| Reproduce the individual test before repair | 1 test, expected existing TypeError, 0.024 s | [before repair](evidence/silver-before-fix.log) |
| Silver pipeline + Gold pipeline/reward tests after repair | **25/25 pass**, 0.148 s | [targeted](evidence/silver-after-fix.log) |
| Full backend discovery after repair | **270/270 pass**, 38.220 s, no errors/failures/skips | [full suite](evidence/full-suite-after-fix.log) |
| Whitespace validation | `git diff --check` passes | Final diff inspected |

The same Python 3.14.9 `.venv` and temporary already-declared API dependencies from the [initial baseline](00_BASELINE_TESTS.md) were used. No further installation occurred. Commands ran from `project/backend` with the environment prefix:

```sh
REPO=/home/thien408/Projects/2026-2027/ai_advanced/ML-Advance-Group
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/tmp/neural-mcts-phase0/deps:$REPO"
PY="$REPO/.venv/bin/python"

# Before repair: reproduces the TypeError.
"$PY" -m unittest tests.data_pipeline.test_silver_pipeline.SilverPipelineTests.test_sample_limit_preserves_full_bronze_provenance -v

# After repair:
"$PY" -m unittest tests.data_pipeline.test_silver_pipeline \
  tests.data_pipeline.test_gold_pipeline tests.data_pipeline.test_gold_rewards -v
"$PY" -m unittest discover -s tests -v
```

The full suite ran with approved sandbox escalation so the existing multiprocessing tests could create their local worker sockets. This uses the same environment adjustment as the original audit, not a workaround in test code.

## Scope and recommendation

HEAD remains `d5ad26067aa29b72fe9a0337e7cf4500593b4619`; no commit or push was made. The sole tracked-code change is the test call above; audit documentation/evidence is updated separately. R09 is resolved for this tested working tree. Remaining model licensing, actual inference, mapping/parity and resource questions are unchanged. **Baseline repair complete; stop and wait for explicit Phase 1 approval.**
