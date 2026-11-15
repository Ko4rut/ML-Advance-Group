# Phase 4 — Neural MCTS search lifecycle

Written for: engineers maintaining the search and reviewers deciding whether Phase 4 meets its gate.

Date: 2026-11-13. Branch `agent/mcts_nn`, HEAD at start `4f5c4db` (Phase 3 committed by the manager). Nothing committed in this phase. Tags: **VERIFIED** = observed in a command run this phase (evidence in [evidence/phase4/](evidence/phase4/)); **INFERRED**; **UNKNOWN**.

## 1. What was built

| Piece | File | Role |
| --- | --- | --- |
| `NeuralMCTSSearch` | [src/search/mcts/neural_search.py](../../src/search/mcts/neural_search.py) | PUCT search whose priors and leaf values come from a policy-value evaluator. Subclasses `MCTSSearch` only to reuse `select`, `backpropagate`, `visit_policy`, `_record`; overrides `search` |
| `NeuralEvaluator` | [src/search/mcts/evaluators.py](../../src/search/mcts/evaluators.py) | Thin wrapper over a Phase 3 `PolicyValueAdapter`: `evaluate_with_priors(state, legal_moves)` (what the search uses) and value-only `evaluate(state)`. Duck-typed: never imports torch or `networks` |
| `NeuralMCTSAgent` | [src/agents/neural_mcts_agent.py](../../src/agents/neural_mcts_agent.py) | `from_config(engine, config, adapter)`; `select_action` returns the most visited root child (ties: mean value, then prior). Refuses a classic search and a missing adapter. Exported from `src.agents` |

Not touched (VERIFIED by `git diff`): `search.py`, `node.py`, `selection.py`, `mcts_agent.py`, `RolloutEvaluator`, the API registry (`neural` still unavailable), the frontend, configs. The only edit to an existing module's behaviour is replacing the stub body of `NeuralEvaluator`.

Design choice: a **separate subclass** instead of adding a neural branch to `MCTSSearch`. Classic UCT + rollout therefore cannot regress through this change; the cost is ~25 duplicated lines of the budget loop and root-shortcut. `PUCTSelection` and `MCTSNode.expand(priors)` already existed and are reused unchanged — the "uniform prior" trap was only in `MCTSSearch.expand`, which the neural search never calls.

## 2. One simulation

```text
select    PUCT from the root:  Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
          (MCTSSearch.select: descend while a node has no untried moves; stop at a leaf)
leaf?     ensure_generated(engine): legal moves + engine outcome, once per visited node
terminal  value = engine reward(side_to_move)  -- the evaluator is NOT called
live      (priors, value) = evaluator.evaluate_with_priors(state, legal_moves)   # ONE network call
          validate: priors cover exactly the legal moves, finite, >= 0, sum 1; value finite in [-1,1]
expand    MCTSNode.expand(priors): ALL legal children, each with its network prior
backup    MCTSSearch.backpropagate(path, value)  -- sign flips at every ply
```

Conventions (unchanged from `node.py`): the evaluator value is from the leaf's `side_to_move`; `node.W/N` are from the view of the player who moved *into* the node; backup flips before updating the leaf and at each level up, so at a parent `child.mean_value()` is already the chooser's view and PUCT takes a plain maximum. No colour-dependent sign flip exists anywhere in the neural path; the network's side-to-move value is used as is.

Consequences (all tested):
- The first simulation only evaluates and expands the root, so `sum(child.N) == simulations - 1` and `root.N == simulations`.
- Unvisited children have Q = 0 (the existing `mean_value()` of an empty node), i.e. first-play value 0, as in the original AlphaZero.
- `last_nodes` counts every created node, including never-visited children, so it is ~35–40× `simulations` (e.g. 218 simulations → 8,120 nodes). Creating a child state costs ≈1 µs; measured: all ~40 children of a node 0.04 ms, whereas legal-move generation for a *visited* node is ≈1.6 ms, paid only when that child becomes a leaf. Memory is therefore proportional to nodes, ≈8–17 k nodes per 3 s move.
- Ties in PUCT go to the first child in dict order = the order of the legal-move list = the engine's `order_moves` order. There is **no randomness** (no Dirichlet noise, no sampling), so a deterministic network gives an identical search; `seed` is accepted for interface parity and has no effect (tested).
- The tree is rebuilt for every move; there is no subtree reuse.

## 3. Engine authority and the root shortcut

- Terminal nodes (checkmate, stalemate, threefold repetition, 120-ply no-capture) get the exact engine result and never reach the network. The evaluator fakes in the tests assert that no finished game is ever passed in; the adapter (Phase 3) also bypasses the network on terminals. If an evaluator ever calls a node the search considers live "terminal" (different rule sets), the search raises `RuntimeError`.
- **Root win shortcut** (on by default, as in the classic search): if some root move ends the game in the mover's favour, it is played with one pseudo-simulation and no network call. This matters because (VERIFIED) the raw network does not reliably single out mating moves: for `distill-v4` in the Black mate-in-one fixture, with the shortcut disabled, 150–1000 simulations never made a mating move the most visited. The values of "mate" (+1 exact, from the engine) and "overwhelmingly winning" (≈0.9999) are nearly equal for PUCT, so visits spread by prior. `distill-v4-large` found the mate in both colours at 150 simulations, but that is not guaranteed in general.
- Not implemented: a mate-in-one detector at *interior* nodes, and proven-win propagation (MCTS-Solver). Both would cost extra move generation per leaf; whether they pay is a Phase 6 question.

## 4. Budgets

`SearchBudget.max_simulations` and `time_limit` are honoured; `max_nodes` / `max_depth` are ignored, exactly as in the classic search. At least one simulation always runs. With neither limit set the default is `NeuralMCTSSearch.DEFAULT_SIMULATIONS = 100`. The deadline is checked before each simulation, so the overshoot is at most one simulation (VERIFIED: ≤ 0.03 s at a 3 s limit). The caller (API, benchmark) is responsible for choosing the limit below its hard ceiling; there is no internal safety margin.

**What ~3 s buys** (VERIFIED, [play_game.py budget](../../experiments/neural_mcts/phase4/play_game.py), 6 consecutive moves from the initial position, Intel i5-13420H, torch default 8 threads, CPU, single-state inference):

| Model | simulations / 3 s (median, range) | per simulation | max time used |
| --- | --- | --- | --- |
| distill-v4-large | 212 (157–230) | ≈14 ms (network ≈10 ms + engine ≈1.8 ms + tree) | 3.024 s |
| distill-v4 | 384 (291–492) | ≈8 ms | 3.028 s |

This meets the provisional 3 s target mechanically; whether 200–400 simulations is *good enough* play is unmeasured (Phase 6). Timings vary ±40 % run to run on this hybrid CPU (see model card). Every simulation was a network call in these runs (`network_evaluations == simulations`, no terminal leaves in the opening).

## 5. Configuration

`NeuralMCTSAgent.from_config(engine, {"c_puct": ..., "seed": ...}, adapter)`; `c_puct` defaults to **1.5**. That value is a conventional choice, **not tuned or validated**: Phase 6 should sweep it. The adapter must be passed in (load it with `load_adapter()`, which fails loudly); there is no fallback to rollouts. `configs/neural_mcts.yaml` and the API/registry wiring belong to Phase 5.

## 6. Batching (deliberately not implemented)

Inference is single-state. `NeuralMCTSSearch._evaluate_leaf` is the only place that calls the evaluator, and the evaluator interface is one method; a batched variant would (a) collect K leaves during selection using virtual loss, (b) call a new `evaluate_with_priors_batch` on `NeuralEvaluator` (and a batch `infer` on the Phase 3 backend), (c) validate/expand/back up each leaf with the existing code. Nothing in the node, selection or agent classes would change. Phase 1 measured ≈1.6–1.7× per-position gain at batch 8; whether it pays inside this search needs virtual-loss design and is left for a decision after Phase 6 profiling.

## 7. Verification (all commands executed; run from `project/backend`, `REPO=$(git rev-parse --show-toplevel)`)

See [evidence/phase4/](evidence/phase4/) for raw logs and `SHA256SUMS.txt`.

| What | Result |
| --- | --- |
| Classic regression: pinned decisions/node counts/policies for 3 seeded searches ([test](../../tests/search/test_classic_mcts_unchanged.py)); the pinned values were also reproduced on the pristine pre-change HEAD via `git stash` | pass |
| Fake-evaluator lifecycle tests ([test_neural_mcts.py](../../tests/search/test_neural_mcts.py), 34 tests): priors at every node, policy bias changes visits and chosen move, sign flip per ply for both colours, material value sign, terminal beats a network that says +1 everywhere, draw rules, legality of every node of 20+ trees, 1/2/100 simulations, time budget, determinism, 5 classes of invalid evaluator output, adapter actually invoked | pass, no torch needed |
| Real-weight tests ([test_neural_mcts_real.py](../../tests/search/test_neural_mcts_real.py), 7 per model): real priors bit-identical at root and below, reproducibility, value sign for both colours, mate shortcut + exact terminal values, 24-ply self-play, neural vs classic with each colour, time budget | pass for distill-v4-large and distill-v4; skip with an explicit reason without torch/checkpoint; error under `XIANGQI_NEURAL_REQUIRE_REAL=1` |
| Real games ([play_game.py](../../experiments/neural_mcts/phase4/play_game.py) `game`): v4-large Red vs classic MCTS, 100 simulations each | Red wins by checkmate in 31 plies, all moves legal, 1,455 network calls |
| same, v4-large as Black | Black wins by checkmate in 16 plies, all moves legal |
| v4 neural self-play, 60 simulations | 80 plies, Black wins by checkmate, all moves legal, 4,680 network calls |
| Full backend suite, standard environment (no torch), `python -m unittest discover -s tests` | 411 tests OK, 21 skipped — every skip is a torch/real-weight test with an explicit reason ([log](evidence/phase4/backend-full-suite-standard.log)) |
| Full backend suite, validation environment (torch + `XIANGQI_NEURAL_REQUIRE_REAL=1`) | 445 tests OK, 0 skipped ([log](evidence/phase4/backend-full-suite-validation.log)). Baseline before Phase 4: 394 (validation) / 374 (standard); the difference is exactly the 34 + 3 + 14 new tests (standard: 37 new, the 14 real ones skipped) |
| Mutation checks ([script](evidence/phase4/mutation_checks.py), [log](evidence/phase4/mutation-checks.log)): 19 injected bugs — uniform priors, value sign, terminal override removed, one child per expansion, wrong legal list, 4 validation checks, UCT accepted, time/simulation limits ignored, shortcut off, best-child tie-break, agent bypass, evaluator not calling the adapter, 3 bugs in the CLASSIC code | 19/19 detected. Two honest notes: the first pass **missed** a classic UCT-term change (my pinned cases were insensitive), so two low-branching 400-simulation cases were added and it is now detected; and two mutations are detected only by a hang (harness timeout) |

Wins above are plumbing evidence only (one game per setting against a classic agent with a tiny budget); they are **not** a strength claim.

## 8. Known limitations / open questions for the manager

1. Strength is unmeasured. `c_puct`, simulation counts, and the first-play value are unvalidated defaults.
2. Near-saturated network values blur "mate" vs "winning" (§3); consider interior mate checks / solver in Phase 6 if games show missed or slow mates.
3. No Dirichlet noise or temperature: play is deterministic, which is right for evaluation, wrong for self-play data (Phase 7).
4. No tree reuse between moves; with ≈200 simulations per move this discards useful statistics.
5. Rule mismatch from Phase 1 (perpetual-check adjudication, no-capture counter) is unchanged: the engine decides terminals, the network never sees repetition state, so the search can walk into repetition draws the network does not understand.
6. `NeuralMCTSAgent` is not reachable from the API (Phase 5) and is not registered anywhere.
