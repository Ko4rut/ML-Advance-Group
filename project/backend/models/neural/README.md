# Pretrained neural checkpoints (not stored in git)

`manifest.json` (tracked) pins the only checkpoints the backend will load: file name, size,
**SHA-256**, architecture, upstream source revision and the encoding contract. The weight
files themselves (`*.pt`) are git-ignored — provision them locally. The backend never
downloads anything at startup.

| Key | File | Size | SHA-256 |
| --- | --- | --- | --- |
| `distill-v4-large` (default) | `distill-v4-large.pt` | 38,033,337 B | `05927ccd874c4d97588ba7d06266fda9ba17700d436fdc03ff6844a0d61d0dff` |
| `distill-v4` | `distill-v4.pt` | 15,500,553 B | `1be52664b216529ce56240043406878a93f95f57e4153ff73c4d63afd772fba4` |

Source: <https://github.com/hskl18/cc-lite> at revision `7d305c0e5adbfae4e23db40997603d8dc86b078b`
(Apache-2.0). Model card: [docs/neural_mcts/03_MODEL_CARD.md](../../docs/neural_mcts/03_MODEL_CARD.md).

## 1. Install the optional runtime

Torch is optional; classic MCTS and the rest of the backend work without it. Validated with
`torch 2.11.0+cpu` on Python 3.14.9. The CPU wheel is much smaller than the default Linux CUDA wheel:

```sh
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0"
# or, from project/backend:  pip install -e ".[neural]"   (pulls the platform default build)
```

## 2. Obtain and verify a checkpoint

```sh
cd project/backend
python scripts/fetch_neural_checkpoint.py --model distill-v4-large   # 38.0 MB
python scripts/fetch_neural_checkpoint.py --model distill-v4         # 15.5 MB (fallback)
```

The script downloads from the pinned upstream revision, refuses a size/SHA-256 mismatch, writes
atomically and never overwrites an existing file that fails verification. You can also copy a file in
by hand; the loader checks size and SHA-256 itself and rejects anything else. Check manually with
`sha256sum models/neural/*.pt` against the table above.

## 3. Configure (all optional)

Set in the process environment or in `project/backend/.env` (see `.env.example`):

| Variable | Meaning | Default |
| --- | --- | --- |
| `XIANGQI_NEURAL_MODEL` | manifest key | `distill-v4-large` |
| `XIANGQI_NEURAL_MODEL_PATH` | checkpoint file; absolute, or relative to `project/backend` (never to the cwd) | `models/neural/<key>.pt` |
| `XIANGQI_NEURAL_MODEL_SHA256` | extra assertion; must equal the manifest's pinned hash. It cannot make an unlisted file trusted | unset |
| `XIANGQI_NEURAL_DEVICE` | only `cpu` is supported/validated | `cpu` |
| `XIANGQI_NEURAL_THREADS` | torch CPU threads (process-global) | torch default |

A new checkpoint (e.g. a Phase 7 fine-tune) becomes loadable only by adding a manifest entry with its
hash in a reviewed change.

## 4. Smoke test

```sh
python experiments/neural_mcts/phase3/demo.py --model distill-v4-large
XIANGQI_NEURAL_REQUIRE_REAL=1 python -m unittest tests.networks.test_real_checkpoint -v
```

`XIANGQI_NEURAL_REQUIRE_REAL=1` turns "torch or checkpoint missing" from a skip into an error — use
it in the model-validation environment so the real-weight tests cannot pass vacuously.

## Failure behaviour

Missing file, wrong size/hash, unsupported format/architecture, missing torch, unsupported device or
bad configuration raise a `NeuralModelError` subclass naming the cause. There is no fallback to random
weights, another checkpoint or a heuristic.

## Application integration (Phase 5)

Install the API and test extras (`pip install -e ".[api,test]"`), provision the checkpoint above,
then run from `project/backend`:

```sh
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, run `npm ci && npm run dev` from `project/frontend`, open
<http://localhost:3000>, select **Neural MCTS**, and create a match. Red moves first;
click **Chạy AI** if the first turn belongs to the AI. After a human move the AI replies
automatically. Both player colors and AI-vs-AI are supported.

`configs/neural_mcts.yaml` controls PUCT and the soft 3-second search target, plus an optional
simulation cap. Phase 3's environment variables override the YAML model fields. Only the
manifest-approved weights are accepted. Neural matches default to `opening_mode=none`; an
explicit opening book remains supported.

The API starts without loading the network. The first `/api/v1/agents` request or neural match
loads and validates it and performs one readiness forward. All matches share that adapter;
each participant gets its own search. Missing runtime/weights, bad integrity or invalid settings
leave other agents usable, expose `unavailable_reason` in the catalog, and return HTTP 503 for
neural matches. Errors never trigger another model or a heuristic fallback.

Readiness and failures are cached for the process lifetime. After installing dependencies,
provisioning weights, or editing config, **restart the backend and reload the page**. For rollback,
set `XIANGQI_NEURAL_ENABLED=false` and restart. Existing classic behavior is preserved.

See [05_APPLICATION_INTEGRATION.md](../../docs/neural_mcts/05_APPLICATION_INTEGRATION.md)
for exact validation commands, evidence, and deployment limitations.
