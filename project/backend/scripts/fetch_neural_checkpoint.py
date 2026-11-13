"""Explicitly download a pinned pretrained checkpoint into models/neural/ and verify it.

Never run automatically: the backend does not download anything at startup.
Downloads from the pinned upstream revision recorded in models/neural/manifest.json,
refuses any file whose size or SHA-256 differs from the manifest, and writes
atomically (a failed or tampered download leaves nothing behind).

    python scripts/fetch_neural_checkpoint.py --model distill-v4-large
    python scripts/fetch_neural_checkpoint.py --model distill-v4 --directory /data/models

Sizes: distill-v4-large 38.0 MB, distill-v4 15.5 MB. Needs network access to
raw.githubusercontent.com only; no credentials. Pure standard library.
"""

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
import urllib.request

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from src.networks.pretrained import load_manifest  # noqa: E402
from src.networks.pretrained.manifest import DEFAULT_MODEL_DIR  # noqa: E402

DEFAULT_BASE = "https://raw.githubusercontent.com/hskl18/cc-lite"


def download(url: str, destination: Path, size: int, sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", delete=False) as t:
        staging = Path(t.name)
    try:
        digest, received = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=120) as response, staging.open("wb") as out:
            while block := response.read(1 << 20):
                received += len(block)
                if received > size:
                    raise ValueError(f"download exceeds the pinned size of {size} bytes")
                digest.update(block)
                out.write(block)
        if received != size or digest.hexdigest() != sha256:
            raise ValueError(f"integrity check failed: got {received} bytes, sha256 {digest.hexdigest()}")
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="manifest key (default: the manifest default)")
    parser.add_argument("--directory", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE, help=argparse.SUPPRESS)  # tests use file://
    args = parser.parse_args()
    manifest = load_manifest()
    spec = manifest.get(args.model or manifest.default_model)
    destination = args.directory / spec.file
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == spec.sha256:
            print(f"already present and verified: {destination}")
            return 0
        print(f"refusing to overwrite {destination}: it exists but does not match the pinned SHA-256; "
              "move it away first", file=sys.stderr)
        return 1
    url = f"{args.base_url}/{spec.source_revision}/{spec.upstream_path}"
    print(f"downloading {url} ({spec.bytes} bytes)")
    try:
        download(url, destination, spec.bytes, spec.sha256)
    except (OSError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"verified {destination}\n  sha256 {spec.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
