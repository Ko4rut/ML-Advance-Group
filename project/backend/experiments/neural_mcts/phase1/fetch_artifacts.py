"""Fetch pinned Phase 1 artifacts into an explicitly supplied external directory.

No runtime installation, training, or backend registration occurs here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


MANIFEST = Path(__file__).with_name("artifacts.json")


def verify(path: Path, record: dict) -> None:
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"Unexpected size: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def fetch(url: str, path: Path, record: dict) -> None:
    if path.exists():
        verify(path, record)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as temp:
        staging = Path(temp.name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, staging.open("wb") as output:
            received = 0
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > record["bytes"]:
                    raise ValueError(f"Download exceeds pinned size: {url}")
                output.write(block)
        verify(staging, record)
        staging.rename(path)
    finally:
        staging.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--with-checkpoint", action="store_true",
                        help="Explicitly opt in to the 15.5 MB checkpoint download")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[5]
    if args.directory.resolve().is_relative_to(repository_root):
        parser.error("Use an external artifact directory; checkpoints do not belong in this repository")
    data = json.loads(MANIFEST.read_text())
    base = f"https://raw.githubusercontent.com/hskl18/cc-lite/{data['revision']}"
    records = dict(data["files"])
    if args.with_checkpoint:
        record = dict(data["checkpoint"])
        records[record.pop("path")] = record
    for relative, record in records.items():
        fetch(f"{base}/{relative}", args.directory / relative, record)
        print(f"verified {relative}: {record['sha256']}")


if __name__ == "__main__":
    main()
