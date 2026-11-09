"""Install the trained Tabular Q checkpoint into backend/models/tabular_q/.

Run from project/backend:

    python -m scripts.download_tabular_q_model --from-file "C:/Users/you/Downloads/q_table.json.gz"
    python -m scripts.download_tabular_q_model  # Drive download, requires Google ADC

The API will use the resulting file with XIANGQI_TABULAR_Q_CHECKPOINT.
This CLI never changes Gold data or trains the Q-table.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from src.api.services.tabular_q_checkpoint import download_drive_checkpoint
from src.training.tabular_qlearning import MODEL_VERSION, TabularQTable

DEFAULT_DRIVE_FILE_ID = "1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c"
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parent.parent / "models" / "tabular_q" / "q_table.json.gz"
)
MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024


def install_checkpoint(
    destination: str | Path = DEFAULT_DESTINATION,
    source_file: str | Path | None = None,
    drive_file_id: str = DEFAULT_DRIVE_FILE_ID,
    replace: bool = False,
    downloader: Callable[[str, Path], None] = download_drive_checkpoint,
) -> dict[str, object]:
    """Copy or fetch a checkpoint, validate it and publish atomically.

    The downloader parameter permits offline unit testing without Google access.
    Existing checkpoints are never overwritten unless replace=True.
    """
    target = Path(destination).expanduser().resolve()
    if target.exists() and not replace:
        raise FileExistsError(
            f"Checkpoint already exists: {target}. Use --replace to overwrite it."
        )
    if source_file is not None:
        source = Path(source_file).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Downloaded checkpoint not found: {source}")
    else:
        source = None

    target.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=".q_table_",
            suffix=".downloading",
            delete=False,
        ) as temporary:
            staging = Path(temporary.name)

        if source is not None:
            shutil.copyfile(source, staging)
        else:
            downloader(drive_file_id, staging)

        size = staging.stat().st_size
        if size == 0 or size > MAX_CHECKPOINT_BYTES:
            raise ValueError("Checkpoint size is empty or exceeds 128 MiB")

        table, payload = TabularQTable.load(staging)
        if payload.get("format") != MODEL_VERSION or not table.values:
            raise ValueError("Invalid or empty Tabular Q model")

        sha256 = hashlib.sha256()
        with staging.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha256.update(chunk)

        os.replace(staging, target)
        staging = None
        return {
            "path": str(target),
            "bytes": size,
            "sha256": sha256.hexdigest(),
            "model_version": MODEL_VERSION,
            "known_states": len(table.values),
            "known_state_action_pairs": sum(
                len(actions) for actions in table.values.values()
            ),
            "source": "local_copy" if source is not None else "google_drive",
        }
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/import a trained Tabular Q model into backend/models"
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--from-file",
        type=Path,
        help="Import a q_table.json.gz you downloaded manually; no Google auth needed",
    )
    source_group.add_argument(
        "--drive-file-id",
        default=DEFAULT_DRIVE_FILE_ID,
        help="Private Google Drive model file ID (default: the team's Gold v2 model)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Output checkpoint path (default: backend/models/tabular_q/q_table.json.gz)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing checkpoint only after validating the new file",
    )
    args = parser.parse_args()

    try:
        info = install_checkpoint(
            destination=args.output,
            source_file=args.from_file,
            drive_file_id=args.drive_file_id,
            replace=args.replace,
        )
    except (OSError, ValueError, RuntimeError, KeyError, EOFError) as exc:
        parser.exit(status=1, message=f"Model installation failed: {exc}\n")

    print(f"Installed: {info['path']}")
    print(f"Source: {info['source']}")
    print(f"Size: {info['bytes']} bytes; SHA-256: {info['sha256']}")
    print(
        f"Q-table: {info['known_states']} states, "
        f"{info['known_state_action_pairs']} state-action pairs"
    )
    print()
    print("In PowerShell, configure the API before starting FastAPI:")
    print(f'$env:XIANGQI_TABULAR_Q_CHECKPOINT="{info["path"]}"')
    print("Remove-Item Env:XIANGQI_TABULAR_Q_DRIVE_FILE_ID -ErrorAction SilentlyContinue")
    print("fastapi dev src/api/main.py")


if __name__ == "__main__":
    main()
