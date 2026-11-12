"""Reproducibility metadata for tournament runs: git state, machine, hashes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

# Output directories must not make the tree look dirty or pollute git.diff.
_GIT_EXCLUDE = (":(exclude)results",)

RELEVANT_PACKAGES = (
    "pydantic", "PyYAML", "fastapi", "python-dotenv", "numpy", "matplotlib", "torch",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    """Stable JSON used for hashing: sorted keys, compact, UTF-8 friendly."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def git_info(cwd: str | Path) -> dict[str, Any]:
    """Commit, branch and dirty flag; ``available`` is False outside a git repo."""
    root = Path(cwd)
    commit = _git(["rev-parse", "HEAD"], root)
    if commit is None:
        return {"available": False, "commit": None, "branch": None, "dirty": None,
                "status": [], "untracked": []}
    branch = (_git(["rev-parse", "--abbrev-ref", "HEAD"], root) or "").strip() or None
    status = _git(["status", "--porcelain", "--", ".", *_GIT_EXCLUDE], root) or ""
    lines = [line for line in status.splitlines() if line.strip()]
    return {
        "available": True,
        "commit": commit.strip(),
        "branch": branch,
        "dirty": bool(lines),
        "status": lines,
        "untracked": [line[3:] for line in lines if line.startswith("??")],
    }


def git_diff(cwd: str | Path) -> str:
    """Uncommitted tracked changes (untracked files are listed in git_info)."""
    return _git(["diff", "HEAD", "--", ".", *_GIT_EXCLUDE], Path(cwd)) or ""


def package_versions(names: tuple[str, ...] = RELEVANT_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def system_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def collect_provenance(cwd: str | Path, argv: list[str] | None = None) -> dict[str, Any]:
    return {
        "git": git_info(cwd),
        "system": system_info(),
        "packages": package_versions(),
        "argv": list(argv if argv is not None else sys.argv),
    }
