"""Backend runtime configuration loaded from project/backend/.env.

Explicit process environment variables always override .env values.
This module is deliberately independent of the working directory.
"""

from pathlib import Path

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[2]


def load_backend_env(env_path: str | Path | None = None) -> bool:
    """Load local backend settings without overwriting existing OS variables."""
    file_path = Path(env_path) if env_path is not None else BACKEND_ROOT / ".env"
    return load_dotenv(dotenv_path=file_path, override=False)
