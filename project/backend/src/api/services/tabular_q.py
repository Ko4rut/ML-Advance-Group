"""Read-only Tabular Q checkpoint service for live Xiangqi matches.

The model is trained offline; API inference must never mutate the Q-table.
Configure XIANGQI_TABULAR_Q_CHECKPOINT to a LOCAL .json.gz model file.
For a Google Drive checkpoint, first mount/download it on the API host.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock

from ...agents.rule_based import RuleBasedAgent
from ...agents.tabular_q_agent import TabularQAgent
from ...environment import GameEngine
from ...training.tabular_qlearning import MODEL_VERSION, TabularQTable


class TabularQService:
    """Lazy-load and share one immutable Q-table across matches."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        configured = checkpoint_path
        if configured is None:
            configured = os.getenv("XIANGQI_TABULAR_Q_CHECKPOINT")
        self.checkpoint_path = Path(configured).expanduser() if configured else None
        self._table: TabularQTable | None = None
        self._error: str | None = None
        self._lock = RLock()

    def _load(self) -> None:
        if self._table is not None or self._error is not None:
            return
        if self.checkpoint_path is None:
            self._error = (
                "Tabular Q checkpoint not configured; set "
                "XIANGQI_TABULAR_Q_CHECKPOINT to a local q_table.json.gz"
            )
            return
        if not self.checkpoint_path.is_file():
            self._error = f"Tabular Q checkpoint not found: {self.checkpoint_path}"
            return
        try:
            table, metadata = TabularQTable.load(self.checkpoint_path)
            if metadata.get("format") != MODEL_VERSION:
                raise ValueError("unsupported Tabular Q checkpoint format")
            if not table.values:
                raise ValueError("Tabular Q checkpoint contains no states")
            self._table = table
        except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
            self._error = f"Unable to load Tabular Q checkpoint: {exc}"

    @property
    def available(self) -> bool:
        with self._lock:
            self._load()
            return self._table is not None

    @property
    def unavailable_reason(self) -> str | None:
        with self._lock:
            self._load()
            return self._error

    def make_agent(self, engine: GameEngine) -> TabularQAgent:
        """Deterministic greedy inference, with rule-based unknown-state fallback."""
        with self._lock:
            self._load()
            if self._table is None:
                raise RuntimeError(self._error or "Tabular Q checkpoint unavailable")
            return TabularQAgent(
                table=self._table,
                engine=engine,
                epsilon=0.0,
                fallback_agent=RuleBasedAgent(engine),
            )
