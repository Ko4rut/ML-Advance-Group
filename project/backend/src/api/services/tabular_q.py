"""Read-only Tabular Q checkpoint service for live Xiangqi matches.

The model is trained offline; API inference must never mutate the Q-table.
Configure a local .json.gz checkpoint, or provide its private Drive file ID.
Drive files are downloaded with Google ADC and cached on the API host.
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock

from ...agents.rule_based import RuleBasedAgent
from ...agents.tabular_q_agent import TabularQAgent
from ...environment import GameEngine
from ..models.tabular_q import TabularQModelStatus
from ...training.tabular_qlearning import MODEL_VERSION, TabularQTable
from .tabular_q_checkpoint import TabularQCheckpointSource, CheckpointDownloader


class TabularQService:
    """Lazy-load and share one immutable Q-table across matches."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        drive_file_id: str | None = None,
        cache_dir: str | Path | None = None,
        downloader: CheckpointDownloader | None = None,
    ) -> None:
        self._source = TabularQCheckpointSource(
            checkpoint_path=checkpoint_path,
            drive_file_id=drive_file_id,
            cache_dir=cache_dir,
            downloader=downloader,
        )
        self.checkpoint_path = self._source.checkpoint_path
        self._table: TabularQTable | None = None
        self._error: str | None = None
        self._lock = RLock()

    def _load(self) -> None:
        if self._table is not None or self._error is not None:
            return
        try:
            resolved_path = self._source.resolve()
            table, metadata = TabularQTable.load(resolved_path)
            if metadata.get("format") != MODEL_VERSION:
                raise ValueError("unsupported Tabular Q checkpoint format")
            if not table.values:
                raise ValueError("Tabular Q checkpoint contains no states")
            self._table = table
        except (OSError, EOFError, ValueError, KeyError, TypeError, RuntimeError) as exc:
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

    def status(self) -> TabularQModelStatus:
        """Inspect shared model readiness without exposing credentials or paths."""
        with self._lock:
            self._load()
            source = "unconfigured"
            if self._source.checkpoint_path is not None:
                source = "local"
            elif self._source.drive_file_id is not None:
                source = "drive"
            table = self._table
            if table is None:
                return TabularQModelStatus(
                    available=False,
                    source=source,
                    model_version=None,
                    known_states=0,
                    known_state_action_pairs=0,
                )
            return TabularQModelStatus(
                available=True,
                source=source,
                model_version=MODEL_VERSION,
                known_states=len(table.values),
                known_state_action_pairs=sum(len(actions) for actions in table.values.values()),
            )

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
