"""Optional neural service: one validated model, fresh agents, explicit failures.

Startup never imports torch or loads/downloads weights. The first catalog lookup
or neural match validates and probes the configured model once, under a lock.
Success and failure are cached; restart after provisioning or changing settings.
"""

import math
import os
from pathlib import Path
from threading import Lock
from typing import Mapping

import yaml

from ...agents import NeuralMCTSAgent, SearchBudget
from ...environment import GameEngine, GameState, Move
from ...networks.pretrained import (
    NeuralConfigurationError, NeuralModelError, NeuralModelSettings, load_adapter,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "neural_mcts.yaml"


class NeuralMCTSService:
    def __init__(self, config: dict | None = None, config_path: Path = DEFAULT_CONFIG_PATH,
                 *, env: Mapping[str, str] | None = None) -> None:
        self._config = dict(config) if config is not None else None
        self._config_path = Path(config_path)
        self._env = dict(os.environ if env is None else env)
        self._lock = Lock()
        self._attempted = False
        self._adapter = None
        self._error: str | None = None

    def _settings(self) -> tuple[dict, NeuralModelSettings]:
        config = self._config
        if config is None:
            config = yaml.safe_load(self._config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise NeuralConfigurationError("Neural MCTS config must be a YAML mapping")
        allowed = {"enabled", "model", "path", "sha256", "device", "threads",
                   "c_puct", "time_limit", "max_simulations", "seed"}
        if set(config) - allowed:
            raise NeuralConfigurationError(f"unknown Neural MCTS settings: {sorted(set(config) - allowed)}")
        enabled = config.get("enabled", True)
        flag = self._env.get("XIANGQI_NEURAL_ENABLED")
        if flag is not None:
            if flag.strip().lower() not in {"true", "false", "1", "0"}:
                raise NeuralConfigurationError("XIANGQI_NEURAL_ENABLED must be true/false or 1/0")
            enabled = flag.strip().lower() in {"true", "1"}
        if not isinstance(enabled, bool):
            raise NeuralConfigurationError("Neural MCTS enabled must be a boolean")
        if not enabled:
            raise NeuralConfigurationError("Neural MCTS is disabled by configuration")
        config = {"c_puct": 1.5, "time_limit": 3.0, "max_simulations": None, **config}
        for key in ("c_puct", "time_limit"):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0:
                raise NeuralConfigurationError(f"Neural MCTS {key} must be a finite positive number")
        for key in ("max_simulations", "seed"):
            value = config.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or value < (1 if key == "max_simulations" else 0)):
                raise NeuralConfigurationError(f"Neural MCTS {key} must be a valid integer or null")
        # Reuse Phase 3's environment parser and trust manifest. Env wins over YAML.
        model_env = dict(self._env)
        for key in ("model", "path", "sha256", "device", "threads"):
            value = config.get(key)
            if value is not None:
                suffix = {"path": "MODEL_PATH", "sha256": "MODEL_SHA256", "model": "MODEL"}.get(key, key.upper())
                model_env.setdefault(f"XIANGQI_NEURAL_{suffix}", str(value))
        settings = NeuralModelSettings.from_env(model_env)
        settings.resolve()  # validate before loading the optional runtime
        return config, settings

    def _ensure_ready(self) -> None:
        with self._lock:
            if self._attempted:
                return
            self._attempted = True
            try:
                config, settings = self._settings()
                adapter = load_adapter(settings)
                adapter.predict(GameState.initial())  # readiness includes an actual forward
                self._config, self._adapter = config, adapter
            except (NeuralModelError, OSError, ImportError, RuntimeError, ValueError, TypeError, yaml.YAMLError) as exc:
                self._error = f"Neural MCTS unavailable: {exc}"

    @property
    def available(self) -> bool:
        self._ensure_ready()
        return self._adapter is not None and self._error is None

    @property
    def unavailable_reason(self) -> str | None:
        self._ensure_ready()
        return self._error

    def make_agent(self, engine: GameEngine) -> NeuralMCTSAgent:
        if not self.available:
            raise NeuralModelError(self._error)
        return NeuralMCTSAgent.from_config(engine, self._config, self._adapter)

    def select_move(self, agent: NeuralMCTSAgent, state: GameState) -> Move:
        if not self.available:
            raise NeuralModelError(self._error)
        budget = SearchBudget(time_limit=self._config["time_limit"],
                              max_simulations=self._config["max_simulations"])
        try:
            return agent.select_action(state, budget).move
        except (NeuralModelError, OSError, RuntimeError, ValueError) as exc:
            with self._lock:
                self._error = f"Neural MCTS inference failed: {exc}"
            raise NeuralModelError(self._error) from exc
