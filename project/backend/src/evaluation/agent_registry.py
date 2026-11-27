"""Single table mapping tournament agent names to factories, budgets and specs.

The tournament measures agents; it never tunes them. It has two budget modes
(config key ``budget_mode``):

``default``
    Unless the tournament config overrides a value, every agent runs with
    **the defaults of its own code** (constructor defaults, or configs/mcts.yaml
    / configs/neural_mcts.yaml), i.e. what the web API serves:

    * ``alphabeta`` / ``minimax``: constructor default depth, no time limit.
    * ``heuristic``: constructor defaults (including its 1 s time limit, so its
      strength and move choice depend on machine speed).
    * ``mcts``: wall-clock ``time_limit`` per move from configs/mcts.yaml;
      ``max_simulations`` is an opt-in for exact replay.
    * ``neural_mcts``: ``c_puct`` / soft ``time_limit`` from configs/neural_mcts.yaml;
      ``max_simulations`` is an opt-in for exact replay.

``equal_time``
    Every *searching* agent gets the same ``SearchBudget(time_limit=T)`` per move
    (``T = time_per_move``) and time is the **only** limit: depth/node/simulation
    caps are lifted (``UNBOUNDED_*``) so that a search stops because the clock ran
    out, never because of a hidden cap. ``random``, ``rules`` and ``tabular_q`` do
    not search; they play as usual and are marked ``budget_applicable: false``.
    Nothing ever sleeps to "use up" a budget.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..agents.base import Agent, SearchBudget
from ..environment import GameEngine
from .provenance import sha256_file

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABULAR_Q_ENV = "XIANGQI_TABULAR_Q_CHECKPOINT"
DEFAULT_TABULAR_Q_CHECKPOINT = "models/tabular_q/q_table.json.gz"
DEFAULT_NEURAL_CONFIG = "configs/neural_mcts.yaml"
DEFAULT_NEURAL_THREADS = 1  # one torch thread per worker process: workers must not fight for CPUs

BUDGET_MODES = ("default", "equal_time")
DEFAULT_TIME_PER_MOVE = 1.0
# "No cap" values used in equal_time so that the clock is the only limit.
# 64 plies of iterative deepening is unreachable in any time budget this
# tournament uses; 10**9 nodes is far beyond what A* can expand in a second.
UNBOUNDED_DEPTH = 64
UNBOUNDED_NODES = 1_000_000_000

# Resolved-parameter keys that are machine specific. They stay in the manifest
# but are left out of the config hash (the content hash replaces them).
NON_HASHED_KEYS = frozenset({"checkpoint", "config_source"})


class AgentUnavailable(Exception):
    """An optional agent cannot be built (e.g. missing checkpoint)."""


@dataclass(frozen=True)
class AgentSpec:
    name: str
    class_name: str
    module: str
    deterministic: bool
    resolve: Callable[[dict[str, Any], "ResolveContext"], dict[str, Any]]
    factory: Callable[[GameEngine, dict[str, Any], int], Agent]
    budget: Callable[[dict[str, Any]], SearchBudget | None] = lambda params: None
    allowed_keys: frozenset[str] = field(default_factory=frozenset)
    # True for agents whose move choice is bounded by a SearchBudget (they search).
    # In equal_time the others (random, rules, tabular_q) simply play as usual.
    budget_applicable: bool = False


@dataclass(frozen=True)
class ResolveContext:
    backend_root: Path
    env: Mapping[str, str]
    budget_mode: str = "default"
    time_per_move: float | None = None

    @property
    def equal_time(self) -> bool:
        return self.budget_mode == "equal_time"


def _require_keys(name: str, params: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"agent {name!r}: unknown parameter(s) {unknown}; allowed {sorted(allowed)}")


def _reject_budget_overrides(
    name: str, params: dict[str, Any], keys: frozenset[str], ctx: ResolveContext,
) -> None:
    """In equal_time the budget is fixed by ``time_per_move``; per-agent caps would break fairness."""
    clash = sorted(set(params) & keys)
    if ctx.equal_time and clash:
        raise ValueError(
            f"agent {name!r}: {clash} cannot be set when budget_mode is 'equal_time' "
            f"(every searching agent gets time_per_move={ctx.time_per_move} s and no other cap)")


def _positive(name: str, key: str, value: Any, *, kind: type) -> Any:
    ok = isinstance(value, kind) and not isinstance(value, bool) and value > 0
    if kind is float and isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return float(value)
    if not ok:
        raise ValueError(f"agent {name!r}: {key} must be a positive {kind.__name__}")
    return value


def _code_default(cls: type, parameter: str) -> Any:
    """The default value of ``cls.__init__(parameter=...)`` as written in the code."""
    return inspect.signature(cls.__init__).parameters[parameter].default


def _seed_mode(name: str, params: dict[str, Any]) -> str:
    mode = params.get("seed_mode", "per_game")
    if mode != "per_game":
        raise ValueError(f"agent {name!r}: only seed_mode 'per_game' is supported")
    return mode


def _time_budget(params: dict[str, Any]) -> SearchBudget | None:
    """equal_time budget of alphabeta / minimax / heuristic (``time_budget`` marks the mode)."""
    if "time_budget" in params:
        return SearchBudget(time_limit=params["time_budget"])
    return None


# --- random ----------------------------------------------------------------

def _resolve_random(params, ctx):
    _require_keys("random", params, frozenset({"seed_mode"}))
    return {"seed_mode": _seed_mode("random", params)}


def _make_random(engine, params, seed):
    from ..agents.random_agent import RandomAgent
    return RandomAgent(seed)


# --- rules -----------------------------------------------------------------

def _resolve_rules(params, ctx):
    _require_keys("rules", params, frozenset())
    return {}


def _make_rules(engine, params, seed):
    from ..agents.rule_based import RuleBasedAgent
    return RuleBasedAgent(engine)


# --- alphabeta -------------------------------------------------------------

def _resolve_alphabeta(params, ctx):
    from ..agents.alphabeta_agent import AlphaBetaAgent

    _require_keys("alphabeta", params, frozenset({"max_depth"}))
    if ctx.equal_time:
        _reject_budget_overrides("alphabeta", params, frozenset({"max_depth"}), ctx)
        # Iterative deepening runs until the clock stops it; the depth cap is only a bound.
        return {"max_depth": UNBOUNDED_DEPTH, "time_budget": ctx.time_per_move}
    default = _code_default(AlphaBetaAgent, "max_depth")
    return {"max_depth": _positive("alphabeta", "max_depth", params.get("max_depth", default), kind=int)}


def _make_alphabeta(engine, params, seed):
    from ..agents.alphabeta_agent import AlphaBetaAgent
    return AlphaBetaAgent(engine, max_depth=params["max_depth"])


# --- minimax ---------------------------------------------------------------

def _resolve_minimax(params, ctx):
    from ..agents.minimax_agent import MinimaxAgent

    _require_keys("minimax", params, frozenset({"max_depth"}))
    if ctx.equal_time:
        _reject_budget_overrides("minimax", params, frozenset({"max_depth"}), ctx)
        return {"max_depth": UNBOUNDED_DEPTH, "time_budget": ctx.time_per_move}
    default = _code_default(MinimaxAgent, "max_depth")
    return {"max_depth": _positive("minimax", "max_depth", params.get("max_depth", default), kind=int)}


def _make_minimax(engine, params, seed):
    from ..agents.minimax_agent import MinimaxAgent
    return MinimaxAgent(engine, max_depth=params["max_depth"])


# --- heuristic (A*) --------------------------------------------------------

def _resolve_heuristic(params, ctx):
    from ..agents.heuristic import HeuristicAgent

    keys = frozenset({"max_depth", "max_nodes", "time_limit"})
    _require_keys("heuristic", params, keys)
    if ctx.equal_time:
        _reject_budget_overrides("heuristic", params, keys, ctx)
        return {
            "max_depth": UNBOUNDED_DEPTH, "max_nodes": UNBOUNDED_NODES,
            "time_limit": ctx.time_per_move, "time_budget": ctx.time_per_move,
        }
    return {
        key: _positive("heuristic", key, params.get(key, _code_default(HeuristicAgent, key)),
                       kind=float if key == "time_limit" else int)
        for key in ("max_depth", "max_nodes", "time_limit")
    }


def _make_heuristic(engine, params, seed):
    from ..agents.heuristic import HeuristicAgent
    return HeuristicAgent(
        engine,
        max_depth=params["max_depth"],
        max_nodes=params["max_nodes"],
        time_limit=params["time_limit"],
    )


# --- mcts ------------------------------------------------------------------

def _resolve_mcts(params, ctx):
    _require_keys("mcts", params, frozenset({"config", "time_limit", "max_simulations", "seed_mode"}))
    import yaml

    _reject_budget_overrides("mcts", params, frozenset({"time_limit", "max_simulations"}), ctx)
    source = str(params.get("config", "configs/mcts.yaml"))
    path = Path(source)
    if not path.is_absolute():
        path = ctx.backend_root / path
    if not path.is_file():
        raise ValueError(f"agent 'mcts': config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"agent 'mcts': {path} must contain a mapping")
    yaml_time_limit = loaded.get("time_limit", 1.0)
    loaded = {key: value for key, value in loaded.items() if key not in {"seed", "time_limit"}}
    max_simulations = params.get("max_simulations")
    if max_simulations is not None:
        max_simulations = _positive("mcts", "max_simulations", max_simulations, kind=int)
    time_limit = _positive("mcts", "time_limit", params.get("time_limit", yaml_time_limit), kind=float)
    if ctx.equal_time:
        time_limit, max_simulations = ctx.time_per_move, None  # time is the only limit
    return {
        "config": loaded,
        "config_source": source,
        "time_limit": time_limit,
        "max_simulations": max_simulations,
        "seed_mode": _seed_mode("mcts", params),
    }


def _make_mcts(engine, params, seed):
    from ..agents.mcts_agent import MCTSAgent
    return MCTSAgent.from_config(engine, {**params["config"], "seed": seed})


def _budget_mcts(params):
    if params["max_simulations"] is not None:
        return SearchBudget(max_simulations=params["max_simulations"])
    return SearchBudget(time_limit=params["time_limit"])


# --- neural_mcts -----------------------------------------------------------

NEURAL_YAML_KEYS = frozenset({
    "enabled", "model", "path", "sha256", "device", "threads",
    "c_puct", "time_limit", "max_simulations", "seed",
})


def _torch_available() -> bool:
    """True if PyTorch can be imported (checked without importing it)."""
    return importlib.util.find_spec("torch") is not None


def _unavailable(required: bool, reason: str):
    """Optional agents are excluded with a reason; required ones stop the run."""
    if required:
        raise ValueError(f"agent 'neural_mcts' is required but {reason}")
    raise AgentUnavailable(reason)


def _resolve_neural_mcts(params, ctx):
    _require_keys(
        "neural_mcts", params,
        frozenset({"config", "c_puct", "time_limit", "max_simulations", "threads",
                   "model", "checkpoint", "required"}),
    )
    import yaml

    from ..networks.pretrained import NeuralModelError, NeuralModelSettings
    from ..networks.pretrained import manifest as neural_manifest

    required = bool(params.get("required", False))
    _reject_budget_overrides("neural_mcts", params, frozenset({"time_limit", "max_simulations"}), ctx)

    source = str(params.get("config", DEFAULT_NEURAL_CONFIG))
    config_path = Path(source)
    if not config_path.is_absolute():
        config_path = ctx.backend_root / config_path
    if not config_path.is_file():
        raise ValueError(f"agent 'neural_mcts': config file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"agent 'neural_mcts': {config_path} must contain a mapping")
    unknown = sorted(set(loaded) - NEURAL_YAML_KEYS)
    if unknown:
        raise ValueError(f"agent 'neural_mcts': unknown key(s) {unknown} in {config_path}")

    c_puct = _positive("neural_mcts", "c_puct", params.get("c_puct", loaded.get("c_puct", 1.5)), kind=float)
    time_limit = _positive(
        "neural_mcts", "time_limit", params.get("time_limit", loaded.get("time_limit", 3.0)), kind=float)
    max_simulations = params.get("max_simulations", loaded.get("max_simulations"))
    if max_simulations is not None:
        max_simulations = _positive("neural_mcts", "max_simulations", max_simulations, kind=int)
    threads = _positive("neural_mcts", "threads", params.get("threads", DEFAULT_NEURAL_THREADS), kind=int)
    if ctx.equal_time:
        time_limit, max_simulations = ctx.time_per_move, None  # time is the only limit

    # The same switches the web service honours: YAML `enabled` and XIANGQI_NEURAL_ENABLED.
    enabled = loaded.get("enabled", True)
    flag = ctx.env.get("XIANGQI_NEURAL_ENABLED")
    if flag is not None:
        if flag.strip().lower() not in {"true", "false", "1", "0"}:
            raise ValueError("XIANGQI_NEURAL_ENABLED must be true/false or 1/0")
        enabled = flag.strip().lower() in {"true", "1"}
    if not enabled:
        _unavailable(required, "Neural MCTS is disabled by configuration (enabled: false)")
    if not _torch_available():
        _unavailable(required, "PyTorch is not installed (see models/neural/README.md)")

    # Model selection follows the API: environment variables win over the YAML;
    # explicit tournament-config values win over both. Threads are always the config's.
    model_env = dict(ctx.env)
    for key in ("model", "path", "sha256", "device"):
        value = loaded.get(key)
        if value is not None:
            suffix = {"path": "MODEL_PATH", "sha256": "MODEL_SHA256", "model": "MODEL"}.get(key, key.upper())
            model_env.setdefault(f"XIANGQI_NEURAL_{suffix}", str(value))
    if params.get("model"):
        model_env[neural_manifest.ENV_MODEL] = str(params["model"])
    if params.get("checkpoint"):
        model_env[neural_manifest.ENV_PATH] = str(params["checkpoint"])
    model_env.pop(neural_manifest.ENV_THREADS, None)
    try:
        settings = dataclasses.replace(NeuralModelSettings.from_env(model_env), threads=threads)
        resolved = settings.resolve()
    except NeuralModelError as exc:
        _unavailable(required, f"neural model configuration is invalid: {exc}")
    spec, path = resolved.spec, resolved.path
    if not path.is_file():
        _unavailable(required, f"checkpoint for {spec.name!r} not found: {path} (see models/neural/README.md)")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != spec.bytes or digest != spec.sha256:
        _unavailable(
            required,
            f"checkpoint {path} fails the manifest integrity check "
            f"(size {size} vs {spec.bytes}, sha256 {digest[:12]}… vs {spec.sha256[:12]}…)")
    return {
        "config_source": source,
        "model": spec.name,
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "device": settings.device,
        "threads": threads,
        "c_puct": c_puct,
        "time_limit": time_limit,
        "max_simulations": max_simulations,
    }


_ADAPTER_CACHE: dict[tuple, Any] = {}


def _load_adapter(params: dict[str, Any]):
    """Heavy policy-value adapter, loaded once per process (agents themselves stay fresh).

    Loading also pins torch to ``params["threads"]`` threads in this process, so
    several worker processes do not oversubscribe the CPUs.
    """
    key = (params["checkpoint"], params["checkpoint_sha256"], params["threads"])
    adapter = _ADAPTER_CACHE.get(key)
    if adapter is None:
        from ..networks.pretrained import NeuralModelSettings, load_adapter
        adapter = load_adapter(NeuralModelSettings(
            model=params["model"], path=Path(params["checkpoint"]),
            sha256=params["checkpoint_sha256"], device=params["device"], threads=params["threads"],
        ))
        _ADAPTER_CACHE[key] = adapter
    return adapter


def _make_neural_mcts(engine, params, seed):
    from ..agents.neural_mcts_agent import NeuralMCTSAgent
    return NeuralMCTSAgent.from_config(
        engine, {"c_puct": params["c_puct"], "seed": seed}, _load_adapter(params))


def _budget_neural_mcts(params):
    if params["max_simulations"] is not None:  # exact-replay opt-in, like mcts
        return SearchBudget(max_simulations=params["max_simulations"])
    return SearchBudget(time_limit=params["time_limit"])


# --- tabular_q -------------------------------------------------------------

def _resolve_tabular_q(params, ctx):
    _require_keys(
        "tabular_q", params,
        frozenset({"checkpoint", "checkpoint_env", "epsilon", "fallback", "required"}),
    )
    required = bool(params.get("required", False))
    epsilon = params.get("epsilon", 0.0)
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not 0 <= epsilon <= 1:
        raise ValueError("agent 'tabular_q': epsilon must be within [0, 1]")
    fallback = params.get("fallback", "rules")
    if fallback != "rules":
        raise ValueError("agent 'tabular_q': only fallback 'rules' is supported")

    configured = params.get("checkpoint")
    source = "config"
    if not configured:
        env_name = params.get("checkpoint_env", DEFAULT_TABULAR_Q_ENV)
        configured = ctx.env.get(env_name)
        source = f"env:{env_name}"
    if not configured:
        configured = DEFAULT_TABULAR_Q_CHECKPOINT
        source = "default"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = ctx.backend_root / path  # same rule as the API: relative to backend root
    path = path.resolve()
    if not path.is_file():
        reason = f"checkpoint not found: {path} (from {source})"
        if required:
            raise ValueError(f"agent 'tabular_q' is required but {reason}")
        raise AgentUnavailable(reason)
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "epsilon": float(epsilon),
        "fallback": fallback,
    }


_TABLE_CACHE: dict[str, Any] = {}


def _load_table(path: str):
    """Read-only Q-table, loaded once per process (agents themselves stay fresh)."""
    table = _TABLE_CACHE.get(path)
    if table is None:
        from ..training.tabular_qlearning import TabularQTable
        table, _ = TabularQTable.load(path)
        _TABLE_CACHE[path] = table
    return table


def _make_tabular_q(engine, params, seed):
    from ..agents.rule_based import RuleBasedAgent
    from ..agents.tabular_q_agent import TabularQAgent
    return TabularQAgent(
        table=_load_table(params["checkpoint"]),
        engine=engine,
        seed=seed,
        epsilon=params["epsilon"],
        fallback_agent=RuleBasedAgent(engine),
    )


def _spec(name, class_name, module, deterministic, resolve, factory, budget=None, keys=(),
          budget_applicable=False):
    return AgentSpec(
        name=name, class_name=class_name, module=module, deterministic=deterministic,
        resolve=resolve, factory=factory,
        budget=budget or (lambda params: None), allowed_keys=frozenset(keys),
        budget_applicable=budget_applicable,
    )


# Config order = schedule order (pairs are itertools.combinations of it).
REGISTRY: dict[str, AgentSpec] = {
    spec.name: spec for spec in (
        _spec("random", "RandomAgent", "src.agents.random_agent", False,
              _resolve_random, _make_random),
        _spec("rules", "RuleBasedAgent", "src.agents.rule_based", True,
              _resolve_rules, _make_rules),
        # Default mode: fixed depth, no time limit (deterministic). equal_time: clock-bound.
        _spec("alphabeta", "AlphaBetaAgent", "src.agents.alphabeta_agent", True,
              _resolve_alphabeta, _make_alphabeta, _time_budget, budget_applicable=True),
        _spec("minimax", "MinimaxAgent", "src.agents.minimax_agent", True,
              _resolve_minimax, _make_minimax, _time_budget, budget_applicable=True),
        # Time-bound by its own 1 s default, hence machine dependent.
        _spec("heuristic", "HeuristicAgent", "src.agents.heuristic", False,
              _resolve_heuristic, _make_heuristic, _time_budget, budget_applicable=True),
        _spec("mcts", "MCTSAgent", "src.agents.mcts_agent", False,
              _resolve_mcts, _make_mcts, _budget_mcts, budget_applicable=True),
        _spec("neural_mcts", "NeuralMCTSAgent", "src.agents.neural_mcts_agent", False,
              _resolve_neural_mcts, _make_neural_mcts, _budget_neural_mcts, budget_applicable=True),
        _spec("tabular_q", "TabularQAgent", "src.agents.tabular_q_agent", True,
              _resolve_tabular_q, _make_tabular_q),
    )
}


def default_env(backend_root: Path = BACKEND_ROOT) -> Mapping[str, str]:
    """Process env after loading project/backend/.env (OS variables win)."""
    try:
        from ..api.config import load_backend_env
    except ImportError:  # python-dotenv not installed: parse .env by hand
        _load_dotenv_fallback(backend_root / ".env")
    else:
        load_backend_env(backend_root / ".env")
    return os.environ


def _load_dotenv_fallback(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def resolve_agents(
    agents_config: Mapping[str, Any],
    backend_root: Path = BACKEND_ROOT,
    env: Mapping[str, str] | None = None,
    *,
    budget_mode: str = "default",
    time_per_move: float | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Resolve every configured agent, in config order.

    Returns ``(resolved, excluded)``. ``excluded`` maps optional agents that
    cannot be built (missing checkpoint, missing torch) to the reason; callers
    must surface it.
    """
    if not isinstance(agents_config, Mapping) or not agents_config:
        raise ValueError("config 'agents' must be a non-empty mapping")
    if budget_mode not in BUDGET_MODES:
        raise ValueError(f"budget_mode must be one of {list(BUDGET_MODES)}, got {budget_mode!r}")
    if budget_mode == "equal_time":
        time_per_move = _equal_time_value(time_per_move)
    ctx = ResolveContext(
        backend_root, env if env is not None else default_env(backend_root),
        budget_mode=budget_mode, time_per_move=time_per_move,
    )
    resolved: dict[str, dict[str, Any]] = {}
    excluded: dict[str, str] = {}
    for name, params in agents_config.items():
        if name not in REGISTRY:
            raise ValueError(f"unknown agent {name!r}; known: {', '.join(REGISTRY)}")
        params = dict(params or {})
        try:
            resolved[name] = REGISTRY[name].resolve(params, ctx)
        except AgentUnavailable as exc:
            excluded[name] = str(exc)
    return resolved, excluded


def _equal_time_value(value: Any) -> float:
    if value is None:
        return DEFAULT_TIME_PER_MOVE
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise ValueError("time_per_move must be a positive finite number of seconds")
    return float(value)


def make_agent(name: str, params: dict[str, Any], engine: GameEngine, seed: int) -> Agent:
    """Always builds a *new* agent; agents are never shared between games."""
    return REGISTRY[name].factory(engine, params, seed)


def budget_for(name: str, params: dict[str, Any]) -> SearchBudget | None:
    return REGISTRY[name].budget(params)


def budget_to_dict(budget: SearchBudget | None) -> dict[str, Any] | None:
    if budget is None:
        return None
    values = {
        "max_depth": budget.max_depth, "max_nodes": budget.max_nodes,
        "max_simulations": budget.max_simulations, "time_limit": budget.time_limit,
    }
    return {key: value for key, value in values.items() if value is not None}


def is_deterministic(name: str, budget_mode: str = "default") -> bool:
    """Move-for-move reproducible? Clock-bound search (equal_time) is not."""
    spec = REGISTRY[name]
    if budget_mode == "equal_time" and spec.budget_applicable:
        return False
    return spec.deterministic


def describe(name: str, params: dict[str, Any], budget_mode: str = "default") -> dict[str, Any]:
    """Class, module, resolved parameters and determinism, for the manifest."""
    spec = REGISTRY[name]
    return {
        "class": spec.class_name,
        "module": spec.module,
        "params": params,
        "budget": budget_to_dict(spec.budget(params)),
        # False => the agent ignores SearchBudget (random/rules/tabular_q); in equal_time it
        # plays as usual and simply is not bound by the common time budget.
        "budget_applicable": spec.budget_applicable,
        "is_deterministic": is_deterministic(name, budget_mode),
    }


def hash_view(agents: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Agent params minus machine-specific paths, for the config hash."""
    return {
        name: {key: value for key, value in params.items() if key not in NON_HASHED_KEYS}
        for name, params in agents.items()
    }
