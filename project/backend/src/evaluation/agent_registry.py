"""Single table mapping tournament agent names to factories, budgets and specs.

The tournament measures agents; it never tunes them. Unless the tournament
config overrides a value, every agent runs with **the defaults of its own
code** (constructor defaults, or configs/mcts.yaml for MCTS):

* ``alphabeta``: constructor default depth, no time limit.
* ``heuristic``: constructor defaults (including its 1 s time limit, so its
  strength and move choice depend on machine speed).
* ``mcts``: wall-clock ``time_limit`` per move from configs/mcts.yaml;
  ``max_simulations`` is an opt-in for exact replay.
"""
from __future__ import annotations

import inspect
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


@dataclass(frozen=True)
class ResolveContext:
    backend_root: Path
    env: Mapping[str, str]


def _require_keys(name: str, params: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"agent {name!r}: unknown parameter(s) {unknown}; allowed {sorted(allowed)}")


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
    default = _code_default(AlphaBetaAgent, "max_depth")
    return {"max_depth": _positive("alphabeta", "max_depth", params.get("max_depth", default), kind=int)}


def _make_alphabeta(engine, params, seed):
    from ..agents.alphabeta_agent import AlphaBetaAgent
    return AlphaBetaAgent(engine, max_depth=params["max_depth"])


# --- heuristic (A*) --------------------------------------------------------

def _resolve_heuristic(params, ctx):
    from ..agents.heuristic import HeuristicAgent

    _require_keys("heuristic", params, frozenset({"max_depth", "max_nodes", "time_limit"}))
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
    return {
        "config": loaded,
        "config_source": source,
        "time_limit": _positive("mcts", "time_limit", params.get("time_limit", yaml_time_limit), kind=float),
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


def _spec(name, class_name, module, deterministic, resolve, factory, budget=None, keys=()):
    return AgentSpec(
        name=name, class_name=class_name, module=module, deterministic=deterministic,
        resolve=resolve, factory=factory,
        budget=budget or (lambda params: None), allowed_keys=frozenset(keys),
    )


REGISTRY: dict[str, AgentSpec] = {
    spec.name: spec for spec in (
        _spec("random", "RandomAgent", "src.agents.random_agent", False,
              _resolve_random, _make_random),
        _spec("rules", "RuleBasedAgent", "src.agents.rule_based", True,
              _resolve_rules, _make_rules),
        _spec("alphabeta", "AlphaBetaAgent", "src.agents.alphabeta_agent", True,
              _resolve_alphabeta, _make_alphabeta),
        # Time-bound by its own 1 s default, hence machine dependent.
        _spec("heuristic", "HeuristicAgent", "src.agents.heuristic", False,
              _resolve_heuristic, _make_heuristic),
        _spec("mcts", "MCTSAgent", "src.agents.mcts_agent", False,
              _resolve_mcts, _make_mcts, _budget_mcts),
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
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Resolve every configured agent, in config order.

    Returns ``(resolved, excluded)``. ``excluded`` maps optional agents that
    cannot be built (missing checkpoint) to the reason; callers must surface it.
    """
    if not isinstance(agents_config, Mapping) or not agents_config:
        raise ValueError("config 'agents' must be a non-empty mapping")
    ctx = ResolveContext(backend_root, env if env is not None else default_env(backend_root))
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


def describe(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Class, module, resolved parameters and determinism, for the manifest."""
    spec = REGISTRY[name]
    return {
        "class": spec.class_name,
        "module": spec.module,
        "params": params,
        "budget": budget_to_dict(spec.budget(params)),
        "is_deterministic": spec.deterministic,
    }


def hash_view(agents: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Agent params minus machine-specific paths, for the config hash."""
    return {
        name: {key: value for key, value in params.items() if key not in NON_HASHED_KEYS}
        for name, params in agents.items()
    }
