"""Public Tabular Q inference health and model coverage contract."""

from typing import Literal

from pydantic import BaseModel, Field


class TabularQModelStatus(BaseModel):
    agent_id: Literal["tabular_q"] = "tabular_q"
    available: bool
    source: Literal["local", "drive", "unconfigured"]
    model_version: str | None
    known_states: int = Field(ge=0)
    known_state_action_pairs: int = Field(ge=0)
    epsilon: Literal[0.0] = 0.0
    unknown_state_fallback: Literal["rule_based"] = "rule_based"
