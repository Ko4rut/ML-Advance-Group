"""Separate Gold contracts for policy imitation and offline RL."""

from __future__ import annotations

from typing import Iterable, Literal

from pydantic import Field, model_validator

from ..models import ActionId, Board, SideName, SourceType, Split, VersionedRecord

ACTION_SPACE_SIZE = 90 * 90


def dense_legal_action_mask(action_ids: Iterable[int]) -> list[bool]:
    """Expand sparse from-to action IDs only when a training batch needs them."""
    mask = [False] * ACTION_SPACE_SIZE
    for action_id in action_ids:
        if isinstance(action_id, bool) or not isinstance(action_id, int) or not 0 <= action_id < ACTION_SPACE_SIZE:
            raise ValueError("action_id must be an integer in [0, 8100)")
        mask[action_id] = True
    return mask


class PolicySample(VersionedRecord):
    game_id: str = Field(min_length=1)
    ply: int = Field(ge=1)
    split: Split
    state: Board
    side: SideName
    target_action_id: ActionId
    legal_action_ids: list[ActionId] = Field(min_length=1)
    behavior_source_type: SourceType
    source_dataset: str | None = None
    action_encoding: Literal["from_to_8100_v1"] = "from_to_8100_v1"

    @model_validator(mode="after")
    def target_is_legal(self) -> "PolicySample":
        if self.target_action_id not in self.legal_action_ids:
            raise ValueError("target action must appear in legal_action_ids")
        if len(self.legal_action_ids) != len(set(self.legal_action_ids)):
            raise ValueError("legal_action_ids must be unique")
        return self


class OfflineRLSample(VersionedRecord):
    game_id: str = Field(min_length=1)
    ply: int = Field(ge=1)
    split: Split
    state: Board
    side: SideName
    action_id: ActionId
    reward: float
    reward_definition: str = Field(min_length=1)
    next_state: Board
    next_side: SideName | None = None
    legal_action_ids: list[ActionId] = Field(default_factory=list)
    next_legal_action_ids: list[ActionId] = Field(default_factory=list)
    terminated: bool
    truncated: bool
    termination_source: Literal["none", "engine", "recorded"] = "none"
    behavior_source_type: SourceType
    source_dataset: str | None = None
    action_encoding: Literal["from_to_8100_v1"] = "from_to_8100_v1"

    @model_validator(mode="after")
    def terminal_flags(self) -> "OfflineRLSample":
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated are mutually exclusive")
        if self.next_side is not None and self.next_side == self.side:
            raise ValueError("next_side must alternate after each move")
        for ids in (self.legal_action_ids, self.next_legal_action_ids):
            if len(ids) != len(set(ids)):
                raise ValueError("legal action IDs must be unique")
        if self.legal_action_ids and self.action_id not in self.legal_action_ids:
            raise ValueError("action_id must be legal in state")
        if self.terminated and self.next_legal_action_ids:
            raise ValueError("terminal transitions must have empty next_legal_action_ids")
        if self.termination_source != "none" and not self.terminated:
            raise ValueError("termination_source requires terminated=True")
        return self
