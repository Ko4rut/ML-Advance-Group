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
    terminated: bool
    truncated: bool
    behavior_source_type: SourceType
    action_encoding: Literal["from_to_8100_v1"] = "from_to_8100_v1"

    @model_validator(mode="after")
    def terminal_flags(self) -> "OfflineRLSample":
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated are mutually exclusive")
        return self
