"""Transport contracts. Square indices always follow the engine's orientation."""

from typing import Literal

from pydantic import BaseModel, Field, StrictInt, model_validator


class ParticipantModel(BaseModel):
    type: Literal["human", "agent"]
    agent_id: str | None = None

    @model_validator(mode="after")
    def validate_participant(self) -> "ParticipantModel":
        if self.type == "human" and self.agent_id is not None:
            raise ValueError("human participants cannot specify agent_id")
        if self.type == "agent" and not self.agent_id:
            raise ValueError("agent participants require agent_id")
        return self


class CreateMatchRequest(BaseModel):
    red: ParticipantModel
    black: ParticipantModel


class MoveModel(BaseModel):
    source: StrictInt = Field(ge=0, le=89)
    target: StrictInt = Field(ge=0, le=89)


class MoveRequest(MoveModel):
    expected_ply: StrictInt = Field(ge=0)


class StepRequest(BaseModel):
    expected_ply: StrictInt = Field(ge=0)


class MatchSnapshot(BaseModel):
    match_id: str
    red: ParticipantModel
    black: ParticipantModel
    board: list[int] = Field(min_length=90, max_length=90)
    side_to_move: Literal["red", "black"]
    ply: int
    status: Literal["playing", "finished"]
    winner: Literal["red", "black"] | None
    reason: str | None
    last_move: MoveModel | None
    legal_moves: list[MoveModel]


class LegalMovesResponse(BaseModel):
    match_id: str
    ply: int
    legal_moves: list[MoveModel]


class AgentInfo(BaseModel):
    id: str
    name: str
    available: bool
