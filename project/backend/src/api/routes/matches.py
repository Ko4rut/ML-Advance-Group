"""Versioned routes backed by MatchService."""

from fastapi import APIRouter, Depends, Query, Request

from ..models.match import (
    AgentInfo, CreateMatchRequest, LegalMovesResponse, MatchSnapshot, MoveRequest, StepRequest,
)
from ..services.matches import MatchService

router = APIRouter(prefix="/api/v1")


def get_service(request: Request) -> MatchService:
    return request.app.state.match_service


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/agents", response_model=list[AgentInfo], tags=["agents"])
def list_agents(service: MatchService = Depends(get_service)) -> list[AgentInfo]:
    return service.agents()


@router.get("/openings", response_model=list[dict[str, str]], tags=["openings"])
def list_openings(service: MatchService = Depends(get_service)) -> list[dict[str, str]]:
    return service.openings()


@router.post("/matches", response_model=MatchSnapshot, status_code=201, tags=["matches"])
def create_match(body: CreateMatchRequest, service: MatchService = Depends(get_service)) -> MatchSnapshot:
    return service.create(body)


@router.get("/matches/{match_id}", response_model=MatchSnapshot, tags=["matches"])
def get_match(match_id: str, service: MatchService = Depends(get_service)) -> MatchSnapshot:
    return service.get(match_id)


@router.get("/matches/{match_id}/legal-moves", response_model=LegalMovesResponse, tags=["matches"])
def get_legal_moves(
    match_id: str,
    source: int | None = Query(default=None, ge=0, le=89),
    service: MatchService = Depends(get_service),
) -> LegalMovesResponse:
    return service.legal_moves(match_id, source)


@router.post("/matches/{match_id}/moves", response_model=MatchSnapshot, tags=["matches"])
def human_move(match_id: str, body: MoveRequest, service: MatchService = Depends(get_service)) -> MatchSnapshot:
    return service.move(match_id, body)


@router.post("/matches/{match_id}/step", response_model=MatchSnapshot, tags=["matches"])
def agent_move(match_id: str, body: StepRequest, service: MatchService = Depends(get_service)) -> MatchSnapshot:
    return service.step(match_id, body)


@router.post("/matches/{match_id}/reset", response_model=MatchSnapshot, tags=["matches"])
def reset_match(match_id: str, service: MatchService = Depends(get_service)) -> MatchSnapshot:
    return service.reset(match_id)
