"""Read-only status endpoint for the existing Tabular Q inference service."""

from fastapi import APIRouter, Depends, Request

from ..models.tabular_q import TabularQModelStatus
from ..services.matches import MatchService

router = APIRouter(prefix="/api/v1/models", tags=["models"])


def get_match_service(request: Request) -> MatchService:
    """Use the SAME service instance as /api/v1/matches and /agents."""
    return request.app.state.match_service


@router.get("/tabular-q/status", response_model=TabularQModelStatus)
def get_tabular_q_status(
    service: MatchService = Depends(get_match_service),
) -> TabularQModelStatus:
    """Report readiness and Q-table size; no credentials or file paths."""
    return service.tabular_q_status()
