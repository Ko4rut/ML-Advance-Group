"""Start with: python -m uvicorn src.api.main:app --reload --port 8000."""

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes.matches import router
from .routes.tabular_q import router as tabular_q_router
from .services.matches import MatchError, MatchService


def create_app(service: MatchService | None = None) -> FastAPI:
    app = FastAPI(title="Xiangqi AI API", version="0.1.0")
    app.state.match_service = service if service is not None else MatchService()
    origins = [item.strip() for item in os.getenv(
        "XIANGQI_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",") if item.strip()]
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(MatchError)
    async def handle_match_error(request: Request, exc: MatchError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    app.include_router(router)
    app.include_router(tabular_q_router)
    return app


app = create_app()
