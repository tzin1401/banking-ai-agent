"""FastAPI application factory.

This module is intentionally thin: it wires together the agent and exposes a
small REST surface. All business logic lives under `app/`.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.agent.orchestrator import BankingAgent
from app.core.schemas import AgentResponse, CustomerRequest
from app.core.settings import get_settings

logger = logging.getLogger(__name__)

EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "examples" / "sample_requests.json"


# ---------------------------------------------------------------------------
# Lifespan: build the agent once and reuse it across requests
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Building BankingAgent (intent_mode=%s, model=%s)",
                settings.intent_mode, settings.ollama_model)
    agent = BankingAgent(settings=settings)
    try:
        agent.warm_up()
    except Exception:
        logger.exception(
            "Agent warm-up failed; intent node will be loaded lazily on first request."
        )
    app.state.agent = agent
    app.state.settings = settings
    yield
    logger.info("Shutting down BankingAgent.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="Banking AI-Agent",
        description=(
            "Agentic workflow for banking customer support — Lab 3, "
            "Applications of NLP in Industry, University of Science, VNU-HCM."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    @app.get("/health", tags=["meta"])
    def health() -> dict[str, Any]:
        settings = app.state.settings
        return {
            "status": "ok",
            "intent_mode": settings.intent_mode,
            "ollama_model": settings.ollama_model,
            "mock_llm": settings.mock_llm,
        }

    # ------------------------------------------------------------------
    @app.get("/examples", tags=["meta"])
    def examples() -> JSONResponse:
        if not EXAMPLES_PATH.exists():
            raise HTTPException(status_code=404, detail="sample_requests.json not found")
        return JSONResponse(content=json.loads(EXAMPLES_PATH.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    @app.post("/process", response_model=AgentResponse, tags=["agent"])
    def process(request: CustomerRequest) -> AgentResponse:
        agent: BankingAgent = app.state.agent
        try:
            return agent.run(request.message)
        except Exception as exc:
            logger.exception("Agent.run failed.")
            raise HTTPException(
                status_code=500, detail=f"Agent failure: {exc}"
            ) from exc

    return app


app = create_app()
