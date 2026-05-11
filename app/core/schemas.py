"""Pydantic schemas for the Banking AI-Agent.

These types are the **contracts** between the workflow nodes and the FastAPI
layer. Every node consumes / produces one of these structured objects so the
orchestrator can collect a transparent trace of the pipeline.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request / response (FastAPI layer)
# ---------------------------------------------------------------------------


class CustomerRequest(BaseModel):
    """Inbound request body for `POST /process`."""

    message: str = Field(..., min_length=1, description="Customer message text.")
    customer_id: str | None = Field(
        default=None, description="Optional customer identifier."
    )


# ---------------------------------------------------------------------------
# Per-node structured outputs
# ---------------------------------------------------------------------------


class IntentResult(BaseModel):
    """Output of the intent detection node."""

    intent: str
    confidence: float | None = None
    source: Literal["unsloth", "mock"] = "unsloth"


PriorityLevel = Literal["low", "medium", "high"]


class PriorityResult(BaseModel):
    """Output of the priority / risk detection node."""

    level: PriorityLevel
    reason: str
    matched_keywords: list[str] = Field(default_factory=list)


class PolicyResult(BaseModel):
    """Output of the policy retrieval node."""

    policy_id: str | None
    content: str
    found: bool


class DraftResult(BaseModel):
    """Output of the response drafting node."""

    draft_reply: str
    missing_info: list[str] = Field(default_factory=list)
    suggested_action: str = ""
    raw_model_output: str | None = None


class ValidationResult(BaseModel):
    """Output of the validation node."""

    is_valid: bool
    issues: list[str] = Field(default_factory=list)


RoutingAction = Literal["reply", "ask_more", "escalate"]


class RoutingDecision(BaseModel):
    """Output of the final router node."""

    action: RoutingAction
    reason: str


# ---------------------------------------------------------------------------
# Aggregate response
# ---------------------------------------------------------------------------


class AgentTrace(BaseModel):
    """Full intermediate trace returned to the caller for transparency."""

    intent: IntentResult
    priority: PriorityResult
    policy: PolicyResult
    draft: DraftResult
    validation: ValidationResult


class AgentResponse(BaseModel):
    """Outbound body for `POST /process`."""

    final_response: str = Field(
        ..., description="Text that should be shown to the customer."
    )
    decision: RoutingDecision
    trace: AgentTrace
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Free-form metadata (latency, etc.)."
    )
