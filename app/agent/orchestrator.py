"""The main workflow controller for the Banking AI-Agent.

`BankingAgent` instantiates every node once and then exposes a single `run`
method that walks the pipeline and collects intermediate outputs into an
`AgentResponse`. Node failures are caught per-step so a faulty draft can still
produce a useful response (escalated to a human).
"""

from __future__ import annotations

import logging
import time
from typing import Final

from app.clients.base import BaseLLMClient
from app.clients.ollama_client import MockOllamaClient, OllamaClient
from app.core.schemas import (
    AgentResponse,
    AgentTrace,
    DraftResult,
    IntentResult,
    PolicyResult,
    PriorityResult,
    RoutingDecision,
    ValidationResult,
)
from app.core.settings import Settings, get_settings
from app.nodes.draft_node import DraftNode
from app.nodes.intent_node import IntentNode
from app.nodes.policy_node import PolicyNode
from app.nodes.priority_node import PriorityNode
from app.nodes.router_node import RouterNode
from app.nodes.validation_node import ValidationNode

logger = logging.getLogger(__name__)

ASK_MORE_PREFIX: Final = (
    "Thanks for getting in touch. To help you faster, could you please share: "
)
ESCALATE_TEXT: Final = (
    "Thanks for your message. Your case has been escalated to one of our "
    "specialists — they will reach out to you shortly."
)


# ---------------------------------------------------------------------------
# Helper: build the LLM client based on settings
# ---------------------------------------------------------------------------


def build_llm_client(settings: Settings) -> BaseLLMClient:
    if settings.mock_llm:
        logger.info("BankingAgent: using MockOllamaClient (MOCK_LLM=1).")
        return MockOllamaClient()
    return OllamaClient(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
        default_max_tokens=settings.llm_max_tokens,
        reasoning_effort=settings.llm_reasoning_effort,
    )


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------


class BankingAgent:
    """Orchestrates the 6 nodes that make up the agentic workflow."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseLLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or build_llm_client(self.settings)

        self.intent_node = IntentNode(self.settings)
        self.priority_node = PriorityNode()
        self.policy_node = PolicyNode()
        self.draft_node = DraftNode(self.llm)
        self.validation_node = ValidationNode(
            min_draft_length=self.settings.min_draft_length,
            intent_confidence_threshold=self.settings.intent_confidence_threshold,
        )
        self.router_node = RouterNode()

    # ------------------------------------------------------------------
    def warm_up(self) -> None:
        """Trigger lazy loads (e.g. the Unsloth classifier) ahead of time."""

        # Calling _ensure_loaded directly avoids running a real prediction.
        self.intent_node._ensure_loaded()  # noqa: SLF001 (private but trusted)

    # ------------------------------------------------------------------
    def run(self, message: str) -> AgentResponse:
        """Execute the full pipeline for a single customer message."""

        started = time.perf_counter()
        message = message.strip()

        intent = self._safe_intent(message)
        priority = self.priority_node.run(intent=intent.intent, message=message)
        policy = self.policy_node.run(intent=intent.intent)
        draft = self.draft_node.run(
            message=message,
            intent=intent.intent,
            priority=priority,
            policy=policy,
        )
        validation = self.validation_node.run(
            draft=draft, intent=intent, policy=policy
        )
        decision = self.router_node.run(
            intent=intent,
            priority=priority,
            draft=draft,
            validation=validation,
        )

        final_response = self._materialise_final_response(
            decision=decision, draft=draft
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return AgentResponse(
            final_response=final_response,
            decision=decision,
            trace=AgentTrace(
                intent=intent,
                priority=priority,
                policy=policy,
                draft=draft,
                validation=validation,
            ),
            extra={"latency_ms": round(elapsed_ms, 2)},
        )

    # ------------------------------------------------------------------
    def _safe_intent(self, message: str) -> IntentResult:
        """Run the intent node but fall back to `default` on failure."""

        try:
            return self.intent_node.run(message)
        except Exception as exc:
            logger.exception("Intent node failed.")
            return IntentResult(
                intent="default",
                confidence=0.0,
                source=self.settings.intent_mode,  # type: ignore[arg-type]
            )

    # ------------------------------------------------------------------
    def _materialise_final_response(
        self,
        *,
        decision: RoutingDecision,
        draft: DraftResult,
    ) -> str:
        if decision.action == "escalate":
            return ESCALATE_TEXT
        if decision.action == "ask_more":
            if draft.missing_info:
                return ASK_MORE_PREFIX + ", ".join(draft.missing_info) + "?"
            return (
                "Thanks for getting in touch. Could you share a few more "
                "details about your issue so we can assist you correctly?"
            )
        return draft.draft_reply


__all__ = [
    "BankingAgent",
    "build_llm_client",
    "AgentResponse",
    "AgentTrace",
    "IntentResult",
    "PriorityResult",
    "PolicyResult",
    "DraftResult",
    "ValidationResult",
    "RoutingDecision",
]
