"""Routing / escalation node.

Decides what the system should do with the drafted reply, combining the
signals produced by the previous nodes:

* `escalate`  — pass to a human agent
* `ask_more`  — ask the customer for missing information
* `reply`     — send the draft directly
"""

from __future__ import annotations

from app.core.schemas import (
    DraftResult,
    IntentResult,
    PriorityResult,
    RoutingDecision,
    ValidationResult,
)


CRITICAL_VALIDATION_KEYWORDS: tuple[str, ...] = (
    "LLM call failed",
    "placeholder",
)


def _is_critical_validation_failure(validation: ValidationResult) -> bool:
    return any(
        any(keyword.lower() in issue.lower() for keyword in CRITICAL_VALIDATION_KEYWORDS)
        for issue in validation.issues
    )


class RouterNode:
    """Reduce node outputs into a single routing decision."""

    def run(
        self,
        *,
        intent: IntentResult,
        priority: PriorityResult,
        draft: DraftResult,
        validation: ValidationResult,
    ) -> RoutingDecision:
        if priority.level == "high":
            return RoutingDecision(
                action="escalate",
                reason=f"High priority issue: {priority.reason}",
            )

        if _is_critical_validation_failure(validation):
            return RoutingDecision(
                action="escalate",
                reason=(
                    "Critical validation failure — draft is unsafe to send: "
                    + "; ".join(validation.issues)
                ),
            )

        if draft.missing_info:
            return RoutingDecision(
                action="ask_more",
                reason="Draft requires missing info: " + ", ".join(draft.missing_info),
            )

        if not validation.is_valid:
            return RoutingDecision(
                action="ask_more",
                reason=(
                    "Validation issues — asking customer to confirm details: "
                    + "; ".join(validation.issues)
                ),
            )

        return RoutingDecision(
            action="reply",
            reason="All checks passed; safe to send the draft directly.",
        )
