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


# ---------------------------------------------------------------------------
# Intent-aware required-fields table
# ---------------------------------------------------------------------------
# For certain intents the customer *must* provide specific details before we
# can actually resolve their issue.  If the message is missing most of these
# fields, the router forces `ask_more` regardless of the LLM's output.

REQUIRED_FIELDS: dict[str, list[str]] = {
    "transfer_not_received_by_recipient": [
        "reference", "beneficiary", "amount",
    ],
    "failed_transfer": [
        "reference", "error", "amount",
    ],
    "Refund_not_showing_up": [
        "merchant", "date", "confirmation",
    ],
    "declined_card_payment": [
        "merchant", "error",
    ],
    "declined_transfer": [
        "reference", "error",
    ],
    "transaction_charged_twice": [
        "date", "amount", "merchant",
    ],
}


def _has_missing_required(intent: str, message: str) -> list[str]:
    """Return the list of required fields absent from *message*.

    Only triggers ``ask_more`` when **at least half+1** of the expected fields
    are missing, so customers who already supply most details aren't blocked.
    """
    fields = REQUIRED_FIELDS.get(intent, [])
    if not fields:
        return []

    lowered = message.lower()
    missing = [
        field for field in fields
        if not any(word in lowered for word in field.split())
    ]
    # Trigger only when the majority of fields are absent
    if len(missing) >= len(fields) // 2 + 1:
        return missing
    return []


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
        message: str,
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

        # --- Intent-aware completeness check (deterministic) ---
        required_missing = _has_missing_required(intent.intent, message)
        if required_missing:
            return RoutingDecision(
                action="ask_more",
                reason=(
                    "Customer message missing required details: "
                    + ", ".join(required_missing)
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
