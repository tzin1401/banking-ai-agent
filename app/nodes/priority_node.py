"""Priority / risk detection node.

A lightweight rule-based classifier that combines:

1. **Intent-based defaults** — some intents are inherently high-risk
   (e.g. lost/stolen card, unauthorised transactions).
2. **Keyword scan** — overrides upward if urgency words appear in the
   raw message (e.g. "fraud", "urgent", "stolen").

The goal is transparency rather than ML sophistication.
"""

from __future__ import annotations

from app.core.schemas import PriorityLevel, PriorityResult


HIGH_PRIORITY_INTENTS: set[str] = {
    "lost_or_stolen_card",
    "compromised_card",
    "card_payment_not_recognised",
    "cash_withdrawal_not_recognised",
    "direct_debit_payment_not_recognised",
    "unable_to_verify_identity",
    "lost_or_stolen_phone",
}

MEDIUM_PRIORITY_INTENTS: set[str] = {
    "failed_transfer",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "pending_card_payment",
    "pending_transfer",
    "pending_top_up",
    "pending_cash_withdrawal",
    "Refund_not_showing_up",
    "request_refund",
    "transfer_not_received_by_recipient",
    "transaction_charged_twice",
    "wrong_amount_of_cash_received",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "top_up_failed",
    "pin_blocked",
    "card_not_working",
    "card_swallowed",
    "virtual_card_not_working",
    "extra_charge_on_statement",
    "card_payment_fee_charged",
    "cash_withdrawal_charge",
    "verify_my_identity",
    "verify_source_of_funds",
}


HIGH_KEYWORDS: tuple[str, ...] = (
    "fraud", "stolen", "stole", "hacked", "unauthori",
    "compromised", "urgent", "emergency", "immediately",
    "missing money", "money is missing", "money gone",
    "lost my card", "lost card",
)


def _scan_keywords(message: str, vocab: tuple[str, ...]) -> list[str]:
    lowered = message.lower()
    return [word for word in vocab if word in lowered]


def assess_priority(intent: str, message: str) -> PriorityResult:
    """Combine intent + keyword scan to return a PriorityResult."""

    keyword_hits = _scan_keywords(message, HIGH_KEYWORDS)

    level: PriorityLevel
    reason: str

    if keyword_hits:
        level = "high"
        reason = (
            f"Detected urgent keywords in message: {', '.join(keyword_hits)}."
        )
    elif intent in HIGH_PRIORITY_INTENTS:
        level = "high"
        reason = f"Intent '{intent}' is classified as high-risk."
    elif intent in MEDIUM_PRIORITY_INTENTS:
        level = "medium"
        reason = f"Intent '{intent}' typically requires moderate attention."
    else:
        level = "low"
        reason = "No urgency signals detected; routine inquiry."

    return PriorityResult(level=level, reason=reason, matched_keywords=keyword_hits)


class PriorityNode:
    """Thin wrapper around `assess_priority` for orchestrator symmetry."""

    def run(self, intent: str, message: str) -> PriorityResult:
        return assess_priority(intent=intent, message=message)
