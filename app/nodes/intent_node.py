"""Intent detection node — the core of the workflow.

Two modes are supported, switched by the `INTENT_MODE` env var:

* `unsloth` (default, production) — loads the fine-tuned Lab 2 model via the
  vendored `IntentClassification` class. Requires CUDA.
* `mock` (dev) — keyword-based classifier so the orchestrator can be tested
  without a GPU.

The Unsloth model is loaded **lazily** on the first call so importing this
module is cheap and side-effect-free.
"""

from __future__ import annotations

import logging
from typing import Callable

from app.core.schemas import IntentResult
from app.core.settings import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock implementation (no GPU required)
# ---------------------------------------------------------------------------


_MOCK_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("card_payment_not_recognised", [
        "unauthori", "unrecognised", "don't recognise", "did not make",
        "didn't make", "i didn't", "i did not",
    ]),
    ("compromised_card", ["compromised", "hacked", "someone used"]),
    ("lost_or_stolen_card", ["lost", "stolen", "stole", "missing card", "can't find my card"]),
    ("cash_withdrawal_not_recognised", ["atm withdrawal", "cash withdrawal i didn", "atm i didn"]),
    ("transfer_not_received_by_recipient", [
        "hasn't received", "still hasn't received", "still has not received",
        "didn't receive", "didn't get the money", "didn't get the transfer",
        "transfer not received", "recipient hasn", "recipient didn",
    ]),
    ("failed_transfer", ["transfer fail", "transfer failed", "transfer error", "transfer rejected", "why did my transfer"]),
    ("pending_transfer", ["transfer pending", "transfer is pending", "pending transfer"]),
    ("cancel_transfer", ["cancel my transfer", "cancel a transfer", "stop a transfer"]),
    ("card_arrival", [
        "waiting on my card", "where is my card", "card arrive", "card delivery",
        "ordered a new card", "still hasn't arrived", "haven't received my card",
    ]),
    ("card_not_working", ["card not working", "card declined", "card doesn't work", "card isn't working"]),
    ("pending_card_payment", ["pending payment", "payment is pending", "charge is pending"]),
    ("Refund_not_showing_up", [
        "refund hasn", "refund not", "still no refund", "missing refund",
        "don't see the money", "refunded me", "refund last week",
    ]),
    ("request_refund", ["want a refund", "request a refund", "get a refund"]),
    ("pin_blocked", [
        "pin blocked", "pin is blocked", "blocked my pin",
        "now it's blocked", "pin wrong", "wrong pin",
    ]),
    ("change_pin", ["change my pin", "change pin", "reset pin"]),
    ("verify_my_identity", ["verify my identity", "verify identity", "kyc"]),
    ("apple_pay_or_google_pay", ["apple pay", "google pay"]),
    ("top_up_failed", ["top up failed", "top-up failed", "topup failed"]),
    ("topping_up_by_card", ["top up", "top-up", "topping up"]),
    ("card_payment_fee_charged", [
        "charged extra", "extra fee", "extra charge", "extra charged",
        "charged a fee", "fee on my", "fee on my last",
    ]),
]


def _mock_predict(message: str) -> IntentResult:
    """Return a best-effort intent from a small keyword table."""

    lower = message.lower()
    for intent, keywords in _MOCK_KEYWORD_RULES:
        if any(kw in lower for kw in keywords):
            return IntentResult(intent=intent, confidence=0.6, source="mock")
    return IntentResult(intent="default", confidence=0.0, source="mock")


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------


class IntentNode:
    """Detect the customer's intent.

    The Unsloth model is loaded lazily on the first call so importing this
    module never touches CUDA.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._classifier: Callable[[str], str] | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load the real classifier on first use."""

        if self._loaded:
            return

        if self.settings.intent_mode == "mock":
            self._classifier = None
            self._loaded = True
            logger.info("IntentNode: running in MOCK mode (no GPU).")
            return

        try:
            from app.nodes._lab2_inference import IntentClassification
        except Exception as exc:
            logger.exception("Failed to import Lab 2 IntentClassification.")
            raise RuntimeError(
                "Cannot import Unsloth-based IntentClassification. "
                "Set INTENT_MODE=mock for local development."
            ) from exc

        config_path = str(self.settings.intent_config_path)
        logger.info("IntentNode: loading Unsloth classifier from %s", config_path)
        clf = IntentClassification(config_path)
        self._classifier = clf  # callable: clf(message) -> label
        self._loaded = True
        logger.info("IntentNode: classifier ready.")

    # ------------------------------------------------------------------
    def run(self, message: str) -> IntentResult:
        """Return the predicted intent for `message`."""

        self._ensure_loaded()

        if self.settings.intent_mode == "mock":
            return _mock_predict(message)

        assert self._classifier is not None, "Classifier not loaded"
        label = self._classifier(message)
        # The vendored class already applies exact + fuzzy matching, so we
        # surface a high-confidence value when a label is returned.
        return IntentResult(
            intent=label or "default",
            confidence=0.95 if label else 0.0,
            source="unsloth",
        )
