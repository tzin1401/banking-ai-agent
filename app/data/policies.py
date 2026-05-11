"""Dummy policy / FAQ snippets keyed by BANKING77 intent.

In a real system this would come from a database or a vector store. For Lab 3
we only need a believable, reasonably-shaped collection of support snippets
that the response-drafting node can ground its replies in.

The keys here must match labels in `sample_data/labels.txt` exactly (case
sensitive). A `default` entry is used when the intent has no dedicated policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Policy:
    """A single FAQ / support snippet."""

    intent: str
    title: str
    content: str
    # Suggested action for the router / drafter.
    suggested_action: str = ""


# ---------------------------------------------------------------------------
# Snippet body builder helpers
# ---------------------------------------------------------------------------


def _para(*lines: str) -> str:
    return "\n".join(line.strip() for line in lines if line.strip())


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------


_POLICIES: dict[str, Policy] = {
    "card_arrival": Policy(
        intent="card_arrival",
        title="Card delivery timeline",
        content=_para(
            "New physical cards are normally delivered within 5–7 working days",
            "from the order date. International addresses can take up to 14 days.",
            "If the card has not arrived after this window, we will cancel the",
            "old order and re-issue a new card free of charge.",
        ),
        suggested_action="Confirm the customer's delivery address and order date.",
    ),
    "card_not_working": Policy(
        intent="card_not_working",
        title="Card not working",
        content=_para(
            "If a card declines, first verify that it is unfrozen in the mobile",
            "app and that the available balance is sufficient. Contactless can",
            "be temporarily disabled after 5 consecutive failures — a chip + PIN",
            "transaction will re-enable it. If the chip itself is damaged we will",
            "replace the card free of charge.",
        ),
        suggested_action="Ask the customer where the card was declined and the exact error message.",
    ),
    "lost_or_stolen_card": Policy(
        intent="lost_or_stolen_card",
        title="Lost or stolen card",
        content=_para(
            "Block the card immediately from the mobile app or by calling our",
            "24/7 hotline. Any transactions performed after the report time are",
            "covered by our zero-liability policy. A replacement card is issued",
            "the same working day and delivered within 5–7 working days.",
        ),
        suggested_action="Escalate to fraud team for immediate card freeze and replacement.",
    ),
    "compromised_card": Policy(
        intent="compromised_card",
        title="Compromised card",
        content=_para(
            "If the customer suspects their card details have been exposed,",
            "we will freeze the card, dispute any unrecognised transactions, and",
            "issue a replacement with a new card number. All disputes are",
            "investigated within 10 working days.",
        ),
        suggested_action="Escalate to fraud team. Freeze card before drafting reply.",
    ),
    "card_payment_not_recognised": Policy(
        intent="card_payment_not_recognised",
        title="Unrecognised card payment",
        content=_para(
            "Many unrecognised transactions are merchant DBA names that differ",
            "from the trading name. Customers can dispute the transaction in the",
            "app within 60 days. If fraud is confirmed, we will refund the amount",
            "and re-issue the card.",
        ),
        suggested_action="Ask the customer for the transaction date, amount, and merchant name.",
    ),
    "cash_withdrawal_not_recognised": Policy(
        intent="cash_withdrawal_not_recognised",
        title="Unrecognised cash withdrawal",
        content=_para(
            "Any ATM withdrawal the customer does not recognise should be",
            "reported within 60 days. We will request the ATM journal and refund",
            "confirmed unauthorised withdrawals within 10 working days.",
        ),
        suggested_action="Escalate to fraud team and ask for the date, amount, and ATM location.",
    ),
    "failed_transfer": Policy(
        intent="failed_transfer",
        title="Failed transfer",
        content=_para(
            "Transfers can fail because of incorrect beneficiary details,",
            "insufficient funds, or the recipient bank rejecting the payment.",
            "Failed transfers are normally reversed to the source account within",
            "1 working day. No fee is charged for a failed transfer.",
        ),
        suggested_action="Ask for the transfer reference and check status in core banking.",
    ),
    "pending_transfer": Policy(
        intent="pending_transfer",
        title="Pending transfer",
        content=_para(
            "Transfers can stay pending for up to 3 working days when sent",
            "outside business hours or to a new beneficiary. Customers can",
            "cancel a pending transfer in the mobile app at any time.",
        ),
        suggested_action="Confirm the transfer reference; offer cancellation if still pending.",
    ),
    "transfer_not_received_by_recipient": Policy(
        intent="transfer_not_received_by_recipient",
        title="Transfer not received by recipient",
        content=_para(
            "If the sender's account is debited but the beneficiary has not",
            "received the funds, we can raise a payment trace with the recipient",
            "bank. Traces typically take 3–5 working days. We can refund the",
            "amount if the transfer is confirmed lost.",
        ),
        suggested_action="Collect the beneficiary IBAN/account number and transfer reference, then trace.",
    ),
    "cancel_transfer": Policy(
        intent="cancel_transfer",
        title="Cancel a transfer",
        content=_para(
            "Pending transfers can be cancelled directly in the mobile app",
            "before they are picked up by the clearing system. Once cleared, we",
            "can only attempt a recall through the recipient's bank.",
        ),
        suggested_action="Check transfer status; cancel if pending or open a recall otherwise.",
    ),
    "pending_card_payment": Policy(
        intent="pending_card_payment",
        title="Pending card payment",
        content=_para(
            "Card authorisations can stay pending for up to 7 days while the",
            "merchant finalises the charge. The funds are reserved but not yet",
            "captured. If the merchant never captures the payment, the amount",
            "is automatically released back to the customer.",
        ),
        suggested_action="Reassure the customer; advise to wait up to 7 days.",
    ),
    "Refund_not_showing_up": Policy(
        intent="Refund_not_showing_up",
        title="Missing refund",
        content=_para(
            "Refunds initiated by the merchant typically settle within 5–10",
            "working days. If the merchant cannot confirm the refund or the",
            "deadline has passed, we can open a chargeback case which is",
            "resolved within 30 days.",
        ),
        suggested_action="Ask for proof of refund from the merchant; open chargeback if missing.",
    ),
    "request_refund": Policy(
        intent="request_refund",
        title="Request a refund",
        content=_para(
            "Refunds must first be requested from the merchant. If the merchant",
            "refuses or does not reply within a reasonable time, we can open a",
            "chargeback case under the card scheme rules within 60 days of the",
            "transaction date.",
        ),
        suggested_action="Confirm the merchant has been contacted before opening a dispute.",
    ),
    "pin_blocked": Policy(
        intent="pin_blocked",
        title="PIN blocked",
        content=_para(
            "A card PIN is blocked after 3 consecutive incorrect entries. The",
            "PIN can be unblocked by making a chip + PIN purchase with the",
            "correct PIN at any merchant, or by visiting an ATM with PIN-change",
            "support. A new PIN can be requested in the mobile app.",
        ),
        suggested_action="Walk the customer through unblocking; offer PIN reset.",
    ),
    "verify_my_identity": Policy(
        intent="verify_my_identity",
        title="Identity verification",
        content=_para(
            "Identity verification uses an ID document (passport or national ID)",
            "plus a short selfie video. Verification normally completes within",
            "5 minutes. If the customer is repeatedly rejected, we can open a",
            "manual review which takes up to 2 working days.",
        ),
        suggested_action="Direct the customer to retry; offer manual review if needed.",
    ),
    "unable_to_verify_identity": Policy(
        intent="unable_to_verify_identity",
        title="Unable to verify identity",
        content=_para(
            "Common reasons are blurry photos, expired documents, or document",
            "types not yet supported in the customer's country. Manual review is",
            "available and resolves within 2 working days.",
        ),
        suggested_action="Escalate to the verification team for manual review.",
    ),
    "topping_up_by_card": Policy(
        intent="topping_up_by_card",
        title="Top up by card",
        content=_para(
            "Top-ups by debit card are credited instantly. Credit card top-ups",
            "may be subject to a small fee depending on the issuer. Daily and",
            "monthly limits apply depending on the customer's verification tier.",
        ),
        suggested_action="Check the customer's verification tier for the applicable limits.",
    ),
    "top_up_failed": Policy(
        intent="top_up_failed",
        title="Top up failed",
        content=_para(
            "Top-up failures are normally caused by 3D Secure authentication",
            "issues or limits set by the funding bank. Funds are not debited from",
            "the source account when a top-up fails. Customers can retry after",
            "approving the transaction with their bank.",
        ),
        suggested_action="Ask for the failure code or screenshot to diagnose.",
    ),
    "change_pin": Policy(
        intent="change_pin",
        title="Change PIN",
        content=_para(
            "PIN can be changed at any ATM that supports PIN change, or via the",
            "mobile app under Cards → PIN. The change takes effect immediately.",
        ),
        suggested_action="Direct the customer to the in-app PIN change flow.",
    ),
    "card_payment_fee_charged": Policy(
        intent="card_payment_fee_charged",
        title="Card payment fee charged",
        content=_para(
            "A small fee may be charged on card payments made in a foreign",
            "currency, at an ATM operated by a third-party, or above the",
            "monthly free-transactions limit. The exact fee can be found in",
            "the Fees section of the mobile app. If the fee is unexpected,",
            "we can review the transaction and refund it when applicable.",
        ),
        suggested_action="Ask for the transaction date, amount, and merchant; review fee schedule.",
    ),
    "extra_charge_on_statement": Policy(
        intent="extra_charge_on_statement",
        title="Extra charge on statement",
        content=_para(
            "Unexpected charges on a statement are usually currency conversion",
            "fees, foreign-ATM fees, or a delayed merchant authorisation. The",
            "Fees section of the app explains each charge. If the customer",
            "still cannot identify a charge, we can investigate the specific",
            "transaction within 5 working days.",
        ),
        suggested_action="Ask for the exact line on the statement (date, amount, description).",
    ),
    "apple_pay_or_google_pay": Policy(
        intent="apple_pay_or_google_pay",
        title="Apple Pay / Google Pay",
        content=_para(
            "All our debit cards support both Apple Pay and Google Pay. Adding",
            "the card to the wallet may require a one-time SMS verification.",
        ),
        suggested_action="Walk the customer through wallet setup.",
    ),
    "default": Policy(
        intent="default",
        title="General support",
        content=_para(
            "Our support team is available 24/7 through the in-app chat and the",
            "hotline. Most queries are resolved within one business day.",
        ),
        suggested_action="Acknowledge the request and offer general support channels.",
    ),
}


# ---------------------------------------------------------------------------
# Public lookup API
# ---------------------------------------------------------------------------


def get_policy(intent: str) -> Policy:
    """Return the policy for `intent`, falling back to the default snippet."""

    return _POLICIES.get(intent) or _POLICIES["default"]


def has_policy(intent: str) -> bool:
    """Return True if a dedicated policy exists for `intent`."""

    return intent in _POLICIES and intent != "default"


def all_policies() -> dict[str, Policy]:
    """Return the full policy registry (for inspection / testing)."""

    return dict(_POLICIES)
