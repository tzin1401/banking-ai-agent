"""Validation node.

Performs cheap, deterministic sanity checks on the generated draft to catch
obviously bad outputs before they reach the routing decision.

Two severity levels are tracked internally:

* **errors** — flip `is_valid` to False so the router downgrades the action.
* **warnings** — informational only; surfaced in `issues` but do not block.
"""

from __future__ import annotations

import re

from app.core.schemas import (
    DraftResult,
    IntentResult,
    PolicyResult,
    ValidationResult,
)


PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}|\[TODO\]|\[FILL[^]]*\]", re.IGNORECASE)


class ValidationNode:
    """Apply a small battery of rules to the drafted reply."""

    def __init__(
        self,
        *,
        min_draft_length: int = 20,
        intent_confidence_threshold: float = 0.0,
    ) -> None:
        self.min_draft_length = min_draft_length
        self.intent_confidence_threshold = intent_confidence_threshold

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        draft: DraftResult,
        intent: IntentResult,
        policy: PolicyResult,
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if len(draft.draft_reply) < self.min_draft_length:
            errors.append(
                f"Draft too short ({len(draft.draft_reply)} < {self.min_draft_length} chars)."
            )

        if PLACEHOLDER_RE.search(draft.draft_reply):
            errors.append("Draft still contains placeholder text.")

        if (
            intent.confidence is not None
            and intent.confidence < self.intent_confidence_threshold
        ):
            errors.append(
                f"Intent confidence below threshold "
                f"({intent.confidence:.2f} < {self.intent_confidence_threshold:.2f})."
            )

        if not policy.found:
            warnings.append(
                f"No dedicated policy for intent '{intent.intent}'; using fallback."
            )

        return ValidationResult(is_valid=not errors, issues=errors + warnings)
