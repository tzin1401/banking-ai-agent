"""Response drafting node.

Calls the configured LLM (Ollama / `gpt-oss:20b` by default) to generate a
customer-facing reply, grounded in:

* the original customer message
* the predicted intent
* the assessed priority level
* the retrieved policy snippet

The LLM is instructed to return a small JSON object so the orchestrator can
extract `draft_reply`, `missing_info`, and a `suggested_action` cleanly. If
parsing fails, the raw text is used as the draft and the failure is recorded
on the result.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.clients.base import BaseLLMClient
from app.core.schemas import DraftResult, PolicyResult, PriorityResult
from app.data.policies import Policy, get_policy

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a senior customer-support agent for an online bank. "
    "Your goal is to write a short, professional, empathetic reply to the "
    "customer based on the supplied policy. Never invent policy details that "
    "are not stated in the snippet. If important information is missing from "
    "the customer's message, list it in `missing_info` instead of guessing."
)


USER_PROMPT_TEMPLATE = """\
Context:
- Detected intent: {intent}
- Priority level: {priority} ({priority_reason})
- Relevant policy ({policy_title}):
\"\"\"
{policy_content}
\"\"\"

Customer message:
\"\"\"
{message}
\"\"\"

Write the reply as a JSON object with EXACTLY these keys:
- "draft_reply": a 2-4 sentence reply to the customer.
- "missing_info": a list of strings naming any info still needed (empty list if none).
- "suggested_action": a one-sentence next step for the support team.

Return ONLY the JSON object, no preamble, no markdown fences.
"""


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_json(text: str) -> str:
    """Extract a JSON object from arbitrary LLM output."""

    text = text.strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        return fence.group(1)
    obj = _JSON_OBJECT_RE.search(text)
    if obj:
        return obj.group(0)
    return text


def _parse_json(raw: str) -> dict[str, Any] | None:
    candidate = _strip_json(raw)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


class DraftNode:
    """Generate a draft reply using the configured LLM client."""

    def __init__(self, llm: BaseLLMClient, max_tokens: int = 512) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        message: str,
        intent: str,
        priority: PriorityResult,
        policy: PolicyResult,
    ) -> str:
        snippet: Policy = get_policy(intent)
        return USER_PROMPT_TEMPLATE.format(
            intent=intent,
            priority=priority.level,
            priority_reason=priority.reason,
            policy_title=snippet.title,
            policy_content=policy.content,
            message=message.strip(),
        )

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        message: str,
        intent: str,
        priority: PriorityResult,
        policy: PolicyResult,
    ) -> DraftResult:
        prompt = self._build_prompt(message, intent, priority, policy)

        try:
            raw = self.llm.generate(
                prompt,
                system=SYSTEM_PROMPT,
                max_tokens=self.max_tokens,
                temperature=0.2,
            )
        except Exception as exc:
            logger.exception("LLM generate failed.")
            return DraftResult(
                draft_reply=(
                    "We're currently unable to draft a personalised reply. "
                    "A human agent will follow up with you shortly."
                ),
                missing_info=["LLM call failed"],
                suggested_action=f"Manual reply needed (LLM error: {exc})",
                raw_model_output=None,
            )

        parsed = _parse_json(raw)
        if parsed is None:
            logger.warning("Could not parse LLM JSON output; using raw text.")
            return DraftResult(
                draft_reply=raw.strip(),
                missing_info=[],
                suggested_action=get_policy(intent).suggested_action,
                raw_model_output=raw,
            )

        draft_reply = str(parsed.get("draft_reply", "")).strip()
        missing_info_raw = parsed.get("missing_info", []) or []
        if isinstance(missing_info_raw, str):
            missing_info_raw = [missing_info_raw]
        missing_info = [str(item).strip() for item in missing_info_raw if str(item).strip()]
        suggested_action = str(
            parsed.get("suggested_action") or get_policy(intent).suggested_action
        ).strip()

        if not draft_reply:
            draft_reply = raw.strip()

        return DraftResult(
            draft_reply=draft_reply,
            missing_info=missing_info,
            suggested_action=suggested_action,
            raw_model_output=raw,
        )
