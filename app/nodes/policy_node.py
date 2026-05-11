"""Policy retrieval node.

Looks up the snippet associated with a predicted intent. In a real system
this would query a vector store or knowledge base. Here we use the static
registry in `app/data/policies.py`.
"""

from __future__ import annotations

from app.core.schemas import PolicyResult
from app.data.policies import get_policy, has_policy


class PolicyNode:
    """Resolve an intent into a structured policy result."""

    def run(self, intent: str) -> PolicyResult:
        policy = get_policy(intent)
        return PolicyResult(
            policy_id=policy.intent if has_policy(intent) else None,
            content=policy.content,
            found=has_policy(intent),
        )
