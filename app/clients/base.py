"""Abstract base class for LLM clients.

Even though Lab 3 only uses Ollama, the workflow goes through this interface
so we can later swap in a different LLM (e.g. an OpenAI-compatible endpoint)
without touching the node code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Minimal LLM client contract used by the drafting node."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Return the model's reply for `prompt`.

        Implementations should raise an exception (rather than returning an
        empty string) on hard failures so the orchestrator can decide whether
        to retry or escalate.
        """

    def health(self) -> bool:
        """Best-effort liveness probe. Default returns True."""

        return True
