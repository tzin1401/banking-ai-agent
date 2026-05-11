"""HTTP client for an Ollama server.

The client uses Ollama's `/api/chat` endpoint which is the most natural fit
for instruction-tuned models like `gpt-oss:20b`. A `MockOllamaClient` is also
provided for offline development.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from app.clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """Real HTTP client talking to `ollama serve`."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        extra: dict[str, Any] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if extra:
            payload["options"].update(extra)

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "").strip()
                if not content:
                    raise RuntimeError(f"Empty content in Ollama response: {data}")
                return content
            except (requests.RequestException, RuntimeError) as exc:
                last_exc = exc
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))

        assert last_exc is not None
        raise RuntimeError(f"Ollama call failed after retries: {last_exc}")

    def health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


class MockOllamaClient(BaseLLMClient):
    """Offline stand-in for `OllamaClient`.

    Returns a small piece of JSON-shaped text so the drafting node's parser
    can be exercised without a real LLM. Activated by setting `MOCK_LLM=1`.
    """

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        extra: dict[str, Any] | None = None,
    ) -> str:
        return (
            '{"draft_reply": "Thanks for reaching out. We have located your '
            "request and our team will follow up shortly. If you need urgent "
            'help, please reply with your account number.", '
            '"missing_info": [], '
            '"suggested_action": "Acknowledge and await customer reply."}'
        )

    def health(self) -> bool:
        return True
