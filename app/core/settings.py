"""Application configuration loaded from environment variables.

All runtime knobs live here so the rest of the code never reads from os.environ
directly. Defaults are tuned for the Colab-only deployment described in
`plan.md` (Scenario C).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. All fields can be overridden via env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Ollama (response drafting LLM) ---
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama HTTP server.",
    )
    ollama_model: str = Field(
        default="gpt-oss:20b",
        description="Model tag served by Ollama for drafting.",
    )
    ollama_timeout: float = Field(
        default=120.0,
        description="Per-request timeout in seconds.",
    )

    # --- Intent detection (Lab 2 fine-tuned model) ---
    intent_mode: Literal["unsloth", "mock"] = Field(
        default="unsloth",
        description="`unsloth` loads the real LoRA adapter; `mock` uses keyword rules.",
    )
    intent_config_path: Path = Field(
        default=PROJECT_ROOT / "configs" / "inference.yaml",
        description="Path to the Lab 2 inference YAML (model_path, labels_path...).",
    )
    intent_labels_path: Path = Field(
        default=PROJECT_ROOT / "sample_data" / "labels.txt",
        description="Path to the 77-line label list used for mock + fuzzy match.",
    )

    # --- Behaviour toggles ---
    mock_llm: bool = Field(
        default=False,
        description="If True, skip Ollama and return a canned draft.",
    )

    # --- Validation thresholds ---
    min_draft_length: int = Field(default=20, ge=1)
    intent_confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached `Settings` instance (env is read only once)."""

    return Settings()
