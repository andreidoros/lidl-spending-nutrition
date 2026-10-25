"""Opt-in configuration for production AI enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ai_eval.providers import (
    GROQ_DEFAULT_MODEL,
    FreeModelError,
    get_api_key,
    get_setting,
    validate_groq_model,
)

AI_MODES = frozenset({"off", "categories", "categories_and_nutrition"})
DEFAULT_BATCH_SIZE = 20
MAX_BATCH_SIZE = 50
GLOBAL_BUDGET_SECONDS = 5 * 60
MAX_REQUEST_ATTEMPTS = 3
REQUEST_TIMEOUT_SECONDS = 30.0


class AIEnrichmentConfigError(ValueError):
    """Safe-to-display AI configuration problem; contains no secret values."""


@dataclass(frozen=True)
class AIEnrichmentConfig:
    mode: str = "off"
    model: str = GROQ_DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    budget_seconds: float = GLOBAL_BUDGET_SECONDS
    request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS
    max_attempts: int = MAX_REQUEST_ATTEMPTS
    api_key: str | None = field(default=None, repr=False, compare=False)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def nutrition_enabled(self) -> bool:
        return self.mode == "categories_and_nutrition"


def load_config(
    *,
    env_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AIEnrichmentConfig:
    """Resolve opt-in settings without mutating process environment.

    Production reads the normal environment / project ``.env`` through the
    existing evaluation provider helpers. Tests can inject an environment and
    temporary env-file path, never the user's credentials.
    """
    raw_mode = get_setting(
        "AI_ENRICHMENT_MODE", "off", env_path=env_path, environ=environ
    )
    mode = (raw_mode or "off").strip().lower()
    if mode not in AI_MODES:
        raise AIEnrichmentConfigError(
            "AI_ENRICHMENT_MODE must be off, categories, or categories_and_nutrition."
        )

    raw_batch_size = get_setting(
        "AI_ENRICHMENT_BATCH_SIZE",
        str(DEFAULT_BATCH_SIZE),
        env_path=env_path,
        environ=environ,
    )
    try:
        batch_size = int(raw_batch_size or DEFAULT_BATCH_SIZE)
    except (TypeError, ValueError) as exc:
        raise AIEnrichmentConfigError("AI_ENRICHMENT_BATCH_SIZE must be an integer.") from exc
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise AIEnrichmentConfigError(
            f"AI_ENRICHMENT_BATCH_SIZE must be between 1 and {MAX_BATCH_SIZE}."
        )

    raw_model = get_setting(
        "GROQ_MODEL", GROQ_DEFAULT_MODEL, env_path=env_path, environ=environ
    )
    try:
        model = validate_groq_model(raw_model, allow_nonfree=False)
    except FreeModelError as exc:
        raise AIEnrichmentConfigError(
            "GROQ_MODEL is not on the configured free-model allowlist."
        ) from exc

    api_key = get_api_key("GROQ_API_KEY", env_path=env_path, environ=environ)
    return AIEnrichmentConfig(mode=mode, model=model, batch_size=batch_size, api_key=api_key)
