"""Free-tier-only provider clients for the evaluation harness.

No production integration lives here. All network activity is explicit and
only happens from :meth:`chat` or the two listing helpers.
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"

# Documented Groq free-tier chat models. Extend only with a documented entry.
GROQ_FREE_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3.8-27b",
)
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"

RETRYABLE_STATUS_CODES = frozenset({429})
MAX_RETRY_AFTER_SECONDS = 120.0
MAX_ATTEMPTS_LIMIT = 3
# Provider error payloads must never be dumped wholesale into results.jsonl.
_MAX_ERROR_MESSAGE_CHARS = 500
# Stable OpenRouter ``error.metadata`` fields that distinguish rate limits from
# provider overload or a guardrail block. ``raw`` is deliberately excluded.
_DIAGNOSTIC_FIELDS = ("error_type", "provider_code", "provider_name")

_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+")
_API_KEY_RE = re.compile(r"\b(?:sk|gsk)[-_][A-Za-z0-9._\-]{6,}")


class ProviderError(RuntimeError):
    """Base class for provider configuration and request errors."""


class FreeModelError(ProviderError):
    """Raised when a model is not on an allowed free-tier path."""


class MissingApiKeyError(ProviderError):
    """Raised when a required API key is not configured."""


def _is_retryable_status(status: int) -> bool:
    """429 and every 5xx are transient; 4xx (except 429) are not."""
    return status in RETRYABLE_STATUS_CODES or 500 <= status <= 599


def redact_text(text: object, secrets: Sequence[str] = ()) -> str:
    """Remove known secrets and common API-key patterns from text."""
    if not isinstance(text, str):
        return str(text)
    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "***")
    cleaned = _BEARER_RE.sub("Bearer ***", cleaned)
    cleaned = _API_KEY_RE.sub("***", cleaned)
    return cleaned


def parse_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Minimal ``KEY=VALUE`` parser; never mutates ``os.environ``."""
    location = Path(path) if path is not None else DEFAULT_ENV_PATH
    try:
        raw = location.read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def get_api_key(
    name: str,
    env_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Environment first, then ``.env``. Returns ``None`` when absent."""
    environment = os.environ if environ is None else environ
    value = environment.get(name)
    if value:
        return value
    return parse_env_file(env_path).get(name) or None


def get_setting(
    name: str,
    default: str | None = None,
    env_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Non-secret setting: environment first, then ``.env``, then ``default``."""
    value = get_api_key(name, env_path=env_path, environ=environ)
    return default if value is None else value


def validate_groq_model(model: str | None, allow_nonfree: bool = False) -> str:
    if not isinstance(model, str) or not model.strip():
        raise FreeModelError("A Groq model ID must be specified.")
    if model not in GROQ_FREE_MODELS and not allow_nonfree:
        raise FreeModelError(
            f"Groq model {model!r} is not on the documented free-tier "
            f"allowlist ({', '.join(GROQ_FREE_MODELS)}). Pass "
            "--allow-nonfree-model only if you explicitly accept possible "
            "billing."
        )
    return model


def validate_openrouter_model(model: str | None) -> str:
    if not isinstance(model, str) or not model.strip():
        raise FreeModelError(
            "No OpenRouter model configured. Run --list-free-models, then set "
            "OPENROUTER_MODEL or pass --model with an ID ending in ':free'."
        )
    if model == "openrouter/free" or model.endswith(":free"):
        return model
    raise FreeModelError(
        f"OpenRouter model {model!r} is not free. Use an ID ending in ':free' "
        "or 'openrouter/free'. Run --list-free-models to discover current "
        "free models."
    )


@dataclass(frozen=True)
class ProviderResponse:
    content: str | None
    status_code: int | None
    attempts: int
    latency_ms: float
    error: str | None = None
    error_kind: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.content is not None


class ProviderClient:
    provider = "base"
    endpoint = ""
    key_env = ""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 90.0,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        session: Any | None = None,
        sleep_fn: Any | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else get_api_key(self.key_env)
        self.model = model
        self.timeout = timeout
        self.max_attempts = max(1, min(int(max_attempts), MAX_ATTEMPTS_LIMIT))
        self.base_delay = max(0.0, base_delay)
        self.session = session if session is not None else requests.Session()
        self.sleep_fn = sleep_fn if sleep_fn is not None else time.sleep
        self.rng = rng if rng is not None else random.Random()

    def build_payload(
        self,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0,
            "max_tokens": max_tokens,
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
        }

    def _error_body(self, response: Any) -> str:
        try:
            payload = response.json()
        except (ValueError, AttributeError):
            payload = None
        message: object = None
        diagnostics: list[str] = []
        if isinstance(payload, Mapping):
            error = payload.get("error")
            if isinstance(error, Mapping):
                message = error.get("message") or error.get("type")
                metadata = error.get("metadata")
                if isinstance(metadata, Mapping):
                    for field_name in _DIAGNOSTIC_FIELDS:
                        value = metadata.get(field_name)
                        if value:
                            diagnostics.append(f"{field_name}={value}")
            elif error:
                message = error
            elif payload.get("message"):
                message = payload.get("message")
        if message is None:
            text = getattr(response, "text", "") or ""
            message = text[:_MAX_ERROR_MESSAGE_CHARS]
        composed = str(message)
        if diagnostics:
            composed = f"{composed} [{', '.join(diagnostics)}]"
        # Redact the whole composed message before truncating, so cutting at
        # the length limit can never leave a partial key behind. Headers are
        # never read or included.
        composed = redact_text(composed, (self.api_key or "",))
        return composed[:_MAX_ERROR_MESSAGE_CHARS]

    @staticmethod
    def _retry_after(response: Any) -> float | None:
        """Advised wait in seconds, never clipped to the retry budget.

        Callers compare it against :data:`MAX_RETRY_AFTER_SECONDS` and stop
        instead of retrying earlier than the provider allows.
        """
        headers = getattr(response, "headers", None) or {}
        raw = headers.get("Retry-After")
        if raw is None:
            return None
        text = str(raw).strip()
        try:
            seconds = float(text)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(text)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
        return max(seconds, 0.0)

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        base = self.base_delay * (2 ** (attempt - 1))
        delay = base + self.rng.uniform(0.0, self.base_delay)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay

    @staticmethod
    def _extract_content(payload: Any) -> str | None:
        if not isinstance(payload, Mapping):
            return None
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, Mapping):
            return None
        message = first.get("message")
        if not isinstance(message, Mapping):
            return None
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, Mapping):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts) if parts else None
        return None

    @staticmethod
    def _first_choice(payload: Any) -> Mapping[str, Any] | None:
        if not isinstance(payload, Mapping):
            return None
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        return first if isinstance(first, Mapping) else None

    @staticmethod
    def _is_empty_content(value: Any) -> bool:
        """True for a present-but-empty content value (``null`` or ``""``)."""
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, list):
            return not any(
                isinstance(part, Mapping)
                and isinstance(part.get("text"), str)
                and part.get("text").strip()
                for part in value
            )
        return False

    @classmethod
    def _empty_content_message(cls, payload: Any, status: int | None) -> str:
        details: list[str] = []
        first = cls._first_choice(payload)
        finish_reason = first.get("finish_reason") if first is not None else None
        if finish_reason is not None:
            details.append(f"finish_reason={finish_reason}")
        usage = payload.get("usage") if isinstance(payload, Mapping) else None
        completion_details = (
            usage.get("completion_tokens_details")
            if isinstance(usage, Mapping)
            else None
        )
        reasoning_tokens = (
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, Mapping)
            else None
        )
        if reasoning_tokens is not None:
            details.append(f"reasoning_tokens={reasoning_tokens}")
        suffix = f" ({', '.join(details)})" if details else ""
        status_text = f" for HTTP {status}" if status else ""
        return (
            f"Provider returned an empty choices[0].message.content{status_text}"
            f"{suffix}. Retrying is not expected to help; the model likely "
            "spent its whole token budget on reasoning."
        )

    def _failure(
        self,
        kind: str,
        message: str,
        attempts: int,
        started: float,
        status_code: int | None = None,
    ) -> ProviderResponse:
        return ProviderResponse(
            content=None,
            status_code=status_code,
            attempts=attempts,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=redact_text(message, (self.api_key or "",)),
            error_kind=kind,
        )

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int = 4096,
        *,
        deadline: float | None = None,
    ) -> ProviderResponse:
        started = time.perf_counter()
        if not self.api_key:
            return self._failure(
                "missing_api_key",
                f"{self.key_env} is not configured (environment or .env).",
                0,
                started,
            )

        payload = self.build_payload(messages, max_tokens=max_tokens)
        last_kind = "request_error"
        last_message = "Provider request failed."
        last_status: int | None = None

        for attempt in range(1, self.max_attempts + 1):
            if deadline is not None and time.monotonic() >= deadline:
                return self._failure(
                    "deadline_exceeded",
                    "Global enrichment time budget expired before the request.",
                    attempt - 1,
                    started,
                    last_status,
                )
            retry_after: float | None = None
            try:
                request_timeout = self.timeout
                if deadline is not None:
                    remaining = max(deadline - time.monotonic(), 0.05)
                    request_timeout = min(request_timeout, remaining)
                response = self.session.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=payload,
                    timeout=request_timeout,
                )
            except requests.Timeout:
                last_kind = "timeout"
                last_message = f"Request timed out after {self.timeout:g}s."
                last_status = None
            except requests.ConnectionError as exc:
                last_kind = "connection_error"
                last_message = f"Connection error: {exc}"
                last_status = None
            except requests.RequestException as exc:
                # InvalidHeader, TooManyRedirects, MissingSchema, ... are not
                # transient: retrying them only burns quota.
                return self._failure(
                    "request_error",
                    f"Non-retryable request error: {exc}",
                    attempt,
                    started,
                )
            else:
                status = int(getattr(response, "status_code", 0) or 0)
                last_status = status or None
                if status >= 400:
                    last_kind = "http_error"
                    last_message = f"HTTP {status}: {self._error_body(response)}"
                    if not _is_retryable_status(status):
                        return self._failure(
                            last_kind, last_message, attempt, started, status
                        )
                    retry_after = self._retry_after(response)
                    if (
                        retry_after is not None
                        and retry_after > MAX_RETRY_AFTER_SECONDS
                    ):
                        return self._failure(
                            "rate_limited",
                            f"Provider advised waiting {retry_after:.0f}s before "
                            f"retrying, above the "
                            f"{MAX_RETRY_AFTER_SECONDS:g}s budget; stopped "
                            "instead of retrying earlier than advised.",
                            attempt,
                            started,
                            status,
                        )
                else:
                    try:
                        body = response.json()
                    except (ValueError, AttributeError):
                        return self._failure(
                            "invalid_json_response",
                            "Provider returned a non-JSON body.",
                            attempt,
                            started,
                            status or None,
                        )
                    content = self._extract_content(body)
                    if content is None or not content.strip():
                        first = self._first_choice(body)
                        message = first.get("message") if first is not None else None
                        if isinstance(message, Mapping) and self._is_empty_content(
                            message.get("content")
                        ):
                            # A 200 with empty content (reasoning models that
                            # exhaust their budget) is not a malformed payload
                            # and retrying does not help: classify it, once.
                            return self._failure(
                                "empty_content",
                                self._empty_content_message(body, status or None),
                                attempt,
                                started,
                                status or None,
                            )
                        return self._failure(
                            "unexpected_response_shape",
                            "Provider response did not contain "
                            "choices[0].message.content.",
                            attempt,
                            started,
                            status or None,
                        )
                    return ProviderResponse(
                        content=content,
                        status_code=status or None,
                        attempts=attempt,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                    )

            if attempt >= self.max_attempts:
                break
            delay = self._backoff_delay(attempt, retry_after)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or delay >= remaining:
                    return self._failure(
                        "deadline_exceeded",
                        "Global enrichment time budget does not allow another retry.",
                        attempt,
                        started,
                        last_status,
                    )
            self.sleep_fn(delay)

        return self._failure(
            last_kind, last_message, self.max_attempts, started, last_status
        )


class GroqClient(ProviderClient):
    provider = "groq"
    endpoint = GROQ_ENDPOINT
    key_env = "GROQ_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        allow_nonfree_model: bool = False,
        **kwargs: Any,
    ) -> None:
        resolved_model = model or GROQ_DEFAULT_MODEL
        validate_groq_model(resolved_model, allow_nonfree=allow_nonfree_model)
        self.allow_nonfree_model = allow_nonfree_model
        super().__init__(api_key=api_key, model=resolved_model, **kwargs)


class OpenRouterClient(ProviderClient):
    provider = "openrouter"
    endpoint = OPENROUTER_ENDPOINT
    key_env = "OPENROUTER_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_model = validate_openrouter_model(model)
        super().__init__(api_key=api_key, model=resolved_model, **kwargs)


def _is_zero_price(value: object) -> bool:
    if value is None:
        return False
    try:
        return float(str(value).strip()) == 0.0
    except (TypeError, ValueError):
        return False


def _openrouter_model_is_usable(model_id: object) -> bool:
    try:
        validate_openrouter_model(str(model_id or ""))
    except FreeModelError:
        return False
    return True


def list_openrouter_free_models(
    session: Any | None = None,
    timeout: float = 30.0,
) -> list[dict[str, object]]:
    """Live listing of models whose prompt and completion prices are zero.

    Every zero-priced model is returned (none are hidden), with ``usable``
    telling whether :func:`validate_openrouter_model` would accept the ID.
    Zero-priced IDs without the ``:free`` suffix are listed but rejected by
    the guard, so the flag matters.
    """
    http = session if session is not None else requests
    try:
        response = http.get(OPENROUTER_MODELS_ENDPOINT, timeout=timeout)
    except requests.RequestException as exc:
        raise ProviderError(f"Could not reach OpenRouter models endpoint: {exc}") from exc
    if int(getattr(response, "status_code", 0) or 0) >= 400:
        raise ProviderError(
            f"OpenRouter models endpoint returned HTTP {response.status_code}."
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError("OpenRouter models endpoint returned invalid JSON.") from exc

    items = payload.get("data") if isinstance(payload, Mapping) else payload
    if not isinstance(items, list):
        raise ProviderError("Unexpected OpenRouter models payload shape.")

    free_models: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        pricing = item.get("pricing") or {}
        if not isinstance(pricing, Mapping):
            continue
        if _is_zero_price(pricing.get("prompt")) and _is_zero_price(
            pricing.get("completion")
        ):
            free_models.append(
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "context_length": item.get("context_length"),
                    "prompt_price": pricing.get("prompt"),
                    "completion_price": pricing.get("completion"),
                    "usable": _openrouter_model_is_usable(item.get("id")),
                }
            )
    return sorted(free_models, key=lambda model: str(model.get("id") or ""))


def list_groq_models(
    api_key: str | None = None,
    session: Any | None = None,
    timeout: float = 30.0,
) -> list[str]:
    """Live listing of Groq model IDs (requires a key)."""
    key = api_key if api_key is not None else get_api_key("GROQ_API_KEY")
    if not key:
        raise MissingApiKeyError(
            "GROQ_API_KEY is not configured; cannot list Groq models."
        )
    http = session if session is not None else requests
    try:
        response = http.get(
            GROQ_MODELS_ENDPOINT,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise ProviderError(f"Could not reach Groq models endpoint: {exc}") from exc
    if int(getattr(response, "status_code", 0) or 0) >= 400:
        raise ProviderError(f"Groq models endpoint returned HTTP {response.status_code}.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError("Groq models endpoint returned invalid JSON.") from exc
    items = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(items, list):
        raise ProviderError("Unexpected Groq models payload shape.")
    return sorted(
        str(item.get("id"))
        for item in items
        if isinstance(item, Mapping) and item.get("id")
    )
