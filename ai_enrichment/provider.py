"""Cancellable Groq transport used by the production enrichment stage."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import time
from typing import Any, Mapping, Sequence

import httpx

from ai_eval.providers import (
    GROQ_DEFAULT_MODEL,
    GROQ_ENDPOINT,
    MAX_ATTEMPTS_LIMIT,
    MAX_RETRY_AFTER_SECONDS,
    ProviderResponse,
    validate_groq_model,
)


class GroqProductionClient:
    """Synchronous Groq client with cancellable, total-duration request limits.

    ``asyncio.wait_for`` cancels an in-flight HTTPX request at the absolute
    deadline. This differs from a Requests read timeout, which only limits
    inactivity and can be extended indefinitely by a slow trickle of bytes.
    ``transport`` is an injection seam for deterministic HTTPX mock tests.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str | None = None,
        timeout: float = 30.0,
        max_attempts: int = 3,
        sleep_fn: Any | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = validate_groq_model(model or GROQ_DEFAULT_MODEL, allow_nonfree=False)
        self.timeout = max(float(timeout), 0.05)
        self.max_attempts = max(1, min(int(max_attempts), MAX_ATTEMPTS_LIMIT))
        self.sleep_fn = sleep_fn or time.sleep
        self.transport = transport

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int = 4096,
        *,
        deadline: float | None = None,
    ) -> ProviderResponse:
        started = time.perf_counter()
        if not self.api_key:
            return self._failure("missing_api_key", 0, started)

        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        return asyncio.run(self._chat_async(payload, deadline, started))

    async def _chat_async(
        self,
        payload: Mapping[str, object],
        deadline: float | None,
        started: float,
    ) -> ProviderResponse:
        last_kind = "request_error"
        last_status: int | None = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            transport=self.transport,
        ) as client:
            for attempt in range(1, self.max_attempts + 1):
                remaining = self.timeout
                if deadline is not None:
                    remaining = min(remaining, deadline - time.monotonic())
                    if remaining <= 0:
                        return self._failure("deadline_exceeded", attempt - 1, started, last_status)
                try:
                    request = self._request_once(client, headers, payload, remaining)
                    if deadline is None:
                        status, response_headers, body = await asyncio.wait_for(
                            request, timeout=remaining
                        )
                    else:
                        status, response_headers, body = await asyncio.wait_for(
                            request, timeout=remaining
                        )
                except asyncio.TimeoutError:
                    if deadline is not None and time.monotonic() >= deadline:
                        return self._failure("deadline_exceeded", attempt, started, last_status)
                    last_kind = "timeout"
                    status = None
                    response_headers = {}
                    body = b""
                except httpx.TimeoutException:
                    last_kind = "timeout"
                    status = None
                    response_headers = {}
                    body = b""
                except httpx.RemoteProtocolError:
                    # A remote peer closing/resetting an otherwise valid HTTP
                    # exchange is a transient transport failure. Keep local
                    # protocol/configuration errors terminal below.
                    last_kind = "connection_error"
                    status = None
                    response_headers = {}
                    body = b""
                except httpx.NetworkError:
                    last_kind = "connection_error"
                    status = None
                    response_headers = {}
                    body = b""
                except httpx.RequestError:
                    return self._failure("request_error", attempt, started, last_status)
                else:
                    last_status = status
                    if status >= 400:
                        last_kind = "http_error"
                        if not (status == 429 or status >= 500):
                            return self._failure(last_kind, attempt, started, status)
                        retry_after = _retry_after_seconds(response_headers.get("retry-after"))
                        if retry_after is not None and retry_after > MAX_RETRY_AFTER_SECONDS:
                            return self._failure("rate_limited", attempt, started, status)
                    else:
                        try:
                            envelope = json.loads(body)
                        except (ValueError, UnicodeDecodeError, RecursionError):
                            return self._failure("invalid_json_response", attempt, started, status)
                        content = _extract_content(envelope)
                        if content is None:
                            kind = "empty_content" if _has_empty_content(envelope) else "unexpected_response_shape"
                            return self._failure(kind, attempt, started, status)
                        return ProviderResponse(
                            content=content,
                            status_code=status or None,
                            attempts=attempt,
                            latency_ms=(time.perf_counter() - started) * 1000.0,
                        )

                if attempt >= self.max_attempts:
                    break
                advised = _retry_after_seconds(response_headers.get("retry-after"))
                delay = max(float(attempt), advised or 0.0)
                if deadline is not None and delay >= deadline - time.monotonic():
                    return self._failure("deadline_exceeded", attempt, started, last_status)
                if self.sleep_fn is time.sleep:
                    await asyncio.sleep(delay)
                else:
                    self.sleep_fn(delay)

        return self._failure(last_kind, self.max_attempts, started, last_status)

    async def _request_once(
        self,
        client: httpx.AsyncClient,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout: float,
    ) -> tuple[int, httpx.Headers, bytes]:
        async with client.stream(
            "POST",
            GROQ_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=httpx.Timeout(timeout),
        ) as response:
            content = await response.aread()
            return response.status_code, response.headers, content

    @staticmethod
    def _failure(
        kind: str,
        attempts: int,
        started: float,
        status_code: int | None = None,
    ) -> ProviderResponse:
        # Persist/report only stable error codes. Never retain provider bodies,
        # which may contain user data or credential-shaped strings.
        return ProviderResponse(
            content=None,
            status_code=status_code,
            attempts=attempts,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=f"Groq request failed ({kind}).",
            error_kind=kind,
        )


def _extract_content(envelope: object) -> str | None:
    if not isinstance(envelope, Mapping):
        return None
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return None


def _has_empty_content(envelope: object) -> bool:
    if not isinstance(envelope, Mapping):
        return False
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return False
    message = choices[0].get("message")
    return isinstance(message, Mapping) and message.get("content") in (None, "")


def _retry_after_seconds(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    try:
        return max(float(text), 0.0)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)
