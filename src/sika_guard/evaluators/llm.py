"""Provider-configurable, OpenAI-compatible chat-completions client and the LLM analyst.

Configuration comes from environment variables; nothing here names a vendor, model or endpoint.
Secrets are never printed or included in error text.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Self
from urllib.parse import urlsplit

import httpx

from sika_guard.evaluators.contract import (
    CallRecord,
    CallStatus,
    EvaluatorRole,
    EvaluatorStatus,
    EvaluatorVerdict,
    EvaluatorView,
    InvalidResponse,
    failure_verdict,
    parse_analysis_normalised,
    verdict_from_analysis,
)
from sika_guard.evaluators.payloads import collect_refs
from sika_guard.evaluators.prompts import GENERALIST, RUBRIC_GENERALIST, SPECIALISTS, RoleSpec, specs_for

ENV_BASE_URL = "SIKA_GUARD_LLM_BASE_URL"
ENV_API_KEY = "SIKA_GUARD_LLM_API_KEY"
ENV_MODEL = "SIKA_GUARD_LLM_MODEL"
ENV_TIMEOUT = "SIKA_GUARD_LLM_TIMEOUT_SECONDS"  # optional, default 60
ENV_TEMPERATURE = "SIKA_GUARD_LLM_TEMPERATURE"  # optional; omitted from requests when unset
ENV_JSON_MODE = "SIKA_GUARD_LLM_JSON_MODE"  # optional; "0" stops sending response_format
ENV_PRICE_IN = "SIKA_GUARD_LLM_PRICE_INPUT_PER_MTOK"  # optional; both prices needed to report cost
ENV_PRICE_OUT = "SIKA_GUARD_LLM_PRICE_OUTPUT_PER_MTOK"
REQUIRED_ENV = (ENV_BASE_URL, ENV_API_KEY, ENV_MODEL)

DEFAULT_TIMEOUT_SECONDS = 60.0
RAW_USAGE_CAP = 4000  # characters of JSON


class ConfigError(Exception):
    """Configuration is missing or malformed. `missing` lists the variable names, never values."""

    def __init__(self, message: str, missing: tuple[str, ...] = ()):
        super().__init__(message)
        self.missing = missing


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = field(repr=False)
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    temperature: float | None = None
    json_mode: bool = True
    price_input_per_mtok: Decimal | None = None
    price_output_per_mtok: Decimal | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Self:
        env = os.environ if env is None else env
        missing = tuple(name for name in REQUIRED_ENV if not env.get(name, "").strip())
        if missing:
            raise ConfigError("missing environment variables: " + ", ".join(missing), missing)
        try:
            return cls(
                base_url=env[ENV_BASE_URL].strip(),
                api_key=env[ENV_API_KEY].strip(),
                model=env[ENV_MODEL].strip(),
                timeout_seconds=float(env.get(ENV_TIMEOUT) or DEFAULT_TIMEOUT_SECONDS),
                temperature=float(env[ENV_TEMPERATURE]) if env.get(ENV_TEMPERATURE) else None,
                json_mode=env.get(ENV_JSON_MODE, "1").strip() != "0",
                price_input_per_mtok=Decimal(env[ENV_PRICE_IN]) if env.get(ENV_PRICE_IN) else None,
                price_output_per_mtok=Decimal(env[ENV_PRICE_OUT]) if env.get(ENV_PRICE_OUT) else None,
            )
        except (ValueError, InvalidOperation) as exc:
            raise ConfigError(f"a numeric environment variable is malformed ({type(exc).__name__})") from None

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def describe(self) -> dict[str, Any]:
        """Safe to print or save: no key, and the URL reduced to scheme, host and path."""
        parts = urlsplit(self.base_url)
        return {
            "base_url": f"{parts.scheme}://{parts.hostname}{parts.path}",
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "json_mode": self.json_mode,
            "pricing_configured": self.price_input_per_mtok is not None and self.price_output_per_mtok is not None,
        }

    def cost(self, prompt_tokens: int | None, completion_tokens: int | None) -> Decimal | None:
        """Cost from configured prices and provider-returned counts only; otherwise None."""
        if None in (prompt_tokens, completion_tokens, self.price_input_per_mtok, self.price_output_per_mtok):
            return None
        return (prompt_tokens * self.price_input_per_mtok + completion_tokens * self.price_output_per_mtok) / Decimal(1_000_000)


@dataclass(frozen=True)
class LLMResponse:
    call: CallRecord
    content: str | None = None
    refusal: str | None = None
    error: str | None = None  # short, credential-free description of a provider-side failure


class LLMClient:
    """One shared async HTTP client; safe to use from concurrent tasks."""

    def __init__(self, config: LLMConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self._http = httpx.AsyncClient(timeout=config.timeout_seconds, transport=transport)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._http.aclose()

    async def complete(self, messages: list[dict[str, str]], *, evaluator_id: str) -> LLMResponse:
        cfg = self.config
        body: dict[str, Any] = {"model": cfg.model, "messages": messages}
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if cfg.json_mode:
            body["response_format"] = {"type": "json_object"}

        started_at = datetime.now(timezone.utc)  # wall clock, for ordering only
        started = time.perf_counter()  # monotonic, for latency

        def record(
            status: CallStatus,
            http_status: int | None = None,
            usage: Mapping[str, Any] | None = None,
            provider: object = None,
            returned_model: object = None,
        ) -> CallRecord:
            usage = usage or {}
            prompt, completion = _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens"))
            provider_cost = _raw_cost(usage.get("cost"))
            return CallRecord(
                evaluator_id=evaluator_id,
                model=cfg.model,
                status=status,
                latency_ms=(time.perf_counter() - started) * 1000,
                http_status=http_status,
                prompt_tokens=prompt,
                completion_tokens=completion,
                cost=cfg.cost(prompt, completion),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                provider=_short(provider),
                endpoint_host=urlsplit(cfg.base_url).hostname,
                provider_cost=provider_cost,
                provider_cost_unit="UNKNOWN" if provider_cost is not None else None,
                returned_model=_short(returned_model),
                raw_usage=_raw_usage(usage),
            )

        try:
            async with asyncio.timeout(cfg.timeout_seconds):
                response = await self._http.post(
                    cfg.endpoint, json=body, headers={"Authorization": f"Bearer {cfg.api_key}"}
                )
        except (TimeoutError, httpx.TimeoutException):
            return LLMResponse(record(CallStatus.TIMEOUT))
        except httpx.HTTPError as exc:
            return LLMResponse(record(CallStatus.PROVIDER_ERROR), error=f"transport error: {type(exc).__name__}")

        if response.status_code >= 400:
            return LLMResponse(record(CallStatus.PROVIDER_ERROR, response.status_code), error=_error_summary(response))
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            return LLMResponse(record(CallStatus.PROVIDER_ERROR, response.status_code), error="unexpected response envelope")

        usage = data.get("usage") if isinstance(data, dict) else None
        usage = usage if isinstance(usage, dict) else None
        provider = data.get("provider") if isinstance(data, dict) else None
        returned_model = data.get("model") if isinstance(data, dict) else None
        refusal = message.get("refusal") if isinstance(message, dict) else None
        if refusal or choice.get("finish_reason") == "content_filter":
            return LLMResponse(record(CallStatus.REFUSAL, response.status_code, usage, provider, returned_model), refusal=str(refusal or "content filtered")[:200])
        content = message.get("content") if isinstance(message, dict) else None
        return LLMResponse(record(CallStatus.OK, response.status_code, usage, provider, returned_model), content=content if isinstance(content, str) else None)


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _short(value: object) -> str | None:
    return value.strip()[:80] if isinstance(value, str) and value.strip() else None


def _raw_usage(usage: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The provider's usage object as returned, if it is plain JSON and small; otherwise None."""
    if not usage:
        return None
    try:
        text = json.dumps(usage, allow_nan=False)
    except (TypeError, ValueError):
        return None
    return json.loads(text) if len(text) <= RAW_USAGE_CAP else None


def _raw_cost(value: object) -> Decimal | None:
    """The provider's `usage.cost` as a number, exactly as returned. Anything else is None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    amount = Decimal(str(value))
    return amount if amount.is_finite() and amount >= 0 else None


def _error_summary(response: httpx.Response) -> str:
    """Status plus the provider's error type or code only; a body can echo credentials."""
    detail = ""
    try:
        err = response.json().get("error")
        if isinstance(err, dict):
            detail = " ".join(str(err[k])[:60] for k in ("type", "code") if err.get(k))
    except (ValueError, AttributeError):
        pass
    return f"HTTP {response.status_code} {detail}".strip()


class LLMAnalyst:
    """An evaluator backed by one chat-completions call per case."""

    def __init__(self, spec: RoleSpec, client: LLMClient):
        self.spec = spec
        self.client = client
        self.evaluator_id = spec.evaluator_id
        self.role = spec.role

    async def evaluate(self, view: EvaluatorView) -> EvaluatorVerdict:
        spec, model = self.spec, self.client.config.model
        payload = spec.build_payload(view)
        messages = [
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": spec.user_message(payload)},
        ]
        response = await self.client.complete(messages, evaluator_id=self.evaluator_id)
        common = dict(
            evaluator_id=self.evaluator_id, role=self.role, prompt_version=spec.prompt_version, model_id=model,
            calls=(response.call,), raw_response=response.content,
        )

        def fail(status: EvaluatorStatus, reason: str) -> EvaluatorVerdict:
            return failure_verdict(view, status=status, reason=reason, **common)

        match response.call.status:
            case CallStatus.TIMEOUT:
                return fail(EvaluatorStatus.TIMEOUT, f"no response within {self.client.config.timeout_seconds:g}s")
            case CallStatus.REFUSAL:
                return fail(EvaluatorStatus.REFUSAL, f"model refused: {response.refusal}")
            case CallStatus.PROVIDER_ERROR:
                return fail(EvaluatorStatus.PROVIDER_ERROR, response.error or "provider error")
        try:
            analysis, normalisations = parse_analysis_normalised(response.content)
        except InvalidResponse as exc:
            return fail(EvaluatorStatus.INVALID_RESPONSE, str(exc))
        return verdict_from_analysis(analysis, view, allowed_refs=collect_refs(payload), normalisations=normalisations, payload=payload, **common)


def generalist(client: LLMClient) -> list[LLMAnalyst]:
    """Condition B."""
    return [LLMAnalyst(GENERALIST, client)]


def specialists(client: LLMClient) -> list[LLMAnalyst]:
    """Condition C."""
    return [LLMAnalyst(spec, client) for spec in SPECIALISTS]


def rubric_generalist(client: LLMClient) -> list[LLMAnalyst]:
    """Condition Bc: one call applying all three specialist rubrics."""
    return [LLMAnalyst(RUBRIC_GENERALIST, client)]


def evaluators_for(condition: str, client: LLMClient | None) -> list:
    """The analysts for any AI condition: B, B3 (three generalist samples), Bc or C. Condition A+ is not an AI condition: its one
    evaluator is the deterministic scanner, which makes no call and needs no client."""
    if condition == "A+":
        from sika_guard.evaluators.scanner import DeterministicScanner

        return [DeterministicScanner()]
    assert client is not None, "an AI condition needs a client"
    return [LLMAnalyst(spec, client) for spec in specs_for(condition)]
