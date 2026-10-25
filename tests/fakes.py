"""A fake OpenAI-compatible endpoint, served through httpx.MockTransport. No network is used."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from sika_guard.evaluators.llm import LLMClient, LLMConfig

SECRET = "sk-test-SECRET-do-not-leak"
ROLE_MARKERS = {
    "generalist": "general-purpose financial-security analyst",
    "rubric_generalist": "rubric-based financial-security analyst",
    "merchant_claim": "merchant-claim analyst",
    "credit_terms": "credit-terms analyst",
    "authority": "authority analyst",
}


def make_config(**overrides: Any) -> LLMConfig:
    fields: dict[str, Any] = dict(base_url="https://llm.example.test/v1", api_key=SECRET, model="test-model", timeout_seconds=5.0)
    return LLMConfig(**{**fields, **overrides})


def analysis_json(verdict: str | None = "ALLOW", **overrides: Any) -> str:
    body: dict[str, Any] = dict(
        verdict=verdict, findings=[], extracted_claims=[], risk_score=0.1, abstain=False, abstain_reason=None, explanation="fine"
    )
    return json.dumps({**body, **overrides})


def completion(content: str | None, *, usage: dict[str, int] | None = None, **message: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": content, **message}, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    return body


def role_of(request_body: dict[str, Any]) -> str:
    system = request_body["messages"][0]["content"]
    return next(role for role, marker in ROLE_MARKERS.items() if marker in system)


Responder = Callable[[str, dict[str, Any]], "httpx.Response | dict[str, Any] | Awaitable[httpx.Response | dict[str, Any]]"]


class FakeProvider:
    """Records every request; `responder(role, body)` decides each reply."""

    def __init__(self, responder: Responder | None = None):
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.responder = responder or (lambda role, body: completion(analysis_json("ALLOW")))

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(request)
        self.bodies.append(body)
        result = self.responder(role_of(body), body)
        if hasattr(result, "__await__"):
            result = await result
        return result if isinstance(result, httpx.Response) else httpx.Response(200, json=result)

    def client(self, **config_overrides: Any) -> LLMClient:
        return self.client_for(make_config(**config_overrides))

    def client_for(self, config: LLMConfig) -> LLMClient:
        return LLMClient(config, transport=httpx.MockTransport(self._handle))

    def bodies_for(self, role: str) -> list[dict[str, Any]]:
        return [b for b in self.bodies if role_of(b) == role]
