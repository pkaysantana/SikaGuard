"""A fake OpenAI-compatible provider for offline end-to-end runs. No network is used.

It plugs into the real `LLMClient` as an httpx transport, so a fake run exercises the production request
building, response parsing, validation, adjudication and coordination. Its replies are scripted caricatures
and are NOT model output: results from a fake run say nothing about any model's accuracy.

Replies are a deterministic function of the request, so a fake run is reproducible. This module must not
import the hidden-label module.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from decimal import Decimal

import httpx

from sika_guard.evaluators.payloads import collect_refs

MODES = ("quiet", "alarmist", "extractor", "flaky")
PROVIDERS = ("FakeProviderA", "FakeProviderB")


def _usage() -> dict:
    return {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.0004}


def _payload(user_message: str) -> dict:
    return json.loads(user_message[user_message.index("{"):])


def _analysis(verdict: str, *, findings=(), claims=(), risk=0.1) -> str:
    return json.dumps(
        {"verdict": verdict, "findings": list(findings), "extracted_claims": list(claims), "risk_score": risk, "abstain": False, "abstain_reason": None, "explanation": "scripted fake reply"}
    )


def _first(refs: set[str], prefix: str) -> str | None:
    return next(iter(sorted(r for r in refs if r.startswith(prefix))), None)


def _content(mode: str, payload: dict, ordinal: int) -> str:
    refs = set(collect_refs(payload))
    if mode == "alarmist":
        findings = []
        if (intent := _first(refs, "intent:")):
            findings.append({"code": "intent_mismatch", "evidence_refs": [intent], "note": "vague unease"})
        if (txn := _first(refs, "txn:")):
            findings.append({"code": "limit_exceeded", "evidence_refs": [txn], "note": "alleged, and false"})
        return _analysis("REVIEW", findings=findings, risk=0.9)
    if mode == "extractor":
        selected = payload["task_context"]["selected_offer_id"]
        listing = next((x for x in payload.get("untrusted_merchant_content", {}).get("offer_listings", []) if x["offer_id"] == selected and x.get("advertised_total_repayable")), None)
        if listing is None:
            return _analysis("ALLOW")
        amount, currency = listing["advertised_total_repayable"].split()
        claim = {"kind": "total_repayable", "offer_id": selected, "source_ref": listing["ref"], "quote": f"advertised_total_repayable: {listing['advertised_total_repayable']}",
                 "asserted_available": None, "amount_minor": int(Decimal(amount) * 100), "currency": currency}
        return _analysis("ALLOW", claims=[claim])
    return _analysis("ALLOW")


class FakeProviderTransport(httpx.AsyncBaseTransport):
    """mode: quiet (valid ALLOW), alarmist (REVIEW with a vague finding and a false limit allegation),
    extractor (ALLOW plus the selected offer's advertised total extracted verbatim), flaky (quiet, with deterministic
    timeouts, HTTP 500s and malformed replies)."""

    def __init__(self, mode: str = "quiet", model: str = "fake-model"):
        if mode not in MODES:
            raise ValueError(f"unknown fake mode {mode!r}; expected one of {MODES}")
        self.mode, self.model = mode, model
        self.requests = 0
        self._seen: Counter[str] = Counter()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        body = json.loads(request.content)
        digest = hashlib.sha256(request.content).hexdigest()
        ordinal = self._seen[digest]
        self._seen[digest] += 1
        draw = int(hashlib.sha256(f"{digest}|{ordinal}".encode()).hexdigest(), 16) % 100
        if self.mode == "flaky":
            if draw < 10:
                raise httpx.ReadTimeout("fake timeout", request=request)
            if draw < 18:
                return httpx.Response(500, json={"error": {"type": "fake_server_error"}})
            if draw < 28:
                return self._reply(body, digest, "this is not json at all")
        content = _content("quiet" if self.mode == "flaky" else self.mode, _payload(body["messages"][1]["content"]), ordinal)
        return self._reply(body, digest, content)

    def _reply(self, body: dict, digest: str, content: str) -> httpx.Response:
        provider = PROVIDERS[int(digest[:2], 16) % len(PROVIDERS)]
        return httpx.Response(
            200,
            json={"id": "fake-" + digest[:8], "model": self.model, "provider": provider,
                  "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}], "usage": _usage()},
        )


class CountingTransport(httpx.AsyncBaseTransport):
    """Counts requests and refuses any beyond `cap`: a hard ceiling on paid calls, enforced where the request leaves the process.
    A refused request surfaces as an explicit provider error in the log. Nothing retries."""

    def __init__(self, inner: httpx.AsyncBaseTransport, cap: int | None):
        self.inner, self.cap, self.calls, self.refused = inner, cap, 0, 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.cap is not None and self.calls >= self.cap:
            self.refused += 1
            raise httpx.ConnectError("call cap reached", request=request)
        self.calls += 1
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()
