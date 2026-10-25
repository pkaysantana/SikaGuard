"""The OpenAI-compatible client and the LLM analyst, against a mocked transport. No network."""

import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from sika_guard.evaluators.contract import CallStatus, EvaluatorStatus, EvaluatorView
from sika_guard.evaluators.llm import ConfigError, LLMAnalyst, LLMConfig
from sika_guard.evaluators.prompts import GENERALIST
from sika_guard.fixtures.cases import get_case
from sika_guard.models import Decision

from fakes import SECRET, FakeProvider, analysis_json, completion, make_config

ENV = {
    "SIKA_GUARD_LLM_BASE_URL": "https://llm.example.test/v1/",
    "SIKA_GUARD_LLM_API_KEY": SECRET,
    "SIKA_GUARD_LLM_MODEL": "test-model",
}


def evaluate(provider: FakeProvider, case_id: str = "case-05", **config):
    async def go():
        async with provider.client(**config) as client:
            return await LLMAnalyst(GENERALIST, client).evaluate(EvaluatorView.from_case(get_case(case_id)))

    return asyncio.run(go())


def reply(content, **kw):
    return FakeProvider(lambda role, body: completion(content, **kw))


# --- configuration ----------------------------------------------------------


def test_config_names_exactly_the_missing_variables():
    with pytest.raises(ConfigError) as caught:
        LLMConfig.from_env({"SIKA_GUARD_LLM_MODEL": "m"})
    assert caught.value.missing == ("SIKA_GUARD_LLM_BASE_URL", "SIKA_GUARD_LLM_API_KEY")
    assert SECRET not in str(caught.value)


def test_config_reads_environment_and_never_exposes_the_key():
    config = LLMConfig.from_env({**ENV, "SIKA_GUARD_LLM_TIMEOUT_SECONDS": "12.5", "SIKA_GUARD_LLM_PRICE_INPUT_PER_MTOK": "2", "SIKA_GUARD_LLM_PRICE_OUTPUT_PER_MTOK": "8"})
    assert config.endpoint == "https://llm.example.test/v1/chat/completions"
    assert config.timeout_seconds == 12.5 and config.model == "test-model"
    for text in (repr(config), json.dumps(config.describe()), str(config.describe())):
        assert SECRET not in text


def test_config_describe_strips_credentials_embedded_in_the_url():
    described = LLMConfig.from_env({**ENV, "SIKA_GUARD_LLM_BASE_URL": "https://user:hunter2@llm.example.test/v1?key=abc"}).describe()
    assert "hunter2" not in str(described) and "key=abc" not in str(described)


def test_malformed_numeric_settings_are_a_config_error_that_does_not_echo_them():
    with pytest.raises(ConfigError) as caught:
        LLMConfig.from_env({**ENV, "SIKA_GUARD_LLM_TIMEOUT_SECONDS": "soon-sentinel"})
    assert "sentinel" not in str(caught.value)


def test_cost_needs_both_prices_and_provider_token_counts():
    priced = make_config(price_input_per_mtok=Decimal(2), price_output_per_mtok=Decimal(8))
    assert priced.cost(1_000_000, 500_000) == Decimal(6)
    assert priced.cost(None, 10) is None and priced.cost(10, None) is None
    assert make_config().cost(10, 10) is None
    assert make_config(price_input_per_mtok=Decimal(2)).cost(10, 10) is None


# --- request ----------------------------------------------------------------


def test_request_shape_and_credentials():
    provider = reply(analysis_json("ALLOW"))
    evaluate(provider)
    request, body = provider.requests[0], provider.bodies[0]
    assert str(request.url) == "https://llm.example.test/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert body["model"] == "test-model" and body["response_format"] == {"type": "json_object"}
    assert "temperature" not in body
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_optional_request_settings_are_honoured():
    provider = reply(analysis_json("ALLOW"))
    evaluate(provider, temperature=0.0, json_mode=False)
    body = provider.bodies[0]
    assert body["temperature"] == 0.0 and "response_format" not in body


# --- success ----------------------------------------------------------------


def test_valid_response_becomes_an_ok_verdict_with_usage_and_cost():
    provider = reply(analysis_json("REVIEW", risk_score=0.7), usage={"prompt_tokens": 1000, "completion_tokens": 200})
    verdict = evaluate(provider, price_input_per_mtok=Decimal(3), price_output_per_mtok=Decimal(15))
    assert verdict.status is EvaluatorStatus.OK and verdict.verdict is Decision.REVIEW
    assert verdict.prompt_version == "generalist-v5" and verdict.model_id == "test-model"
    (call,) = verdict.calls
    assert (call.status, call.prompt_tokens, call.completion_tokens) == (CallStatus.OK, 1000, 200)
    assert call.cost == Decimal("0.006") and call.latency_ms >= 0


def test_missing_usage_is_recorded_as_unknown_never_estimated():
    verdict = evaluate(reply(analysis_json("ALLOW")), price_input_per_mtok=Decimal(3), price_output_per_mtok=Decimal(15))
    (call,) = verdict.calls
    assert call.prompt_tokens is None and call.completion_tokens is None and call.cost is None


def test_fenced_json_is_accepted():
    assert evaluate(reply("```json\n" + analysis_json("ALLOW") + "\n```")).status is EvaluatorStatus.OK


# --- explicit failures ------------------------------------------------------


def test_transport_timeout_is_an_explicit_timeout():
    def raises(role, body):
        raise httpx.ReadTimeout("slow")

    verdict = evaluate(FakeProvider(raises))
    assert verdict.status is EvaluatorStatus.TIMEOUT and verdict.verdict is None and verdict.abstained
    assert verdict.calls[0].status is CallStatus.TIMEOUT


def test_a_stalled_provider_is_cut_off_by_the_overall_deadline():
    async def stalls(role, body):
        await asyncio.sleep(5)
        return completion(analysis_json("ALLOW"))

    verdict = evaluate(FakeProvider(stalls), timeout_seconds=0.05)
    assert verdict.status is EvaluatorStatus.TIMEOUT


@pytest.mark.parametrize("kwargs", [{"refusal": "I can't help with that."}])
def test_model_refusal_is_explicit(kwargs):
    verdict = evaluate(reply(None, **kwargs))
    assert verdict.status is EvaluatorStatus.REFUSAL and verdict.abstained
    assert verdict.calls[0].status is CallStatus.REFUSAL


def test_content_filter_is_a_refusal():
    body = completion("")
    body["choices"][0]["finish_reason"] = "content_filter"
    assert evaluate(FakeProvider(lambda role, b: body)).status is EvaluatorStatus.REFUSAL


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_http_errors_are_provider_errors_and_never_leak_the_response_body(status):
    provider = FakeProvider(
        lambda role, body: httpx.Response(status, json={"error": {"type": "server_problem", "message": f"bad key {SECRET}"}})
    )
    verdict = evaluate(provider)
    assert verdict.status is EvaluatorStatus.PROVIDER_ERROR
    assert verdict.calls[0].http_status == status
    assert str(status) in verdict.abstain_reason and SECRET not in verdict.model_dump_json()


def test_unexpected_envelope_is_a_provider_error():
    assert evaluate(FakeProvider(lambda role, body: {"unexpected": True})).status is EvaluatorStatus.PROVIDER_ERROR
    assert evaluate(FakeProvider(lambda role, body: httpx.Response(200, text="<html>not json</html>"))).status is EvaluatorStatus.PROVIDER_ERROR


@pytest.mark.parametrize("content", ["Sure! The answer is ALLOW.", "", None, analysis_json(verdict="MAYBE"), '{"verdict": "ALLOW"}'])
def test_unparseable_or_schema_invalid_content_is_an_invalid_response(content):
    verdict = evaluate(reply(content))
    assert verdict.status is EvaluatorStatus.INVALID_RESPONSE and verdict.verdict is None
    assert verdict.calls  # the call is still accounted for


def test_citing_a_reference_that_was_not_shown_is_an_invalid_response():
    finding = {"code": "limit_exceeded", "evidence_refs": ["evidence:INVENTED"], "note": "n"}
    assert evaluate(reply(analysis_json("REVIEW", findings=[finding]))).status is EvaluatorStatus.INVALID_RESPONSE


def test_a_model_may_abstain():
    verdict = evaluate(reply(analysis_json(None, abstain=True, abstain_reason="cannot judge")))
    assert verdict.status is EvaluatorStatus.ABSTAINED and verdict.abstain_reason == "cannot judge"


def test_the_api_key_never_reaches_a_verdict_record():
    for provider in (reply(analysis_json("ALLOW")), reply(None, refusal="no"), FakeProvider(lambda r, b: httpx.Response(500, text=SECRET))):
        assert SECRET not in evaluate(provider).model_dump_json()
