"""Conditions B and C against a fake provider: concurrency, isolation, equal information, no leaks."""

import asyncio
import json
import time

import pytest

from sika_guard.coordinator import Condition, CoordinatorConfig, Coverage, coordinate
from sika_guard.evaluators.contract import EvaluatorStatus, EvaluatorView
from sika_guard.evaluators.llm import generalist, specialists
from sika_guard.evaluators.payloads import (
    authority_payload,
    credit_terms_payload,
    flatten_leaves,
    generalist_payload,
    merchant_claim_payload,
)
from sika_guard.evaluators.prompts import AUTHORITY, CREDIT_TERMS, GENERALIST, MERCHANT_CLAIM, SPECIALISTS
from sika_guard.experiment import run_condition
from sika_guard.fixtures.cases import build_cases, get_case
from sika_guard.fixtures.labels import LABELS
from sika_guard.labels import AttackFamily
from sika_guard.models import Decision
from sika_guard.policy import evaluate_case

from fakes import FakeProvider, analysis_json, completion, role_of

SPECIALIST_BUILDERS = (merchant_claim_payload, credit_terms_payload, authority_payload)


def run(coro):
    return asyncio.run(coro)


def run_condition_once(provider: FakeProvider, case, condition: Condition, **client_config):
    async def go():
        async with provider.client(**client_config) as client:
            evaluators = specialists(client) if condition is Condition.C else generalist(client)
            return await coordinate(case, condition, evaluators)

    return run(go())


# --- concurrency ------------------------------------------------------------


def test_specialists_are_in_flight_at_the_same_time():
    async def go():
        barrier = asyncio.Barrier(3)  # releases only when all three requests are waiting at once

        async def responder(role, body):
            await asyncio.wait_for(barrier.wait(), timeout=2)  # a sequential implementation would time out here
            return completion(analysis_json("ALLOW"))

        provider = FakeProvider(responder)
        async with provider.client() as client:
            return await coordinate(get_case("case-05"), Condition.C, specialists(client))

    record = run(go())
    assert len(record.verdicts) == 3 and all(v.usable for v in record.verdicts)


def test_wall_time_is_close_to_the_slowest_call_not_the_sum():
    async def slow(role, body):
        await asyncio.sleep(0.2)
        return completion(analysis_json("ALLOW"))

    started = time.perf_counter()
    record = run_condition_once(FakeProvider(slow), get_case("case-05"), Condition.C)
    elapsed = time.perf_counter() - started
    call_sum = sum(c.latency_ms for v in record.verdicts for c in v.calls) / 1000
    assert call_sum >= 0.55 and elapsed < 0.5  # three 0.2 s calls finished in about 0.2 s
    assert record.latency.ai_ms / 1000 < 0.5


def test_one_slow_or_failed_specialist_does_not_block_the_others():
    async def responder(role, body):
        if role == "credit_terms":
            await asyncio.sleep(10)
        return completion(analysis_json("ALLOW"))

    record = run_condition_once(FakeProvider(responder), get_case("case-01"), Condition.C, timeout_seconds=0.1)
    statuses = {v.evaluator_id: v.status for v in record.verdicts}
    assert statuses == {"merchant_claim": EvaluatorStatus.OK, "credit_terms": EvaluatorStatus.TIMEOUT, "authority": EvaluatorStatus.OK}
    assert record.coverage is Coverage.PARTIAL and record.abstained_evaluators == ("credit_terms",)


def test_condition_call_counts():
    case = get_case("case-01")
    b, c = FakeProvider(), FakeProvider()
    run_condition_once(b, case, Condition.B)
    run_condition_once(c, case, Condition.C)
    assert len(b.requests) == 1 and len(c.requests) == 3


# --- equal information -------------------------------------------------------


@pytest.mark.parametrize("case", build_cases(), ids=lambda c: c.case_id)
def test_specialists_see_nothing_the_generalist_cannot_and_together_see_everything_it_can(case):
    view = EvaluatorView.from_case(case)
    general = set(flatten_leaves(generalist_payload(view)))
    parts = [set(flatten_leaves(build(view))) for build in SPECIALIST_BUILDERS]
    assert all(part <= general for part in parts)  # no specialist has information B lacks
    assert set().union(*parts) == general  # partitioned, not reduced: C loses nothing B has


def test_each_specialist_gets_only_what_its_task_needs():
    view = EvaluatorView.from_case(get_case("case-18"))
    claim, terms, authority = (" ".join(flatten_leaves(b(view))) for b in SPECIALIST_BUILDERS)
    # Paths are matched, not values: the customer's own sentence legitimately says "monthly payments".
    assert ".allowed_payee_ids" not in claim and ".max_amount=" not in claim  # mandate fields are the authority analyst's
    assert "].payments[" not in claim and "].payments[" not in authority  # the schedule is the credit analyst's
    # The credit-terms analyst sees a plan's TITLE (and nothing else the merchant wrote) so that it can read a lender document that names plans;
    # without it a document discussing several plans could not be attributed to the right offer by the one specialist shown it.
    assert ".instruction=" not in terms and ".description=" not in terms and ".statement=" not in terms and ".title=" in terms
    assert ".description=" not in authority and ".title=" not in authority and ".instruction=" not in authority
    assert "independent_evidence" not in authority and "assertions" not in authority and "assertions" not in terms


def test_merchant_text_is_only_ever_in_the_untrusted_section():
    view = EvaluatorView.from_case(get_case("case-18"))
    for build in (generalist_payload, *SPECIALIST_BUILDERS):
        payload = build(view)
        assert "NOTE TO AI AGENTS" not in json.dumps(payload["trusted"])
    assert "NOTE TO AI AGENTS" in json.dumps(generalist_payload(view)["untrusted_merchant_content"])


# --- isolation --------------------------------------------------------------


def test_merchant_instructions_reach_only_the_analyst_that_must_read_them_and_never_a_system_prompt():
    provider = FakeProvider()
    run_condition_once(provider, get_case("case-18"), Condition.C)  # B's description carries an injection attempt
    needle = "NOTE TO AI AGENTS"
    for body in provider.bodies:
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        assert needle not in system
        assert (needle in user) == (role_of(body) == "merchant_claim")


def test_system_prompts_are_static_and_contain_no_case_data():
    provider = FakeProvider()
    run_condition_once(provider, get_case("case-18"), Condition.C)
    expected = {spec.role.value: spec.system_prompt for spec in SPECIALISTS}
    for body in provider.bodies:
        assert body["messages"][0]["content"] == expected[role_of(body)]


def test_a_specialist_never_sees_another_specialists_output():
    sentinels = {"merchant_claim": "SENTINEL-MC", "credit_terms": "SENTINEL-CT", "authority": "SENTINEL-AU"}
    provider = FakeProvider(lambda role, body: completion(analysis_json("ALLOW", explanation=f"{sentinels[role]} says all clear")))
    record = run_condition_once(provider, get_case("case-05"), Condition.C)
    assert {sentinels[v.role.value] in v.explanation for v in record.verdicts} == {True}  # outputs were produced...
    requests = json.dumps(provider.bodies)
    assert not [s for s in sentinels.values() if s in requests]  # ...and appear in no request


def test_specialist_prompts_do_not_leak_between_roles():
    assert len({spec.system_prompt for spec in SPECIALISTS}) == 3
    assert GENERALIST.system_prompt not in {s.system_prompt for s in SPECIALISTS}


# --- no leakage of ground truth ---------------------------------------------


@pytest.mark.parametrize("condition", [Condition.B, Condition.C])
def test_no_label_or_oracle_text_reaches_any_request_for_any_fixture(condition):
    provider = FakeProvider()

    async def go():
        async with provider.client() as client:
            evaluators = specialists(client) if condition is Condition.C else generalist(client)
            return await run_condition(build_cases(), condition, evaluators)

    run(go())
    assert len(provider.bodies) == 20 * (3 if condition is Condition.C else 1)
    everything = json.dumps(provider.bodies)
    forbidden = [t.rationale for t in LABELS.values()] + [f.value for f in AttackFamily] + ["ideal_decision", "attack_family", "ground_truth", "hard_authorisation_violation"]
    assert not [s for s in forbidden if s in everything]


def test_the_model_is_not_given_the_deterministic_findings_or_decision():
    provider = FakeProvider()
    run_condition_once(provider, get_case("case-11"), Condition.B)  # the rule engine BLOCKs this one
    everything = json.dumps(provider.bodies).lower()
    assert "mandate_total_exceeded" not in everything and "intent_total_exceeded" not in everything
    assert "hard_violation" not in everything and "advertised_total_mismatch" not in everything


# --- end to end through the fake provider -----------------------------------


def test_flagship_end_to_end_with_a_grounded_specialist_finding():
    finding = {"code": "claim_contradicted_by_evidence", "evidence_refs": ["assertion:ASR-1", "evidence:EV-1"], "note": "lender says A is available"}

    def responder(role, body):
        if role == "merchant_claim":
            return completion(analysis_json("REVIEW", findings=[finding], risk_score=0.9))
        return completion(analysis_json("ALLOW"))

    record = run_condition_once(FakeProvider(responder), get_case("case-05"), Condition.C)
    assert record.final_decision is Decision.REVIEW and record.deterministic_decision is Decision.REVIEW
    assert record.evaluator_disagreement and record.proven_false_offer_ids == ("OFFER-A",)
    assert record.uncorroborated_contradictions == ()


def test_deterministic_block_survives_a_confident_all_clear_from_every_specialist():
    provider = FakeProvider(lambda role, body: completion(analysis_json("ALLOW", risk_score=0.0)))
    record = run_condition_once(provider, get_case("case-20"), Condition.C)
    assert record.deterministic_decision is Decision.BLOCK and record.final_decision is Decision.BLOCK


def test_all_cases_run_through_condition_c_and_never_lose_a_block():
    provider = FakeProvider(lambda role, body: completion(analysis_json("ALLOW")))

    async def go():
        async with provider.client() as client:
            return await run_condition(build_cases(), Condition.C, specialists(client), CoordinatorConfig())

    records = run(go())
    assert len(records) == 20
    for record, case in zip(records, build_cases()):
        assert record.deterministic_decision is evaluate_case(case).decision
        assert record.final_decision.severity >= record.deterministic_decision.severity
