"""Fair comparison contracts for B, B3 (matched compute), Bc (matched information) and C. Offline only."""

import asyncio
import itertools
import json
from decimal import Decimal

import pytest

from sika_guard import evaluate
from sika_guard.coordinator import AGGREGATION_POLICY, Condition, CoordinatorConfig, Coverage, coordinate
from sika_guard.evaluators.adjudicate import FindingEffect
from sika_guard.evaluators.contract import ClaimKind, ConcernCode, EvaluatorRole, EvaluatorStatus, EvaluatorView, ExtractedClaim
from sika_guard.evaluators.llm import evaluators_for
from sika_guard.evaluators.mocks import Scripted, make_analysis
from sika_guard.evaluators.payloads import authority_payload, credit_terms_payload, flatten_leaves, generalist_payload, merchant_claim_payload
from sika_guard.evaluators.prompts import (
    B3_SAMPLES,
    GENERALIST,
    GENERALIST_SAMPLES,
    RUBRIC_GENERALIST,
    SPECIALISTS,
    specs_for,
)
from sika_guard.experiment import RunMetadata, format_experiment_report, run_condition, score
from sika_guard.fixtures.builders import fee, level, make_case, make_offer
from sika_guard.fixtures.cases import build_cases, get_case
from sika_guard.fixtures.labels import LABELS
from sika_guard.labels import AttackFamily, GroundTruth
from sika_guard.models import Decision, DecisionBasis, FeeKind
from sika_guard.policy import evaluate_case

from fakes import FakeProvider, analysis_json, completion, role_of

AI_CONDITIONS = [Condition.B, Condition.B3, Condition.BC, Condition.C]
HARD_BLOCK_CASES = ["case-11", "case-12", "case-13", "case-20"]


def run(coro):
    return asyncio.run(coro)


def live(provider: FakeProvider, case, condition: Condition, **client_config):
    async def go():
        async with provider.client(**client_config) as client:
            return await coordinate(case, condition, evaluators_for(condition.value, client))

    return run(go())


def scripted_trio(*verdicts, case_id="case-01", **analysis):
    """Three separately identified mock evaluators, each returning the given verdict for the case."""
    return [Scripted({case_id: make_analysis(v, **analysis)}, evaluator_id=f"mock-{i}") for i, v in enumerate(verdicts)]


def decide(case, evaluators, condition=Condition.B3, **config):
    return run(coordinate(case, condition, evaluators, CoordinatorConfig(**config)))


def trigger_case():
    """The rule engine says ALLOW. The listing text understates the lender's total, so a model that extracts
    the claim lets trusted code prove a contradiction; one that does not leaves the case as it was."""
    a = make_offer("OFFER-A", level("55", fees=(fee(FeeKind.ADMINISTRATION, "20"),)), advertised=None, description="Total repayable £660.")
    return make_case("t-trigger", [a, make_offer("OFFER-B", level("60"))], "OFFER-A")


CLAIM = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="Total repayable £660", amount_minor=66000, currency="GBP")


def trio(*finds, risk=0.5):
    """Evaluators that do (True) or do not (False) extract the contradicting claim. Their verdicts are always ALLOW."""
    return [
        Scripted({"t-trigger": make_analysis(Decision.ALLOW, claims=[CLAIM] if f else [], risk_score=risk)}, evaluator_id=f"mock-{i}")
        for i, f in enumerate(finds)
    ]


# --- the conditions are defined, matched where claimed ------------------------------


def test_each_condition_has_its_declared_evaluators():
    assert [s.evaluator_id for s in specs_for("A")] == []
    assert [s.evaluator_id for s in specs_for("B")] == ["generalist"]
    assert [s.evaluator_id for s in specs_for("B3")] == ["generalist_1", "generalist_2", "generalist_3"]
    assert [s.evaluator_id for s in specs_for("Bc")] == ["rubric_generalist"]
    assert [s.evaluator_id for s in specs_for("C")] == ["merchant_claim", "credit_terms", "authority"]


def test_b3_is_matched_to_c_in_calls_and_bc_to_b():
    assert len(specs_for("B3")) == len(specs_for("C")) == B3_SAMPLES == 3
    assert len(specs_for("Bc")) == len(specs_for("B")) == 1


def test_b3_samples_are_the_same_prompt_and_payload_under_distinct_ids():
    assert {s.system_prompt for s in GENERALIST_SAMPLES} == {GENERALIST.system_prompt}
    assert {s.prompt_version for s in GENERALIST_SAMPLES} == {GENERALIST.prompt_version} == {"generalist-v5"}
    assert len({s.evaluator_id for s in GENERALIST_SAMPLES}) == 3
    assert all(s.build_payload is GENERALIST.build_payload for s in GENERALIST_SAMPLES)


def test_every_condition_aggregates_with_the_one_declared_policy():
    assert AGGREGATION_POLICY == "max_severity_of_adjudicated_verdicts"
    report = format_experiment_report(
        RunMetadata(Condition.B3, "live", (), None, False, {}), score(run(run_condition(build_cases()[:2], Condition.B3, scripted_trio(Decision.ALLOW))), LABELS)
    )
    assert AGGREGATION_POLICY in report and "no vote, no risk scores" in report


# --- equal information -----------------------------------------------------------------


@pytest.mark.parametrize("case", build_cases(), ids=lambda c: c.case_id)
def test_b_b3_and_bc_receive_identical_authoritative_information_and_c_partitions_it(case):
    view = EvaluatorView.from_case(case)
    reference = generalist_payload(view)
    for spec in (GENERALIST, RUBRIC_GENERALIST, *GENERALIST_SAMPLES):
        assert spec.build_payload(view) == reference
    parts = [set(flatten_leaves(b(view))) for b in (merchant_claim_payload, credit_terms_payload, authority_payload)]
    assert all(p <= set(flatten_leaves(reference)) for p in parts)
    assert set().union(*parts) == set(flatten_leaves(reference))


def test_the_requests_that_reach_the_model_carry_the_same_user_message_for_b_b3_and_bc():
    case = get_case("case-18")
    messages = {}
    for condition in (Condition.B, Condition.B3, Condition.BC):
        provider = FakeProvider()
        live(provider, case, condition)
        messages[condition] = {body["messages"][1]["content"] for body in provider.bodies}
    assert messages[Condition.B] == messages[Condition.B3] == messages[Condition.BC]
    assert len(messages[Condition.B]) == 1


def test_the_rubric_prompt_covers_all_three_specialist_remits_and_reuses_the_shared_rules():
    text = RUBRIC_GENERALIST.system_prompt
    for rubric in ("Rubric 1, merchant claims", "Rubric 2, credit terms", "Rubric 3, authority"):
        assert rubric in text
    assert RUBRIC_GENERALIST.prompt_version == "rubric-generalist-v5" and RUBRIC_GENERALIST.role is EvaluatorRole.RUBRIC_GENERALIST
    shared = GENERALIST.system_prompt.split("Your role:")[0]
    assert text.startswith(shared)  # the same evidence, grounding and format rules as every other condition
    assert all(s.system_prompt.split("Your role:")[0] == shared for s in SPECIALISTS)


# --- aggregation fairness -----------------------------------------------------------------


@pytest.mark.parametrize("finds", list(itertools.product([False, True], repeat=3)))
def test_aggregation_is_any_verified_contradiction_with_no_vote(finds):
    result = decide(trigger_case(), trio(*finds))
    assert result.final_decision is (Decision.REVIEW if any(finds) else Decision.ALLOW)  # two silent samples do not outvote one finding


def test_aggregation_ignores_order_and_risk_scores():
    case = trigger_case()
    outcomes = set()
    for order in itertools.permutations([(False, 1.0), (False, 0.9), (True, 0.0)]):
        evaluators = [
            Scripted({"t-trigger": make_analysis(Decision.ALLOW, claims=[CLAIM] if f else [], risk_score=r)}, evaluator_id=f"m{i}")
            for i, (f, r) in enumerate(order)
        ]
        outcomes.add(decide(case, evaluators).final_decision)
    assert outcomes == {Decision.REVIEW}  # the evaluator holding the proof had the lowest risk score and still decided


def test_loud_verdicts_and_risk_scores_without_a_verified_finding_decide_nothing():
    case = get_case("case-01")
    loud = [Scripted({"case-01": make_analysis(Decision.BLOCK, risk_score=1.0)}, evaluator_id=f"loud-{i}") for i in range(3)]
    result = decide(case, loud)
    assert result.final_decision is Decision.ALLOW and result.deterministic_decision is Decision.ALLOW


@pytest.mark.parametrize("finds", [(False, False, False), (False, True, False), (True, True, True)])
def test_b3_and_c_decide_identically_given_the_same_findings(finds):
    b3 = decide(trigger_case(), trio(*finds), Condition.B3)
    c = decide(trigger_case(), trio(*finds), Condition.C)
    assert b3.final_decision is c.final_decision and b3.escalated_by_ai == c.escalated_by_ai and b3.decision_bases == c.decision_bases


@pytest.mark.parametrize("finds, expected", [((True,), Decision.REVIEW), ((False,), Decision.ALLOW)])
def test_a_single_call_decides_by_the_same_rule_in_b_and_bc(finds, expected):
    assert {c: decide(trigger_case(), trio(*finds), c).final_decision for c in (Condition.B, Condition.BC)} == {Condition.B: expected, Condition.BC: expected}


# --- model disagreement ---------------------------------------------------------------------


def test_disagreement_is_reported_separately_from_abstention():
    case = get_case("case-01")
    evaluators = scripted_trio(Decision.ALLOW, Decision.REVIEW) + [Scripted({}, evaluator_id="mock-silent")]  # the third has no script and abstains
    result = decide(case, evaluators)
    assert result.evaluator_disagreement and result.abstained_evaluators == ("mock-silent",)
    # Disagreement and abstention are recorded as such. Neither is an attack finding and neither escalates by default.
    assert result.coverage is Coverage.PARTIAL and result.final_decision is Decision.ALLOW
    assert result.decision_bases == (DecisionBasis.ALLOW_WITH_UNCERTAINTY,)


def test_agreement_is_not_disagreement_and_fail_closed_applies_to_every_condition():
    case = get_case("case-01")
    agree = decide(case, scripted_trio(Decision.ALLOW, Decision.ALLOW, Decision.ALLOW))
    assert not agree.evaluator_disagreement and agree.final_decision is Decision.ALLOW
    silent = [Scripted({}, evaluator_id=f"mock-{i}") for i in range(3)]
    assert decide(case, silent).final_decision is Decision.ALLOW  # a missing reply is not evidence about the merchant
    assert decide(case, silent, fail_closed=True).final_decision is Decision.REVIEW


# --- unsupported allegation suppression, in every condition -----------------------------------


LIMIT_LIE = {"code": "limit_exceeded", "evidence_refs": ["txn:TXN-1", "mandate:MND-1"], "note": "55.00 GBP exceeds 200.00 GBP"}


@pytest.mark.parametrize("condition", AI_CONDITIONS)
def test_an_escalation_resting_on_a_contradicted_allegation_is_ignored_in_every_condition(condition):
    provider = FakeProvider(lambda role, body: completion(analysis_json("BLOCK", findings=[LIMIT_LIE])))
    result = live(provider, get_case("case-01"), condition)
    assert result.final_decision is Decision.ALLOW and not result.escalated_by_ai
    usable = {v.evaluator_id for v in result.verdicts if v.usable}  # a specialist shown no mandate cannot cite it, so its reply is rejected
    assert usable and set(result.unsupported_limit_allegations) == {f"{e}:limit_exceeded" for e in usable}
    assert all(t.counted_as is Decision.ALLOW for t in result.verdict_treatments)


@pytest.mark.parametrize("condition", AI_CONDITIONS)
def test_an_unverifiable_concern_is_recorded_but_enforces_nothing_in_every_condition(condition):
    other = {"code": "other_concern", "evidence_refs": ["merchant:MER-1"], "note": "something odd"}
    provider = FakeProvider(lambda role, body: completion(analysis_json("REVIEW", findings=[other])))
    result = live(provider, get_case("case-01"), condition)
    assert result.final_decision is Decision.ALLOW and not result.escalated_by_ai
    assert result.decision_bases == (DecisionBasis.ALLOW_WITH_UNCERTAINTY,)
    assert result.adjudications and {a.effect for a in result.adjudications} == {FindingEffect.UNVERIFIED_RECORDED}


# --- structured-output validation ---------------------------------------------------------------


def test_one_malformed_sample_abstains_without_hiding_the_others():
    replies = iter([analysis_json("REVIEW"), "not json at all", analysis_json("ALLOW")])
    provider = FakeProvider(lambda role, body: completion(next(replies)))
    result = live(provider, get_case("case-01"), Condition.B3)
    statuses = sorted(v.status.value for v in result.verdicts)
    assert statuses == ["invalid_response", "ok", "ok"] and result.coverage is Coverage.PARTIAL
    assert result.final_decision is Decision.ALLOW and result.abstained_evaluators  # the failure is listed, not treated as an attack


@pytest.mark.parametrize("condition", AI_CONDITIONS)
def test_every_condition_rejects_the_same_schema_violations(condition):
    bad = analysis_json("REVIEW", findings=[{"code": "limit_exceeded", "evidence_refs": [], "note": "n"}])
    result = live(FakeProvider(lambda role, body: completion(bad)), get_case("case-01"), condition)
    assert {v.status for v in result.verdicts} == {EvaluatorStatus.INVALID_RESPONSE} and result.coverage is Coverage.UNAVAILABLE
    assert result.final_decision is Decision.ALLOW and all(v.raw_response == bad for v in result.verdicts)


# --- isolation ------------------------------------------------------------------------------------


def test_samples_never_see_each_others_output():
    counter = itertools.count(1)
    provider = FakeProvider(lambda role, body: completion(analysis_json("ALLOW", explanation=f"SENTINEL-{next(counter)} all clear")))
    result = live(provider, get_case("case-05"), Condition.B3)
    assert {v.explanation.split()[0] for v in result.verdicts} == {"SENTINEL-1", "SENTINEL-2", "SENTINEL-3"}
    assert "SENTINEL" not in json.dumps(provider.bodies)


@pytest.mark.parametrize("condition", [Condition.B3, Condition.BC])
def test_no_label_or_oracle_text_reaches_any_request_for_any_fixture(condition):
    provider = FakeProvider()
    for case in build_cases():
        live(provider, case, condition)
    requests = json.dumps(provider.bodies)
    truths = [LABELS[c.case_id] for c in build_cases()]
    forbidden = {t.rationale for t in truths} | set(GroundTruth.model_fields) - {"case_id"} | {f.value for f in AttackFamily}
    assert not [s for s in forbidden if s in requests]


def test_case_ids_stay_neutral_and_the_rubric_and_sample_prompts_are_label_free():
    forbidden = [f.value for f in AttackFamily] + sorted(set(GroundTruth.model_fields) - {"case_id"})
    for spec in (RUBRIC_GENERALIST, *GENERALIST_SAMPLES):
        assert not [s for s in forbidden if s in spec.system_prompt]
    assert all(c.case_id.startswith("case-") and c.case_id[5:].isdigit() for c in build_cases())


# --- deterministic precedence ----------------------------------------------------------------------


@pytest.mark.parametrize("condition", AI_CONDITIONS)
@pytest.mark.parametrize("case_id", HARD_BLOCK_CASES)
def test_unanimous_model_allow_cannot_lower_a_deterministic_block_in_any_condition(condition, case_id):
    provider = FakeProvider(lambda role, body: completion(analysis_json("ALLOW")))
    result = live(provider, get_case(case_id), condition)
    assert evaluate_case(get_case(case_id)).decision is Decision.BLOCK
    assert result.final_decision is Decision.BLOCK and not result.authorised


@pytest.mark.parametrize("condition", AI_CONDITIONS)
def test_a_model_block_creates_a_block_in_no_condition(condition):
    other = {"code": "other_concern", "evidence_refs": ["merchant:MER-1"], "note": "n"}
    provider = FakeProvider(lambda role, body: completion(analysis_json("BLOCK", findings=[other])))
    for case in build_cases():
        result = live(provider, case, condition)
        assert (result.final_decision is Decision.BLOCK) == (result.deterministic_decision is Decision.BLOCK)


# --- latency and usage accounting -----------------------------------------------------------------------


USAGE = {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.0004}


@pytest.mark.parametrize(
    "condition, calls_per_case", [(Condition.B, 1), (Condition.B3, 3), (Condition.BC, 1), (Condition.C, 3)]
)
def test_usage_is_counted_per_call_and_summed_without_estimates(condition, calls_per_case):
    provider = FakeProvider(lambda role, body: {**completion(analysis_json("ALLOW"), usage=USAGE), "provider": "UpstreamX", "model": "returned-model"})
    cases = build_cases()[:2]

    async def go():
        async with provider.client(price_input_per_mtok=Decimal(2), price_output_per_mtok=Decimal(10)) as client:
            return await run_condition(cases, condition, evaluators_for(condition.value, client))

    records = run(go())
    n = 2 * calls_per_case
    assert len(provider.requests) == n and all(len(r.verdicts) == calls_per_case for r in records)
    report = format_experiment_report(RunMetadata(condition, "live", (), {"model": "test-model"}, False, {}), score(records, LABELS))
    assert f"tokens reported on {n} of {n} calls: prompt {1000 * n}, completion {100 * n}" in report
    assert f"provider-reported usage.cost on {n} of {n} calls: {Decimal('0.0004') * n:.7f} (unit UNKNOWN" in report
    assert f"{Decimal('0.003') * n:.6f}" in report  # 1000*2/1e6 + 100*10/1e6 per call, computed locally and separately
    assert "returned model(s): returned-model x" in report and "UpstreamX x" in report


def test_calls_of_a_sampled_condition_run_concurrently_and_wall_time_is_not_their_sum():
    async def slow(role, body):
        await asyncio.sleep(0.15)
        return completion(analysis_json("ALLOW"))

    result = live(FakeProvider(slow), get_case("case-05"), Condition.B3)
    calls = [c for v in result.verdicts for c in v.calls]
    assert len(calls) == 3 and max(c.started_at for c in calls) < min(c.finished_at for c in calls)
    assert sum(c.latency_ms for c in calls) >= 400 and result.latency.ai_ms < 350


def test_provider_variation_is_reported_as_a_reproducibility_limitation_and_not_controlled():
    names = itertools.cycle(["Azure", "OpenAI", "OpenAI"])
    provider = FakeProvider(lambda role, body: {**completion(analysis_json("ALLOW"), usage=USAGE), "provider": next(names), "model": "m"})

    async def go():
        async with provider.client() as client:
            return await run_condition(build_cases()[:2], Condition.B3, evaluators_for("B3", client))

    report = format_experiment_report(RunMetadata(Condition.B3, "live", (), {"model": "test-model"}, False, {}), score(run(go()), LABELS))
    assert "Azure x2" in report and "OpenAI x4" in report and "reproducibility limitation (not controlled)" in report
    assert not any(k in b for b in provider.bodies for k in ("provider", "route", "order"))  # no attempt to force an upstream


def test_call_records_keep_identity_timing_and_raw_usage():
    provider = FakeProvider(lambda role, body: {**completion(analysis_json("ALLOW"), usage=USAGE), "provider": "UpstreamX", "model": "returned-model"})
    (call,) = [c for v in live(provider, get_case("case-01"), Condition.B).verdicts for c in v.calls]
    assert call.model == "test-model" and call.returned_model == "returned-model"
    assert call.provider == "UpstreamX" and call.endpoint_host == "llm.example.test"
    assert call.raw_usage == USAGE and call.started_at <= call.finished_at
    assert call.provider_cost == Decimal("0.0004") and call.provider_cost_unit == "UNKNOWN"


# --- CLI -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("condition, evaluator_count", [("B3", 3), ("Bc", 1)])
def test_offline_cli_runs_the_new_conditions_without_credentials(capsys, monkeypatch, condition, evaluator_count):
    for name in ("SIKA_GUARD_LLM_BASE_URL", "SIKA_GUARD_LLM_API_KEY", "SIKA_GUARD_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert evaluate.main(["--condition", condition, "--offline", "allow", "--limit", "4"]) == 0
    out = capsys.readouterr().out
    assert f"Condition {condition}" in out and "NOT model output" in out
    assert out.count("  evaluator mock-allow-") == evaluator_count
