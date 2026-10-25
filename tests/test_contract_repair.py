"""Contract repair after the first live smoke test.

Covers: genuine finding references, claim-kind constraints, unverified-claim provenance,
structured-field grounding, unsupported numeric allegations, call timestamps, raw provider cost,
deterministic precedence, and label isolation. No network and no credentials.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from sika_guard.coordinator import Condition, CaseRecord, CoordinatorConfig, coordinate
from sika_guard.evaluators import llm as llm_module
from sika_guard.evaluators.contract import (
    AIFinding,
    CallStatus,
    ClaimKind,
    ConcernCode,
    EvaluatorStatus,
    EvaluatorView,
    ExtractedClaim,
    InvalidResponse,
    parse_analysis,
)
from sika_guard.evaluators.crosscheck import Corroboration, CrossCheckOutcome
from sika_guard.evaluators.llm import LLMAnalyst, generalist, specialists
from sika_guard.evaluators.mocks import Scripted, make_analysis
from sika_guard.evaluators.payloads import generalist_payload
from sika_guard.evaluators.prompts import GENERALIST, SPECIALISTS
from sika_guard.experiment import RunMetadata, format_experiment_report, run_condition, score
from sika_guard.fixtures.builders import gbp, level, make_case, make_offer
from sika_guard.fixtures.cases import build_cases, get_case
from sika_guard.fixtures.labels import LABELS
from sika_guard.labels import AttackFamily, GroundTruth
from sika_guard.models import Decision, DecisionBasis
from sika_guard.policy import evaluate_case

from fakes import SECRET, FakeProvider, analysis_json, completion


def run(coro):
    return asyncio.run(coro)


def coordinate_sync(case, evaluators, condition=Condition.B):
    return run(coordinate(case, condition, evaluators, CoordinatorConfig()))


def finding(code, refs, note="n") -> AIFinding:
    return AIFinding(code=code, evidence_refs=tuple(refs), note=note)


def scripted(case_id, verdict=Decision.ALLOW, *, findings=(), claims=()):
    return Scripted({case_id: make_analysis(verdict, findings=findings, claims=claims)})


def evaluate_with(provider: FakeProvider, case_id="case-05", spec=GENERALIST, **config):
    async def go():
        async with provider.client(**config) as client:
            return await LLMAnalyst(spec, client).evaluate(EvaluatorView.from_case(get_case(case_id)))

    return run(go())


# --- prompts: v2 and still static, label-free ---------------------------------


def test_every_role_prompt_is_version_4():
    assert [s.prompt_version for s in (GENERALIST, *SPECIALISTS)] == [
        "generalist-v4", "merchant-claim-v4", "credit-terms-v4", "authority-v4",
    ]


def test_the_prompt_states_the_contract_the_schema_enforces():
    text = GENERALIST.system_prompt
    for required in (
        "at least one \"ref\"",
        "not corroboration of it",
        "Never label an availability statement as an amount",
        "A value less than or equal to its limit is not a violation",
        "abstain rather than guess",
    ):
        assert required in text, required


@pytest.mark.parametrize("case", build_cases(), ids=lambda c: c.case_id)
def test_prompts_and_payloads_carry_no_label_vocabulary(case):
    truth = LABELS[case.case_id]
    forbidden = [truth.rationale, *(set(GroundTruth.model_fields) - {"case_id"}), *(f.value for f in AttackFamily)]
    view = EvaluatorView.from_case(case)
    for spec in (GENERALIST, *SPECIALISTS):
        haystack = spec.system_prompt + spec.user_message(spec.build_payload(view))
        assert not [s for s in forbidden if s in haystack], (spec.role, case.case_id)


def test_case_records_carry_no_ground_truth_fields():
    assert not (set(GroundTruth.model_fields) - {"case_id"}) & set(CaseRecord.model_fields)


# --- finding references -------------------------------------------------------


def test_a_finding_without_a_reference_is_still_rejected_not_loosened():
    raw = analysis_json("REVIEW", findings=[{"code": "limit_exceeded", "evidence_refs": [], "note": "n"}])
    with pytest.raises(InvalidResponse):
        parse_analysis(raw)
    with pytest.raises(ValidationError):
        AIFinding(code=ConcernCode.LIMIT_EXCEEDED, evidence_refs=(), note="n")


def test_through_the_client_an_ungrounded_finding_is_an_explicit_abstention():
    raw = analysis_json("REVIEW", findings=[{"code": "claim_unverified", "evidence_refs": [], "note": "n"}])
    verdict = evaluate_with(FakeProvider(lambda role, body: completion(raw)))
    assert verdict.status is EvaluatorStatus.INVALID_RESPONSE and verdict.verdict is None and verdict.abstained


def test_a_finding_citing_a_real_reference_is_accepted_and_recorded():
    refs = ["assertion:ASR-1", "evidence:EV-1"]
    raw = analysis_json("REVIEW", findings=[{"code": "claim_contradicted_by_evidence", "evidence_refs": refs, "note": "n"}])
    verdict = evaluate_with(FakeProvider(lambda role, body: completion(raw)))
    assert verdict.status is EvaluatorStatus.OK and verdict.evidence_refs == tuple(sorted(refs))


def test_a_model_may_abstain_instead_of_citing_nothing():
    raw = analysis_json(None, abstain=True, abstain_reason="no reference supports a concern")
    verdict = evaluate_with(FakeProvider(lambda role, body: completion(raw)))
    assert verdict.status is EvaluatorStatus.ABSTAINED and verdict.abstain_reason == "no reference supports a concern"


# --- claim kinds --------------------------------------------------------------


def claim(kind, **fields):
    return dict(kind=kind, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="some quote", **fields)


@pytest.mark.parametrize(
    "bad",
    [
        claim("availability", asserted_available=False, amount_minor=66000, currency="GBP"),  # availability with an amount
        claim("availability"),  # availability with no value
        claim("total_repayable"),  # total with no amount
        claim("total_repayable", amount_minor=100),  # total with no currency
        claim("no_additional_fees", amount_minor=6000, currency="GBP"),  # fee claim with an amount
    ],
)
def test_claim_values_must_match_the_claim_kind(bad):
    with pytest.raises(ValidationError):
        ExtractedClaim(**bad)


@pytest.mark.parametrize(
    "good",
    [
        claim("availability", asserted_available=False),
        claim("total_repayable", amount_minor=66000, currency="GBP"),
        claim("total_repayable", amount_minor=66000, currency="GBP", asserted_available=True),  # stray flag is ignored, not fatal
        claim("no_additional_fees"),
    ],
)
def test_matching_claims_are_accepted(good):
    assert ExtractedClaim(**good).kind in ClaimKind


def test_an_availability_statement_labelled_as_a_total_is_an_invalid_response():
    """The shape the first live run returned: an availability statement filed as total_repayable."""
    mislabelled = claim("total_repayable", asserted_available=False, amount_minor=None, currency=None)
    with pytest.raises(InvalidResponse):
        parse_analysis(analysis_json("REVIEW", extracted_claims=[mislabelled]))


# --- unverified claims: provenance without accusation -------------------------


def test_unverified_claim_keeps_the_merchant_reference_and_marks_corroboration_absent():
    case = get_case("case-14")  # the merchant says A is withdrawn; no independent evidence exists
    assert evaluate_case(case).decision is Decision.REVIEW  # held by the evidence requirement, not by any false-claim finding
    evaluator = scripted("case-14", findings=[finding(ConcernCode.CLAIM_UNVERIFIED, ["assertion:ASR-1"])])
    result = coordinate_sync(case, [evaluator])
    (provenance,) = result.claim_provenance
    assert provenance.merchant_refs == ("assertion:ASR-1",) and provenance.offer_id == "OFFER-A"
    assert provenance.corroboration is Corroboration.ABSENT and provenance.corroborating_refs == ()
    assert provenance.claim_status == "unverified"
    # unverified is not dishonest: nothing is proven false or alleged contradicted. The purchase is held for
    # corroboration by the rule engine's evidence requirement, which is a different thing from a false-claim finding.
    assert result.proven_false_offer_ids == () and result.uncorroborated_contradictions == ()
    assert result.final_decision is Decision.REVIEW and result.decision_bases == (DecisionBasis.REVIEW_EVIDENCE_REQUIREMENT,)
    assert not result.escalated_by_ai


def test_a_listing_sourced_unverified_finding_keeps_its_listing_reference():
    evaluator = scripted("case-01", findings=[finding(ConcernCode.CLAIM_UNVERIFIED, ["listing:OFFER-A"])])
    (provenance,) = coordinate_sync(get_case("case-01"), [evaluator]).claim_provenance
    assert provenance.merchant_refs == ("listing:OFFER-A",) and provenance.offer_id == "OFFER-A"
    assert provenance.corroboration is Corroboration.ABSENT and provenance.claim_status == "unverified"


def test_an_unverified_finding_that_cites_no_merchant_source_is_recorded_as_such():
    evaluator = scripted("case-01", findings=[finding(ConcernCode.CLAIM_UNVERIFIED, ["terms:OFFER-A"])])
    (provenance,) = coordinate_sync(get_case("case-01"), [evaluator]).claim_provenance
    assert provenance.merchant_refs == () and provenance.offer_id is None


def test_stale_evidence_is_not_corroboration():
    case = get_case("case-16")
    assertion_ref = f"assertion:{case.assertions[0].assertion_id}"
    evaluator = scripted("case-16", findings=[finding(ConcernCode.CLAIM_UNVERIFIED, [assertion_ref])])
    (provenance,) = coordinate_sync(case, [evaluator]).claim_provenance
    assert provenance.corroboration is Corroboration.STALE_ONLY and provenance.claim_status == "stale_only"


def test_trusted_code_notes_when_the_model_says_unverified_but_fresh_evidence_exists():
    case = get_case("case-05")
    evaluator = scripted("case-05", findings=[finding(ConcernCode.CLAIM_UNVERIFIED, ["assertion:ASR-1"])])
    (provenance,) = coordinate_sync(case, [evaluator]).claim_provenance
    assert provenance.corroboration is Corroboration.CONTRADICTING
    assert provenance.corroborating_refs == ("evidence:EV-1",) and provenance.claim_status == "confirmed_false"


def test_only_unverified_findings_produce_provenance():
    evaluator = scripted("case-05", findings=[finding(ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE, ["assertion:ASR-1", "evidence:EV-1"])])
    assert coordinate_sync(get_case("case-05"), [evaluator]).claim_provenance == ()


# --- structured-field grounding ------------------------------------------------


def structured_case(*, advertised="auto"):
    """OFFER-A truly totals 660.00 GBP. Its title and description say nothing about money."""
    kwargs = {} if advertised == "auto" else {"advertised": advertised}
    return make_case("t-struct", [make_offer("OFFER-A", level("55"), **kwargs), make_offer("OFFER-B", level("60"))], "OFFER-A")


def total_claim(quote, amount_minor=66000, source_ref="listing:OFFER-A", currency="GBP") -> ExtractedClaim:
    return ExtractedClaim(
        kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-A", source_ref=source_ref, quote=quote,
        amount_minor=amount_minor, currency=currency,
    )


def crosscheck_of(case, extracted):
    result = coordinate_sync(case, [Scripted({case.case_id: make_analysis(Decision.ALLOW, claims=[extracted])})])
    (check,) = result.crosschecks
    return check, result


@pytest.mark.parametrize(
    "quote",
    ["660.00 GBP", "advertised_total_repayable: 660.00 GBP", '"advertised_total_repayable": "660.00 GBP"', 'advertised_total_repayable: "660.00 GBP"'],
)
def test_a_quote_of_the_structured_advertised_total_is_grounded_with_its_field_provenance(quote):
    check, result = crosscheck_of(structured_case(), total_claim(quote))
    assert check.outcome is CrossCheckOutcome.CONSISTENT
    assert check.grounded_in == "listing:OFFER-A.advertised_total_repayable"
    assert result.final_decision is Decision.ALLOW


def test_a_disagreeing_extraction_is_reported_and_the_source_value_is_not_replaced():
    check, result = crosscheck_of(structured_case(), total_claim("660.00 GBP", amount_minor=6600))
    assert check.outcome is CrossCheckOutcome.EXTRACTION_DISAGREES and not check.contradicted
    assert "66.00 GBP" in check.detail and "660.00 GBP" in check.detail  # both values are shown
    assert result.proven_false_offer_ids == () and result.final_decision is Decision.ALLOW


def test_a_wrong_currency_in_the_extraction_is_a_disagreement_too():
    check, _ = crosscheck_of(structured_case(), total_claim("660.00 GBP", currency="USD"))
    assert check.outcome is CrossCheckOutcome.EXTRACTION_DISAGREES


def invalid_reply(case, extracted):
    """Audit F8: a claim the evaluator cannot have quoted from what it was shown invalidates the reply, so it counts against coverage."""
    result = coordinate_sync(case, [Scripted({case.case_id: make_analysis(Decision.ALLOW, claims=[extracted])})])
    (verdict,) = result.verdicts
    assert verdict.status is EvaluatorStatus.INVALID_RESPONSE and result.crosschecks == () and result.proven_false_offer_ids == ()
    return verdict


def test_a_quote_that_is_in_neither_the_text_nor_the_field_makes_the_reply_invalid():
    verdict = invalid_reply(structured_case(), total_claim("700.00 GBP", amount_minor=70000))
    assert "not in listing:OFFER-A" in verdict.abstain_reason and "700" not in verdict.abstain_reason  # names the source, never echoes the quote


def test_there_is_no_structured_field_to_ground_in_when_no_total_was_advertised():
    invalid_reply(structured_case(advertised=None), total_claim("660.00 GBP"))


def test_lender_terms_are_not_merchant_text_and_cannot_ground_a_claim():
    verdict = invalid_reply(structured_case(), total_claim("660.00 GBP", source_ref="terms:OFFER-A"))
    assert "is not a source in this evaluator's input" in verdict.abstain_reason


def test_a_structured_total_that_differs_from_the_lender_terms_is_contradicted_by_terms():
    case = structured_case(advertised=gbp(600))
    check, _ = crosscheck_of(case, total_claim("600.00 GBP", amount_minor=60000))
    assert check.outcome is CrossCheckOutcome.CONTRADICTED_BY_TERMS and check.grounded_in.endswith("advertised_total_repayable")
    assert "600.00 GBP" in check.detail and "660.00 GBP" in check.detail


def test_an_availability_check_names_the_assertion_statement_as_its_source():
    claim = ExtractedClaim(
        kind=ClaimKind.AVAILABILITY, offer_id="OFFER-A", source_ref="assertion:ASR-1", quote="has been withdrawn", asserted_available=False
    )
    check, _ = crosscheck_of(get_case("case-05"), claim)
    assert check.outcome is CrossCheckOutcome.CONTRADICTED_BY_EVIDENCE and check.grounded_in == "assertion:ASR-1.statement"


def test_free_text_grounding_still_works_and_is_labelled_as_text():
    a = make_offer("OFFER-A", level("55"), advertised=None, description="Total repayable £660 and no extra fees.")
    case = make_case("t-text", [a, make_offer("OFFER-B", level("60"))], "OFFER-A")
    check, _ = crosscheck_of(case, total_claim("Total repayable £660"))
    assert check.outcome is CrossCheckOutcome.CONSISTENT and check.grounded_in == "listing:OFFER-A.description"


# --- unsupported numeric allegations -------------------------------------------


def limit_scripted(case_id, verdict, refs):
    return scripted(case_id, verdict, findings=[finding(ConcernCode.LIMIT_EXCEEDED, refs, "55.00 GBP exceeds 200.00 GBP")])


def test_a_contradicted_limit_allegation_is_flagged_and_cannot_escalate():
    case = get_case("case-01")
    assert evaluate_case(case).decision is Decision.ALLOW
    result = coordinate_sync(case, [limit_scripted("case-01", Decision.BLOCK, ["mandate:MND-1", "txn:TXN-1"])])
    assert result.unsupported_limit_allegations == ("mock-scripted:limit_exceeded",)
    assert result.final_decision is Decision.ALLOW and result.authorised and not result.escalated_by_ai
    assert result.deterministic_decision is Decision.ALLOW


def test_a_limit_allegation_the_rule_engine_supports_is_not_flagged():
    case = get_case("case-12")
    assert evaluate_case(case).decision is Decision.BLOCK
    result = coordinate_sync(case, [limit_scripted("case-12", Decision.BLOCK, ["intent:CUST-1"])])
    assert result.unsupported_limit_allegations == ()


def test_the_allegation_is_judged_against_the_offers_it_cites():
    over = make_offer("OFFER-B", level("65"))  # totals 780.00 GBP, above the 750.00 limits
    case = make_case("t-limits", [make_offer("OFFER-A", level("55")), over], "OFFER-A")
    assert evaluate_case(case).decision is Decision.ALLOW
    cites_b = coordinate_sync(case, [limit_scripted("t-limits", Decision.REVIEW, ["terms:OFFER-B"])])
    assert cites_b.unsupported_limit_allegations == ()  # B really does exceed a limit
    cites_selected = coordinate_sync(case, [limit_scripted("t-limits", Decision.REVIEW, ["txn:TXN-1"])])
    assert cites_selected.unsupported_limit_allegations == ("mock-scripted:limit_exceeded",)


def test_other_finding_codes_are_never_judged_as_limit_allegations():
    evaluator = scripted("case-01", Decision.REVIEW, findings=[finding(ConcernCode.OTHER_CONCERN, ["txn:TXN-1"])])
    assert coordinate_sync(get_case("case-01"), [evaluator]).unsupported_limit_allegations == ()


# --- call timestamps -------------------------------------------------------------


def only_call(verdict):
    (call,) = verdict.calls
    return call


def test_calls_record_utc_start_and_finish_times_in_order():
    before = datetime.now(timezone.utc)
    call = only_call(evaluate_with(FakeProvider()))
    after = datetime.now(timezone.utc)
    assert call.started_at.utcoffset() == timedelta(0) and call.finished_at.utcoffset() == timedelta(0)
    assert before <= call.started_at <= call.finished_at <= after


def test_latency_is_monotonic_and_not_derived_from_the_wall_clock(monkeypatch):
    class JumpingClock:
        ticks = 0

        @classmethod
        def now(cls, tz=None):
            cls.ticks += 1
            return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=cls.ticks)

    monkeypatch.setattr(llm_module, "datetime", JumpingClock)
    call = only_call(evaluate_with(FakeProvider()))
    assert call.finished_at - call.started_at == timedelta(hours=1)  # the wall clock jumped an hour
    assert 0 <= call.latency_ms < 5000  # the monotonic duration did not


def test_failed_calls_are_timestamped_too():
    import httpx

    def raises(role, body):
        raise httpx.ReadTimeout("slow")

    call = only_call(evaluate_with(FakeProvider(raises)))
    assert call.status is CallStatus.TIMEOUT and call.started_at <= call.finished_at


def test_concurrent_specialist_calls_overlap_in_recorded_time():
    async def slow(role, body):
        await asyncio.sleep(0.05)
        return completion(analysis_json("ALLOW"))

    provider = FakeProvider(slow)

    async def go():
        async with provider.client() as client:
            return await coordinate(get_case("case-05"), Condition.C, specialists(client))

    calls = [c for v in run(go()).verdicts for c in v.calls]
    assert len(calls) == 3
    assert max(c.started_at for c in calls) < min(c.finished_at for c in calls)  # all three were in flight together


def test_timestamps_survive_json_export_and_no_credentials_do():
    verdict = evaluate_with(FakeProvider())
    dumped = verdict.model_dump_json()
    assert "started_at" in dumped and "finished_at" in dumped and SECRET not in dumped


# --- raw provider-reported cost ----------------------------------------------------


USAGE = {"prompt_tokens": 1000, "completion_tokens": 200, "cost": 0.0005361}


def with_provider(body, name="ExampleUpstream"):
    return {**body, "provider": name}


def test_provider_cost_is_kept_raw_with_an_unknown_unit_and_the_provider_identity():
    provider = FakeProvider(lambda role, body: with_provider(completion(analysis_json("ALLOW"), usage=USAGE)))
    call = only_call(evaluate_with(provider))
    assert call.provider_cost == Decimal("0.0005361") and call.provider_cost_unit == "UNKNOWN"
    assert call.provider == "ExampleUpstream" and call.endpoint_host == "llm.example.test"
    assert call.cost is None  # nothing is computed locally without prices; the provider figure is not promoted


def test_provider_cost_and_locally_computed_cost_are_separate_numbers():
    provider = FakeProvider(lambda role, body: completion(analysis_json("ALLOW"), usage=USAGE))
    call = only_call(evaluate_with(provider, price_input_per_mtok=Decimal(3), price_output_per_mtok=Decimal(15)))
    assert call.cost == Decimal("0.006") and call.provider_cost == Decimal("0.0005361")


@pytest.mark.parametrize("bad", [None, "0.5", True, -1, [1]])
def test_a_provider_cost_that_is_not_a_plain_non_negative_number_is_ignored(bad):
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "cost": bad}
    call = only_call(evaluate_with(FakeProvider(lambda role, body: completion(analysis_json("ALLOW"), usage=usage))))
    assert call.provider_cost is None and call.provider_cost_unit is None


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_provider_costs_are_ignored(bad):
    assert llm_module._raw_cost(bad) is None
    assert llm_module._raw_cost(0) == Decimal(0) and llm_module._raw_cost(2) == Decimal(2)


def test_no_usage_means_no_provider_cost_and_a_non_object_usage_is_survivable():
    for usage in (None, [1, 2]):
        body = completion(analysis_json("ALLOW"))
        if usage is not None:
            body["usage"] = usage
        call = only_call(evaluate_with(FakeProvider(lambda role, b, body=body: body)))
        assert call.provider_cost is None and call.prompt_tokens is None


def test_the_report_shows_provider_cost_separately_and_never_adds_it_to_computed_cost():
    provider = FakeProvider(lambda role, body: completion(analysis_json("ALLOW"), usage=USAGE))

    async def go():
        async with provider.client(price_input_per_mtok=Decimal(3), price_output_per_mtok=Decimal(15)) as client:
            return await run_condition(build_cases()[:2], Condition.B, generalist(client))

    meta = RunMetadata(Condition.B, "live", (), {"model": "test-model"}, False, {})
    report = format_experiment_report(meta, score(run(go()), LABELS))
    assert "in the units of the configured prices: 0.012000" in report  # 2 x 0.006, computed locally
    assert "provider-reported usage.cost on 2 of 2 calls: 0.0010722 (unit UNKNOWN; not a computed cost and not added to it)" in report
    assert "0.013072" not in report  # the two figures are never summed


def test_the_report_says_when_the_provider_returned_no_cost():
    provider = FakeProvider()

    async def go():
        async with provider.client() as client:
            return await run_condition(build_cases()[:1], Condition.B, generalist(client))

    meta = RunMetadata(Condition.B, "live", (), {"model": "test-model"}, False, {})
    assert "provider-reported usage.cost: none returned" in format_experiment_report(meta, score(run(go()), LABELS))


# --- deterministic precedence through the real client path -----------------------------


def role_responder(verdicts):
    return lambda role, body: completion(analysis_json(verdicts[role]))


def test_a_model_block_on_an_allowed_case_has_no_effect():
    provider = FakeProvider(role_responder({"merchant_claim": "ALLOW", "credit_terms": "ALLOW", "authority": "BLOCK"}))

    async def go():
        async with provider.client() as client:
            return await coordinate(get_case("case-01"), Condition.C, specialists(client))

    result = run(go())
    assert result.deterministic_decision is Decision.ALLOW and result.final_decision is Decision.ALLOW  # a model BLOCK enforces nothing


@pytest.mark.parametrize("case_id", ["case-11", "case-12", "case-13", "case-20"])
def test_unanimous_model_allow_cannot_lower_a_deterministic_block(case_id):
    provider = FakeProvider(role_responder({"merchant_claim": "ALLOW", "credit_terms": "ALLOW", "authority": "ALLOW"}))

    async def go():
        async with provider.client() as client:
            return await coordinate(get_case(case_id), Condition.C, specialists(client))

    result = run(go())
    assert result.deterministic_decision is Decision.BLOCK and result.final_decision is Decision.BLOCK and not result.authorised


def test_unsupported_allegations_never_become_a_block_or_lower_a_review():
    case = get_case("case-05")  # deterministic REVIEW
    raw = analysis_json("ALLOW", findings=[{"code": "limit_exceeded", "evidence_refs": ["txn:TXN-1", "mandate:MND-1"], "note": "n"}])
    provider = FakeProvider(lambda role, body: completion(raw))

    async def go():
        async with provider.client() as client:
            return await coordinate(case, Condition.B, generalist(client))

    result = run(go())
    assert result.unsupported_limit_allegations == ("generalist:limit_exceeded",)
    assert result.final_decision is Decision.REVIEW and result.deterministic_decision is Decision.REVIEW
