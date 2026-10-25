"""Finding adjudication, presentation-only normalisation and source grounding. No network."""

import asyncio
import json

import pytest

from sika_guard.coordinator import Condition, CoordinatorConfig, coordinate
from sika_guard.evaluators.adjudicate import AllegationStatus, FindingEffect, ref_source
from sika_guard.evaluators.amounts import parse_amounts
from sika_guard.evaluators.contract import (
    AIFinding,
    ClaimKind,
    ConcernCode,
    EvaluatorStatus,
    EvaluatorView,
    ExtractedClaim,
    InvalidResponse,
    parse_analysis,
    parse_analysis_normalised,
)
from sika_guard.evaluators.crosscheck import CrossCheckOutcome
from sika_guard.evaluators.llm import LLMAnalyst
from sika_guard.evaluators.mocks import Scripted, make_analysis
from sika_guard.evaluators.prompts import GENERALIST
from sika_guard.fixtures.builders import MERCHANT_SOURCE, fee, gbp, level, make_assertion, make_case, make_evidence, make_intent, make_offer
from sika_guard.fixtures.cases import get_case
from sika_guard.models import Decision, DecisionBasis, FeeKind, Objective
from sika_guard.policy import evaluate_case

from fakes import FakeProvider, analysis_json, completion

A, V, U = AllegationStatus.CONTRADICTED, AllegationStatus.VERIFIED, AllegationStatus.UNVERIFIED


def run(coro):
    return asyncio.run(coro)


def coordinate_sync(case, evaluators, condition=Condition.B, **config):
    return run(coordinate(case, condition, evaluators, CoordinatorConfig(**config)))


def finding(code, refs, note="n") -> AIFinding:
    return AIFinding(code=code, evidence_refs=tuple(refs), note=note)


def judged(case, verdict, findings, claims=()):
    return coordinate_sync(case, [Scripted({case.case_id: make_analysis(verdict, findings=findings, claims=claims)})])


def statuses(result):
    return [a.status for a in result.adjudications]


def two_offer_case(**kwargs):
    """OFFER-A (55/month, 660.00 total) is selected; OFFER-B (60/month, 720.00 total) is the alternative."""
    return make_case("t-two", [make_offer("OFFER-A", level("55")), make_offer("OFFER-B", level("60"))], "OFFER-A", **kwargs)


# --- numeric limit allegations ------------------------------------------------


def test_55_exceeding_200_is_contradicted_and_cannot_escalate():
    case = get_case("case-01")
    f = finding(ConcernCode.LIMIT_EXCEEDED, ["mandate:MND-1", "txn:TXN-1"], "55.00 GBP exceeds 200.00 GBP")
    result = judged(case, Decision.BLOCK, [f])
    (a,) = result.adjudications
    assert a.status is A and a.method == "limit_checks" and "55.00 GBP" in a.reason and "200.00 GBP" in a.reason
    assert result.final_decision is Decision.ALLOW and not result.escalated_by_ai
    (t,) = result.verdict_treatments
    assert (t.model_verdict, t.counted_as) == (Decision.BLOCK, Decision.ALLOW) and "no enforcement effect" in t.reason
    assert a.effect is FindingEffect.MODEL_ERROR  # kept in the log as a model error


def test_60_exceeding_an_authenticated_65_limit_is_contradicted():
    case = two_offer_case()
    f = finding(ConcernCode.LIMIT_EXCEEDED, ["intent:CUST-1", "terms:OFFER-B"], "60.00 GBP exceeds 65.00 GBP")
    result = judged(case, Decision.REVIEW, [f])
    assert statuses(result) == [A] and result.final_decision is Decision.ALLOW
    assert result.unsupported_limit_allegations == ("mock-scripted:limit_exceeded",)


def test_a_limit_that_really_is_exceeded_is_verified_and_the_rule_engine_still_blocks():
    case = get_case("case-12")
    assert evaluate_case(case).decision is Decision.BLOCK
    result = judged(case, Decision.ALLOW, [finding(ConcernCode.LIMIT_EXCEEDED, ["intent:CUST-1"])])
    assert statuses(result) == [V] and "intent_total_exceeded" in result.adjudications[0].reason
    assert result.final_decision is Decision.BLOCK and not result.authorised


def test_an_allegation_about_a_cited_offer_is_judged_against_that_offer():
    over = make_offer("OFFER-B", level("65"))  # 780.00 total, above the 750.00 limits
    case = make_case("t-over", [make_offer("OFFER-A", level("55")), over], "OFFER-A")
    assert statuses(judged(case, Decision.REVIEW, [finding(ConcernCode.LIMIT_EXCEEDED, ["terms:OFFER-B"])])) == [V]
    assert statuses(judged(case, Decision.REVIEW, [finding(ConcernCode.LIMIT_EXCEEDED, ["txn:TXN-1"])])) == [A]


# --- valid findings in the same reply are not discarded -----------------------------


def test_a_contradicted_finding_does_not_discard_the_other_findings_in_the_same_reply():
    case = get_case("case-01")
    findings = [
        finding(ConcernCode.LIMIT_EXCEEDED, ["mandate:MND-1", "txn:TXN-1"]),
        finding(ConcernCode.OTHER_CONCERN, ["merchant:MER-1"]),
    ]
    result = judged(case, Decision.REVIEW, findings)
    assert statuses(result) == [A, U]
    assert [a.effect for a in result.adjudications] == [FindingEffect.MODEL_ERROR, FindingEffect.UNVERIFIED_RECORDED]  # both are kept
    assert result.final_decision is Decision.ALLOW and DecisionBasis.ALLOW_WITH_UNCERTAINTY in result.decision_bases
    assert "1 contradicted" in result.verdict_treatments[0].reason and "1 unverified" in result.verdict_treatments[0].reason


def test_a_contradicted_finding_does_not_discard_a_verified_one_in_the_same_reply():
    case = get_case("case-05")
    findings = [
        finding(ConcernCode.LIMIT_EXCEEDED, ["txn:TXN-1", "mandate:MND-1"]),
        finding(ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE, ["assertion:ASR-1", "evidence:EV-1"]),
    ]
    result = judged(case, Decision.REVIEW, findings)
    assert statuses(result) == [A, V] and result.final_decision is Decision.REVIEW
    assert [a.effect for a in result.adjudications] == [FindingEffect.MODEL_ERROR, FindingEffect.ENFORCING]
    assert result.verdict_treatments[0].counted_as is Decision.REVIEW and result.verdict_treatments[0].enforcing == 1


def test_a_verdict_with_no_findings_enforces_nothing():
    result = judged(get_case("case-01"), Decision.REVIEW, [])
    assert result.final_decision is Decision.ALLOW and "offered no findings" in result.verdict_treatments[0].reason


def test_an_allow_verdict_is_never_raised_by_adjudication():
    result = judged(get_case("case-01"), Decision.ALLOW, [finding(ConcernCode.OTHER_CONCERN, ["merchant:MER-1"])])
    assert result.final_decision is Decision.ALLOW and result.verdict_treatments[0].counted_as is Decision.ALLOW


# --- model BLOCK cannot escape deterministic control ----------------------------------


@pytest.mark.parametrize("findings", [[], [finding(ConcernCode.LIMIT_EXCEEDED, ["txn:TXN-1"])], [finding(ConcernCode.OTHER_CONCERN, ["txn:TXN-1"])]])
def test_a_model_block_never_becomes_a_block(findings):
    case = get_case("case-01")
    result = judged(case, Decision.BLOCK, findings)
    assert result.final_decision is not Decision.BLOCK and result.deterministic_decision is Decision.ALLOW and result.authorised


@pytest.mark.parametrize("case_id", ["case-11", "case-12", "case-13", "case-20"])
def test_adjudication_cannot_lower_a_deterministic_block(case_id):
    refuting = [finding(ConcernCode.LIMIT_EXCEEDED, ["txn:TXN-1"])]
    for verdict in (Decision.ALLOW, Decision.REVIEW, Decision.BLOCK):
        assert judged(get_case(case_id), verdict, refuting).final_decision is Decision.BLOCK


# --- contradiction allegations ----------------------------------------------------------


def test_contradiction_confirmed_by_fresh_independent_evidence_is_verified():
    result = judged(get_case("case-05"), Decision.REVIEW, [finding(ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE, ["assertion:ASR-1", "evidence:EV-1"])])
    assert statuses(result) == [V] and result.adjudications[0].method == "claim_assessment"


@pytest.mark.parametrize("case_id", ["case-14", "case-16", "case-17"])  # no evidence, only stale evidence, conflicting evidence
def test_a_contradiction_that_evidence_does_not_establish_stays_unverified(case_id):
    case = get_case(case_id)
    ref = f"assertion:{case.assertions[0].assertion_id}"
    result = judged(case, Decision.REVIEW, [finding(ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE, [ref])])
    assert statuses(result) == [U] and result.adjudications[0].effect is FindingEffect.UNVERIFIED_RECORDED
    assert result.proven_false_offer_ids == ()  # an unverified allegation proves nothing
    assert result.final_decision is evaluate_case(case).decision  # and the model adds no enforcement of its own


def test_a_contradiction_alleged_against_a_claim_that_evidence_supports_is_contradicted():
    case = get_case("case-07")  # the merchant truthfully reports a change
    ref = f"assertion:{case.assertions[0].assertion_id}"
    result = judged(case, Decision.REVIEW, [finding(ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE, [ref])])
    assert statuses(result) == [A] and result.final_decision is evaluate_case(case).decision


def test_an_advertised_total_equal_to_the_lender_total_is_not_a_contradiction():
    f = finding(ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE, ["listing:OFFER-A", "terms:OFFER-A"])
    result = judged(two_offer_case(), Decision.REVIEW, [f])
    assert statuses(result) == [A] and result.final_decision is Decision.ALLOW


def test_an_advertised_total_that_differs_from_the_lender_total_is_verified():
    case = make_case("t-adv", [make_offer("OFFER-A", level("55"), advertised=gbp(600)), make_offer("OFFER-B", level("60"))], "OFFER-A")
    result = judged(case, Decision.REVIEW, [finding(ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE, ["listing:OFFER-A", "terms:OFFER-A"])])
    assert statuses(result) == [V] and "600.00 GBP" in result.adjudications[0].reason


def test_matching_totals_do_not_clear_a_listing_that_has_free_text_nobody_checked():
    a = make_offer("OFFER-A", level("55", fees=(fee(FeeKind.ADMINISTRATION, "20"),)), advertised=gbp(680), description="No extra fees at all.")
    case = make_case("t-text", [a, make_offer("OFFER-B", level("60"))], "OFFER-A")
    result = judged(case, Decision.REVIEW, [finding(ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE, ["listing:OFFER-A"])])
    assert statuses(result) == [U] and result.final_decision is Decision.ALLOW  # recorded as uncertainty, not enforced


def test_a_contradiction_citing_no_merchant_claim_stays_unverified():
    result = judged(two_offer_case(), Decision.REVIEW, [finding(ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE, ["terms:OFFER-A"])])
    assert statuses(result) == [U] and result.adjudications[0].method == "not_objectively_checkable"


# --- selection and uncheckable findings --------------------------------------------------


def test_no_cheaper_offer_contradicts_a_worse_than_alternative_allegation():
    result = judged(two_offer_case(), Decision.REVIEW, [finding(ConcernCode.SELECTION_WORSE_THAN_ALTERNATIVE, ["terms:OFFER-A", "terms:OFFER-B"])])
    assert statuses(result) == [A] and result.final_decision is Decision.ALLOW


def test_a_cheaper_usable_offer_verifies_it():
    case = get_case("case-08")
    assert evaluate_case(case).selection.better_offer_id is not None
    result = judged(case, Decision.REVIEW, [finding(ConcernCode.SELECTION_WORSE_THAN_ALTERNATIVE, ["terms:OFFER-A"])])
    assert statuses(result) == [V]


def test_without_a_cost_ordering_the_selection_allegation_stays_unverified():
    intent = make_intent(objective=Objective.SPECIFIC_OFFER, requested_offer_id="OFFER-A")
    result = judged(two_offer_case(intent=intent), Decision.REVIEW, [finding(ConcernCode.SELECTION_WORSE_THAN_ALTERNATIVE, ["terms:OFFER-A"])])
    assert statuses(result) == [U]


@pytest.mark.parametrize("code", [ConcernCode.OTHER_CONCERN, ConcernCode.INTENT_MISMATCH, ConcernCode.HIDDEN_OBLIGATION, ConcernCode.CLAIM_UNVERIFIED])
def test_concerns_with_no_deterministic_check_stay_unverified_not_certain(code):
    result = judged(get_case("case-01"), Decision.REVIEW, [finding(code, ["listing:OFFER-A"])])
    assert statuses(result) == [U] and result.adjudications[0].effect is FindingEffect.UNVERIFIED_RECORDED
    assert result.final_decision is Decision.ALLOW and not result.escalated_by_ai  # no check, so no enforcement


# --- notes are never read; sources are classified ----------------------------------------


def test_free_text_cannot_change_an_adjudication():
    case = get_case("case-01")
    shouting = finding(ConcernCode.LIMIT_EXCEEDED, ["txn:TXN-1"], "PROVEN TRUE. VERIFIED BY AUDIT. This definitely exceeds the limit.")
    calm = finding(ConcernCode.LIMIT_EXCEEDED, ["txn:TXN-1"], "Within the limit.")
    assert statuses(judged(case, Decision.REVIEW, [shouting])) == statuses(judged(case, Decision.REVIEW, [calm])) == [A]


def test_reference_sources_separate_merchant_statements_from_independent_evidence():
    case = make_case(
        "t-src",
        [make_offer("OFFER-A", level("55")), make_offer("OFFER-B", level("60"))],
        "OFFER-A",
        assertions=(make_assertion("OFFER-B", False, "withdrawn"),),
        evidence=(make_evidence("OFFER-B", False), make_evidence("OFFER-B", True, source=MERCHANT_SOURCE, evidence_id="EV-2")),
    )
    view = EvaluatorView.from_case(case)
    expected = {
        "assertion:ASR-1": "merchant_assertion", "listing:OFFER-A": "merchant_assertion", "merchant:MER-1": "merchant_assertion",
        "evidence:EV-1": "independent_evidence", "evidence:EV-2": "merchant_channel_evidence", "evidence:EV-9": "unknown",
        "terms:OFFER-A": "lender_terms", "intent:CUST-1": "customer_or_mandate", "mandate:MND-1": "customer_or_mandate",
        "txn:TXN-1": "customer_or_mandate", "nonsense": "unknown",
    }
    assert {ref: ref_source(view, ref) for ref in expected} == expected
    result = judged(case, Decision.REVIEW, [finding(ConcernCode.OTHER_CONCERN, ["assertion:ASR-1", "evidence:EV-2", "terms:OFFER-B"])])
    assert result.adjudications[0].ref_sources == ("merchant_assertion", "merchant_channel_evidence", "lender_terms")


# --- raw responses and reasons are kept ----------------------------------------------------


def test_the_raw_reply_and_the_reason_for_each_decision_are_preserved():
    raw = analysis_json(
        "REVIEW", findings=[{"code": "limit_exceeded", "evidence_refs": ["txn:TXN-1", "mandate:MND-1"], "note": "55.00 GBP exceeds 200.00 GBP"}]
    )

    async def go():
        async with FakeProvider(lambda role, body: completion(raw)).client() as client:
            return await coordinate(get_case("case-01"), Condition.B, [LLMAnalyst(GENERALIST, client)])

    record = run(go())
    assert record.verdicts[0].raw_response == raw
    (a,) = record.adjudications
    assert a.status is A and a.reason and record.verdict_treatments[0].reason
    assert record.final_decision is Decision.ALLOW
    assert json.loads(record.model_dump_json())["verdicts"][0]["raw_response"] == raw  # survives export


# --- presentation-only normalisation ---------------------------------------------------------


LONG = "x" * 450


def reply(**overrides) -> str:
    finding_ = {"code": "limit_exceeded", "evidence_refs": ["txn:TXN-1"], "note": "short"}
    return analysis_json("REVIEW", findings=[finding_], **overrides)


def reply_with_long_note(**finding_overrides) -> str:
    finding_ = {"code": "limit_exceeded", "evidence_refs": ["txn:TXN-1"], "note": LONG, **finding_overrides}
    return analysis_json("REVIEW", findings=[finding_])


def test_strict_parsing_is_unchanged_and_still_rejects_a_long_note():
    with pytest.raises(InvalidResponse):
        parse_analysis(reply_with_long_note())


def test_an_over_long_note_is_truncated_for_display_and_the_original_is_kept():
    analysis, norms = parse_analysis_normalised(reply_with_long_note())
    (norm,) = norms
    assert norm.path == "findings.0.note" and norm.original_text == LONG and norm.original_length == 450 and norm.kept_length == 300
    note = analysis.findings[0].note
    assert len(note) == 300 and note.endswith("…") and note[:-1] == LONG[:299]


def test_an_over_long_explanation_and_abstention_reason_are_presentation_text_too():
    _, norms = parse_analysis_normalised(reply(explanation="e" * 900))
    assert [n.path for n in norms] == ["explanation"] and norms[0].kept_length == 800
    _, norms = parse_analysis_normalised(analysis_json(None, abstain=True, abstain_reason="r" * 400))
    assert [n.path for n in norms] == ["abstain_reason"]


def test_a_reply_that_fits_needs_no_repair():
    analysis, norms = parse_analysis_normalised(reply())
    assert norms == () and analysis.findings[0].note == "short"


def test_normalisation_changes_nothing_but_the_over_long_text():
    raw = reply_with_long_note()
    analysis, _ = parse_analysis_normalised(raw)
    original = json.loads(raw)["findings"][0]
    repaired = analysis.findings[0]
    assert (repaired.code.value, list(repaired.evidence_refs)) == (original["code"], original["evidence_refs"])
    assert analysis.verdict is Decision.REVIEW and analysis.risk_score == 0.1


CRITICAL_DEFECTS = {
    "unknown finding code": lambda: reply_with_long_note(code="made_up_code"),
    "empty evidence references": lambda: reply_with_long_note(evidence_refs=[]),
    "missing evidence references": lambda: json.dumps({**json.loads(reply_with_long_note()), "findings": [{"code": "limit_exceeded", "note": LONG}]}),
    "invalid verdict": lambda: json.dumps({**json.loads(reply_with_long_note()), "verdict": "MAYBE"}),
    "risk score out of range": lambda: json.dumps({**json.loads(reply_with_long_note()), "risk_score": 7}),
    "unexpected extra field": lambda: json.dumps({**json.loads(reply_with_long_note()), "payee_id": "PAY-EVIL"}),
    "availability claim filed with an amount": lambda: json.dumps(
        {**json.loads(reply_with_long_note()), "extracted_claims": [
            {"kind": "availability", "offer_id": "OFFER-A", "source_ref": "assertion:ASR-1", "quote": "withdrawn",
             "asserted_available": False, "amount_minor": 66000, "currency": "GBP"}]}
    ),
    "total claim without an amount": lambda: json.dumps(
        {**json.loads(reply_with_long_note()), "extracted_claims": [
            {"kind": "total_repayable", "offer_id": "OFFER-A", "source_ref": "listing:OFFER-A", "quote": "660 GBP"}]}
    ),
    "negative amount": lambda: json.dumps(
        {**json.loads(reply_with_long_note()), "extracted_claims": [
            {"kind": "total_repayable", "offer_id": "OFFER-A", "source_ref": "listing:OFFER-A", "quote": "660 GBP",
             "amount_minor": -1, "currency": "GBP"}]}
    ),
    "over-long quote": lambda: json.dumps(
        {**json.loads(reply_with_long_note()), "extracted_claims": [
            {"kind": "no_additional_fees", "offer_id": "OFFER-A", "source_ref": "listing:OFFER-A", "quote": LONG}]}
    ),
    "abstaining with a verdict": lambda: json.dumps({**json.loads(reply_with_long_note()), "abstain": True, "abstain_reason": "x"}),
}


@pytest.mark.parametrize("defect", CRITICAL_DEFECTS)
def test_a_long_note_never_carries_a_malformed_financial_field_through(defect):
    raw = CRITICAL_DEFECTS[defect]()
    with pytest.raises(InvalidResponse) as caught:
        parse_analysis_normalised(raw)
    assert LONG not in str(caught.value)  # the error never echoes the model's text


@pytest.mark.parametrize(
    "mutation",
    [
        {"evidence_refs": ["evidence:INVENTED"]},  # an identifier that was never shown
    ],
)
def test_an_ungrounded_reference_is_rejected_even_when_the_note_was_repaired(mutation):
    provider = FakeProvider(lambda role, body: completion(reply_with_long_note(**mutation)))

    async def go():
        async with provider.client() as client:
            return await LLMAnalyst(GENERALIST, client).evaluate(EvaluatorView.from_case(get_case("case-01")))

    verdict = run(go())
    assert verdict.status is EvaluatorStatus.INVALID_RESPONSE and verdict.verdict is None
    assert verdict.raw_response is not None and LONG in verdict.raw_response  # the audit copy is intact
    assert verdict.findings == () and verdict.extracted_claims == ()  # nothing from a rejected reply is used


def test_through_the_client_a_repaired_reply_is_flagged_and_keeps_its_raw_text():
    raw = reply_with_long_note()
    provider = FakeProvider(lambda role, body: completion(raw))

    async def go():
        async with provider.client() as client:
            return await LLMAnalyst(GENERALIST, client).evaluate(EvaluatorView.from_case(get_case("case-01")))

    verdict = run(go())
    assert verdict.status is EvaluatorStatus.OK and verdict.normalised
    assert verdict.raw_response == raw and verdict.normalisations[0].original_text == LONG
    assert len(verdict.findings[0].note) == 300


# --- amount reading and source grounding ----------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Total repayable £660 and no extra fees.", ["660.00 GBP"]),
        ('"advertised_total_repayable": "660.00 GBP"', ["660.00 GBP"]),
        ("GBP 1,234.50 or 12 GBP", ["1234.50 GBP", "12.00 GBP"]),
        ("$5 and €7.5", ["5.00 USD", "7.50 EUR"]),
        ("£55 a month for 12 months, £660 in all", ["55.00 GBP", "660.00 GBP"]),
        ("about 660 pounds", []),
        ("1234.567 GBP", []),
        ("no amounts here", []),
    ],
)
def test_amounts_are_read_exactly_and_never_guessed(text, expected):
    assert [m.plain() for m in parse_amounts(text)] == expected


def claim(kind, quote, **fields) -> ExtractedClaim:
    return ExtractedClaim(kind=kind, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote=quote, **fields)


def text_case(description):
    a = make_offer("OFFER-A", level("55", fees=(fee(FeeKind.ADMINISTRATION, "20"),)), advertised=None, description=description)
    return make_case("t-grd", [a, make_offer("OFFER-B", level("60"))], "OFFER-A")  # lender total 680.00, fees 20.00


def crosscheck_of(case, extracted):
    result = coordinate_sync(case, [Scripted({case.case_id: make_analysis(Decision.ALLOW, claims=[extracted])})])
    (check,) = result.crosschecks
    return check, result


def test_an_extracted_total_must_be_the_amount_its_quote_states():
    case = text_case("Total repayable £660.")
    check, result = crosscheck_of(case, claim(ClaimKind.TOTAL_REPAYABLE, "Total repayable £660", amount_minor=68000, currency="GBP"))
    assert check.outcome is CrossCheckOutcome.EXTRACTION_DISAGREES and not check.contradicted
    assert "680.00 GBP" in check.detail and "660.00 GBP" in check.detail
    assert result.proven_false_offer_ids == ()  # a model cannot manufacture a verified contradiction from a misread quote


def test_a_total_that_is_read_correctly_and_contradicts_the_lender_is_verified_by_code():
    check, result = crosscheck_of(text_case("Total repayable £660."), claim(ClaimKind.TOTAL_REPAYABLE, "Total repayable £660", amount_minor=66000, currency="GBP"))
    assert check.outcome is CrossCheckOutcome.CONTRADICTED_BY_TERMS and check.grounded_in == "listing:OFFER-A.description"
    assert result.proven_false_offer_ids == ("OFFER-A",) and result.final_decision is Decision.REVIEW


def test_a_quote_without_an_explicit_amount_cannot_support_an_extracted_figure():
    check, _ = crosscheck_of(text_case("Total repayable is very reasonable."), claim(ClaimKind.TOTAL_REPAYABLE, "Total repayable is very reasonable", amount_minor=66000, currency="GBP"))
    assert check.outcome is CrossCheckOutcome.EXTRACTION_DISAGREES


def test_an_explicit_total_is_verified_even_when_the_quote_also_states_other_amounts():
    """Audit F4: previously any second amount in the quote made the whole claim inconclusive, so a real understatement was missed."""
    text = "£55 a month for 12 months, £660 in all"
    check, result = crosscheck_of(text_case(text), claim(ClaimKind.TOTAL_REPAYABLE, text, amount_minor=66000, currency="GBP"))
    assert check.outcome is CrossCheckOutcome.CONTRADICTED_BY_TERMS and result.final_decision is Decision.REVIEW


def test_a_quote_whose_only_matching_amount_is_an_instalment_is_not_a_total():
    text = "£55 a month for 12 months, £660 in all"
    check, result = crosscheck_of(text_case(text), claim(ClaimKind.TOTAL_REPAYABLE, text, amount_minor=5500, currency="GBP"))
    assert check.outcome is CrossCheckOutcome.EXTRACTION_DISAGREES and result.final_decision is Decision.ALLOW


def test_fees_are_compared_with_the_lender_fee_schedule():
    case = text_case("An administration fee of £20 applies. Total repayable £680.")
    consistent, _ = crosscheck_of(case, claim(ClaimKind.FEE_AMOUNT, "administration fee of £20", amount_minor=2000, currency="GBP"))
    assert consistent.outcome is CrossCheckOutcome.CONSISTENT and "20.00 GBP" in consistent.detail
    understated, result = crosscheck_of(case, claim(ClaimKind.FEE_AMOUNT, "administration fee of £20", amount_minor=500, currency="GBP"))
    assert understated.outcome is CrossCheckOutcome.EXTRACTION_DISAGREES  # the quote says £20, the model says £5


def test_a_stated_fee_total_that_differs_from_the_lender_is_a_verified_contradiction():
    case = text_case("A fee of £5 applies.")
    check, _ = crosscheck_of(case, claim(ClaimKind.FEE_AMOUNT, "A fee of £5", amount_minor=500, currency="GBP"))
    assert check.outcome is CrossCheckOutcome.CONTRADICTED_BY_TERMS and "20.00 GBP" in check.detail


def test_a_fee_claim_needs_an_amount_and_an_ungrounded_quote_is_still_discarded():
    with pytest.raises(ValueError):
        claim(ClaimKind.FEE_AMOUNT, "A fee of £5")
    extracted = claim(ClaimKind.FEE_AMOUNT, "a fee of £20 applies", amount_minor=2000, currency="GBP")
    result = coordinate_sync(text_case("A fee of £5 applies."), [Scripted({"t-grd": make_analysis(Decision.ALLOW, claims=[extracted])})])
    (verdict,) = result.verdicts
    assert verdict.status is EvaluatorStatus.INVALID_RESPONSE and "not in listing:OFFER-A" in verdict.abstain_reason and result.crosschecks == ()


def test_a_merchant_assertion_is_not_independent_evidence_and_does_not_verify_itself():
    """A merchant's own say-so, even with a matching extracted claim, never becomes a verified contradiction."""
    case = get_case("case-14")  # an uncorroborated withdrawal claim
    extracted = ExtractedClaim(kind=ClaimKind.AVAILABILITY, offer_id="OFFER-A", source_ref="assertion:ASR-1", quote="withdrawn", asserted_available=False)
    result = coordinate_sync(case, [Scripted({"case-14": make_analysis(Decision.ALLOW, claims=[extracted], findings=[finding(ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE, ["assertion:ASR-1"])])})])
    assert [c.outcome for c in result.crosschecks] == [CrossCheckOutcome.NOT_COMPARABLE]
    assert statuses(result) == [U] and result.proven_false_offer_ids == ()
