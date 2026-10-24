"""Deterministic coordination: precedence, escalation limits, failures, disagreement and cross-checks."""

import asyncio

import pytest

from sika_guard.coordinator import Condition, Coverage, CoordinatorConfig, coordinate
from sika_guard.evaluators.contract import (
    AIFinding,
    ClaimKind,
    ConcernCode,
    EvaluatorRole,
    EvaluatorStatus,
    ExtractedClaim,
    verdict_from_analysis,
)
from sika_guard.evaluators.crosscheck import CrossCheckOutcome
from sika_guard.evaluators.mocks import AlwaysAllow, AlwaysReview, MalformedResponse, Scripted, Timeout, make_analysis
from sika_guard.evaluators.payloads import collect_refs, generalist_payload
from sika_guard.fixtures.builders import fee, gbp, level, make_case, make_offer
from sika_guard.fixtures.cases import build_cases, get_case
from sika_guard.models import Decision, FeeKind
from sika_guard.policy import evaluate_case

HARD_BLOCK_CASES = ["case-11", "case-12", "case-13", "case-20"]


def run(coro):
    return asyncio.run(coro)


def coordinate_sync(case, evaluators=(), condition=Condition.B, **config):
    return run(coordinate(case, condition, evaluators, CoordinatorConfig(**config)))


def contradiction(refs, offer="OFFER-A") -> AIFinding:
    return AIFinding(code=ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE, evidence_refs=tuple(refs), note=f"{offer} is claimed contradicted")


# --- deterministic precedence -----------------------------------------------


@pytest.mark.parametrize("case_id", HARD_BLOCK_CASES)
@pytest.mark.parametrize(
    "make_evaluator",
    [AlwaysAllow, AlwaysReview, Timeout, MalformedResponse, lambda: Scripted({}), lambda: Scripted({"x": make_analysis(Decision.ALLOW, risk_score=1.0)})],
)
def test_deterministic_block_remains_block_whatever_the_evaluators_say(case_id, make_evaluator):
    case = get_case(case_id)
    assert evaluate_case(case).decision is Decision.BLOCK
    for fail_closed in (False, True):
        assert coordinate_sync(case, [make_evaluator()], fail_closed=fail_closed).final_decision is Decision.BLOCK


def test_ai_can_raise_allow_to_review_but_never_to_block():
    case = get_case("case-01")
    assert coordinate_sync(case, [AlwaysReview()]).final_decision is Decision.REVIEW
    shouted_block = Scripted({"case-01": make_analysis(Decision.BLOCK, risk_score=1.0)})
    result = coordinate_sync(case, [shouted_block])
    assert result.final_decision is Decision.REVIEW and result.escalated_by_ai


def test_ai_allow_cannot_lower_a_deterministic_review():
    result = coordinate_sync(get_case("case-05"), [AlwaysAllow()])
    assert result.deterministic_decision is Decision.REVIEW and result.final_decision is Decision.REVIEW
    assert not result.escalated_by_ai


def test_high_confidence_gives_a_finding_no_authority():
    case = get_case("case-01")
    confident_but_allowing = Scripted(
        {"case-01": make_analysis(Decision.ALLOW, risk_score=1.0, findings=[contradiction(["listing:OFFER-A"])])}
    )
    result = coordinate_sync(case, [confident_but_allowing])
    assert result.final_decision is Decision.ALLOW
    assert result.proven_false_offer_ids == ()
    unconfident_but_reviewing = Scripted({"case-01": make_analysis(Decision.REVIEW, risk_score=0.0)})
    assert coordinate_sync(case, [unconfident_but_reviewing]).final_decision is Decision.REVIEW


def test_unverified_claim_is_not_proven_false_by_a_model_allegation():
    case = get_case("case-14")  # merchant says A is withdrawn; no independent evidence exists
    alleging = Scripted({"case-14": make_analysis(Decision.REVIEW, findings=[contradiction(["assertion:ASR-1"])])})
    result = coordinate_sync(case, [alleging])
    assert "no_independent_evidence" in result.deterministic_finding_codes
    assert result.proven_false_offer_ids == ()
    assert result.uncorroborated_contradictions == ("mock-scripted:OFFER-A",)
    assert result.final_decision is Decision.REVIEW  # escalated for a human, without a fraud finding


def test_allegation_matching_a_deterministic_contradiction_is_corroborated():
    case = get_case("case-05")
    alleging = Scripted({"case-05": make_analysis(Decision.REVIEW, findings=[contradiction(["evidence:EV-1", "assertion:ASR-1"])])})
    result = coordinate_sync(case, [alleging])
    assert result.proven_false_offer_ids == ("OFFER-A",)
    assert result.uncorroborated_contradictions == ()


def test_condition_a_equals_the_deterministic_policy():
    for case in build_cases():
        record = coordinate_sync(case, [], Condition.A)
        assert record.final_decision is evaluate_case(case).decision
        assert record.coverage is Coverage.NOT_APPLICABLE and record.verdicts == ()


# --- failures, abstention and disagreement ----------------------------------


@pytest.mark.parametrize("make_evaluator", [Timeout, MalformedResponse])
def test_failures_are_explicit_abstentions_and_not_evidence_of_fraud(make_evaluator):
    result = coordinate_sync(get_case("case-14"), [make_evaluator()])
    assert result.coverage is Coverage.UNAVAILABLE
    assert len(result.abstained_evaluators) == 1
    assert result.verdicts[0].abstained and result.verdicts[0].verdict is None
    assert result.proven_false_offer_ids == () and result.uncorroborated_contradictions == ()
    assert result.final_decision is Decision.ALLOW  # fail-open by default: the rule engine's decision stands


def test_fail_closed_turns_missing_judgement_into_review_without_accusing_anyone():
    result = coordinate_sync(get_case("case-01"), [Timeout()], fail_closed=True)
    assert result.final_decision is Decision.REVIEW
    assert result.proven_false_offer_ids == ()


def test_disagreement_and_abstention_are_separate_outcomes():
    case = get_case("case-01")
    disagree = coordinate_sync(case, [AlwaysAllow("a"), AlwaysReview("b")], Condition.C)
    assert disagree.evaluator_disagreement and disagree.abstained_evaluators == ()
    abstain = coordinate_sync(case, [AlwaysAllow("a"), Timeout("b")], Condition.C)
    assert not abstain.evaluator_disagreement and abstain.abstained_evaluators == ("b",)
    assert abstain.coverage is Coverage.PARTIAL


def test_there_is_no_majority_vote():
    result = coordinate_sync(get_case("case-01"), [AlwaysAllow("a"), AlwaysAllow("b"), AlwaysReview("c")], Condition.C)
    assert result.final_decision is Decision.REVIEW  # a single REVIEW is enough; it is not outvoted


def test_a_crashing_evaluator_is_recorded_as_a_failure():
    class Crashes:
        evaluator_id, role = "crashes", EvaluatorRole.MOCK

        async def evaluate(self, view):
            raise RuntimeError("secret-sentinel-should-not-be-recorded")

    result = coordinate_sync(get_case("case-01"), [Crashes(), AlwaysAllow()], Condition.C)
    crashed = result.verdicts[0]
    assert crashed.status is EvaluatorStatus.PROVIDER_ERROR and "RuntimeError" in crashed.abstain_reason
    assert "sentinel" not in crashed.model_dump_json()
    assert result.verdicts[1].usable


def test_a_verdict_about_another_case_is_rejected():
    other = get_case("case-08")

    class Confused:
        evaluator_id, role = "confused", EvaluatorRole.MOCK

        async def evaluate(self, view):
            from sika_guard.evaluators.contract import EvaluatorView

            wrong = EvaluatorView.from_case(other)
            return verdict_from_analysis(
                make_analysis(Decision.ALLOW), wrong, evaluator_id="confused", role=EvaluatorRole.MOCK,
                allowed_refs=collect_refs(generalist_payload(wrong)),
            )

    result = coordinate_sync(get_case("case-01"), [Confused()])
    assert result.verdicts[0].status is EvaluatorStatus.INVALID_RESPONSE


# --- trusted cross-checks of extracted claims --------------------------------


def _free_text_case():
    """The listing's prose understates the cost and denies fees; the lender terms say otherwise."""
    a = make_offer(
        "OFFER-A", level("55", fees=(fee(FeeKind.ADMINISTRATION, "20"),)), advertised=None,
        description="Total repayable £660 and no extra fees.",
    )
    b = make_offer("OFFER-B", level("60"))
    return make_case("t-free", [a, b], "OFFER-A")


def _claim(kind, quote, **fields) -> ExtractedClaim:
    return ExtractedClaim(kind=kind, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote=quote, **fields)


def test_a_claim_contradicted_by_lender_terms_escalates_even_if_the_model_says_allow():
    case = _free_text_case()
    assert evaluate_case(case).decision is Decision.ALLOW  # nothing structured is wrong
    claims = [_claim(ClaimKind.TOTAL_REPAYABLE, "Total repayable £660", amount_minor=66000, currency="GBP"), _claim(ClaimKind.NO_ADDITIONAL_FEES, "no extra fees")]
    result = coordinate_sync(case, [Scripted({"t-free": make_analysis(Decision.ALLOW, claims=claims)})])
    assert {c.outcome for c in result.crosschecks} == {CrossCheckOutcome.CONTRADICTED_BY_TERMS}
    assert result.final_decision is Decision.REVIEW and result.proven_false_offer_ids == ("OFFER-A",)


def test_an_extraction_whose_quote_is_not_in_the_source_is_discarded():
    claim = _claim(ClaimKind.NO_ADDITIONAL_FEES, "guaranteed zero fees, honestly")
    result = coordinate_sync(_free_text_case(), [Scripted({"t-free": make_analysis(Decision.ALLOW, claims=[claim])})])
    assert [c.outcome for c in result.crosschecks] == [CrossCheckOutcome.UNGROUNDED]
    assert result.final_decision is Decision.ALLOW


def test_a_claim_that_matches_the_lender_terms_is_consistent():
    claim = _claim(ClaimKind.TOTAL_REPAYABLE, "Total repayable £660", amount_minor=68000, currency="GBP")
    result = coordinate_sync(_free_text_case(), [Scripted({"t-free": make_analysis(Decision.ALLOW, claims=[claim])})])
    assert [c.outcome for c in result.crosschecks] == [CrossCheckOutcome.CONSISTENT]
    assert result.final_decision is Decision.ALLOW


def test_availability_extraction_is_checked_against_deterministic_evidence_rules():
    case = get_case("case-05")
    claim = ExtractedClaim(
        kind=ClaimKind.AVAILABILITY, offer_id="OFFER-A", source_ref="assertion:ASR-1", quote="has been withdrawn", asserted_available=False
    )
    result = coordinate_sync(case, [Scripted({"case-05": make_analysis(Decision.ALLOW, claims=[claim])})])
    assert [c.outcome for c in result.crosschecks] == [CrossCheckOutcome.CONTRADICTED_BY_EVIDENCE]

    uncorroborated = get_case("case-14")
    result = coordinate_sync(uncorroborated, [Scripted({"case-14": make_analysis(Decision.ALLOW, claims=[claim])})])
    assert [c.outcome for c in result.crosschecks] == [CrossCheckOutcome.NOT_COMPARABLE]  # absence of evidence is not a contradiction
    assert result.final_decision is Decision.ALLOW


def test_extraction_that_misreads_the_structured_assertion_is_discarded():
    claim = ExtractedClaim(
        kind=ClaimKind.AVAILABILITY, offer_id="OFFER-A", source_ref="assertion:ASR-1", quote="has been withdrawn", asserted_available=True
    )
    result = coordinate_sync(get_case("case-05"), [Scripted({"case-05": make_analysis(Decision.ALLOW, claims=[claim])})])
    assert [c.outcome for c in result.crosschecks] == [CrossCheckOutcome.UNGROUNDED]
