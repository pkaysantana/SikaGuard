"""Direction B, authorisation trust boundaries and financial correctness: Opus findings R1, R2, R3, R4, R7, R8, R9 and the resolved F6/F7.

Each finding has the counterexample (which must now be handled) and a control showing a legitimate input of the same shape still works.
Offline and synthetic; no model is called.
"""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from sika_guard.evaluators.contract import EvaluatorView
from sika_guard.evaluators.payloads import generalist_payload
from sika_guard.fixtures.builders import (
    AGGREGATOR_SOURCE,
    LENDER_SOURCE,
    MERCHANT_SOURCE,
    NOW,
    T0,
    fee,
    gbp,
    level,
    make_assertion,
    make_case,
    make_evidence,
    make_intent,
    make_mandate,
    make_offer,
)
from sika_guard.models import (
    AvailabilitySubject,
    Decision,
    DecisionBasis,
    EvidenceSource,
    FeeKind,
    FindingCategory,
    Money,
    RepaymentSchedule,
    Scope,
    SourceKind,
)
from sika_guard.policy import (
    ClaimStatus,
    Direction,
    PolicyConfig,
    assess_claim,
    compare_figures,
    decision_bases,
    evaluate_case,
    hidden_obligation_is_material,
    understatement_is_material,
)


def pair(monthly_a="55", monthly_b="62"):
    return make_offer("OFFER-A", level(monthly_a)), make_offer("OFFER-B", level(monthly_b))


def codes(result):
    return {f.code.value for f in result.findings}


def decide(case, config=None):
    return evaluate_case(case) if config is None else evaluate_case(case, config)


# --- R1: the decision clock is the trusted one ---------------------------------------------------------------------------------------------


def test_r1_an_expired_mandate_cannot_be_rescued_by_a_proposal_back_dated_into_its_window():
    a, b = pair()
    expired = make_mandate(issued_at=NOW - timedelta(days=60), expires_at=NOW - timedelta(days=30))
    result = decide(make_case("r1", [a, b], "OFFER-A", mandate=expired, proposed_at=NOW - timedelta(days=45), evaluated_at=NOW))
    assert result.decision is Decision.BLOCK and "mandate_expired" in codes(result)


def test_r1_a_proposal_dated_after_the_decision_clock_is_a_hard_violation():
    a, b = pair()
    result = decide(make_case("r1b", [a, b], "OFFER-A", proposed_at=NOW + timedelta(hours=2), evaluated_at=NOW))
    assert result.decision is Decision.BLOCK and "proposal_from_the_future" in codes(result)


def test_r1_back_dating_a_proposal_does_not_make_fresh_contradicting_evidence_irrelevant():
    a, b = pair()
    contradicting = (make_evidence("OFFER-A", False),)  # fresh lender observation at T0+3m: A is NOT available
    claim = (make_assertion("OFFER-A", True, "In stock, available now", at=T0),)
    honest = make_case("r1c", [a, b], "OFFER-A", assertions=claim, evidence=contradicting)
    backdated = make_case("r1d", [a, b], "OFFER-A", assertions=claim, evidence=contradicting, proposed_at=T0 + timedelta(minutes=1), evaluated_at=NOW)
    assert decide(honest).decision is Decision.REVIEW
    assert decide(backdated).decision is Decision.REVIEW and "availability_contradicted" in codes(decide(backdated))


def test_r1_control_a_proposal_a_little_before_the_decision_clock_is_ordinary():
    a, b = pair()
    result = decide(make_case("r1e", [a, b], "OFFER-A", proposed_at=NOW - timedelta(minutes=2), evaluated_at=NOW))
    assert result.decision is Decision.ALLOW and result.authorised


def test_r1_every_evaluator_judges_time_at_the_trusted_clock_not_the_proposals():
    a, b = pair()
    case = make_case("r1f", [a, b], "OFFER-A", proposed_at=NOW - timedelta(hours=3), evaluated_at=NOW)
    view = EvaluatorView.from_case(case)
    assert view.as_of == NOW and generalist_payload(view)["task_context"]["as_of"] == NOW.isoformat()
    assert generalist_payload(view)["trusted"]["trusted_context"]["evaluated_at"] == NOW.isoformat()


# --- R2: scope comes from the lender's product, not from the proposal ------------------------------------------------------------------------


def test_r2_a_credit_agreement_declared_as_a_card_purchase_is_a_scope_mismatch_even_under_a_card_only_mandate():
    a, b = pair()
    card_only = make_mandate(permitted_scopes=frozenset({Scope.CARD_PURCHASE}))
    result = decide(make_case("r2", [a, b], "OFFER-A", mandate=card_only, scope=Scope.CARD_PURCHASE))
    assert result.decision is Decision.BLOCK
    assert {"scope_offer_mismatch", "scope_not_permitted"} <= codes(result)  # the declared scope is a lie AND the real one is not permitted


def test_r2_declaring_a_permitted_scope_does_not_change_the_offers_own():
    a, b = pair()
    both = make_mandate(permitted_scopes=frozenset({Scope.CARD_PURCHASE}))
    result = decide(make_case("r2b", [a, b], "OFFER-A", mandate=both, scope=Scope.CARD_PURCHASE))
    assert "scope_offer_mismatch" in codes(result) and result.decision is Decision.BLOCK


def test_r2_control_an_offer_whose_own_scope_the_mandate_permits_is_allowed():
    card_offer = make_offer("OFFER-A", level("55"), scope=Scope.CARD_PURCHASE)
    mandate = make_mandate(permitted_scopes=frozenset({Scope.CARD_PURCHASE}))
    assert decide(make_case("r2c", [card_offer, make_offer("OFFER-B", level("62"), scope=Scope.CARD_PURCHASE)], "OFFER-A", mandate=mandate, scope=Scope.CARD_PURCHASE)).decision is Decision.ALLOW
    a, b = pair()
    assert decide(make_case("r2d", [a, b], "OFFER-A")).decision is Decision.ALLOW


def test_r2_an_alternative_of_a_scope_the_mandate_does_not_permit_is_not_a_better_option():
    cheap_card = make_offer("OFFER-A", level("40"), scope=Scope.CARD_PURCHASE)  # much cheaper, but outside the mandate's scopes
    result = decide(make_case("r2e", [cheap_card, make_offer("OFFER-B", level("62"))], "OFFER-B"))
    assert "costlier_than_usable_offer" not in codes(result) and result.decision is Decision.ALLOW


# --- R3: the acting agent is bound to the mandate through the trusted context ----------------------------------------------------------------


def test_r3_a_mandate_issued_to_a_different_agent_is_a_hard_violation():
    a, b = pair()
    result = decide(make_case("r3", [a, b], "OFFER-A", mandate=make_mandate(agent_id="AGENT-SOMEONE-ELSE")))
    assert result.decision is Decision.BLOCK and "agent_mismatch" in codes(result)


def test_r3_the_proposal_cannot_vouch_for_the_agent_because_it_carries_no_identity():
    from sika_guard.models import ProposedTransaction

    assert "agent_id" not in ProposedTransaction.model_fields  # the only identity that counts is the one the trusted context supplies


def test_r3_control_the_agent_the_trusted_context_names_may_use_its_own_mandate():
    a, b = pair()
    assert decide(make_case("r3b", [a, b], "OFFER-A", mandate=make_mandate(agent_id="AGENT-9"), acting_agent_id="AGENT-9")).decision is Decision.ALLOW


# --- F7: every individually committed payment stays within max_amount -----------------------------------------------------------------------


def test_f7_a_later_instalment_above_the_ceiling_blocks_even_when_the_first_payment_is_within_it():
    sched = RepaymentSchedule(payments=(gbp("30"),) + (gbp("250"),) * 2 + (gbp("30"),) * 9)
    case = make_case("f7", [make_offer("OFFER-A", sched)], "OFFER-A", intent=make_intent(max_monthly_payment=None, max_total_repayment=gbp(2000)),
                     mandate=make_mandate(max_monthly_payment=None, max_total_repayment=gbp(2000), max_amount=gbp(200)))
    result = decide(case)
    assert result.decision is Decision.BLOCK and "committed_payment_exceeds_mandate" in codes(result)


def test_f7_a_fee_that_is_not_due_upfront_is_a_committed_payment_of_its_own():
    sched = level("55", fees=(fee(FeeKind.SERVICING, "250", due_upfront=False),))
    case = make_case("f7b", [make_offer("OFFER-A", sched)], "OFFER-A", intent=make_intent(max_total_repayment=gbp(2000)),
                     mandate=make_mandate(max_total_repayment=gbp(2000), max_amount=gbp(200)))
    assert "committed_payment_exceeds_mandate" in codes(decide(case))


def test_f7_controls_the_ceiling_is_inclusive_an_upfront_fee_is_part_of_the_initial_charge_and_ordinary_schedules_pass():
    at_ceiling = RepaymentSchedule(payments=(gbp("200"),) + (gbp("55"),) * 11)
    case = make_case("f7c", [make_offer("OFFER-A", at_ceiling)], "OFFER-A", intent=make_intent(max_monthly_payment=gbp(200), max_total_repayment=gbp(2000)),
                     mandate=make_mandate(max_monthly_payment=gbp(200), max_total_repayment=gbp(2000), max_amount=gbp(200)))
    assert decide(case).decision is Decision.ALLOW and "committed_payment_exceeds_mandate" not in codes(decide(case))
    upfront = level("55", fees=(fee(FeeKind.ARRANGEMENT, "20", due_upfront=True),))  # initial charge 75, under 200
    assert decide(make_case("f7d", [make_offer("OFFER-A", upfront), make_offer("OFFER-B", level("80"))], "OFFER-A")).decision is Decision.ALLOW
    a, b = pair()
    assert decide(make_case("f7e", [a, b], "OFFER-A")).decision is Decision.ALLOW


def test_f7_an_alternative_with_a_later_instalment_above_the_ceiling_is_not_a_usable_better_option():
    steep = RepaymentSchedule(payments=(gbp("30"),) + (gbp("250"),) + (gbp("20"),) * 10)  # total 480, far cheaper than B, but breaks the ceiling later
    case = make_case("f7f", [make_offer("OFFER-A", steep), make_offer("OFFER-B", level("62"))], "OFFER-B")
    assert "costlier_than_usable_offer" not in codes(decide(case))


def test_f7_max_total_repayment_is_a_separate_limit_that_is_unchanged():
    a, b = pair()
    over_total = make_case("f7g", [a, b], "OFFER-A", mandate=make_mandate(max_total_repayment=gbp(600)), intent=make_intent(max_total_repayment=gbp(750)))
    result = decide(over_total)
    assert "mandate_total_exceeded" in codes(result) and "committed_payment_exceeds_mandate" not in codes(result)


# --- F6: an incompatible headline currency on the selected offer is a REVIEW with its own basis ---------------------------------------------


def test_f6_an_advertised_total_in_another_currency_on_the_selected_offer_is_held_without_any_conversion():
    a = make_offer("OFFER-A", level("55"), advertised=Money(amount_minor=70000, currency="USD"))
    result = decide(make_case("f6", [a, make_offer("OFFER-B", level("62"))], "OFFER-A"))
    assert result.decision is Decision.REVIEW and decision_bases(result) == (DecisionBasis.REVIEW_INCOMPARABLE_REPRESENTATION,)
    finding = next(f for f in result.findings if f.code.value == "advertised_currency_incompatible")
    assert finding.category is FindingCategory.INCOMPARABLE_REPRESENTATION and "nothing was converted" in finding.message
    assert "not evidence that the merchant is dishonest" in finding.message
    assert not result.has(FindingCategory.CONFIRMED_FALSE_CLAIM)  # nobody is accused


def test_f6_control_the_same_mismatch_on_an_offer_that_is_not_being_bought_is_only_recorded():
    other = make_offer("OFFER-B", level("62"), advertised=Money(amount_minor=70000, currency="USD"))
    result = decide(make_case("f6b", [make_offer("OFFER-A", level("55")), other], "OFFER-A"))
    assert result.decision is Decision.ALLOW and decision_bases(result) == (DecisionBasis.ALLOW_WITH_UNCERTAINTY,)


# --- R4: direction and predeclared materiality ----------------------------------------------------------------------------------------------------


def test_r4_compare_figures_has_a_direction_and_never_converts_currencies():
    assert compare_figures(gbp("600"), gbp("680")).direction is Direction.UNDERSTATES_COST
    assert compare_figures(gbp("700"), gbp("680")).direction is Direction.OVERSTATES_COST
    assert compare_figures(gbp("680"), gbp("680")).direction is Direction.EQUAL
    assert compare_figures(gbp("680"), Money(amount_minor=68000, currency="USD")) is None


def test_r4_an_advertised_total_above_the_real_one_is_recorded_and_never_escalated():
    result = decide(make_case("r4", [make_offer("OFFER-A", level("55"), advertised=gbp("700")), make_offer("OFFER-B", level("62"))], "OFFER-A"))
    assert result.decision is Decision.ALLOW and "advertised_total_overstated" in codes(result)
    assert decision_bases(result) == (DecisionBasis.ALLOW_WITH_UNCERTAINTY,)


def test_r4_an_understatement_is_material_from_the_predeclared_size_and_the_standard_is_the_configs_not_a_hidden_constant():
    a = make_offer("OFFER-A", level("55"), advertised=gbp("659.99"))  # one penny under the real 660
    case = make_case("r4b", [a, make_offer("OFFER-B", level("62"))], "OFFER-A")
    assert decide(case).decision is Decision.REVIEW  # default standard: any exact shortfall (unchanged from the rule engine's old behaviour)
    lenient = PolicyConfig(understatement_material_minor=500)
    result = decide(case, lenient)
    assert result.decision is Decision.ALLOW and not result.has(FindingCategory.CONFIRMED_FALSE_CLAIM, material_only=True)
    assert understatement_is_material(compare_figures(gbp("659.99"), gbp("660")), PolicyConfig()) is True


def test_r4_a_hidden_obligation_is_material_from_the_gap_or_at_any_size_if_it_carries_the_total_over_a_limit():
    intent, mandate = make_intent(max_total_repayment=gbp(750)), make_mandate(max_total_repayment=gbp(750))
    assert hidden_obligation_is_material(gbp("5.00"), gbp("660"), intent, mandate) is True
    assert hidden_obligation_is_material(gbp("4.99"), gbp("660"), intent, mandate) is False
    assert hidden_obligation_is_material(gbp("4.99"), gbp("748"), intent, mandate) is True  # 748 + 4.99 crosses the 750 limit
    assert hidden_obligation_is_material(gbp("0"), gbp("748"), intent, mandate) is False
    assert hidden_obligation_is_material(Money(amount_minor=9000, currency="USD"), gbp("660"), intent, mandate) is False  # no conversion, no opinion


# --- R7: product availability is not customer eligibility ------------------------------------------------------------------------------------


def elig_assertion(**kw):
    return make_assertion("OFFER-A", False, "Plan A is not open to you.", subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, **kw)


def test_r7_product_evidence_cannot_call_an_eligibility_statement_false():
    a, b = pair()
    case = make_case("r7", [a, b], "OFFER-B", assertions=(elig_assertion(),), evidence=(make_evidence("OFFER-A", True),))  # the lender still sells A
    result = decide(case)
    assert not result.has(FindingCategory.CONFIRMED_FALSE_CLAIM) and "availability_contradicted" not in codes(result)
    assert "eligibility_unresolved" in codes(result)
    assert result.decision is Decision.REVIEW and decision_bases(result) == (DecisionBasis.REVIEW_EVIDENCE_REQUIREMENT,)  # held, for the right reason


def test_r7_a_product_statement_is_still_contradicted_by_product_evidence():
    a, b = pair()
    claim = make_assertion("OFFER-A", False, "Plan A is withdrawn.")
    assert decide(make_case("r7b", [a, b], "OFFER-B", assertions=(claim,), evidence=(make_evidence("OFFER-A", True),))).has(FindingCategory.CONFIRMED_FALSE_CLAIM, material_only=True)


def test_r7_a_lender_decision_about_this_customer_verifies_a_decline():
    a, b = pair()
    decline = make_evidence("OFFER-A", False, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-1")
    result = decide(make_case("r7c", [a, b], "OFFER-B", assertions=(elig_assertion(),), evidence=(decline,)))
    assert result.decision is Decision.ALLOW and "costlier_than_usable_offer" not in codes(result)  # A is verifiably not open to this customer


def test_r7_a_verified_decline_removes_an_alternative_even_when_the_merchant_said_nothing():
    a, b = pair()
    decline = make_evidence("OFFER-A", False, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-1")
    result = decide(make_case("r7d", [a, b], "OFFER-B", evidence=(decline,)))
    assert result.decision is Decision.ALLOW and result.selection.excluded_as_not_open == ("OFFER-A",)


@pytest.mark.parametrize(
    "name, evidence",
    [
        ("a decline about another customer", make_evidence("OFFER-A", False, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-OTHER")),
        ("a decline from an aggregator, which cannot decide eligibility", make_evidence("OFFER-A", False, source=AGGREGATOR_SOURCE, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-1")),
        ("a decline from the merchant's own channel", make_evidence("OFFER-A", False, source=MERCHANT_SOURCE, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-1")),
        ("a decline from another lender", make_evidence("OFFER-A", False, source=EvidenceSource(source_id="LND-OTHER", name="x", kind=SourceKind.LENDER, independent_of_merchant=True), subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-1")),
    ],
)
def test_r7_only_the_offers_own_lender_speaking_about_this_customer_counts_as_a_decline(name, evidence):
    a, b = pair()
    result = decide(make_case("r7e", [a, b], "OFFER-B", evidence=(evidence,)))
    assert result.selection.excluded_as_not_open == () and "costlier_than_usable_offer" in codes(result), name


def test_r7_a_later_approval_supersedes_an_earlier_decline_from_the_same_lender():
    a, b = pair()
    decline = make_evidence("OFFER-A", False, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-1", at=T0 + timedelta(minutes=1), evidence_id="EV-1")
    approval = make_evidence("OFFER-A", True, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-1", at=T0 + timedelta(minutes=3), evidence_id="EV-2")
    assert decide(make_case("r7f", [a, b], "OFFER-B", evidence=(decline, approval))).selection.excluded_as_not_open == ()


def test_r7_a_decline_for_the_offer_being_bought_holds_the_purchase_under_its_own_basis():
    a, b = pair()
    decline = make_evidence("OFFER-A", False, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-1")
    result = decide(make_case("r7g", [a, b], "OFFER-A", evidence=(decline,)))
    assert result.decision is Decision.REVIEW and decision_bases(result) == (DecisionBasis.REVIEW_SELECTED_OFFER_NOT_OPEN,)


def test_r7_an_eligibility_decision_must_name_its_customer_and_a_product_observation_must_not():
    with pytest.raises(ValidationError):
        make_evidence("OFFER-A", False, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY)
    with pytest.raises(ValidationError):
        make_evidence("OFFER-A", False, customer_id="CUST-1")


# --- R8: impossible merchant timestamps ---------------------------------------------------------------------------------------------------------


def test_r8_a_claim_dated_after_the_decision_clock_is_ignored_and_recorded_and_cannot_launder_evidence():
    a, b = pair()
    evidence = (make_evidence("OFFER-A", False),)  # A is really unavailable
    honest = make_case("r8", [a, b], "OFFER-A", assertions=(make_assertion("OFFER-A", True, "In stock", at=T0),), evidence=evidence)
    future = make_case("r8b", [a, b], "OFFER-A", assertions=(make_assertion("OFFER-A", True, "In stock", at=NOW + timedelta(days=1)),), evidence=evidence)
    assert decide(honest).decision is Decision.REVIEW
    result = decide(future)
    assert "impossible_timestamp" in codes(result) and "evidence_stale" not in codes(result)  # it was set aside, not turned into 'stale' evidence
    assert assess_claim(future.assertions[0], future.evidence, a, NOW).status is ClaimStatus.INVALID_CHRONOLOGY


def test_r8_a_future_claim_that_hid_a_cheaper_offer_does_not_hide_it():
    a, b = pair()
    future_withdrawal = make_assertion("OFFER-A", False, "Plan A is withdrawn.", at=NOW + timedelta(hours=1))
    result = decide(make_case("r8c", [a, b], "OFFER-B", assertions=(future_withdrawal,)))
    assert "costlier_than_usable_offer" in codes(result) and "impossible_timestamp" in codes(result)  # A stays a usable alternative


def test_r8_evidence_and_documents_dated_after_the_decision_clock_are_recorded_too():
    from sika_guard.fixtures.builders import make_document

    a, b = pair()
    future_evidence = make_evidence("OFFER-A", False, at=NOW + timedelta(minutes=30))
    future_doc = make_document("LD-9", "An arrangement fee of £45 applies.", offer_id="OFFER-A", at=NOW + timedelta(hours=1))
    result = decide(make_case("r8d", [a, b], "OFFER-A", evidence=(future_evidence,), documents=(future_doc,)))
    assert {f.subject_id for f in result.findings if f.code.value == "impossible_timestamp"} == {"EV-1", "LD-9"} and result.decision is Decision.ALLOW


def test_r8_control_a_claim_made_before_the_decision_clock_is_assessed_normally():
    a, b = pair()
    claim = make_assertion("OFFER-A", True, "In stock", at=NOW - timedelta(minutes=1))
    assert "impossible_timestamp" not in codes(decide(make_case("r8e", [a, b], "OFFER-A", assertions=(claim,))))


# --- R9: a source-authority and conflict hierarchy ----------------------------------------------------------------------------------------------------


def hierarchy_case(lender_at, agg_at, *, lender_says=True, agg_says=False, claim_available=False):
    a, b = pair()
    return make_case("r9", [a, b], "OFFER-B", assertions=(make_assertion("OFFER-A", claim_available, "Plan A is withdrawn.", at=T0),), evidence=(
        make_evidence("OFFER-A", lender_says, at=lender_at, evidence_id="EV-L"),
        make_evidence("OFFER-A", agg_says, source=AGGREGATOR_SOURCE, at=agg_at, evidence_id="EV-G"),
    ))


def status_of(case):
    return assess_claim(case.assertions[0], case.evidence, case.offers[0], NOW).status


def test_r9_the_offers_own_lender_outranks_an_aggregator_that_is_not_newer():
    case = hierarchy_case(T0 + timedelta(minutes=4), T0 + timedelta(minutes=2))
    assert status_of(case) is ClaimStatus.CONFIRMED_FALSE  # the lender says A is available, so the merchant's withdrawal claim is false
    assessment = assess_claim(case.assertions[0], case.evidence, case.offers[0], NOW)
    assert assessment.dissent == ("EV-G",)  # the outranked observation is kept for the record
    result = decide(case)
    assert any("outranked: EV-G" in f.message for f in result.findings if f.code.value == "availability_contradicted")


def test_r9_a_newer_observation_from_a_lower_authority_leaves_the_conflict_open():
    assert status_of(hierarchy_case(T0 + timedelta(minutes=2), T0 + timedelta(minutes=4))) is ClaimStatus.CONFLICTED


def test_r9_two_independent_sources_of_the_same_rank_that_disagree_stay_in_conflict():
    other = EvidenceSource(source_id="AGG-2", name="Other feed", kind=SourceKind.AGGREGATOR, independent_of_merchant=True)
    a, b = pair()
    case = make_case("r9b", [a, b], "OFFER-B", assertions=(make_assertion("OFFER-A", False, "withdrawn", at=T0),), evidence=(
        make_evidence("OFFER-A", True, source=AGGREGATOR_SOURCE, evidence_id="EV-1"), make_evidence("OFFER-A", False, source=other, evidence_id="EV-2")))
    assert status_of(case) is ClaimStatus.CONFLICTED


def test_r9_one_sources_later_observation_supersedes_its_own_earlier_one():
    a, b = pair()
    case = make_case("r9c", [a, b], "OFFER-B", assertions=(make_assertion("OFFER-A", False, "withdrawn", at=T0),), evidence=(
        make_evidence("OFFER-A", True, at=T0 + timedelta(minutes=1), evidence_id="EV-1"), make_evidence("OFFER-A", False, at=T0 + timedelta(minutes=3), evidence_id="EV-2")))
    assert status_of(case) is ClaimStatus.CONFIRMED_TRUE  # the lender later said A is unavailable, which is what the merchant claimed


def test_r9_the_merchants_own_channel_is_never_a_source_of_authority():
    a, b = pair()
    case = make_case("r9d", [a, b], "OFFER-B", assertions=(make_assertion("OFFER-A", False, "withdrawn", at=T0),), evidence=(
        make_evidence("OFFER-A", True, source=MERCHANT_SOURCE),))
    assert status_of(case) is ClaimStatus.UNVERIFIED


def test_r9_agreement_between_ranks_needs_no_resolution():
    case = hierarchy_case(T0 + timedelta(minutes=2), T0 + timedelta(minutes=3), lender_says=False, agg_says=False)
    assert status_of(case) is ClaimStatus.CONFIRMED_TRUE


def test_r9_the_development_fixture_with_conflicting_sources_still_holds_the_purchase():
    from sika_guard.fixtures.cases import get_case

    result = decide(get_case("case-17"))  # the aggregator is newer than the lender here, so the conflict is genuinely unresolved
    assert result.decision is Decision.REVIEW and result.has(FindingCategory.UNRESOLVED_CONTRADICTION)


# --- a fact the rule engine now acts on by itself: fresh independent evidence that the offer being bought is not on sale ------------------------------


def test_the_selected_offer_reported_unavailable_by_its_own_lender_holds_the_purchase_whether_or_not_anyone_claimed_anything():
    a, b = pair()
    result = decide(make_case("u1", [a, b], "OFFER-A", evidence=(make_evidence("OFFER-A", False),)))
    assert result.decision is Decision.REVIEW and decision_bases(result) == (DecisionBasis.REVIEW_SELECTED_OFFER_NOT_OPEN,)
    finding = next(f for f in result.findings if f.code.value == "offer_reported_unavailable")
    assert "not evidence that the merchant is dishonest" in finding.message and not result.has(FindingCategory.CONFIRMED_FALSE_CLAIM)


def test_a_future_dated_claim_no_longer_lets_a_withdrawn_selected_offer_through():
    a, b = pair()
    future = make_assertion("OFFER-A", True, "In stock", at=NOW + timedelta(days=1))
    result = decide(make_case("u2", [a, b], "OFFER-A", assertions=(future,), evidence=(make_evidence("OFFER-A", False),)))
    assert result.decision is Decision.REVIEW and {"impossible_timestamp", "offer_reported_unavailable"} <= codes(result)


@pytest.mark.parametrize(
    "name, evidence",
    [
        ("stale (outside the freshness window)", make_evidence("OFFER-A", False, at=NOW - timedelta(hours=3))),
        ("from the merchant's own channel", make_evidence("OFFER-A", False, source=MERCHANT_SOURCE)),
        ("from another lender", make_evidence("OFFER-A", False, source=EvidenceSource(source_id="LND-OTHER", name="x", kind=SourceKind.LENDER, independent_of_merchant=True))),
        ("observed after the decision clock", make_evidence("OFFER-A", False, at=NOW + timedelta(minutes=5))),
    ],
)
def test_evidence_that_cannot_establish_unavailability_does_not_hold_the_purchase(name, evidence):
    a, b = pair()
    result = decide(make_case("u3", [a, b], "OFFER-A", evidence=(evidence,)))
    assert result.decision is Decision.ALLOW and "offer_reported_unavailable" not in codes(result), name


def test_conflicting_fresh_evidence_about_the_selected_offer_is_not_treated_as_unavailability():
    a, b = pair()
    newer_aggregator = make_evidence("OFFER-A", False, source=AGGREGATOR_SOURCE, at=NOW - timedelta(minutes=1), evidence_id="EV-2")
    result = decide(make_case("u4", [a, b], "OFFER-A", evidence=(make_evidence("OFFER-A", True, at=T0 + timedelta(minutes=2)), newer_aggregator)))
    assert "offer_reported_unavailable" not in codes(result)


def test_an_alternative_that_independent_evidence_says_is_withdrawn_is_not_a_better_option_to_compare_against():
    a, b = pair()  # A is cheaper, but its own lender says it is withdrawn; nobody made a claim either way
    result = decide(make_case("u5", [a, b], "OFFER-B", evidence=(make_evidence("OFFER-A", False),)))
    assert result.decision is Decision.ALLOW and result.selection.excluded_as_not_open == ("OFFER-A",) and "costlier_than_usable_offer" not in codes(result)


def test_the_lender_saying_a_cheaper_alternative_is_available_does_not_exclude_it():
    a, b = pair()
    result = decide(make_case("u6", [a, b], "OFFER-B", evidence=(make_evidence("OFFER-A", True),)))
    assert "costlier_than_usable_offer" in codes(result) and result.selection.excluded_as_not_open == ()


# --- defences a mutation run showed no test exercised ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, evidence",
    [
        ("about another customer", make_evidence("OFFER-A", False, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-OTHER")),
        ("from an aggregator, which cannot decide eligibility", make_evidence("OFFER-A", False, source=AGGREGATOR_SOURCE, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-1")),
        ("from the merchant's own channel", make_evidence("OFFER-A", False, source=MERCHANT_SOURCE, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-1")),
    ],
)
def test_r7_a_merchants_eligibility_statement_is_confirmed_only_by_the_offers_own_lender_speaking_about_this_customer(name, evidence):
    a, b = pair()
    claim = make_assertion("OFFER-A", False, "Plan A is not open to you.", subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY)
    case = make_case("r7h", [a, b], "OFFER-B", assertions=(claim,), evidence=(evidence,))
    assert assess_claim(claim, case.evidence, a, NOW, customer_id="CUST-1").status is ClaimStatus.UNVERIFIED, name
    assert "eligibility_unresolved" in codes(decide(case)), name


def test_r7_the_same_statement_is_confirmed_by_the_offers_own_lender_about_this_customer():
    a, _ = pair()
    claim = make_assertion("OFFER-A", False, "Plan A is not open to you.", subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY)
    decline = make_evidence("OFFER-A", False, subject=AvailabilitySubject.CUSTOMER_ELIGIBILITY, customer_id="CUST-1")
    assert assess_claim(claim, (decline,), a, NOW, customer_id="CUST-1").status is ClaimStatus.CONFIRMED_TRUE
    assert assess_claim(claim, (decline,), a, NOW).status is ClaimStatus.UNVERIFIED  # without knowing whose decision it is, it tests nothing
