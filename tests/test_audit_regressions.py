"""Regression tests for the independent audit's trusted-verification findings (F1-F8). Offline.

Each finding has (a) the audit's counterexample, which must now be rejected or surfaced as uncertainty, and (b) a control
showing a legitimate input of the same shape still succeeds, so a fix cannot hide behind refusing everything.
Scenarios live in `audit_scenarios.py` and are shared with the before/after counterexample matrix.
"""

import pytest
from pydantic import ValidationError

import audit_scenarios as S
from sika_guard.coordinator import Condition
from sika_guard.evaluators.contract import ClaimKind, EvaluatorRole, EvaluatorStatus, EvaluatorView, ExtractedClaim, verdict_from_analysis
from sika_guard.evaluators.crosscheck import CrossCheckOutcome, crosscheck_claims
from sika_guard.evaluators.mocks import Scripted, make_analysis
from sika_guard.evaluators.payloads import PAYLOAD_BUILDERS, collect_refs
from sika_guard.fixtures.builders import fee, gbp, level, make_assertion, make_case, make_evidence, make_offer
from sika_guard.models import Decision, DecisionBasis, EvidenceSource, FeeKind, Money, SourceKind
from sika_guard.policy import ClaimStatus, assess_claim, classify_evidence, EvidenceRelevance

CONTRADICTED = {"contradicted_by_terms", "contradicted_by_evidence"}


def outcomes(obs):
    return [c.split(":", 1)[1] for c in obs["crosschecks"]]


def no_false_accusation(obs):
    assert obs["final"] == "ALLOW", obs
    assert obs["proven_false"] == [] and not (set(outcomes(obs)) & CONTRADICTED), obs


# --- F1: a quote must express the proposition ---------------------------------------------------------------------------


def test_f1a_a_no_fees_claim_from_unrelated_text_is_not_a_verified_contradiction_in_any_condition():
    result = S.f1a_no_fees_claim_from_unrelated_text()
    no_false_accusation(result["scripted"])
    for condition, obs in result["live"].items():
        no_false_accusation(obs)
        assert "consistent" not in outcomes(obs), condition


def test_f1a_the_unrelated_quote_is_recorded_as_uncertainty_not_discarded_silently():
    obs = S.f1a_no_fees_claim_from_unrelated_text()["scripted"]
    assert obs["bases"] == ["allow_with_recorded_uncertainty"] and obs["uncertainty"]


def test_f1_control_a_genuine_no_fees_statement_is_still_checked_against_the_lender_terms():
    case = S.fee_case(description="No extra fees.")
    claim = ExtractedClaim(kind=ClaimKind.NO_ADDITIONAL_FEES, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="No extra fees")
    obs = S.scripted(case, [claim])
    assert outcomes(obs) == ["contradicted_by_terms"] and obs["final"] == "REVIEW" and obs["proven_false"] == ["OFFER-A"]  # the lender lists a 20 fee
    honest = S.scripted(make_case("honest", [make_offer("OFFER-A", level("55"), advertised=None, description="No extra fees."), make_offer("OFFER-B", level("62"))], "OFFER-A"), [claim])
    assert outcomes(honest) == ["consistent"] and honest["final"] == "ALLOW"


def test_f1b_a_zero_fee_amount_against_a_quote_stating_20_is_a_disagreement_not_a_contradiction():
    obs = S.f1b_zero_fee_amount_against_a_quote_stating_20()["scripted"]
    no_false_accusation(obs)
    assert outcomes(obs) == ["extraction_disagrees_with_source"]


def test_f1_control_a_correct_fee_amount_is_consistent_and_a_stated_zero_is_checked():
    case = S.fee_case(description="An administration fee of £20 applies.")
    right = ExtractedClaim(kind=ClaimKind.FEE_AMOUNT, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="An administration fee of £20 applies", amount_minor=2000, currency="GBP")
    assert outcomes(S.scripted(case, [right])) == ["consistent"]
    zero_case = S.fee_case(description="No fees at all.")
    zero = ExtractedClaim(kind=ClaimKind.FEE_AMOUNT, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="No fees at all", amount_minor=0, currency="GBP")
    assert outcomes(S.scripted(zero_case, [zero])) == ["contradicted_by_terms"]  # the merchant says zero, the lender lists 20


# --- F2: subject agreement and what each evaluator was actually shown ---------------------------------------------------------


def test_f2a_a_truthful_quote_about_offer_y_cannot_be_attributed_to_offer_x():
    obs = S.f2a_quote_from_offer_y_attributed_to_offer_x()["scripted"]
    no_false_accusation(obs)
    assert obs["verdict_statuses"] == ["invalid_response"] and obs["coverage"] == "unavailable"


def test_f2a_the_trusted_cross_check_rejects_it_even_if_an_evaluator_skipped_the_grounding():
    x = make_offer("OFFER-X", level("55", fees=(fee(FeeKind.ADMINISTRATION, "20"),)), advertised=None)
    y = make_offer("OFFER-Y", level("58"), advertised=None, description="Total repayable £696.")
    case = make_case("c", [x, y], "OFFER-X")
    view = EvaluatorView.from_case(case)
    claim = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-X", source_ref="listing:OFFER-Y", quote="Total repayable £696", amount_minor=69600, currency="GBP")
    unchecked = verdict_from_analysis(make_analysis(Decision.ALLOW, claims=[claim]), view, evaluator_id="e", role=EvaluatorRole.GENERALIST, allowed_refs=collect_refs(PAYLOAD_BUILDERS[EvaluatorRole.GENERALIST](view)))
    (check,) = crosscheck_claims(view, [unchecked])
    assert check.outcome is CrossCheckOutcome.UNGROUNDED and "OFFER-Y" in check.detail


def test_f2_control_a_truthful_quote_about_the_right_offer_is_still_verified():
    x = make_offer("OFFER-X", level("55", fees=(fee(FeeKind.ADMINISTRATION, "20"),)), advertised=None, description="Total repayable £660.")
    case = make_case("c", [x, make_offer("OFFER-Y", level("62"), advertised=None)], "OFFER-X")
    claim = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-X", source_ref="listing:OFFER-X", quote="Total repayable £660", amount_minor=66000, currency="GBP")
    obs = S.scripted(case, [claim])
    assert outcomes(obs) == ["contradicted_by_terms"] and obs["final"] == "REVIEW"


def test_f2b_a_specialist_that_quotes_a_field_it_was_never_shown_is_an_invalid_response():
    result = S.f2b_specialist_quotes_a_field_it_was_never_shown()
    bad = result["credit_terms_specialist_quotes_the_description"]
    no_false_accusation(bad)
    assert bad["verdict_statuses"].count("invalid_response") == 1 and bad["coverage"] == "partial"
    assert any(u.startswith("evaluator_failure:credit_terms") for u in bad["uncertainty"])


def test_f2b_controls_evaluators_that_do_see_the_description_still_verify_a_genuine_misstatement():
    result = S.f2b_specialist_quotes_a_field_it_was_never_shown()
    for name in ("control_generalist_sees_the_description", "control_merchant_claim_specialist_sees_it"):
        assert result[name]["final"] == "REVIEW" and "contradicted_by_terms" in outcomes(result[name]), name


def test_f2_the_advertised_total_is_visible_to_the_credit_specialist_and_may_be_quoted_there():
    case = S.fee_case(advertised=gbp("600"))  # lender total 680, advertised 600
    claim = S.claim_json("total_repayable", "OFFER-A", "listing:OFFER-A", "advertised_total_repayable: 600.00 GBP", 60000, "GBP")
    obs = S.live(case, Condition.C, {"credit_terms": {"verdict": "ALLOW", "extracted_claims": [claim]}, "*": {}})
    assert obs["verdict_statuses"].count("invalid_response") == 0 and "contradicted_by_terms" in outcomes(obs)


# --- F3: provenance is not established by a merchant-controlled flag ----------------------------------------------------------------


def test_f3_merchant_channel_evidence_that_declares_itself_independent_is_rejected_outright():
    with pytest.raises(ValidationError, match="merchant-channel"):
        EvidenceSource(source_id="MER-1", name="Merchant portal", kind=SourceKind.MERCHANT_CHANNEL, independent_of_merchant=True)


def test_f3_a_source_carrying_the_merchants_own_identity_cannot_be_independent():
    mine = EvidenceSource(source_id="MER-1", name="Looks like a register", kind=SourceKind.AGGREGATOR, independent_of_merchant=True)
    with pytest.raises(ValidationError, match="independent"):
        make_case("c", [make_offer("OFFER-A", level("55")), make_offer("OFFER-B", level("62"))], "OFFER-B", evidence=(make_evidence("OFFER-A", True, source=mine),))


def test_f3_the_scenario_no_longer_lets_the_merchants_channel_corroborate_or_contradict_its_own_claim():
    result = S.f3_merchant_channel_declaring_itself_independent()
    assert result["corroborates_the_merchants_own_claim"] == {"rejected_at_construction": "ValidationError"}
    assert result["contradicts_the_merchants_own_claim"] == {"rejected_at_construction": "ValidationError"}


def test_f3_defence_in_depth_even_a_forced_object_is_not_treated_as_independent():
    forged = EvidenceSource.model_construct(source_id="MER-1", name="forced", kind=SourceKind.MERCHANT_CHANNEL, independent_of_merchant=True)
    evidence = make_evidence("OFFER-A", False).model_copy(update={"source": forged})
    assertion = make_assertion("OFFER-A", False, "Offer A has been withdrawn.")
    offer = make_offer("OFFER-A", level("55"))
    from sika_guard.fixtures.builders import NOW

    assert classify_evidence(assertion, evidence, offer, NOW) is EvidenceRelevance.IRRELEVANT
    assert assess_claim(assertion, (evidence,), offer, NOW).status is ClaimStatus.UNVERIFIED


def test_f3_controls_genuine_independent_sources_and_honestly_flagged_merchant_channels_behave_as_before():
    result = S.f3_merchant_channel_declaring_itself_independent()
    assert result["control_no_evidence_at_all"]["bases"] == ["review_evidence_requirement_unmet"]
    assert result["control_honest_merchant_channel_flag"]["bases"] == ["review_evidence_requirement_unmet"]  # a merchant's own channel is not corroboration
    lender = make_case("c", [make_offer("OFFER-A", level("55")), make_offer("OFFER-B", level("62"))], "OFFER-B",
                       assertions=(make_assertion("OFFER-A", False, "Offer A has been withdrawn."),), evidence=(make_evidence("OFFER-A", False),))
    from sika_guard.policy import evaluate_case

    assert evaluate_case(lender).decision is Decision.ALLOW  # the lender really does confirm the withdrawal


def test_f3_the_evaluator_payload_does_not_present_merchant_channel_evidence_as_independent():
    case = S._withdrawal_case((make_evidence("OFFER-A", False, source=S._merchant_source(False)),))
    payload = PAYLOAD_BUILDERS[EvaluatorRole.GENERALIST](EvaluatorView.from_case(case))
    assert payload["trusted"]["independent_evidence"] == [] and len(payload["untrusted_merchant_content"]["merchant_channel_evidence"]) == 1


# --- F4: other amounts in a quote must not hide an explicit total or fee --------------------------------------------------------------


def test_f4a_an_explicit_total_is_verified_even_when_the_quote_also_states_an_instalment():
    obs = S.f4a_total_claim_whose_quote_also_states_an_instalment()["scripted"]
    assert outcomes(obs) == ["contradicted_by_terms"] and obs["final"] == "REVIEW" and obs["proven_false"] == ["OFFER-A"]


def test_f4b_an_explicit_fee_total_is_verified_even_when_the_quote_also_states_an_instalment():
    obs = S.f4b_fee_claim_whose_quote_also_states_an_instalment()["scripted"]
    assert outcomes(obs) == ["contradicted_by_terms"] and obs["final"] == "REVIEW"


def test_f5a_a_monthly_payment_is_never_taken_as_the_total_repayable():
    obs = S.f5a_monthly_payment_filed_as_total_repayment()["scripted"]
    no_false_accusation(obs)
    assert outcomes(obs) == ["extraction_disagrees_with_source"]


def test_f5b_scaled_or_malformed_amounts_are_never_partially_parsed_into_a_verdict():
    for label, obs in S.f5b_malformed_and_scaled_amounts().items():
        no_false_accusation(obs)
        assert outcomes(obs) == ["not_comparable"], (label, obs["crosschecks"])
        assert obs["bases"] == ["allow_with_recorded_uncertainty"], label


def test_f6a_an_advertised_total_in_another_currency_on_the_selected_offer_is_a_review_with_its_own_basis():
    """Direction B resolves the open F6 decision: the headline figure for the offer being bought cannot be compared with the lender's
    terms, so the purchase is held, under a basis of its own. Nothing is converted and nobody is called a liar."""
    obs = S.f6a_advertised_total_in_another_currency()["rule_engine_only"]
    assert obs["final"] == "REVIEW" and obs["bases"] == ["review_incomparable_representation"]
    assert obs["proven_false"] == []


def test_f6b_a_model_figure_that_disagrees_with_the_source_field_is_recorded_uncertainty():
    obs = S.f6b_model_extraction_disagrees_with_the_structured_figure()["scripted"]
    assert obs["final"] == "ALLOW" and obs["bases"] == ["allow_with_recorded_uncertainty"]
    assert any("extraction_disagrees_with_source" in u for u in obs["uncertainty"])


def test_f6_a_case_with_nothing_unresolved_is_still_a_plain_allow():
    case = make_case("clean", [make_offer("OFFER-A", level("55")), make_offer("OFFER-B", level("62"))], "OFFER-A")
    assert S.scripted(case)["bases"] == ["allow"] and S.scripted(case)["uncertainty"] == []


def test_f6_incompatibility_never_becomes_an_accusation():
    for obs in (S.f6a_advertised_total_in_another_currency()["rule_engine_only"], S.f6b_model_extraction_disagrees_with_the_structured_figure()["scripted"]):
        assert obs["proven_false"] == []


def test_f6_an_incompatible_currency_on_an_offer_that_is_not_being_bought_is_only_recorded():
    other = make_offer("OFFER-B", level("62"), advertised=Money(amount_minor=70000, currency="USD"))
    case = make_case("f6c", [make_offer("OFFER-A", level("55")), other], "OFFER-A")
    obs = S.scripted(case)
    assert obs["final"] == "ALLOW" and obs["bases"] == ["allow_with_recorded_uncertainty"]


# --- F8: an invented quotation is an evaluator failure -------------------------------------------------------------------------------------------


def test_f8_an_invented_quotation_invalidates_the_reply_and_counts_against_coverage():
    result = S.f8_invented_quotation()
    for condition in ("B", "B3", "Bc"):
        obs = result["open"][condition]
        assert set(obs["verdict_statuses"]) == {"invalid_response"} and obs["coverage"] == "unavailable", condition
        assert obs["final"] == "ALLOW" and obs["bases"] == ["allow_with_recorded_uncertainty"]  # fail-open by default, visibly
    assert "invalid_response" in result["open"]["C"]["verdict_statuses"] and result["open"]["C"]["coverage"] != "complete"


def test_f8_under_fail_closed_an_invented_quotation_triggers_the_documented_failure_review():
    result = S.f8_invented_quotation()
    for condition, obs in result["fail_closed"].items():
        assert obs["final"] == "REVIEW" and obs["bases"] == ["review_evaluator_failure"], condition
        assert obs["proven_false"] == []  # a failure, never an accusation


def test_f8_a_genuine_quotation_is_unaffected():
    case = S.fee_case(description="Standard instalment plan.")
    good = S.claim_json("no_additional_fees", "OFFER-A", "listing:OFFER-A", "Standard instalment plan")
    obs = S.live(case, Condition.B, {"*": {"verdict": "ALLOW", "extracted_claims": [good]}})
    assert obs["verdict_statuses"] == ["ok"] and obs["coverage"] == "complete"


# --- F7: what max_amount covers: a product decision is needed ------------------------------------------------------------------------------------------


def test_f7_every_individually_committed_payment_must_stay_within_max_amount():
    """Resolved by the mandate's own wording ("ceiling for any single payment"): a later instalment above it is a hard violation even
    though the initial payment is within it. `policy.check_committed_payments`."""
    obs = S.f7_later_instalment_above_max_amount()["rule_engine_only"]
    assert obs["final"] == "BLOCK" and obs["bases"] == ["block_hard_violation"]


# --- the two defences that a mutation run showed were not isolated by the tests above --------------------------------------------------------


def test_f2_the_cross_check_itself_grounds_in_the_evaluators_own_request_not_the_full_case():
    """Bypass the reply validation entirely (payload=None) and hand the cross-check a credit-terms verdict that quotes the listing's
    description, a field that role is never sent. The trusted stage must still refuse it."""
    case = S.fee_case(description="Total repayable £660.")  # lender total 680
    view = EvaluatorView.from_case(case)
    claim = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="Total repayable £660", amount_minor=66000, currency="GBP")
    seen_by_credit_terms = collect_refs(PAYLOAD_BUILDERS[EvaluatorRole.CREDIT_TERMS](view))
    assert "listing:OFFER-A" in seen_by_credit_terms  # the listing IS in its request, but only its advertised total
    verdict = verdict_from_analysis(make_analysis(Decision.ALLOW, claims=[claim]), view, evaluator_id="credit_terms", role=EvaluatorRole.CREDIT_TERMS, allowed_refs=seen_by_credit_terms)
    (check,) = crosscheck_claims(view, [verdict])
    assert check.outcome is CrossCheckOutcome.UNGROUNDED and "as this evaluator was shown it" in check.detail
    general = verdict_from_analysis(make_analysis(Decision.ALLOW, claims=[claim]), view, evaluator_id="g", role=EvaluatorRole.GENERALIST, allowed_refs=collect_refs(PAYLOAD_BUILDERS[EvaluatorRole.GENERALIST](view)))
    assert crosscheck_claims(view, [general])[0].outcome is CrossCheckOutcome.CONTRADICTED_BY_TERMS  # the generalist was shown the description


def test_f1_a_zero_fee_claim_whose_quote_denies_extra_fees_but_states_a_fee_is_not_a_contradiction():
    case = S.fee_case(description="No extra fees, only an administration fee of £20.")
    claim = ExtractedClaim(kind=ClaimKind.FEE_AMOUNT, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="No extra fees, only an administration fee of £20", amount_minor=0, currency="GBP")
    obs = S.scripted(case, [claim])
    assert outcomes(obs) == ["extraction_disagrees_with_source"] and obs["final"] == "ALLOW" and obs["proven_false"] == []
