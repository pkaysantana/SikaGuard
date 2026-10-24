"""The flagship scenario (case-05): authorised, inferior, and built on a false merchant claim.

Each test isolates one of the properties that must not be conflated.
"""

import pytest

from sika_guard.fixtures.builders import NOW, gbp
from sika_guard.fixtures.cases import get_case
from sika_guard.models import Decision, FindingCategory
from sika_guard.policy import ClaimStatus, assess_claim, check_transaction, evaluate_case


@pytest.fixture
def flagship():
    return get_case("case-05")


def _offer(case, offer_id):
    return next(o for o in case.offers if o.offer_id == offer_id)


def test_offer_b_is_inside_the_total_repayment_limit(flagship):
    offer_b = _offer(flagship, "OFFER-B")
    assert offer_b.schedule.total() == gbp(720)
    assert offer_b.schedule.total() <= flagship.intent.max_total_repayment
    assert offer_b.schedule.max_payment() <= flagship.intent.max_monthly_payment
    assert check_transaction(flagship.transaction, offer_b, flagship) == []


def test_offer_b_is_authorised_yet_sixty_pounds_worse_than_offer_a(flagship):
    result = evaluate_case(flagship)
    assert result.authorised  # authorisation and decision quality are separate questions
    assert result.decision is Decision.REVIEW  # not BLOCK: it is inside the mandate
    assert result.selection.better_offer_id == "OFFER-A"
    assert result.selection.objective_gap == gbp(60)
    assert result.selection.inferior
    assert not result.has(FindingCategory.HARD_VIOLATION)


def test_fresh_lender_statement_contradicts_the_merchant_claim(flagship):
    claim = flagship.assertions[0]
    assessment = assess_claim(claim, flagship.evidence, _offer(flagship, "OFFER-A"), NOW)
    assert assessment.status is ClaimStatus.CONFIRMED_FALSE
    assert evaluate_case(flagship).has(FindingCategory.CONFIRMED_FALSE_CLAIM, material_only=True)


def test_missing_evidence_does_not_prove_the_merchant_lied():
    case = get_case("case-14")  # same claim, no independent evidence at all
    assessment = assess_claim(case.assertions[0], case.evidence, _offer(case, "OFFER-A"), NOW)
    result = evaluate_case(case)
    assert assessment.status is ClaimStatus.UNVERIFIED
    assert not result.has(FindingCategory.CONFIRMED_FALSE_CLAIM)
    assert result.has(FindingCategory.UNVERIFIED_CLAIM)
    assert result.decision is Decision.ALLOW


def test_stale_evidence_does_not_establish_current_availability():
    case = get_case("case-16")  # three-day-old observation that A was available
    assessment = assess_claim(case.assertions[0], case.evidence, _offer(case, "OFFER-A"), NOW)
    result = evaluate_case(case)
    assert assessment.status is ClaimStatus.STALE_ONLY
    assert not result.has(FindingCategory.CONFIRMED_FALSE_CLAIM)
    assert result.decision is Decision.ALLOW
    assert result.selection.better_offer_id is None  # A was not treated as usable
