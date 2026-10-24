"""Trusted comparison of claims extracted by a model against authenticated data.

A model may extract what a merchant claimed. It is never asked whether the claim is true. Code
checks that the quote really appears in the cited merchant text, then compares the claim with the
lender terms or with the deterministic evidence assessment, so a contradiction is established by
code over validated structured fields and not by a model's say-so.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, ValidationError

from sika_guard.evaluators.contract import ClaimKind, EvaluatorVerdict, EvaluatorView, ExtractedClaim
from sika_guard.models import FinancingOffer
from sika_guard.money import Money
from sika_guard.policy import DEFAULT_CONFIG, ClaimStatus, PolicyConfig, assess_claim


class CrossCheckOutcome(StrEnum):
    CONTRADICTED_BY_TERMS = "contradicted_by_terms"
    CONTRADICTED_BY_EVIDENCE = "contradicted_by_evidence"
    CONSISTENT = "consistent"
    NOT_COMPARABLE = "not_comparable"  # nothing authenticated to compare against; says nothing about truth
    UNGROUNDED = "ungrounded"  # the extraction does not match the source text; discarded


class CrossCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluator_id: str
    offer_id: str
    kind: ClaimKind
    outcome: CrossCheckOutcome
    detail: str

    @property
    def contradicted(self) -> bool:
        return self.outcome in (CrossCheckOutcome.CONTRADICTED_BY_TERMS, CrossCheckOutcome.CONTRADICTED_BY_EVIDENCE)


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


def _source_text(view: EvaluatorView, source_ref: str) -> str | None:
    kind, _, ident = source_ref.partition(":")
    if kind == "assertion":
        return next((a.statement for a in view.assertions if a.assertion_id == ident), None)
    if kind == "listing":
        return next((f"{o.title} {o.description}" for o in view.offers if o.offer_id == ident), None)
    return None


def _check(claim: ExtractedClaim, view: EvaluatorView, offer: FinancingOffer, config: PolicyConfig) -> tuple[CrossCheckOutcome, str]:
    text = _source_text(view, claim.source_ref)
    if text is None or _squash(claim.quote) not in _squash(text):
        return CrossCheckOutcome.UNGROUNDED, "quote not found in the cited merchant text"

    if claim.kind is ClaimKind.AVAILABILITY:
        kind, _, ident = claim.source_ref.partition(":")
        if kind != "assertion":
            return CrossCheckOutcome.NOT_COMPARABLE, "free-text availability claims carry no timestamp to test against"
        assertion = next(a for a in view.assertions if a.assertion_id == ident)
        if assertion.offer_id != claim.offer_id or assertion.claims_available != claim.asserted_available:
            return CrossCheckOutcome.UNGROUNDED, "extraction disagrees with the structured assertion"
        status = assess_claim(assertion, view.evidence, offer, view.as_of, config).status
        if status is ClaimStatus.CONFIRMED_FALSE:
            return CrossCheckOutcome.CONTRADICTED_BY_EVIDENCE, "fresh independent evidence says otherwise"
        if status is ClaimStatus.CONFIRMED_TRUE:
            return CrossCheckOutcome.CONSISTENT, "fresh independent evidence agrees"
        return CrossCheckOutcome.NOT_COMPARABLE, f"claim is {status.value}"

    if claim.kind is ClaimKind.TOTAL_REPAYABLE:
        try:
            claimed = Money(amount_minor=claim.amount_minor, currency=claim.currency)
        except ValidationError:
            return CrossCheckOutcome.NOT_COMPARABLE, "claimed amount is not a valid sum of money"
        real = offer.schedule.total()
        if claimed.currency != real.currency:
            return CrossCheckOutcome.NOT_COMPARABLE, "currencies differ"
        if claimed == real:
            return CrossCheckOutcome.CONSISTENT, f"total repayable is {real.plain()}"
        return CrossCheckOutcome.CONTRADICTED_BY_TERMS, f"merchant text says {claimed.plain()}; lender terms total {real.plain()}"

    if offer.schedule.fees:
        return CrossCheckOutcome.CONTRADICTED_BY_TERMS, f"lender terms include {len(offer.schedule.fees)} mandatory fee(s) totalling {offer.schedule.total_fees().plain()}"
    return CrossCheckOutcome.CONSISTENT, "lender terms list no fees"


def crosscheck_claims(
    view: EvaluatorView, verdicts: Iterable[EvaluatorVerdict], config: PolicyConfig = DEFAULT_CONFIG
) -> tuple[CrossCheck, ...]:
    offers = {o.offer_id: o for o in view.offers}
    results = []
    for verdict in verdicts:
        if not verdict.usable:
            continue
        for claim in verdict.extracted_claims:
            offer = offers[claim.offer_id]  # grounding already confirmed the offer exists
            outcome, detail = _check(claim, view, offer, config)
            results.append(CrossCheck(evaluator_id=verdict.evaluator_id, offer_id=claim.offer_id, kind=claim.kind, outcome=outcome, detail=detail))
    return tuple(results)
