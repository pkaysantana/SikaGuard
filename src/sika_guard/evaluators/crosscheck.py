"""Trusted verification of facts a model extracted, and the only route by which an extraction can reach a REVIEW.

A model may say what a passage states. It is never asked whether that is true. Everything between "the model said X" and "the policy
treats X as a verified contradiction" happens here, in code, over structured data, in stages that are kept apart on purpose:

  1. candidate claim        what the model extracted (`ExtractedClaim`): a pointer, a quote, a kind and a figure. Untrusted.
  2. grounded fact          `ground_claim`: the claimed source was in THIS evaluator's request, concerns the claimed offer, is dated
                            no later than the decision clock, and really contains the quote. The kind of source (the merchant's
                            own representation, the offer's own lender, some other independent party) is decided from trusted
                            data, never from anything the model or the document says about itself. The passage's amounts are read
                            by `amounts` with their quantity, and the claim's scope, currency, period and kind must match what the
                            passage states. Anything else is a rejection that is recorded, not a finding.
  3. independent support    `_judge`: the grounded figure is compared with an authoritative one, with a DIRECTION (does it show the
                            customer less than they will pay, or more?) and without converting currencies. A merchant's
                            representation is compared with the lender's structured terms. A lender's own document is compared
                            with the structured terms too, but two lender sources disagreeing is a CONFLICT to be resolved by a
                            person, not a proof that either is false. A document can never corroborate its own author.
  4. policy materiality     the predeclared standard in `PolicyConfig` decides whether the discrepancy matters.
  5. enforcement            `CrossCheck.enforces`: verified, adverse to the customer, material, and about the offer being bought.
                            The coordinator turns that, and only that, into an ALLOW -> REVIEW escalation.

Nothing here can produce a BLOCK, and a failed stage is never an accusation.

One further trusted-code helper lives here because it needs the same data and the same rules: `claim_provenance` records, for each
model finding that a claim is unverified, which merchant source it came from and whether any independent evidence exists. It changes
no decision. Judging whether a finding is true is `adjudicate`'s job.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, ValidationError

from sika_guard.evaluators.amounts import AmountMention, Quantity, no_fee_expression, scan_amounts
from sika_guard.evaluators.contract import ClaimKind, ConcernCode, EvaluatorVerdict, EvaluatorView, ExtractedClaim
from sika_guard.evaluators.grounding import claim_problem, locate_quote_detail, reading_hazards, source_items
from sika_guard.evaluators.payloads import PAYLOAD_BUILDERS, generalist_payload
from sika_guard.models import FinancingOffer, SourceDocument, SourceKind
from sika_guard.money import Money, sum_money
from sika_guard.policy import (
    DEFAULT_CONFIG,
    ClaimStatus,
    Direction,
    EvidenceRelevance,
    PolicyConfig,
    assess_claim,
    classify_evidence,
    compare_figures,
    hidden_obligation_is_material,
    understatement_is_material,
)


class CrossCheckOutcome(StrEnum):
    CONTRADICTED_BY_TERMS = "contradicted_by_terms"  # a merchant representation shows the customer less than the lender's terms say they will pay
    CONTRADICTED_BY_EVIDENCE = "contradicted_by_evidence"
    SOURCE_CONFLICT = "source_conflict"  # a lender's own document states an obligation the structured terms omit: two lender sources disagree
    CONSISTENT = "consistent"
    DISCREPANT_NOT_ADVERSE = "discrepant_not_adverse"  # the figures differ, but in the direction that does not harm the customer: recorded only
    NOT_COMPARABLE = "not_comparable"  # nothing authenticated to compare against, or the passage cannot be safely read; says nothing about truth
    UNGROUNDED = "ungrounded"  # the quote is not in the cited source as this evaluator was shown it; discarded
    EXTRACTION_DISAGREES = "extraction_disagrees_with_source"  # the quote is in the source but the model's figure or kind differs from what it states


class SourceClass(StrEnum):
    """What kind of source a grounded passage is, decided by trusted code from the case's own data."""

    MERCHANT_REPRESENTATION = "merchant_representation"  # the merchant's listing, statement or document: a claim, never corroboration
    LENDER_DOCUMENT = "lender_document"  # a document of the lender that issued the offer it is bound to
    INDEPENDENT_DOCUMENT = "independent_document"  # an independent party with no authority over this lender's terms


class CrossCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluator_id: str
    offer_id: str
    kind: ClaimKind
    outcome: CrossCheckOutcome
    detail: str
    grounded_in: str | None = None  # e.g. "listing:OFFER-A.advertised_total_repayable" or "doc:LD-1.text"
    source_class: SourceClass | None = None
    direction: Direction | None = None  # which way the stated figure is wrong, relative to the authoritative one
    stated_minor: int | None = None  # the figure the grounded passage states
    reference_minor: int | None = None  # the authoritative figure it was compared with
    currency: str | None = None
    material: bool = False  # under the predeclared standard (PolicyConfig); meaningful only for an adverse discrepancy
    concerns_selected_offer: bool = False  # the claim is about the offer actually being authorised
    folds: tuple[str, ...] = ()  # Unicode folds that were needed to match or read the passage; the original text is untouched

    @property
    def contradicted(self) -> bool:
        return self.outcome in (CrossCheckOutcome.CONTRADICTED_BY_TERMS, CrossCheckOutcome.CONTRADICTED_BY_EVIDENCE)

    @property
    def enforces(self) -> bool:
        """May this verified extraction raise an ALLOW to REVIEW? Only an adverse, material discrepancy about the selected offer. The same
        test applies to every condition's extractions; no condition has a route the others lack."""
        adverse = self.outcome in (CrossCheckOutcome.CONTRADICTED_BY_TERMS, CrossCheckOutcome.SOURCE_CONFLICT)
        return adverse and self.material and self.concerns_selected_offer


@dataclass(frozen=True)
class GroundedFact:
    """A claim whose passage has been found, classified and read. Still not a finding: it has not been compared with anything."""

    claim: ExtractedClaim
    ref: str
    field: str
    source_class: SourceClass
    folds: tuple[str, ...]
    hazards: tuple[str, ...]
    document: SourceDocument | None = None

    @property
    def where(self) -> str:
        return f"{self.ref}.{self.field}"


@dataclass(frozen=True)
class _Result:
    outcome: CrossCheckOutcome
    detail: str
    direction: Direction | None = None
    stated: Money | None = None
    reference: Money | None = None
    material: bool = False


def _plain(outcome: CrossCheckOutcome, detail: str) -> _Result:
    return _Result(outcome, detail)


# --- stage 2: from candidate to grounded fact ----------------------------------------------------------------------------------------


def _source_class(ref: str, view: EvaluatorView, offer: FinancingOffer, claim: ExtractedClaim) -> tuple[SourceClass, SourceDocument | None, str | None]:
    """(class, the document if any, a reason the source is unusable). Decided from the case, never from what the model or the text claims."""
    if not ref.startswith("doc:"):
        return SourceClass.MERCHANT_REPRESENTATION, None, None
    document = next((d for d in view.documents if f"doc:{d.document_id}" == ref), None)
    if document is None:
        return SourceClass.MERCHANT_REPRESENTATION, None, f"{ref} is not a document of this case"
    if document.published_at > view.as_of:
        return SourceClass.MERCHANT_REPRESENTATION, document, f"{ref} is dated after the decision clock"
    if not document.source.is_independent_of(view.merchant.merchant_id):
        return SourceClass.MERCHANT_REPRESENTATION, document, None  # the merchant's own document: a claim, whatever it calls itself
    if document.source.kind is SourceKind.LENDER and document.source.source_id == offer.lender_id and document.offer_id in (None, claim.offer_id):
        return SourceClass.LENDER_DOCUMENT, document, None
    return SourceClass.INDEPENDENT_DOCUMENT, document, None


def ground_claim(claim: ExtractedClaim, view: EvaluatorView, offer: FinancingOffer, payload: dict, config: PolicyConfig = DEFAULT_CONFIG) -> GroundedFact | _Result:
    problem = claim_problem(payload, source_ref=claim.source_ref, offer_id=claim.offer_id, quote=claim.quote)
    if problem is not None:
        return _plain(CrossCheckOutcome.UNGROUNDED, problem)
    item = source_items(payload)[claim.source_ref]
    field, folds = locate_quote_detail(item, claim.quote)  # claim_problem found it
    source_class, document, unusable = _source_class(claim.source_ref, view, offer, claim)
    if unusable is not None:
        return _plain(CrossCheckOutcome.UNGROUNDED, unusable)
    if document is not None and source_class is not SourceClass.MERCHANT_REPRESENTATION and view.as_of - document.published_at > config.max_document_age:
        return _plain(CrossCheckOutcome.NOT_COMPARABLE, f"{claim.source_ref} is older than the policy's document window, so it may describe superseded terms")
    hazards = tuple(sorted(set(reading_hazards(str(item.get(field, "")))) | set(reading_hazards(claim.quote))))
    return GroundedFact(claim, claim.source_ref, field, source_class, folds, hazards, document)


# --- reading the figure the passage states --------------------------------------------------------------------------------------------


def _as_money(claim: ExtractedClaim) -> Money | None:
    try:
        return Money(amount_minor=claim.amount_minor, currency=claim.currency)
    except ValidationError:
        return None


def _select_amount(extracted: Money, mentions: list[AmountMention], wanted: set[Quantity], noun: str) -> tuple[Money | None, Quantity | None, _Result | None]:
    """Find, among the amounts a quote states, the one that is the claimed kind of figure and equals the extracted one.

    Returns (amount, the quantity it was stated as, None) on success, or (None, None, an early result) when the quote does not
    support the extraction. Other amounts in the quote are ignored unless they are the claimed kind: an instalment next to a
    total does not hide the total, and an instalment is never accepted as a total. An amount the passage makes conditional or optional
    is not a mandatory obligation and is never counted as one."""
    if any(m.problem for m in mentions):
        return None, None, _plain(CrossCheckOutcome.NOT_COMPARABLE, "the quote contains an amount expression that cannot be read in full (scaled, mis-grouped or over-precise)")
    valid = [m for m in mentions if m.money is not None]
    matching = [m for m in valid if m.quantity in wanted]
    for m in matching:
        if m.money == extracted:
            if m.conditional:
                return None, None, _plain(CrossCheckOutcome.NOT_COMPARABLE, "the passage makes this amount conditional or optional, so it is not an obligation on everyone who takes the agreement")
            return m.money, m.quantity, None
    if matching:
        stated = ", ".join(sorted({m.money.plain() for m in matching}))
        return None, None, _plain(CrossCheckOutcome.EXTRACTION_DISAGREES, f"model extracted {extracted.plain()}; the quote states {noun} {stated}")
    same_figure = next((m for m in valid if m.money == extracted), None)
    if same_figure is not None:
        if same_figure.quantity is Quantity.UNKNOWN:
            return None, None, _plain(CrossCheckOutcome.NOT_COMPARABLE, "the quote states the amount without saying what it is")
        return None, None, _plain(CrossCheckOutcome.EXTRACTION_DISAGREES, f"the quote states {extracted.plain()} as a {same_figure.quantity.value}, not as {noun}")
    if valid:
        return None, None, _plain(CrossCheckOutcome.EXTRACTION_DISAGREES, f"model extracted {extracted.plain()}; the quote states {', '.join(sorted({m.money.plain() for m in valid}))}")
    return None, None, _plain(CrossCheckOutcome.EXTRACTION_DISAGREES, "the quote states no explicit amount")


# --- stages 3 and 4: independent support, direction, materiality ---------------------------------------------------------------------------


def _judge(fact: GroundedFact, view: EvaluatorView, claimed: Money, reference: Money, noun: str, config: PolicyConfig, *, whole: Money) -> _Result:
    """Compare a grounded figure with the authoritative one. `whole` is the schedule's total repayable, used to see whether an omitted
    obligation would carry the customer over a total limit."""
    d = compare_figures(claimed, reference)
    if d is None:
        return _plain(CrossCheckOutcome.NOT_COMPARABLE, f"currencies differ: the passage is in {claimed.currency}, the lender's terms in {reference.currency}; nothing was converted")
    if d.direction is Direction.EQUAL:
        return _Result(CrossCheckOutcome.CONSISTENT, f"{noun} is {reference.plain()}", Direction.EQUAL, claimed, reference)
    if fact.source_class is SourceClass.MERCHANT_REPRESENTATION:
        if d.direction is Direction.UNDERSTATES_COST:
            return _Result(CrossCheckOutcome.CONTRADICTED_BY_TERMS, f"merchant shows {noun} as {claimed.plain()}; lender terms say {reference.plain()} ({d.magnitude.plain()} more)",
                           d.direction, claimed, reference, understatement_is_material(d, config))
        return _Result(CrossCheckOutcome.DISCREPANT_NOT_ADVERSE, f"merchant shows {noun} as {claimed.plain()}; lender terms say {reference.plain()}: the customer would pay less than shown",
                       d.direction, claimed, reference)
    # A lender's own document against the lender's structured terms. The document stating MORE than the terms is the adverse case.
    if d.direction is Direction.OVERSTATES_COST:
        material = hidden_obligation_is_material(d.magnitude, whole, view.intent, view.mandate, config)
        return _Result(CrossCheckOutcome.SOURCE_CONFLICT,
                       f"the lender's document states {noun} of {claimed.plain()} but the structured terms say {reference.plain()} ({d.magnitude.plain()} not in the terms); which governs is unresolved",
                       d.direction, claimed, reference, material)
    return _Result(CrossCheckOutcome.DISCREPANT_NOT_ADVERSE, f"the lender's document states {noun} of {claimed.plain()}, below the structured terms' {reference.plain()}: the terms already assume the higher figure",
                   d.direction, claimed, reference)


def _check_total(fact: GroundedFact, offer: FinancingOffer, view: EvaluatorView, config: PolicyConfig) -> _Result:
    claim = fact.claim
    extracted = _as_money(claim)
    if extracted is None:
        return _plain(CrossCheckOutcome.NOT_COMPARABLE, "extracted amount is not a valid sum of money")
    real = offer.schedule.total()
    if fact.field == "advertised_total_repayable" and offer.advertised_total_repayable is not None and fact.source_class is SourceClass.MERCHANT_REPRESENTATION:
        # The structured field is authoritative for what the merchant advertised; the model's figure is only checked against it.
        if extracted != offer.advertised_total_repayable:
            return _plain(CrossCheckOutcome.EXTRACTION_DISAGREES, f"model extracted {extracted.plain()}; the source field says {offer.advertised_total_repayable.plain()}")
        claimed = offer.advertised_total_repayable
    else:
        claimed, _, early = _select_amount(extracted, scan_amounts(claim.quote), {Quantity.TOTAL}, "a total repayable")
        if early is not None:
            return early
    return _judge(fact, view, claimed, real, "total repayable", config, whole=real)


def _check_fee_amount(fact: GroundedFact, offer: FinancingOffer, view: EvaluatorView, config: PolicyConfig) -> _Result:
    claim = fact.claim
    extracted = _as_money(claim)
    if extracted is None:
        return _plain(CrossCheckOutcome.NOT_COMPARABLE, "extracted amount is not a valid sum of money")
    mentions = scan_amounts(claim.quote)
    real = offer.schedule.total_fees()
    if extracted.currency != real.currency:
        return _plain(CrossCheckOutcome.NOT_COMPARABLE, f"currencies differ: the claim is in {extracted.currency}, the lender's terms in {real.currency}; nothing was converted")
    stated_as: Quantity | None
    if extracted.amount_minor == 0:
        # "Zero fees" must be what the quote says. A quote that states a fee, or says nothing about fees, does not support zero.
        stated = [m for m in mentions if m.money is not None and m.quantity in (Quantity.FEE, Quantity.FEE_TOTAL) and m.money.amount_minor > 0]
        if stated:
            return _plain(CrossCheckOutcome.EXTRACTION_DISAGREES, f"model extracted zero fees; the quote states a fee of {stated[0].money.plain()}")
        if not no_fee_expression(claim.quote) and not any(m.money is not None and m.money.amount_minor == 0 and m.quantity in (Quantity.FEE, Quantity.FEE_TOTAL) for m in mentions):
            return _plain(CrossCheckOutcome.EXTRACTION_DISAGREES, "the quote does not state that there are no fees")
        claimed, stated_as = extracted, Quantity.FEE_TOTAL
    else:
        claimed, stated_as, early = _select_amount(extracted, mentions, {Quantity.FEE, Quantity.FEE_TOTAL}, "a fee")
        if early is not None and early.outcome is CrossCheckOutcome.EXTRACTION_DISAGREES:
            # Several separate mandatory fees in one passage: the claimed total may be their sum, which is plain arithmetic.
            parts = [m.money for m in mentions if m.money is not None and m.quantity is Quantity.FEE and not m.conditional]
            if len(parts) > 1 and not any(m.problem for m in mentions) and len({p.currency for p in parts}) == 1 and sum_money(parts, parts[0].currency) == extracted:
                claimed, stated_as, early = extracted, Quantity.FEE_TOTAL, None
        if early is not None:
            return early
    if claimed != real and stated_as is Quantity.FEE and any(f.amount == claimed for f in offer.schedule.fees):
        return _plain(CrossCheckOutcome.NOT_COMPARABLE, f"the claim matches one fee, but the lender terms list {len(offer.schedule.fees)} fees totalling {real.plain()}")
    return _judge(fact, view, claimed, real, "mandatory fees", config, whole=offer.schedule.total())


def _check_no_fees(fact: GroundedFact, offer: FinancingOffer, view: EvaluatorView, config: PolicyConfig) -> _Result:
    claim = fact.claim
    if not no_fee_expression(claim.quote):
        return _plain(CrossCheckOutcome.EXTRACTION_DISAGREES, "the quote does not state that there are no (additional) fees")
    mentions = scan_amounts(claim.quote)
    if any(m.problem for m in mentions) or any(m.money is not None and m.quantity in (Quantity.FEE, Quantity.FEE_TOTAL) and m.money.amount_minor > 0 for m in mentions):
        return _plain(CrossCheckOutcome.NOT_COMPARABLE, "the quote both denies fees and states one, so it cannot be tested")
    zero = Money(amount_minor=0, currency=offer.schedule.currency)
    return _judge(fact, view, zero, offer.schedule.total_fees(), "mandatory fees", config, whole=offer.schedule.total())


def _check_availability(fact: GroundedFact, view: EvaluatorView, offer: FinancingOffer, config: PolicyConfig) -> _Result:
    claim = fact.claim
    kind, _, ident = claim.source_ref.partition(":")
    if kind != "assertion":
        return _plain(CrossCheckOutcome.NOT_COMPARABLE, "availability stated in free text or in a document carries no structured timestamp or subject to test against")
    assertion = next(a for a in view.assertions if a.assertion_id == ident)
    if assertion.offer_id != claim.offer_id or assertion.claims_available != claim.asserted_available:
        return _plain(CrossCheckOutcome.UNGROUNDED, "extraction disagrees with the structured assertion")
    status = assess_claim(assertion, view.evidence, offer, view.as_of, config, view.intent.customer_id).status
    if status is ClaimStatus.CONFIRMED_FALSE:
        return _plain(CrossCheckOutcome.CONTRADICTED_BY_EVIDENCE, "fresh independent evidence says otherwise")
    if status is ClaimStatus.CONFIRMED_TRUE:
        return _plain(CrossCheckOutcome.CONSISTENT, "fresh independent evidence agrees")
    return _plain(CrossCheckOutcome.NOT_COMPARABLE, f"claim is {status.value}")


def compare_fact(fact: GroundedFact, view: EvaluatorView, offer: FinancingOffer, config: PolicyConfig) -> _Result:
    claim = fact.claim
    if claim.kind is ClaimKind.AVAILABILITY:
        return _check_availability(fact, view, offer, config)
    if fact.source_class is SourceClass.INDEPENDENT_DOCUMENT:
        return _plain(CrossCheckOutcome.NOT_COMPARABLE, "this independent source has no authority over the offer's own lender's terms, so it cannot confirm or contradict them")
    if fact.hazards:
        return _plain(CrossCheckOutcome.NOT_COMPARABLE, f"the passage cannot be read safely ({', '.join(fact.hazards)}); no amount was taken from it")
    if claim.kind is ClaimKind.TOTAL_REPAYABLE:
        return _check_total(fact, offer, view, config)
    if claim.kind is ClaimKind.FEE_AMOUNT:
        return _check_fee_amount(fact, offer, view, config)
    return _check_no_fees(fact, offer, view, config)


def _record(evaluator_id: str, claim: ExtractedClaim, view: EvaluatorView, r: _Result, fact: GroundedFact | None) -> CrossCheck:
    money = r.reference or r.stated
    return CrossCheck(
        evaluator_id=evaluator_id, offer_id=claim.offer_id, kind=claim.kind, outcome=r.outcome, detail=r.detail,
        grounded_in=fact.where if fact is not None else None, source_class=fact.source_class if fact is not None else None,
        direction=r.direction, stated_minor=r.stated.amount_minor if r.stated else None, reference_minor=r.reference.amount_minor if r.reference else None,
        currency=money.currency if money else None, material=r.material,
        concerns_selected_offer=claim.offer_id == view.transaction.offer_id, folds=fact.folds if fact is not None else (),
    )


def crosscheck_claims(
    view: EvaluatorView, verdicts: Iterable[EvaluatorVerdict], config: PolicyConfig = DEFAULT_CONFIG
) -> tuple[CrossCheck, ...]:
    offers = {o.offer_id: o for o in view.offers}
    results = []
    for verdict in verdicts:
        if not verdict.usable:
            continue
        payload = PAYLOAD_BUILDERS.get(verdict.role, generalist_payload)(view)  # what THIS evaluator was shown
        for claim in verdict.extracted_claims:
            offer = offers[claim.offer_id]  # grounding already confirmed the offer exists
            grounded = ground_claim(claim, view, offer, payload, config)
            if isinstance(grounded, _Result):
                results.append(_record(verdict.evaluator_id, claim, view, grounded, None))
                continue
            results.append(_record(verdict.evaluator_id, claim, view, compare_fact(grounded, view, offer, config), grounded))
    return tuple(results)


# --- Provenance of "unverified" findings ------------------------------------

_MERCHANT_REF_KINDS = ("listing", "assertion", "merchant")


class Corroboration(StrEnum):
    ABSENT = "absent"  # no independent evidence about the claim exists in the input
    STALE_ONLY = "stale_only"
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    CONFLICTED = "conflicted"


_CORROBORATION = {
    ClaimStatus.UNVERIFIED: Corroboration.ABSENT,
    ClaimStatus.STALE_ONLY: Corroboration.STALE_ONLY,
    ClaimStatus.CONFIRMED_TRUE: Corroboration.SUPPORTING,
    ClaimStatus.CONFIRMED_FALSE: Corroboration.CONTRADICTING,
    ClaimStatus.CONFLICTED: Corroboration.CONFLICTED,
    ClaimStatus.INVALID_CHRONOLOGY: Corroboration.ABSENT,  # a statement from the future is not assessed against anything
}


class ClaimProvenance(BaseModel):
    """Where a claim a model called unverified came from, and what trusted code finds around it.

    The merchant's own reference is retained and is never counted as corroboration. An unverified
    claim stays unverified: nothing here says the merchant lied, and absence of evidence is not a
    contradiction. `claim_status` is the deterministic status (`unverified` when no independent
    evidence exists), which can differ from the model's label if evidence turns out to exist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluator_id: str
    offer_id: str | None
    merchant_refs: tuple[str, ...]  # merchant-originated refs the finding cited; empty if it cited none
    corroboration: Corroboration
    corroborating_refs: tuple[str, ...]  # independent evidence about the same claim, derived by trusted code
    claim_status: str


def claim_provenance(
    view: EvaluatorView, verdicts: Iterable[EvaluatorVerdict], config: PolicyConfig = DEFAULT_CONFIG
) -> tuple[ClaimProvenance, ...]:
    offers = {o.offer_id: o for o in view.offers}
    assertions = {f"assertion:{a.assertion_id}": a for a in view.assertions}
    records = []
    for verdict in verdicts:
        if not verdict.usable:
            continue
        for finding in verdict.findings:
            if finding.code is not ConcernCode.CLAIM_UNVERIFIED:
                continue
            merchant_refs = tuple(r for r in finding.evidence_refs if r.partition(":")[0] in _MERCHANT_REF_KINDS)
            assertion = next((assertions[r] for r in merchant_refs if r in assertions), None)
            if assertion is None:
                # A listing or merchant reference: independent evidence in this data model concerns
                # availability only, so there is none to find for it.
                offer_id = next((r.partition(":")[2] for r in merchant_refs if r.startswith("listing:")), None)
                records.append(
                    ClaimProvenance(
                        evaluator_id=verdict.evaluator_id, offer_id=offer_id, merchant_refs=merchant_refs,
                        corroboration=Corroboration.ABSENT, corroborating_refs=(), claim_status=ClaimStatus.UNVERIFIED.value,
                    )
                )
                continue
            offer = offers.get(assertion.offer_id)
            status = assess_claim(assertion, view.evidence, offer, view.as_of, config, view.intent.customer_id).status
            relevant = tuple(
                f"evidence:{e.evidence_id}"
                for e in view.evidence
                if classify_evidence(assertion, e, offer, view.as_of, config, view.intent.customer_id) is not EvidenceRelevance.IRRELEVANT
            )
            records.append(
                ClaimProvenance(
                    evaluator_id=verdict.evaluator_id, offer_id=assertion.offer_id, merchant_refs=merchant_refs,
                    corroboration=_CORROBORATION[status], corroborating_refs=relevant, claim_status=status.value,
                )
            )
    return tuple(records)
