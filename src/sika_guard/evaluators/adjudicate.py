"""Deterministic adjudication of model findings.

Three things are kept apart:

  1. a model allegation        what a finding says; recorded exactly as given
  2. an adjudicated finding    what trusted code can establish about it from structured data
  3. the executable decision   decided in `coordinator`, never here

Each finding gets one of three statuses:

  verified                          trusted code confirms the allegation from structured data
  contradicted_by_structured_data   trusted code shows the structured data says otherwise
  unverified                        trusted code can neither confirm nor disprove it; it stays an
                                    uncertainty and is not turned into either certainty

Only findings whose subject is objectively checkable are decided (numeric limits, whether a merchant
total equals the lender total, whether an availability claim is contradicted, whether a cheaper
offer exists). Everything else stays unverified. The text of a note or an explanation is never read:
only the finding's code and the references it cites.

Enforcement (the frozen decision policy, see DECISION_POLICY.md). A model verdict is advisory and has no
enforcement effect of its own. Each finding gets an effect:

  enforcing              verified AND material: it may support REVIEW
  verified_not_material  verified, but too small or irrelevant to the purchase to act on
  model_error            contradicted by authenticated structured facts: logged as a model error, no effect
  unverified_recorded    could not be verified or disproved: recorded as uncertainty, no effect

A verdict counts as REVIEW only if at least one of its findings is enforcing; otherwise it counts as ALLOW,
whatever it said and however confident it was. Vague suspicion, unverifiable allegations and self-reported
risk can therefore never escalate on their own. Materiality is read from the rule engine's own result, so the
two cannot disagree about it. Nothing here can create a BLOCK or lower one.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from sika_guard.evaluators.amounts import mentions_money
from sika_guard.evaluators.contract import AIFinding, ConcernCode, EvaluatorVerdict, EvaluatorView
from sika_guard.models import Decision, FinancingOffer, FindingCategory, FindingCode, SecurityDecision
from sika_guard.policy import (
    DEFAULT_CONFIG,
    ClaimStatus,
    PolicyConfig,
    assess_claim,
    check_offer_limits,
    check_payment_cap,
    objective_cost_minor,
)


class AllegationStatus(StrEnum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted_by_structured_data"
    UNVERIFIED = "unverified"


class FindingEffect(StrEnum):
    ENFORCING = "enforcing"
    VERIFIED_NOT_MATERIAL = "verified_not_material"
    MODEL_ERROR = "model_error"
    UNVERIFIED_RECORDED = "unverified_recorded"


class FindingAdjudication(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluator_id: str
    finding_index: int
    code: ConcernCode
    evidence_refs: tuple[str, ...]  # as cited by the model
    ref_sources: tuple[str, ...]  # what kind of source each cited ref is, in the same order
    status: AllegationStatus
    method: str  # which deterministic check decided it, or "not_objectively_checkable"
    reason: str
    effect: FindingEffect = FindingEffect.UNVERIFIED_RECORDED


class VerdictTreatment(BaseModel):
    """How one usable verdict entered the decision, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluator_id: str
    model_verdict: Decision
    counted_as: Decision
    findings: int
    contradicted: int
    reason: str
    enforcing: int = 0


# --- What kind of source is a reference? ------------------------------------


def ref_source(view: EvaluatorView, ref: str) -> str:
    """merchant_assertion | independent_evidence | merchant_channel_evidence | lender_terms |
    customer_or_mandate | unknown. Separates what a merchant said from what an independent source observed."""
    kind, _, ident = ref.partition(":")
    if kind in ("assertion", "listing", "merchant"):
        return "merchant_assertion"
    if kind == "evidence":
        evidence = next((e for e in view.evidence if e.evidence_id == ident), None)
        if evidence is None:
            return "unknown"
        return "independent_evidence" if evidence.source.is_independent_of(view.merchant.merchant_id) else "merchant_channel_evidence"
    if kind == "terms":
        return "lender_terms"
    if kind in ("intent", "mandate", "txn"):
        return "customer_or_mandate"
    return "unknown"


# --- Individual checks -------------------------------------------------------

_LIMIT_CODES = frozenset(
    {
        FindingCode.AMOUNT_EXCEEDS_MANDATE,
        FindingCode.COMMITTED_PAYMENT_EXCEEDS_MANDATE,
        FindingCode.MANDATE_TOTAL_EXCEEDED,
        FindingCode.MANDATE_MONTHLY_EXCEEDED,
        FindingCode.INTENT_TOTAL_EXCEEDED,
        FindingCode.INTENT_MONTHLY_EXCEEDED,
    }
)

Outcome = tuple[AllegationStatus, str, str]  # status, method, reason


def _offer_ids(refs: Iterable[str], *kinds: str) -> list[str]:
    return [ident for kind, _, ident in (r.partition(":") for r in refs) if kind in kinds]


def _check_limit(view: EvaluatorView, finding: AIFinding) -> Outcome:
    """Is any numeric limit exceeded by the payment, the selected offer or an offer the finding cites?
    Limits are objective, so this uses the rule engine's own limit functions."""
    offers = {o.offer_id: o for o in view.offers}
    scope = sorted(({view.transaction.offer_id} | set(_offer_ids(finding.evidence_refs, "terms", "listing"))) & offers.keys())
    found = list(check_payment_cap(view.mandate, view.transaction.amount, view.transaction.transaction_id))
    for offer_id in scope:
        found += check_offer_limits(offers[offer_id], view.intent, view.mandate)
    exceeded = sorted({f.code.value for f in found if f.code in _LIMIT_CODES})
    if exceeded:
        return AllegationStatus.VERIFIED, "limit_checks", f"the rule engine's limit checks find: {', '.join(exceeded)}"
    return (
        AllegationStatus.CONTRADICTED,
        "limit_checks",
        f"no limit is exceeded: payment {view.transaction.amount.plain()} against mandate cap {view.mandate.max_amount.plain()}; "
        f"offers checked: {', '.join(scope) or 'none'}",
    )


def _check_contradiction(view: EvaluatorView, finding: AIFinding, config: PolicyConfig) -> Outcome:
    """Does the cited material show the merchant claim to be false?"""
    refs = finding.evidence_refs
    offers = {o.offer_id: o for o in view.offers}

    assertion = next((a for a in view.assertions if f"assertion:{a.assertion_id}" in refs), None)
    if assertion is not None:
        status = assess_claim(assertion, view.evidence, offers.get(assertion.offer_id), view.as_of, config, view.intent.customer_id).status
        if status is ClaimStatus.CONFIRMED_FALSE:
            return AllegationStatus.VERIFIED, "claim_assessment", "fresh independent evidence contradicts the assertion"
        if status is ClaimStatus.CONFIRMED_TRUE:
            return AllegationStatus.CONTRADICTED, "claim_assessment", "fresh independent evidence supports the assertion"
        return AllegationStatus.UNVERIFIED, "claim_assessment", f"the assertion is {status.value}; that does not prove a contradiction"

    listing_ids = list(dict.fromkeys(_offer_ids(refs, "listing")))
    if listing_ids:
        method = "advertised_vs_lender_total"
        outcomes = [_check_listing(offers.get(i)) for i in listing_ids]
        for wanted in (AllegationStatus.VERIFIED, AllegationStatus.UNVERIFIED):  # any real mismatch outranks an unchecked listing
            hit = next((o for o in outcomes if o[0] is wanted), None)
            if hit:
                return wanted, method, hit[1]
        return AllegationStatus.CONTRADICTED, method, "; ".join(o[1] for o in outcomes)

    return AllegationStatus.UNVERIFIED, "not_objectively_checkable", "the cited sources do not identify a checkable merchant claim"


def _check_listing(offer: FinancingOffer | None) -> tuple[AllegationStatus, str]:
    if offer is None or offer.advertised_total_repayable is None:
        return AllegationStatus.UNVERIFIED, "no structured advertised figure to compare"
    advertised, real = offer.advertised_total_repayable, offer.schedule.total()
    if advertised.currency != real.currency:
        return AllegationStatus.UNVERIFIED, "advertised and lender totals are in different currencies"
    if advertised != real:
        return AllegationStatus.VERIFIED, f"{offer.offer_id}: advertised {advertised.plain()} differs from lender total {real.plain()}"
    if offer.description.strip() or mentions_money(offer.title):
        # Free text could carry a false claim that the headline figure does not cover, so nothing is concluded.
        return AllegationStatus.UNVERIFIED, f"{offer.offer_id}: the figures agree but its free text was not checked"
    return AllegationStatus.CONTRADICTED, f"{offer.offer_id}: advertised total equals lender total ({real.plain()}) and the listing has no other text"


def _check_selection(view: EvaluatorView, det: SecurityDecision) -> Outcome:
    """Is there an offer that is cheaper under the customer's own objective?"""
    offers = {o.offer_id: o for o in view.offers}
    selected = offers.get(view.transaction.offer_id)
    cost = objective_cost_minor(selected, view.intent.objective) if selected else None
    if selected is None or cost is None:
        return AllegationStatus.UNVERIFIED, "selection_cost", "the customer's objective gives no cost ordering"
    cheaper = [
        o.offer_id
        for o in view.offers
        if o.offer_id != selected.offer_id
        and o.schedule.currency == selected.schedule.currency
        and (c := objective_cost_minor(o, view.intent.objective)) is not None
        and c < cost
    ]
    if not cheaper:
        return AllegationStatus.CONTRADICTED, "selection_cost", "no other offer costs less under the customer's objective"
    if det.selection is not None and det.selection.better_offer_id is not None:
        return AllegationStatus.VERIFIED, "selection_cost", f"the rule engine finds a cheaper usable offer, {det.selection.better_offer_id}"
    return AllegationStatus.UNVERIFIED, "selection_cost", f"a cheaper offer exists ({', '.join(sorted(cheaper))}) but is not confirmed usable"


def adjudicate_finding(
    view: EvaluatorView, det: SecurityDecision, finding: AIFinding, config: PolicyConfig = DEFAULT_CONFIG
) -> Outcome:
    code = finding.code
    if code is ConcernCode.LIMIT_EXCEEDED:
        return _check_limit(view, finding)
    if code is ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE:
        return _check_contradiction(view, finding, config)
    if code is ConcernCode.SELECTION_WORSE_THAN_ALTERNATIVE:
        return _check_selection(view, det)
    if code is ConcernCode.CLAIM_UNVERIFIED:
        return AllegationStatus.UNVERIFIED, "unverified_claim", "reported as an uncertainty; nothing is alleged to be false"
    return AllegationStatus.UNVERIFIED, "not_objectively_checkable", "this kind of concern has no deterministic check"


# --- Materiality and effect --------------------------------------------------


def _is_material(view: EvaluatorView, det: SecurityDecision, finding: AIFinding) -> bool:
    """Is a *verified* finding relevant enough to the purchase to act on? Read from the rule engine's result."""
    if finding.code is ConcernCode.LIMIT_EXCEEDED:
        return not det.authorised  # a verified exceeded limit on the purchase is a hard violation
    if finding.code is ConcernCode.SELECTION_WORSE_THAN_ALTERNATIVE:
        return det.selection is not None and det.selection.inferior
    if finding.code is ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE:
        subjects = {a.offer_id for a in view.assertions if f"assertion:{a.assertion_id}" in finding.evidence_refs}
        subjects |= set(_offer_ids(finding.evidence_refs, "listing"))
        return any(f.material and f.subject_id in subjects and f.category is FindingCategory.CONFIRMED_FALSE_CLAIM for f in det.findings)
    return False


def _effect(status: AllegationStatus, material: bool) -> FindingEffect:
    if status is AllegationStatus.CONTRADICTED:
        return FindingEffect.MODEL_ERROR
    if status is AllegationStatus.VERIFIED:
        return FindingEffect.ENFORCING if material else FindingEffect.VERIFIED_NOT_MATERIAL
    return FindingEffect.UNVERIFIED_RECORDED


# --- Per-case adjudication ----------------------------------------------------


def _treat(verdict: EvaluatorVerdict, adjudications: list[FindingAdjudication]) -> VerdictTreatment:
    assert verdict.verdict is not None
    n = len(adjudications)
    count = lambda effect: sum(a.effect is effect for a in adjudications)  # noqa: E731
    enforcing, errors = count(FindingEffect.ENFORCING), count(FindingEffect.MODEL_ERROR)
    base = dict(evaluator_id=verdict.evaluator_id, model_verdict=verdict.verdict, findings=n, contradicted=errors, enforcing=enforcing)
    if enforcing:
        return VerdictTreatment(
            **base, counted_as=Decision.REVIEW,
            reason=f"REVIEW: {enforcing} finding(s) verified by trusted code and material to this purchase (the model said {verdict.verdict.value})",
        )
    if verdict.verdict is Decision.ALLOW:
        return VerdictTreatment(**base, counted_as=Decision.ALLOW, reason="no escalation requested and none enforced")
    detail = (
        f"{errors} contradicted by structured data (logged as model errors), {count(FindingEffect.VERIFIED_NOT_MATERIAL)} verified but not material, "
        f"{count(FindingEffect.UNVERIFIED_RECORDED)} unverified (recorded as uncertainty)"
    )
    return VerdictTreatment(
        **base, counted_as=Decision.ALLOW,
        reason=f"model asked for {verdict.verdict.value} but no finding is both verified and material, so it has no enforcement effect"
        + (f": {detail}" if n else "; it offered no findings"),
    )


def adjudicate(
    view: EvaluatorView, det: SecurityDecision, verdicts: Iterable[EvaluatorVerdict], config: PolicyConfig = DEFAULT_CONFIG
) -> tuple[tuple[FindingAdjudication, ...], tuple[VerdictTreatment, ...]]:
    """Adjudicate every finding of every usable verdict and decide how each verdict counts.
    A verdict that is not usable has no findings to adjudicate and is not a treatment."""
    all_adjudications: list[FindingAdjudication] = []
    treatments: list[VerdictTreatment] = []
    for verdict in verdicts:
        if not verdict.usable:
            continue
        mine = []
        for index, finding in enumerate(verdict.findings):
            status, method, reason = adjudicate_finding(view, det, finding, config)
            effect = _effect(status, status is AllegationStatus.VERIFIED and _is_material(view, det, finding))
            mine.append(
                FindingAdjudication(
                    evaluator_id=verdict.evaluator_id, finding_index=index, code=finding.code,
                    evidence_refs=finding.evidence_refs, ref_sources=tuple(ref_source(view, r) for r in finding.evidence_refs),
                    status=status, method=method, reason=reason, effect=effect,
                )
            )
        all_adjudications += mine
        treatments.append(_treat(verdict, mine))
    return tuple(all_adjudications), tuple(treatments)


def unsupported_limit_allegations(adjudications: Iterable[FindingAdjudication]) -> tuple[str, ...]:
    """`evaluator:limit_exceeded` for every limit allegation that structured data contradicts."""
    return tuple(
        sorted(
            {
                f"{a.evaluator_id}:limit_exceeded"
                for a in adjudications
                if a.code is ConcernCode.LIMIT_EXCEEDED and a.status is AllegationStatus.CONTRADICTED
            }
        )
    )
