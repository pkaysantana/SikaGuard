"""Deterministic policy baseline.

Rules here cover only objectively checkable properties: numeric limits, money
arithmetic, identifier equality and timestamps. Nothing here calls a model, and
nothing here may import ground-truth labels.

Invariants (see CLAUDE.md):
  * A hard violation always yields BLOCK, and nothing downstream may lower it.
  * An in-mandate transaction is never BLOCKed merely for being costlier than
    another offer; that is a decision-quality finding, which yields REVIEW.
  * Missing or stale evidence is not a contradiction and is never reported as one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from sika_guard.models import (
    Case,
    CustomerIntent,
    Decision,
    Evidence,
    FinancingOffer,
    Finding,
    FindingCategory,
    FindingCode,
    Mandate,
    MerchantAssertion,
    Objective,
    ProposedTransaction,
    Scope,
    SecurityDecision,
    SelectionAssessment,
    SourceKind,
)
from sika_guard.money import Money

C = FindingCategory
F = FindingCode


@dataclass(frozen=True)
class PolicyConfig:
    max_evidence_age: timedelta = timedelta(hours=1)
    material_gap_minor: int = 500  # smaller cost differences are not worth escalating


DEFAULT_CONFIG = PolicyConfig()


# --- Hard authorisation checks ---------------------------------------------


def _hard(code: FindingCode, message: str, subject_id: str | None = None) -> Finding:
    return Finding(category=C.HARD_VIOLATION, code=code, message=message, subject_id=subject_id)


def _limit_findings(
    value: Money, limit: Money | None, exceeded: FindingCode, label: str, subject_id: str
) -> list[Finding]:
    if limit is None:
        return []
    if value.currency != limit.currency:
        return [_hard(F.CURRENCY_MISMATCH, f"{label} is in {value.currency} but the limit is in {limit.currency}", subject_id)]
    if value > limit:
        return [_hard(exceeded, f"{label} {value} exceeds the limit {limit}", subject_id)]
    return []


def check_mandate_window(mandate: Mandate, as_of: datetime) -> list[Finding]:
    if as_of < mandate.issued_at:
        return [_hard(F.MANDATE_NOT_YET_VALID, "mandate is not yet valid", mandate.mandate_id)]
    if as_of >= mandate.expires_at:
        return [_hard(F.MANDATE_EXPIRED, "mandate has expired", mandate.mandate_id)]
    return []


def check_scope(mandate: Mandate, scope: Scope) -> list[Finding]:
    if scope not in mandate.permitted_scopes:
        return [_hard(F.SCOPE_NOT_PERMITTED, f"scope {scope.value} is outside the mandate", mandate.mandate_id)]
    return []


def check_payee(mandate: Mandate, payee_id: str) -> list[Finding]:
    if payee_id not in mandate.allowed_payee_ids:
        return [_hard(F.PAYEE_NOT_ALLOWED, f"payee {payee_id} is not permitted by the mandate", payee_id)]
    return []


def check_payment_cap(mandate: Mandate, amount: Money, subject_id: str) -> list[Finding]:
    return _limit_findings(amount, mandate.max_amount, F.AMOUNT_EXCEEDS_MANDATE, "payment", subject_id)


def check_offer_limits(offer: FinancingOffer, intent: CustomerIntent, mandate: Mandate) -> list[Finding]:
    """Total-repayment, monthly-payment and term limits from the mandate and the customer."""
    total = offer.schedule.total()
    monthly = offer.schedule.max_payment()
    found = [
        *_limit_findings(total, mandate.max_total_repayment, F.MANDATE_TOTAL_EXCEEDED, "total repayable", offer.offer_id),
        *_limit_findings(monthly, mandate.max_monthly_payment, F.MANDATE_MONTHLY_EXCEEDED, "monthly payment", offer.offer_id),
        *_limit_findings(total, intent.max_total_repayment, F.INTENT_TOTAL_EXCEEDED, "total repayable", offer.offer_id),
        *_limit_findings(monthly, intent.max_monthly_payment, F.INTENT_MONTHLY_EXCEEDED, "monthly payment", offer.offer_id),
    ]
    if intent.term_months is not None and offer.schedule.term_months != intent.term_months:
        found.append(
            _hard(F.INTENT_TERM_MISMATCH, f"term is {offer.schedule.term_months} months, customer asked for {intent.term_months}", offer.offer_id)
        )
    return found


def check_offer_authorised(offer: FinancingOffer, case: Case) -> list[Finding]:
    """Would accepting this offer, instead of the proposed one, have been authorised?"""
    as_of = case.transaction.proposed_at
    return [
        *check_mandate_window(case.mandate, as_of),
        *check_scope(case.mandate, case.transaction.scope),
        *check_payee(case.mandate, offer.payee_id),
        *check_payment_cap(case.mandate, offer.schedule.initial_charge(), offer.offer_id),
        *check_offer_limits(offer, case.intent, case.mandate),
    ]


def check_transaction(txn: ProposedTransaction, offer: FinancingOffer, case: Case) -> list[Finding]:
    """All hard checks for the proposed transaction, deduplicated."""
    mandate, intent = case.mandate, case.intent
    found: list[Finding] = []
    if txn.mandate_id != mandate.mandate_id:
        found.append(_hard(F.MANDATE_MISMATCH, "transaction cites a different mandate", txn.mandate_id))
    if intent.customer_id != mandate.customer_id:
        found.append(_hard(F.CUSTOMER_MISMATCH, "mandate belongs to a different customer than the instruction", mandate.mandate_id))
    found += check_mandate_window(mandate, txn.proposed_at)
    found += check_scope(mandate, txn.scope)
    found += check_payee(mandate, txn.payee_id)
    if txn.payee_id != offer.payee_id:
        found.append(_hard(F.PAYEE_OFFER_MISMATCH, f"pays {txn.payee_id} but the offer is payable to {offer.payee_id}", txn.payee_id))
    found += check_payee(mandate, offer.payee_id)
    found += check_payment_cap(mandate, txn.amount, txn.transaction_id)
    if txn.amount != offer.schedule.initial_charge():
        found.append(
            _hard(F.AMOUNT_OFFER_MISMATCH, f"charges {txn.amount} but the offer's initial charge is {offer.schedule.initial_charge()}", txn.transaction_id)
        )
    found += check_offer_limits(offer, intent, mandate)
    if intent.objective is Objective.SPECIFIC_OFFER and intent.requested_offer_id != offer.offer_id:
        found.append(_hard(F.INTENT_OFFER_MISMATCH, f"customer asked for {intent.requested_offer_id}, not {offer.offer_id}", offer.offer_id))
    return _dedupe(found)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[FindingCode, str | None]] = set()
    unique = []
    for f in findings:
        if (f.code, f.subject_id) not in seen:
            seen.add((f.code, f.subject_id))
            unique.append(f)
    return unique


# --- Merchant claims versus independent evidence ---------------------------


class EvidenceRelevance(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    IRRELEVANT = "irrelevant"


class ClaimStatus(StrEnum):
    CONFIRMED_TRUE = "confirmed_true"
    CONFIRMED_FALSE = "confirmed_false"
    CONFLICTED = "conflicted"  # fresh independent sources disagree with each other
    STALE_ONLY = "stale_only"  # only out-of-date evidence exists
    UNVERIFIED = "unverified"  # no compatible independent evidence


@dataclass(frozen=True)
class ClaimAssessment:
    assertion: MerchantAssertion
    status: ClaimStatus


def classify_evidence(
    assertion: MerchantAssertion,
    evidence: Evidence,
    offer: FinancingOffer | None,
    as_of: datetime,
    config: PolicyConfig = DEFAULT_CONFIG,
) -> EvidenceRelevance:
    """Evidence can test a claim only if identifiers match and it was observed after the claim, recently."""
    if evidence.offer_id != assertion.offer_id or not evidence.source.independent_of_merchant:
        return EvidenceRelevance.IRRELEVANT
    if evidence.source.kind is SourceKind.LENDER and (offer is None or evidence.source.source_id != offer.lender_id):
        return EvidenceRelevance.IRRELEVANT
    if evidence.observed_at > as_of:
        return EvidenceRelevance.IRRELEVANT
    if evidence.observed_at < assertion.made_at or as_of - evidence.observed_at > config.max_evidence_age:
        return EvidenceRelevance.STALE
    return EvidenceRelevance.FRESH


def assess_claim(
    assertion: MerchantAssertion,
    evidence: tuple[Evidence, ...],
    offer: FinancingOffer | None,
    as_of: datetime,
    config: PolicyConfig = DEFAULT_CONFIG,
) -> ClaimAssessment:
    relevance = [(e, classify_evidence(assertion, e, offer, as_of, config)) for e in evidence]
    fresh = {e.offer_available for e, r in relevance if r is EvidenceRelevance.FRESH}
    if len(fresh) > 1:
        return ClaimAssessment(assertion, ClaimStatus.CONFLICTED)
    if fresh:
        status = ClaimStatus.CONFIRMED_TRUE if fresh == {assertion.claims_available} else ClaimStatus.CONFIRMED_FALSE
        return ClaimAssessment(assertion, status)
    if any(r is EvidenceRelevance.STALE for _, r in relevance):
        return ClaimAssessment(assertion, ClaimStatus.STALE_ONLY)
    return ClaimAssessment(assertion, ClaimStatus.UNVERIFIED)


def _latest_claims(assessments: list[ClaimAssessment]) -> dict[str, ClaimAssessment]:
    latest: dict[str, ClaimAssessment] = {}
    for a in sorted(assessments, key=lambda a: a.assertion.made_at):
        latest[a.assertion.offer_id] = a
    return latest


def effective_availability(offer_id: str, latest: dict[str, ClaimAssessment]) -> bool | None:
    """True or False when known; None when a claim about the offer cannot be resolved."""
    a = latest.get(offer_id)
    if a is None:
        return True  # listed, and nobody disputes it
    if a.status is ClaimStatus.CONFIRMED_TRUE:
        return a.assertion.claims_available
    if a.status is ClaimStatus.CONFIRMED_FALSE:
        return not a.assertion.claims_available
    if a.assertion.claims_available and a.status in (ClaimStatus.UNVERIFIED, ClaimStatus.STALE_ONLY):
        return True  # the merchant is not disputing the listing
    return None


# --- Decision quality (separate from authorisation) ------------------------


def objective_cost_minor(offer: FinancingOffer, objective: Objective) -> int | None:
    """The quantity the customer wants minimised, or None when there is no ordering."""
    if objective is Objective.MINIMISE_TOTAL_REPAYMENT:
        return offer.schedule.total().amount_minor
    if objective is Objective.MINIMISE_MONTHLY_PAYMENT:
        return offer.schedule.max_payment().amount_minor
    return None


def _cost_gap_minor(selected: FinancingOffer, other: FinancingOffer, objective: Objective) -> int | None:
    if selected.schedule.currency != other.schedule.currency:
        return None
    a, b = objective_cost_minor(selected, objective), objective_cost_minor(other, objective)
    return None if a is None or b is None else a - b


def assess_selection(
    selected: FinancingOffer,
    usable: list[FinancingOffer],
    intent: CustomerIntent,
    config: PolicyConfig = DEFAULT_CONFIG,
) -> SelectionAssessment:
    """Compare the selected offer with usable alternatives under the customer's objective."""
    gaps = [(g, o) for o in usable if (g := _cost_gap_minor(selected, o, intent.objective)) is not None]
    if not gaps:
        return SelectionAssessment(selected_offer_id=selected.offer_id)
    gap, best = min(gaps, key=lambda go: (-go[0], go[1].offer_id))  # largest gap; offer_id breaks ties
    if gap <= 0:
        return SelectionAssessment(selected_offer_id=selected.offer_id)
    return SelectionAssessment(
        selected_offer_id=selected.offer_id,
        better_offer_id=best.offer_id,
        objective_gap=Money(amount_minor=gap, currency=selected.schedule.currency),
        inferior=gap >= config.material_gap_minor,
    )


def advertised_total_discrepancy(offer: FinancingOffer) -> Money | None:
    """How much the real total exceeds the advertised headline figure, if it does."""
    advertised = offer.advertised_total_repayable
    if advertised is None or advertised.currency != offer.schedule.currency:
        return None
    real = offer.schedule.total()
    return Money(amount_minor=real.amount_minor - advertised.amount_minor, currency=real.currency) if real > advertised else None


# --- Suspicious text -------------------------------------------------------

_INSTRUCTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|other|earlier)\b",
        r"\bdisregard\b.{0,40}\b(?:instructions?|offers?|rules?|limits?)\b",
        r"\b(?:system|developer)\s+(?:prompt|message|instructions?)\b",
        r"\b(?:ai|llm|agent|assistant)\b.{0,40}\b(?:must|should|need to)\b.{0,30}\b(?:select|choose|pick|buy|pay|accept)\b",
        r"\bdo not\s+(?:compare|consider|check|verify)\b",
    )
]


def find_instruction_like_text(text: str) -> str | None:
    for pattern in _INSTRUCTION_PATTERNS:
        if match := pattern.search(text):
            return match.group(0)
    return None


# --- Putting it together ---------------------------------------------------

_ESCALATING = {C.CONFIRMED_FALSE_CLAIM, C.UNRESOLVED_CONTRADICTION, C.INFERIOR_SELECTION}


def evaluate_case(case: Case, config: PolicyConfig = DEFAULT_CONFIG) -> SecurityDecision:
    txn, as_of = case.transaction, case.transaction.proposed_at
    offers = {o.offer_id: o for o in case.offers}
    selected = offers.get(txn.offer_id)
    findings: list[Finding] = []

    if selected is None:
        findings.append(_hard(F.OFFER_NOT_FOUND, f"offer {txn.offer_id} is not in the case", txn.offer_id))
        return SecurityDecision(decision=Decision.BLOCK, authorised=False, findings=tuple(findings))

    findings += check_transaction(txn, selected, case)

    # Claims versus independent evidence, for this merchant's assertions only.
    assessments = [
        assess_claim(a, case.evidence, offers.get(a.offer_id), as_of, config)
        for a in case.assertions
        if a.merchant_id == case.merchant.merchant_id
    ]
    latest = _latest_claims(assessments)
    candidates = {o.offer_id: o for o in case.offers if o.offer_id != selected.offer_id and not check_offer_authorised(o, case)}

    def material(offer_id: str) -> bool:
        """A claim matters if it concerns the selected offer or hides a materially better authorised one."""
        if offer_id == selected.offer_id:
            return True
        other = candidates.get(offer_id)
        gap = None if other is None else _cost_gap_minor(selected, other, case.intent.objective)
        return gap is not None and gap >= config.material_gap_minor

    for a in assessments:
        subject, is_material = a.assertion.offer_id, material(a.assertion.offer_id)
        if a.status is ClaimStatus.CONFIRMED_FALSE:
            findings.append(Finding(category=C.CONFIRMED_FALSE_CLAIM, code=F.AVAILABILITY_CONTRADICTED, subject_id=subject, material=is_material,
                                    message=f"merchant says {subject} available={a.assertion.claims_available}; fresh independent evidence says otherwise"))
        elif a.status is ClaimStatus.CONFLICTED:
            findings.append(Finding(category=C.UNRESOLVED_CONTRADICTION, code=F.EVIDENCE_CONFLICT, subject_id=subject, material=is_material,
                                    message=f"fresh independent sources disagree about {subject}"))
        elif a.status is ClaimStatus.STALE_ONLY:
            findings.append(Finding(category=C.UNVERIFIED_CLAIM, code=F.EVIDENCE_STALE, subject_id=subject, material=is_material,
                                    message=f"only stale evidence exists for the claim about {subject}"))
        elif a.status is ClaimStatus.UNVERIFIED:
            findings.append(Finding(category=C.UNVERIFIED_CLAIM, code=F.NO_INDEPENDENT_EVIDENCE, subject_id=subject, material=is_material,
                                    message=f"no independent evidence for the claim about {subject}"))

    # The merchant's headline figure against its own schedule.
    for offer in case.offers:
        if (shortfall := advertised_total_discrepancy(offer)) is not None:
            findings.append(Finding(category=C.CONFIRMED_FALSE_CLAIM, code=F.ADVERTISED_TOTAL_MISMATCH, subject_id=offer.offer_id,
                                    material=offer.offer_id == selected.offer_id,
                                    message=f"{offer.offer_id} advertises {offer.advertised_total_repayable} but its schedule totals {offer.schedule.total()} ({shortfall} more)"))

    # Decision quality: only offers that were authorised and not in doubt count as better options.
    usable = [o for o in candidates.values() if effective_availability(o.offer_id, latest) is True]
    selection = assess_selection(selected, usable, case.intent, config)
    if selection.inferior:
        findings.append(Finding(category=C.INFERIOR_SELECTION, code=F.COSTLIER_THAN_USABLE_OFFER, subject_id=selected.offer_id,
                                message=f"{selected.offer_id} costs {selection.objective_gap} more than {selection.better_offer_id}, which was usable"))

    # Instruction-like text is noted, not treated as proof of an attack.
    texts = [(o.offer_id, f"{o.title} {o.description}") for o in case.offers] + [(a.assertion_id, a.statement) for a in case.assertions]
    for subject, text in texts:
        if (hit := find_instruction_like_text(text)) is not None:
            findings.append(Finding(category=C.SUSPICIOUS_BEHAVIOUR, code=F.INSTRUCTION_LIKE_TEXT, subject_id=subject, material=False,
                                    message=f"instruction-like text in merchant content: {hit!r}"))

    authorised = not any(f.category is C.HARD_VIOLATION for f in findings)
    if not authorised:
        decision = Decision.BLOCK
    elif any(f.category in _ESCALATING and f.material for f in findings):
        decision = Decision.REVIEW
    else:
        decision = Decision.ALLOW
    return SecurityDecision(decision=decision, authorised=authorised, findings=tuple(findings), selection=selection)


def resolve_final_decision(deterministic: Decision, advisory: Decision | None = None) -> Decision:
    """Merge the deterministic decision with an advisory (e.g. LLM) verdict.

    An advisory verdict can only escalate ALLOW to REVIEW. It can never lower a
    deterministic decision, and never raises one to BLOCK: BLOCK is reserved for
    hard violations found by deterministic rules.
    """
    if advisory is None or deterministic is Decision.BLOCK:
        return deterministic
    capped = advisory if advisory.severity <= Decision.REVIEW.severity else Decision.REVIEW
    return max(deterministic, capped, key=lambda d: d.severity)
