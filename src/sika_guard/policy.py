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
from enum import IntEnum, StrEnum

from sika_guard.models import (
    AvailabilitySubject,
    Case,
    CustomerIntent,
    Decision,
    DecisionBasis,
    Evidence,
    EvidenceSource,
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
    """Provisional policy assumptions. None of these values is calibrated or tuned to any dataset; they
    are stated so they can be challenged and varied (see `evaluate --sensitivity`)."""

    max_evidence_age: timedelta = timedelta(hours=1)
    material_gap_minor: int = 500  # smaller cost differences are not worth escalating
    # Evidence requirement (rule 5): an uncorroborated "unavailable" claim that rules out a cheaper authorised
    # offer must be corroborated before payment, if the customer asked to minimise cost and the gap is this large.
    require_corroboration: bool = True
    evidence_requirement_gap_minor: int = 500
    stale_evidence_counts_as_uncorroborated: bool = True  # evidence older than the claim or the window is not corroboration
    # Predeclared materiality of a financial discrepancy (AUDIT R4). Neither value is tuned to any data.
    #  * understatement_material_minor: a merchant representation that shows the customer LESS than the lender's terms say they will pay is
    #    material from this size. 1 minor unit means any exact arithmetic shortfall, which is what the rule engine already did for the
    #    structured advertised total; the figures are checkable facts, not judgements, so there is no tolerance to tune.
    #  * hidden_obligation_material_minor: an obligation that a lender document states and the structured terms omit is material from this
    #    size (the pre-existing 5.00 gap), or at any size if including it would breach the mandate's or the customer's total limit.
    # An OVERstatement (the merchant shows the customer MORE than they will pay) is never material: it does not harm the customer.
    understatement_material_minor: int = 1
    hidden_obligation_material_minor: int = 500
    # A lender document older than this may describe superseded terms, so it is not compared with the current structured terms. Provisional;
    # not tuned to any data.
    max_document_age: timedelta = timedelta(days=365)


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


def check_committed_payments(offer: FinancingOffer, mandate: Mandate) -> list[Finding]:
    """`max_amount` bounds every individually committed payment, not only the first (AUDIT F7, resolved by the mandate's own wording:
    "ceiling for any single payment"). The first payment plus upfront fees is the transaction's initial charge and is checked against the
    same ceiling by `check_payment_cap`; this covers each later instalment and each fee that is not due upfront. The timing of a fee that
    is not due upfront is not modelled, so such a fee is treated as a payment of its own. Total commitment (`max_total_repayment`) is
    a separate limit and unchanged; cumulative spending across agreements is not modelled."""
    sched = offer.schedule
    later = [*sched.payments[1:], *(f.amount for f in sched.fees if not f.due_upfront)]
    if not later:
        return []
    worst = max(later, key=lambda m: m.amount_minor)
    return _limit_findings(worst, mandate.max_amount, F.COMMITTED_PAYMENT_EXCEEDS_MANDATE, "committed payment", offer.offer_id)


def check_offer_limits(offer: FinancingOffer, intent: CustomerIntent, mandate: Mandate) -> list[Finding]:
    """Total-repayment, monthly-payment and term limits from the mandate and the customer, and the per-payment ceiling on every
    committed payment."""
    total = offer.schedule.total()
    monthly = offer.schedule.max_payment()
    found = [
        *check_committed_payments(offer, mandate),
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
    """Would accepting this offer, instead of the proposed one, have been authorised? Judged at the trusted clock and for the offer's own scope."""
    return [
        *check_mandate_window(case.mandate, case.trusted.evaluated_at),
        *check_scope(case.mandate, offer.scope),
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
    # The decision clock, the scope and the acting agent come from the trusted context and the lender's product, never from the proposal.
    if case.trusted.acting_agent_id != mandate.agent_id:
        found.append(_hard(F.AGENT_MISMATCH, "the mandate was issued to a different agent than the one acting", mandate.mandate_id))
    if txn.proposed_at > case.trusted.evaluated_at:
        found.append(_hard(F.PROPOSAL_FROM_THE_FUTURE, "the proposal is dated after the decision clock", txn.transaction_id))
    found += check_mandate_window(mandate, case.trusted.evaluated_at)
    found += check_scope(mandate, offer.scope)
    if txn.scope != offer.scope:
        found.append(_hard(F.SCOPE_OFFER_MISMATCH, f"the proposal declares scope {txn.scope.value} but the offer is a {offer.scope.value}", txn.transaction_id))
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
    CONFLICTED = "conflicted"  # fresh independent sources disagree and neither outranks the other
    STALE_ONLY = "stale_only"  # only out-of-date evidence exists
    UNVERIFIED = "unverified"  # no compatible independent evidence
    INVALID_CHRONOLOGY = "invalid_chronology"  # the statement is dated after the decision clock: it cannot be assessed and is not trusted


class SourceAuthority(IntEnum):
    """How much a source's word counts about THIS offer (AUDIT R9). An authority never makes a source true; it only decides which of two
    disagreeing, equally recent observations to prefer, and a newer observation from a lower authority prevents the preference."""

    NONE = 0  # the merchant's own channel, or anything not independent of the merchant: never evidence
    INDEPENDENT = 1  # an independent third party, such as an aggregator
    OFFER_LENDER = 2  # the lender that issued this very offer


def source_authority(source: EvidenceSource, offer: FinancingOffer | None, merchant_id: str) -> SourceAuthority:
    if not source.is_independent_of(merchant_id):
        return SourceAuthority.NONE
    if source.kind is SourceKind.LENDER:
        return SourceAuthority.OFFER_LENDER if offer is not None and source.source_id == offer.lender_id else SourceAuthority.NONE
    return SourceAuthority.INDEPENDENT


@dataclass(frozen=True)
class ClaimAssessment:
    assertion: MerchantAssertion
    status: ClaimStatus
    dissent: tuple[str, ...] = ()  # fresh observations that disagreed with the resolved value and were outranked, for the record


def classify_evidence(
    assertion: MerchantAssertion,
    evidence: Evidence,
    offer: FinancingOffer | None,
    as_of: datetime,
    config: PolicyConfig = DEFAULT_CONFIG,
    customer_id: str | None = None,
) -> EvidenceRelevance:
    """Evidence can test a claim only if it is about the same fact (same offer, same subject), comes from an independent source, and was
    observed after the claim, recently. A decision about a customer must name that customer and come from the offer's own lender."""
    if evidence.offer_id != assertion.offer_id or evidence.subject is not assertion.subject:
        return EvidenceRelevance.IRRELEVANT
    if not evidence.source.is_independent_of(assertion.merchant_id):
        return EvidenceRelevance.IRRELEVANT
    if evidence.source.kind is SourceKind.LENDER and (offer is None or evidence.source.source_id != offer.lender_id):
        return EvidenceRelevance.IRRELEVANT
    if evidence.subject is AvailabilitySubject.CUSTOMER_ELIGIBILITY and (
        customer_id is None or evidence.customer_id != customer_id or source_authority(evidence.source, offer, assertion.merchant_id) is not SourceAuthority.OFFER_LENDER
    ):
        return EvidenceRelevance.IRRELEVANT
    if evidence.observed_at > as_of:
        return EvidenceRelevance.IRRELEVANT
    if evidence.observed_at < assertion.made_at or as_of - evidence.observed_at > config.max_evidence_age:
        return EvidenceRelevance.STALE
    return EvidenceRelevance.FRESH


def _resolve_fresh(fresh: list[tuple[Evidence, SourceAuthority]]) -> tuple[bool | None, tuple[str, ...]]:
    """The value fresh observations settle on, or None when they cannot be reconciled; plus the ids of any outranked dissent.

    Authority, scope and time decide, never a flag. The highest-authority observations speak first: if they disagree, the latest of ONE
    source supersedes that source's earlier observation, but two sources of the same rank that disagree stay in conflict. A lower-authority
    observation that is NEWER than the winner's latest means the state may have changed since, so it also leaves the conflict open."""
    top = max(a for _, a in fresh)
    top_items = [e for e, a in fresh if a == top]
    top_values = {e.offer_available for e in top_items}
    if len(top_values) > 1:
        if len({e.source.source_id for e in top_items}) != 1:
            return None, ()
        winner = max(top_items, key=lambda e: e.observed_at).offer_available
    else:
        winner = next(iter(top_values))
    winner_time = max(e.observed_at for e in top_items if e.offer_available == winner)
    opposing = [e for e, _ in fresh if e.offer_available != winner]
    if any(e.observed_at > winner_time for e in opposing):
        return None, ()
    return winner, tuple(sorted(e.evidence_id for e in opposing))


def assess_claim(
    assertion: MerchantAssertion,
    evidence: tuple[Evidence, ...],
    offer: FinancingOffer | None,
    as_of: datetime,
    config: PolicyConfig = DEFAULT_CONFIG,
    customer_id: str | None = None,
) -> ClaimAssessment:
    if assertion.made_at > as_of:
        return ClaimAssessment(assertion, ClaimStatus.INVALID_CHRONOLOGY)  # a statement from the future is not a statement
    relevance = [(e, classify_evidence(assertion, e, offer, as_of, config, customer_id)) for e in evidence]
    fresh = [(e, source_authority(e.source, offer, assertion.merchant_id)) for e, r in relevance if r is EvidenceRelevance.FRESH]
    if fresh:
        value, dissent = _resolve_fresh(fresh) if len({e.offer_available for e, _ in fresh}) > 1 else (fresh[0][0].offer_available, ())
        if value is None:
            return ClaimAssessment(assertion, ClaimStatus.CONFLICTED)
        return ClaimAssessment(assertion, ClaimStatus.CONFIRMED_TRUE if value == assertion.claims_available else ClaimStatus.CONFIRMED_FALSE, dissent)
    if any(r is EvidenceRelevance.STALE for _, r in relevance):
        return ClaimAssessment(assertion, ClaimStatus.STALE_ONLY)
    return ClaimAssessment(assertion, ClaimStatus.UNVERIFIED)


ClaimKey = tuple[str, AvailabilitySubject]


def _latest_claims(assessments: list[ClaimAssessment]) -> dict[ClaimKey, ClaimAssessment]:
    """The latest statement about each (offer, subject): a later claim about eligibility does not replace an earlier one about the product."""
    latest: dict[ClaimKey, ClaimAssessment] = {}
    for a in sorted(assessments, key=lambda a: a.assertion.made_at):
        latest[(a.assertion.offer_id, a.assertion.subject)] = a
    return latest


def _one_subject_availability(a: ClaimAssessment | None) -> bool | None:
    if a is None:
        return True  # listed, and nobody disputes it
    if a.status is ClaimStatus.CONFIRMED_TRUE:
        return a.assertion.claims_available
    if a.status is ClaimStatus.CONFIRMED_FALSE:
        return not a.assertion.claims_available
    if a.assertion.claims_available and a.status in (ClaimStatus.UNVERIFIED, ClaimStatus.STALE_ONLY):
        return True  # the merchant is not disputing the listing
    return None


def effective_availability(offer_id: str, latest: dict[ClaimKey, ClaimAssessment]) -> bool | None:
    """True or False when known; None when a claim about the offer cannot be resolved. Product availability and customer eligibility
    must both hold for the offer to be usable: a verified failure of either makes it unusable, an unresolved one makes it uncertain."""
    states = [_one_subject_availability(latest.get((offer_id, subject))) for subject in AvailabilitySubject]
    if False in states:
        return False
    return None if None in states else True


def unopen_offers(case: Case, as_of: datetime, config: PolicyConfig = DEFAULT_CONFIG) -> dict[str, FindingCode]:
    """Offers that fresh independent evidence says CANNOT be taken, with why, whatever any merchant claimed (or failed to claim).

    * the offer's own lender has, in a decision about THIS customer, declined them (the latest decision from a lender counts, so a later
      approval supersedes an earlier decline); a merchant's say-so is never a decision;
    * fresh independent observations, settled by authority, scope and time (`_resolve_fresh`), say the product itself is not on sale.

    Stale or conflicting evidence proves nothing and leaves an offer open (invariant 4)."""
    offers = {o.offer_id: o for o in case.offers}
    merchant_id = case.merchant.merchant_id
    out: dict[str, FindingCode] = {}
    decisions: dict[tuple[str, str], Evidence] = {}
    fresh: dict[str, list[tuple[Evidence, SourceAuthority]]] = {}
    for e in sorted(case.evidence, key=lambda e: e.observed_at):
        offer = offers.get(e.offer_id)
        authority = source_authority(e.source, offer, merchant_id)
        if offer is None or authority is SourceAuthority.NONE or e.observed_at > as_of:
            continue
        if e.subject is AvailabilitySubject.CUSTOMER_ELIGIBILITY:
            if e.customer_id == case.intent.customer_id and authority is SourceAuthority.OFFER_LENDER:
                decisions[(e.offer_id, e.source.source_id)] = e
        elif as_of - e.observed_at <= config.max_evidence_age:
            fresh.setdefault(e.offer_id, []).append((e, authority))
    for (offer_id, _), e in decisions.items():
        if not e.offer_available:
            out[offer_id] = F.ELIGIBILITY_DECLINED
    for offer_id, items in fresh.items():
        value, _ = _resolve_fresh(items) if len({e.offer_available for e, _ in items}) > 1 else (items[0][0].offer_available, ())
        if value is False and offer_id not in out:
            out[offer_id] = F.OFFER_REPORTED_UNAVAILABLE
    return out


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


class Direction(StrEnum):
    """Which way a figure is wrong, relative to the authoritative one. Only a representation that shows the customer LESS than they will
    pay can mislead them into paying more (AUDIT R4); showing more is a discrepancy but not a harm."""

    UNDERSTATES_COST = "understates_cost"
    OVERSTATES_COST = "overstates_cost"
    EQUAL = "equal"


@dataclass(frozen=True)
class Discrepancy:
    direction: Direction
    magnitude: Money  # absolute difference, always non-negative


def compare_figures(shown: Money, authoritative: Money) -> Discrepancy | None:
    """How a figure someone showed the customer differs from the authoritative one. None when the currencies differ: nothing is converted."""
    if shown.currency != authoritative.currency:
        return None
    diff = authoritative.amount_minor - shown.amount_minor
    direction = Direction.UNDERSTATES_COST if diff > 0 else Direction.OVERSTATES_COST if diff < 0 else Direction.EQUAL
    return Discrepancy(direction, Money(amount_minor=abs(diff), currency=authoritative.currency))


def understatement_is_material(d: Discrepancy, config: PolicyConfig = DEFAULT_CONFIG) -> bool:
    """A merchant representation that shows the customer less than the lender's terms say is material from the predeclared size."""
    return d.direction is Direction.UNDERSTATES_COST and d.magnitude.amount_minor >= config.understatement_material_minor


def hidden_obligation_is_material(extra: Money, total: Money, intent: CustomerIntent, mandate: Mandate, config: PolicyConfig = DEFAULT_CONFIG) -> bool:
    """An obligation a lender document states and the structured terms omit: material from the predeclared size, or at any size if
    adding it would carry the total repayable over the mandate's or the customer's total limit (which would change the authorisation)."""
    if extra.currency != total.currency or extra.amount_minor <= 0:
        return False
    if extra.amount_minor >= config.hidden_obligation_material_minor:
        return True
    limits = [limit for limit in (intent.max_total_repayment, mandate.max_total_repayment) if limit is not None and limit.currency == total.currency]
    return any(total.amount_minor <= limit.amount_minor < total.amount_minor + extra.amount_minor for limit in limits)


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

_ESCALATING = {
    C.CONFIRMED_FALSE_CLAIM, C.UNRESOLVED_CONTRADICTION, C.INFERIOR_SELECTION, C.EVIDENCE_REQUIREMENT_UNMET,
    C.INCOMPARABLE_REPRESENTATION, C.SELECTED_OFFER_NOT_OPEN,
}


def evaluate_case(case: Case, config: PolicyConfig = DEFAULT_CONFIG) -> SecurityDecision:
    txn, as_of = case.transaction, case.trusted.evaluated_at  # the decision clock is the trusted one, never the proposal's
    offers = {o.offer_id: o for o in case.offers}
    selected = offers.get(txn.offer_id)
    findings: list[Finding] = []

    if selected is None:
        findings.append(_hard(F.OFFER_NOT_FOUND, f"offer {txn.offer_id} is not in the case", txn.offer_id))
        return SecurityDecision(decision=Decision.BLOCK, authorised=False, findings=tuple(findings))

    findings += check_transaction(txn, selected, case)

    # Chronology. A statement or an observation dated after the decision clock is impossible: it is ignored (it can neither confirm
    # nor refute anything), and it is recorded. It is a note, not proof of an attack (CLAUDE.md invariant 10).
    mine = [a for a in case.assertions if a.merchant_id == case.merchant.merchant_id]
    for a in mine:
        if a.made_at > as_of:
            findings.append(Finding(category=C.SUSPICIOUS_BEHAVIOUR, code=F.IMPOSSIBLE_TIMESTAMP, subject_id=a.assertion_id, material=False,
                                    message=f"merchant statement {a.assertion_id} is dated after the decision clock; it was ignored"))
    for e in case.evidence:
        if e.observed_at > as_of:
            findings.append(Finding(category=C.SUSPICIOUS_BEHAVIOUR, code=F.IMPOSSIBLE_TIMESTAMP, subject_id=e.evidence_id, material=False,
                                    message=f"evidence {e.evidence_id} is dated after the decision clock; it was ignored"))

    for d in case.documents:
        if d.published_at > as_of:
            findings.append(Finding(category=C.SUSPICIOUS_BEHAVIOUR, code=F.IMPOSSIBLE_TIMESTAMP, subject_id=d.document_id, material=False,
                                    message=f"document {d.document_id} is dated after the decision clock; it was not shown to any evaluator and is ignored"))

    # Claims versus independent evidence, for this merchant's assertions only.
    assessments = [
        x for x in (assess_claim(a, case.evidence, offers.get(a.offer_id), as_of, config, case.intent.customer_id) for a in mine)
        if x.status is not ClaimStatus.INVALID_CHRONOLOGY
    ]
    latest = _latest_claims(assessments)
    unopen = unopen_offers(case, as_of, config)  # fresh independent evidence says these cannot be taken: not usable alternatives, and not buyable
    candidates = {
        o.offer_id: o for o in case.offers if o.offer_id != selected.offer_id and not check_offer_authorised(o, case) and o.offer_id not in unopen
    }

    def material(offer_id: str) -> bool:
        """A claim matters if it concerns the selected offer or hides a materially better authorised one."""
        if offer_id == selected.offer_id:
            return True
        other = candidates.get(offer_id)
        gap = None if other is None else _cost_gap_minor(selected, other, case.intent.objective)
        return gap is not None and gap >= config.material_gap_minor

    for a in assessments:
        subject, is_material = a.assertion.offer_id, material(a.assertion.offer_id)
        about = "customer eligibility for" if a.assertion.subject is AvailabilitySubject.CUSTOMER_ELIGIBILITY else "availability of"
        if a.status is ClaimStatus.CONFIRMED_FALSE:
            overridden = f" (outranked: {', '.join(a.dissent)})" if a.dissent else ""
            findings.append(Finding(category=C.CONFIRMED_FALSE_CLAIM, code=F.AVAILABILITY_CONTRADICTED, subject_id=subject, material=is_material,
                                    message=f"merchant says {about} {subject} is {a.assertion.claims_available}; fresh independent evidence says otherwise{overridden}"))
        elif a.status is ClaimStatus.CONFLICTED:
            findings.append(Finding(category=C.UNRESOLVED_CONTRADICTION, code=F.EVIDENCE_CONFLICT, subject_id=subject, material=is_material,
                                    message=f"fresh independent sources disagree about {about} {subject}, and neither outranks the other"))
        elif a.status is ClaimStatus.STALE_ONLY:
            findings.append(Finding(category=C.UNVERIFIED_CLAIM, code=F.EVIDENCE_STALE, subject_id=subject, material=is_material,
                                    message=f"only stale evidence exists for the claim about {about} {subject}"))
        elif a.status is ClaimStatus.UNVERIFIED:
            eligibility = a.assertion.subject is AvailabilitySubject.CUSTOMER_ELIGIBILITY
            findings.append(Finding(category=C.UNVERIFIED_CLAIM, code=F.ELIGIBILITY_UNRESOLVED if eligibility else F.NO_INDEPENDENT_EVIDENCE,
                                    subject_id=subject, material=is_material,
                                    message=(f"the merchant says the customer is not eligible for {subject}; only the offer's own lender can decide that, and no decision about this customer is available"
                                             if eligibility else f"no independent evidence for the claim about {subject}")))

    if (why := unopen.get(selected.offer_id)) is not None:
        reason = ("the offer's own lender has declined this customer" if why is F.ELIGIBILITY_DECLINED
                  else "fresh independent evidence says the product is not on sale")
        findings.append(Finding(category=C.SELECTED_OFFER_NOT_OPEN, code=why, subject_id=selected.offer_id,
                                message=f"{reason}; the selected offer {selected.offer_id} cannot be taken. This is not evidence that the merchant is dishonest."))

    # Evidence requirement. Missing corroboration is never proof of dishonesty, so this is not a false-claim
    # finding; it says only that a claim the purchase leans on has not been independently confirmed. It needs
    # all of: the customer asked to minimise cost, the claim says an offer is unavailable, that offer is
    # authorised and costs materially less, and no fresh independent evidence confirms the claim.
    uncorroborated = (ClaimStatus.UNVERIFIED, ClaimStatus.STALE_ONLY) if config.stale_evidence_counts_as_uncorroborated else (ClaimStatus.UNVERIFIED,)
    if config.require_corroboration:
        for a in assessments:
            other = candidates.get(a.assertion.offer_id)
            if a.status not in uncorroborated or a.assertion.claims_available or other is None or latest.get((a.assertion.offer_id, a.assertion.subject)) is not a:
                continue
            gap = _cost_gap_minor(selected, other, case.intent.objective)
            if gap is not None and gap >= config.evidence_requirement_gap_minor:
                eligibility = a.assertion.subject is AvailabilitySubject.CUSTOMER_ELIGIBILITY
                findings.append(
                    Finding(
                        category=C.EVIDENCE_REQUIREMENT_UNMET, code=F.DECISION_CRITICAL_CLAIM_UNCORROBORATED, subject_id=other.offer_id,
                        message=(
                            f"the merchant's claim that {other.offer_id} is {'not open to this customer' if eligibility else 'unavailable'} is not independently corroborated ({a.status.value}); "
                            f"the customer asked to {case.intent.objective.value.replace('_', ' ')} and {other.offer_id} would cost "
                            f"{Money(amount_minor=gap, currency=selected.schedule.currency)} less. Corroboration is required before payment. "
                            "This is not evidence that the merchant is dishonest."
                        ),
                    )
                )

    # The merchant's headline figure against its own schedule: direction first, then predeclared materiality, and no currency conversion.
    for offer in case.offers:
        advertised = offer.advertised_total_repayable
        if advertised is None:
            continue
        is_selected = offer.offer_id == selected.offer_id
        d = compare_figures(advertised, offer.schedule.total())
        if d is None:
            findings.append(Finding(
                category=C.INCOMPARABLE_REPRESENTATION if is_selected else C.UNVERIFIED_CLAIM, code=F.ADVERTISED_CURRENCY_INCOMPATIBLE,
                subject_id=offer.offer_id, material=is_selected,
                message=f"{offer.offer_id} advertises {advertised} but its lender terms are in {offer.schedule.currency}; the figures cannot be compared and nothing was converted. "
                        "This is not evidence that the merchant is dishonest."))
        elif d.direction is Direction.UNDERSTATES_COST:
            findings.append(Finding(category=C.CONFIRMED_FALSE_CLAIM, code=F.ADVERTISED_TOTAL_MISMATCH, subject_id=offer.offer_id,
                                    material=is_selected and understatement_is_material(d, config),
                                    message=f"{offer.offer_id} advertises {advertised} but its schedule totals {offer.schedule.total()} ({d.magnitude} more)"))
        elif d.direction is Direction.OVERSTATES_COST:
            findings.append(Finding(category=C.UNVERIFIED_CLAIM, code=F.ADVERTISED_TOTAL_OVERSTATED, subject_id=offer.offer_id, material=False,
                                    message=f"{offer.offer_id} advertises {advertised} but its schedule totals {offer.schedule.total()}; the customer would pay {d.magnitude} less than advertised, which is not a harm"))

    # Decision quality: only offers that were authorised and not in doubt count as better options.
    usable = [o for o in candidates.values() if effective_availability(o.offer_id, latest) is True]
    selection = assess_selection(selected, usable, case.intent, config)
    if unopen:
        selection = selection.model_copy(update={"excluded_as_not_open": tuple(sorted(set(unopen) - {selected.offer_id}))})
    if selection.inferior:
        findings.append(Finding(category=C.INFERIOR_SELECTION, code=F.COSTLIER_THAN_USABLE_OFFER, subject_id=selected.offer_id,
                                message=f"{selected.offer_id} costs {selection.objective_gap} more than {selection.better_offer_id}, which was usable"))

    # Instruction-like text is noted, not treated as proof of an attack.
    texts = [(o.offer_id, f"{o.title} {o.description}") for o in case.offers] + [(a.assertion_id, a.statement) for a in case.assertions]
    texts += [(d.document_id, d.text) for d in case.documents]
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


_BASIS_BY_CATEGORY = {
    C.CONFIRMED_FALSE_CLAIM: DecisionBasis.REVIEW_VERIFIED_CONTRADICTION,
    C.UNRESOLVED_CONTRADICTION: DecisionBasis.REVIEW_UNRESOLVED_CONFLICT,
    C.INFERIOR_SELECTION: DecisionBasis.REVIEW_INFERIOR_SELECTION,
    C.EVIDENCE_REQUIREMENT_UNMET: DecisionBasis.REVIEW_EVIDENCE_REQUIREMENT,
    C.INCOMPARABLE_REPRESENTATION: DecisionBasis.REVIEW_INCOMPARABLE_REPRESENTATION,
    C.SELECTED_OFFER_NOT_OPEN: DecisionBasis.REVIEW_SELECTED_OFFER_NOT_OPEN,
}


def decision_bases(result: SecurityDecision) -> tuple[DecisionBasis, ...]:
    """Why the rule engine decided as it did: every distinct reason, most specific first.

    A BLOCK has one basis. A REVIEW lists each kind of material finding behind it. An ALLOW is
    `ALLOW_WITH_UNCERTAINTY` when anything unverified or suspicious was recorded but not enforced."""
    if result.decision is Decision.BLOCK:
        return (DecisionBasis.BLOCK_HARD_VIOLATION,)
    if result.decision is Decision.REVIEW:
        found = {_BASIS_BY_CATEGORY[f.category] for f in result.findings if f.category in _ESCALATING and f.material}
        return tuple(b for b in _BASIS_BY_CATEGORY.values() if b in found)
    recorded = any(f.category in (C.UNVERIFIED_CLAIM, C.SUSPICIOUS_BEHAVIOUR) or (f.category in _ESCALATING and not f.material) for f in result.findings)
    return (DecisionBasis.ALLOW_WITH_UNCERTAINTY if recorded else DecisionBasis.ALLOW,)


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
