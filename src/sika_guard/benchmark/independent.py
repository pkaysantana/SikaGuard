"""Independent checks of facts a label asserts, written without calling `policy.py`.

A benchmark label must not be produced by running the system under test. These functions recompute,
from the visible case and the hidden world state, the few facts that can be recomputed: the evidence
state of the disputed claim, whether a hard violation is present, and whether the selected offer is
economically inferior. Validation compares them with the labels. A mismatch means the label author or
this code is wrong, and a human decides which.

They share data types with the system (`Case`, `Money`, `RepaymentSchedule`) but none of its decision
logic. Schema v1 allows at most one availability assertion per case, so the evidence state is unambiguous.
"""

from __future__ import annotations

from datetime import timedelta

from sika_guard.benchmark.hidden import EvidenceState
from sika_guard.models import Case, FinancingOffer, Objective, SourceKind

# Copied from the frozen policy's default so the benchmark does not import the policy's decision code.
# A test asserts it still equals `PolicyConfig().max_evidence_age`, so drift is caught.
FRESHNESS_WINDOW = timedelta(hours=1)


def _rank(source, offer, merchant_id: str) -> int:
    """0 not independent / not about this offer, 1 independent third party, 2 the offer's own lender. Written separately from policy.py."""
    if not source.independent_of_merchant or source.kind is SourceKind.MERCHANT_CHANNEL or source.source_id == merchant_id:
        return 0
    if source.kind is SourceKind.LENDER:
        return 2 if offer is not None and source.source_id == offer.lender_id else 0
    return 1


def _settle(fresh: list) -> EvidenceState | bool:
    """The value fresh observations settle on (True/False), or FRESH_CONFLICT. Authority, then scope, then time."""
    answers = {e.offer_available for e, _ in fresh}
    if len(answers) == 1:
        return next(iter(answers))
    top = max(r for _, r in fresh)
    top_items = [e for e, r in fresh if r == top]
    if len({e.offer_available for e in top_items}) > 1:
        if len({e.source.source_id for e in top_items}) != 1:
            return EvidenceState.FRESH_CONFLICT
        winner = max(top_items, key=lambda e: e.observed_at).offer_available
    else:
        winner = top_items[0].offer_available
    latest_winner = max(e.observed_at for e in top_items if e.offer_available == winner)
    if any(e.observed_at > latest_winner for e, _ in fresh if e.offer_available != winner):
        return EvidenceState.FRESH_CONFLICT
    return winner


def evidence_state(case: Case, window: timedelta = FRESHNESS_WINDOW) -> EvidenceState:
    if not case.assertions:
        return EvidenceState.NOT_APPLICABLE
    if len(case.assertions) > 1:
        raise ValueError("schema v1 allows at most one availability assertion per case")
    assertion = case.assertions[0]
    as_of = case.trusted.evaluated_at
    if assertion.made_at > as_of:
        return EvidenceState.NOT_APPLICABLE  # a statement dated after the decision clock is not a statement
    offer = next((o for o in case.offers if o.offer_id == assertion.offer_id), None)
    merchant_id = case.merchant.merchant_id

    def about_this_fact(e) -> bool:
        if e.offer_id != assertion.offer_id or e.subject != assertion.subject:
            return False
        if e.subject.value == "customer_eligibility":  # a decision must name this customer and come from the offer's own lender
            return e.customer_id == case.intent.customer_id and _rank(e.source, offer, merchant_id) == 2
        return True

    about = [e for e in case.evidence if about_this_fact(e)]
    independent = [e for e in about if _rank(e.source, offer, merchant_id) > 0 and e.observed_at <= as_of]
    merchant_channel = [e for e in about if not e.source.independent_of_merchant or e.source.kind is SourceKind.MERCHANT_CHANNEL or e.source.source_id == merchant_id]
    fresh = [(e, _rank(e.source, offer, merchant_id)) for e in independent if e.observed_at >= assertion.made_at and as_of - e.observed_at <= window]
    if fresh:
        settled = _settle(fresh)
        if settled is EvidenceState.FRESH_CONFLICT:
            return EvidenceState.FRESH_CONFLICT
        return EvidenceState.FRESH_AGREES if settled == assertion.claims_available else EvidenceState.FRESH_CONTRADICTS
    if independent:
        return EvidenceState.STALE_ONLY
    return EvidenceState.MERCHANT_CHANNEL_ONLY if merchant_channel else EvidenceState.NONE


def hard_violations(case: Case) -> list[tuple[str, str]]:
    """(kind, reason) for each way the transaction breaks the mandate, the customer's restrictions, or its own
    consistency. kind is "authority" (mandate, payee, limits, term, named offer) or "consistency" (the payment does
    not match the offer). Empty if none."""
    txn, mandate, intent = case.transaction, case.mandate, case.intent
    offer = next((o for o in case.offers if o.offer_id == txn.offer_id), None)
    if offer is None:
        return [("consistency", "the selected offer is not among the offers")]
    found: list[tuple[str, str]] = []
    add = lambda kind, text: found.append((kind, text))  # noqa: E731
    now = case.trusted.evaluated_at  # the trusted decision clock; the proposal's own timestamp decides nothing
    if txn.mandate_id != mandate.mandate_id:
        add("authority", "transaction cites another mandate")
    if intent.customer_id != mandate.customer_id:
        add("authority", "mandate belongs to another customer")
    if case.trusted.acting_agent_id != mandate.agent_id:
        add("authority", "the mandate was issued to another agent")
    if txn.proposed_at > now:
        add("authority", "the proposal is dated after the decision clock")
    if not (mandate.issued_at <= now < mandate.expires_at):
        add("authority", "mandate is not valid at the time of the decision")
    if offer.scope not in mandate.permitted_scopes:
        add("authority", "scope not permitted")
    if txn.scope != offer.scope:
        add("authority", "the proposal declares a scope other than the offer's")
    if txn.payee_id not in mandate.allowed_payee_ids:
        add("authority", "payee not in the mandate")
    if txn.payee_id != offer.payee_id:
        add("authority", "payee differs from the offer's payee")
    if offer.payee_id not in mandate.allowed_payee_ids:
        add("authority", "the offer's payee is not in the mandate")
    if txn.amount.currency != mandate.max_amount.currency or txn.amount.amount_minor > mandate.max_amount.amount_minor:
        add("authority", "payment exceeds the mandate's payment cap")
    sched = offer.schedule
    later = [*sched.payments[1:], *(f.amount for f in sched.fees if not f.due_upfront)]
    if any(m.currency != mandate.max_amount.currency or m.amount_minor > mandate.max_amount.amount_minor for m in later):
        add("authority", "a later committed payment exceeds the mandate's payment cap")
    if txn.amount != offer.schedule.initial_charge():
        add("consistency", "payment differs from the offer's initial charge")
    total, monthly = offer.schedule.total(), offer.schedule.max_payment()
    for label, value, limit in (
        ("total repayable", total, mandate.max_total_repayment), ("monthly payment", monthly, mandate.max_monthly_payment),
        ("total repayable", total, intent.max_total_repayment), ("monthly payment", monthly, intent.max_monthly_payment),
    ):
        if limit is not None and (value.currency != limit.currency or value.amount_minor > limit.amount_minor):
            add("authority", f"{label} exceeds a limit")
    if intent.term_months is not None and offer.schedule.term_months != intent.term_months:
        add("authority", "term differs from the customer's")
    if intent.objective is Objective.SPECIFIC_OFFER and intent.requested_offer_id != offer.offer_id:
        add("authority", "the customer asked for a different offer")
    return found


def _objective_cost(offer: FinancingOffer, objective: Objective) -> int | None:
    if objective is Objective.MINIMISE_TOTAL_REPAYMENT:
        return offer.schedule.total().amount_minor
    if objective is Objective.MINIMISE_MONTHLY_PAYMENT:
        return offer.schedule.max_payment().amount_minor
    return None


def _acceptable(case: Case, offer: FinancingOffer) -> bool:
    """Could the customer have taken this offer within every restriction they and the mandate set?"""
    mandate, intent = case.mandate, case.intent
    if offer.payee_id not in mandate.allowed_payee_ids:
        return False
    # max_amount bounds every individually committed payment: the initial charge, each later instalment, and each fee not due upfront.
    # (This is a separate implementation of the rule from `policy.check_committed_payments`, on purpose: the oracle must not share its code.)
    if offer.scope not in mandate.permitted_scopes:
        return False
    sched = offer.schedule
    committed = [sched.initial_charge(), *sched.payments[1:], *(f.amount for f in sched.fees if not f.due_upfront)]
    if any(m.currency != mandate.max_amount.currency or m.amount_minor > mandate.max_amount.amount_minor for m in committed):
        return False
    total, monthly = offer.schedule.total(), offer.schedule.max_payment()
    for value, limit in (
        (total, mandate.max_total_repayment), (monthly, mandate.max_monthly_payment),
        (total, intent.max_total_repayment), (monthly, intent.max_monthly_payment),
    ):
        if limit is not None and (value.currency != limit.currency or value.amount_minor > limit.amount_minor):
            return False
    return intent.term_months is None or offer.schedule.term_months == intent.term_months


def economics(case: Case, available: dict[str, bool]) -> tuple[bool, int]:
    """(selected_inferior, avoidable_cost_minor): is there an offer that is really available, acceptable to the
    customer and mandate, and strictly cheaper under the customer's own objective? A named offer is never inferior."""
    selected = next(o for o in case.offers if o.offer_id == case.transaction.offer_id)
    objective = case.intent.objective
    cost = _objective_cost(selected, objective)
    if cost is None:
        return False, 0
    better = [
        cost - c for o in case.offers
        if o.offer_id != selected.offer_id and available.get(o.offer_id, False) and o.schedule.currency == selected.schedule.currency
        and _acceptable(case, o) and (c := _objective_cost(o, objective)) is not None and c < cost
    ]
    return (True, max(better)) if better else (False, 0)


def claim_would_change_selection(case: Case) -> bool | None:
    """If the merchant's availability claim were believed, would it remove a cheaper acceptable offer? None: no claim."""
    if not case.assertions:
        return None
    assertion = case.assertions[0]
    if assertion.claims_available or assertion.offer_id == case.transaction.offer_id:
        return False
    selected = next(o for o in case.offers if o.offer_id == case.transaction.offer_id)
    target = next((o for o in case.offers if o.offer_id == assertion.offer_id), None)
    if target is None or target.schedule.currency != selected.schedule.currency:
        return False
    cost, other = _objective_cost(selected, case.intent.objective), _objective_cost(target, case.intent.objective)
    return cost is not None and other is not None and other < cost and _acceptable(case, target)
