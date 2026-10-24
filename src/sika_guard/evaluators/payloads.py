"""What each analyst is shown.

Every payload is a projection of the same underlying facts, split into two trust domains:

  trusted                    authenticated customer intent, the delegated mandate, the proposed
                             transaction, lender-originated financing terms, and evidence from sources
                             independent of the merchant
  untrusted_merchant_content text and figures the merchant supplied, including any evidence that
                             arrived over a merchant-controlled channel

The generalist sees everything. Each specialist sees only what its task needs, and a specialist's
fields are always a subset of the generalist's, so Condition C never sees anything Condition B
cannot. Trusted code derives `computed` figures from the lender schedule, so no model has to
do arithmetic.

Assumption: `FinancingOffer.schedule` stands for terms taken from the lender's agreement, while the
offer's title, description and advertised total are the merchant's representation of it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from sika_guard.evaluators.contract import EvaluatorRole, EvaluatorView
from sika_guard.models import Evidence, FinancingOffer, MerchantAssertion

Payload = dict[str, Any]


def _pick(source: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: source[k] for k in keys}


# --- Facts, as plain JSON-able dicts ----------------------------------------


def _task(view: EvaluatorView) -> Payload:
    return {"case_id": view.case_id, "as_of": view.as_of.isoformat(), "selected_offer_id": view.transaction.offer_id}


def _intent(view: EvaluatorView) -> Payload:
    i = view.intent
    return {
        "ref": f"intent:{i.customer_id}",
        "customer_id": i.customer_id,
        "instruction": i.instruction,
        "objective": i.objective.value,
        "term_months": i.term_months,
        "max_total_repayment": i.max_total_repayment.plain() if i.max_total_repayment else None,
        "max_monthly_payment": i.max_monthly_payment.plain() if i.max_monthly_payment else None,
        "requested_offer_id": i.requested_offer_id,
    }


def _mandate(view: EvaluatorView) -> Payload:
    m = view.mandate
    return {
        "ref": f"mandate:{m.mandate_id}",
        "mandate_id": m.mandate_id,
        "customer_id": m.customer_id,
        "agent_id": m.agent_id,
        "allowed_payee_ids": sorted(m.allowed_payee_ids),
        "permitted_scopes": sorted(s.value for s in m.permitted_scopes),
        "max_amount": m.max_amount.plain(),
        "max_total_repayment": m.max_total_repayment.plain() if m.max_total_repayment else None,
        "max_monthly_payment": m.max_monthly_payment.plain() if m.max_monthly_payment else None,
        "issued_at": m.issued_at.isoformat(),
        "expires_at": m.expires_at.isoformat(),
    }


def _transaction(view: EvaluatorView) -> Payload:
    t = view.transaction
    return {
        "ref": f"txn:{t.transaction_id}",
        "transaction_id": t.transaction_id,
        "mandate_id": t.mandate_id,
        "offer_id": t.offer_id,
        "payee_id": t.payee_id,
        "amount": t.amount.plain(),
        "scope": t.scope.value,
        "proposed_at": t.proposed_at.isoformat(),
    }


def _computed(o: FinancingOffer) -> Payload:
    s = o.schedule
    return {
        "total_payments": s.total_payments().plain(),
        "total_fees": s.total_fees().plain(),
        "total_repayable": s.total().plain(),
        "max_monthly_payment": s.max_payment().plain(),
        "initial_charge": s.initial_charge().plain(),
    }


def _terms(o: FinancingOffer) -> Payload:
    s = o.schedule
    return {
        "ref": f"terms:{o.offer_id}",
        "offer_id": o.offer_id,
        "merchant_id": o.merchant_id,
        "lender_id": o.lender_id,
        "payee_id": o.payee_id,
        "term_months": s.term_months,
        "payments": [p.plain() for p in s.payments],
        "fees": [{"kind": f.kind.value, "amount": f.amount.plain(), "due_upfront": f.due_upfront} for f in s.fees],
        "computed": _computed(o),
    }


def _listing(o: FinancingOffer) -> Payload:
    adv = o.advertised_total_repayable
    return {
        "ref": f"listing:{o.offer_id}",
        "offer_id": o.offer_id,
        "title": o.title,
        "description": o.description,
        "advertised_total_repayable": adv.plain() if adv else None,
    }


def _assertion(a: MerchantAssertion) -> Payload:
    return {
        "ref": f"assertion:{a.assertion_id}",
        "offer_id": a.offer_id,
        "claims_available": a.claims_available,
        "statement": a.statement,
        "made_at": a.made_at.isoformat(),
    }


def _evidence(e: Evidence) -> Payload:
    return {
        "ref": f"evidence:{e.evidence_id}",
        "offer_id": e.offer_id,
        "offer_available": e.offer_available,
        "observed_at": e.observed_at.isoformat(),
        "source": {
            "source_id": e.source.source_id,
            "name": e.source.name,
            "kind": e.source.kind.value,
            "independent_of_merchant": e.source.independent_of_merchant,
        },
    }


def _independent(view: EvaluatorView) -> list[Payload]:
    return [_evidence(e) for e in view.evidence if e.source.independent_of_merchant]


def _merchant_channel(view: EvaluatorView) -> list[Payload]:
    return [_evidence(e) for e in view.evidence if not e.source.independent_of_merchant]


def _merchant(view: EvaluatorView) -> Payload:
    return {"ref": f"merchant:{view.merchant.merchant_id}", "merchant_id": view.merchant.merchant_id, "name": view.merchant.name}


# --- Payload per role -------------------------------------------------------


def generalist_payload(view: EvaluatorView) -> Payload:
    return {
        "task_context": _task(view),
        "trusted": {
            "customer_intent": _intent(view),
            "mandate": _mandate(view),
            "proposed_transaction": _transaction(view),
            "lender_terms": [_terms(o) for o in view.offers],
            "independent_evidence": _independent(view),
        },
        "untrusted_merchant_content": {
            "merchant": _merchant(view),
            "offer_listings": [_listing(o) for o in view.offers],
            "assertions": [_assertion(a) for a in view.assertions],
            "merchant_channel_evidence": _merchant_channel(view),
        },
    }


def merchant_claim_payload(view: EvaluatorView) -> Payload:
    intent = _intent(view)
    txn = _transaction(view)
    return {
        "task_context": _task(view),
        "trusted": {
            "customer_intent": _pick(intent, "ref", "objective", "instruction"),
            "proposed_transaction": _pick(txn, "ref", "offer_id"),
            "lender_terms": [
                {**_pick(_terms(o), "ref", "offer_id", "lender_id"), "computed": _pick(_computed(o), "total_repayable")}
                for o in view.offers
            ],
            "independent_evidence": _independent(view),
        },
        "untrusted_merchant_content": {
            "merchant": _merchant(view),
            "offer_listings": [_listing(o) for o in view.offers],
            "assertions": [_assertion(a) for a in view.assertions],
            "merchant_channel_evidence": _merchant_channel(view),
        },
    }


def credit_terms_payload(view: EvaluatorView) -> Payload:
    intent = _intent(view)
    txn = _transaction(view)
    return {
        "task_context": _task(view),
        "trusted": {
            "customer_intent": _pick(intent, "ref", "objective", "term_months", "max_total_repayment", "max_monthly_payment"),
            "proposed_transaction": _pick(txn, "ref", "offer_id"),
            "lender_terms": [_terms(o) for o in view.offers],
        },
        "untrusted_merchant_content": {
            "offer_listings": [_pick(_listing(o), "ref", "offer_id", "advertised_total_repayable") for o in view.offers],
        },
    }


def authority_payload(view: EvaluatorView) -> Payload:
    intent = _intent(view)
    merchant = _merchant(view)
    return {
        "task_context": _task(view),
        "trusted": {
            "customer_intent": _pick(
                intent, "ref", "customer_id", "term_months", "max_total_repayment", "max_monthly_payment", "requested_offer_id"
            ),
            "mandate": _mandate(view),
            "proposed_transaction": _transaction(view),
            "lender_terms": [
                {
                    **_pick(_terms(o), "ref", "offer_id", "merchant_id", "lender_id", "payee_id", "term_months"),
                    "computed": _pick(_computed(o), "total_repayable", "max_monthly_payment", "initial_charge"),
                }
                for o in view.offers
            ],
        },
        "untrusted_merchant_content": {"merchant": _pick(merchant, "ref", "merchant_id")},
    }


PAYLOAD_BUILDERS: dict[EvaluatorRole, Callable[[EvaluatorView], Payload]] = {
    EvaluatorRole.GENERALIST: generalist_payload,
    EvaluatorRole.MOCK: generalist_payload,
    EvaluatorRole.MERCHANT_CLAIM: merchant_claim_payload,
    EvaluatorRole.CREDIT_TERMS: credit_terms_payload,
    EvaluatorRole.AUTHORITY: authority_payload,
}


# --- Inspection helpers (used for grounding and tests) ----------------------


def collect_refs(payload: Any) -> frozenset[str]:
    """Every `ref` an analyst is allowed to cite."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("ref"), str):
                found.add(node["ref"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return frozenset(found)


def flatten_leaves(payload: Any, prefix: str = "") -> Iterator[str]:
    """Stable path for every scalar in a payload; list items with a `ref` are keyed by it."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield from flatten_leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            key = item["ref"] if isinstance(item, dict) and "ref" in item else str(index)
            yield from flatten_leaves(item, f"{prefix}[{key}]")
    else:
        yield f"{prefix}={payload!r}"
