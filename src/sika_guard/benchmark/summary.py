"""A flat, per-record summary that keeps four things apart: model errors, provider failures, explicit
policy reviews and genuine verified contradictions.

Nothing here reads a label; it only describes what one condition did on one case in one repetition.
This module must not import the hidden-label module.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from sika_guard.coordinator import CaseRecord
from sika_guard.evaluators.adjudicate import AllegationStatus
from sika_guard.evaluators.contract import EvaluatorStatus
from sika_guard.models import DecisionBasis

_POLICY_REVIEW_BASES = {
    DecisionBasis.REVIEW_EVIDENCE_REQUIREMENT, DecisionBasis.REVIEW_INFERIOR_SELECTION, DecisionBasis.REVIEW_UNRESOLVED_CONFLICT,
    DecisionBasis.REVIEW_INCOMPARABLE_REPRESENTATION, DecisionBasis.REVIEW_SELECTED_OFFER_NOT_OPEN,
}
_PROVIDER_FAILURES = {EvaluatorStatus.TIMEOUT, EvaluatorStatus.PROVIDER_ERROR, EvaluatorStatus.REFUSAL}


class RecordSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: str
    deterministic_decision: str
    decision_bases: tuple[str, ...]
    escalated_by_ai: bool
    verdict_statuses: dict[str, int]  # every verdict's status, so nothing is dropped
    provider_failures: int  # timeouts, provider errors, refusals: the service, not the model's judgement
    schema_failures: int  # replies that failed the strict output contract
    abstentions: int  # the model chose not to judge
    supported_findings: int  # model findings that trusted code verified
    unsupported_findings: int  # model findings contradicted by structured data: these are model errors
    unverified_findings: int  # could be neither verified nor disproved
    model_errors: int  # == unsupported_findings; named separately because a reader looks for it
    verified_contradictions: int  # offers trusted code proved the merchant misdescribed
    verified_contradiction_review: bool  # the decision rests on a verified contradiction
    explicit_policy_review: bool  # the decision rests on an evidence requirement, inferior selection or unresolved conflict
    evaluator_failure_review: bool  # fail-closed only: not an attack finding
    verified_extraction_added_review: bool = False  # the AI layer raised ALLOW to REVIEW through a verified extraction (not through anything the rule engine already held)
    calls: int
    latency_ms_total: float
    prompt_tokens: int | None  # None unless every call reported counts
    completion_tokens: int | None
    provider_cost: str | None  # the provider's own figure, unit UNKNOWN; None unless every call reported one
    providers: tuple[str, ...]
    returned_models: tuple[str, ...]


def summarize(record: CaseRecord) -> RecordSummary:
    verdicts = record.verdicts
    statuses = Counter(v.status for v in verdicts)
    adjudications = Counter(a.status for a in record.adjudications)
    calls = [c for v in verdicts for c in v.calls]
    bases = set(record.decision_bases)
    counted = all(c.prompt_tokens is not None and c.completion_tokens is not None for c in calls) and bool(calls)
    costed = all(c.provider_cost is not None for c in calls) and bool(calls)
    return RecordSummary(
        decision=record.final_decision.value,
        deterministic_decision=record.deterministic_decision.value,
        decision_bases=tuple(b.value for b in record.decision_bases),
        escalated_by_ai=record.escalated_by_ai,
        verdict_statuses={s.value: n for s, n in sorted(statuses.items(), key=lambda kv: kv[0].value)},
        provider_failures=sum(n for s, n in statuses.items() if s in _PROVIDER_FAILURES),
        schema_failures=statuses[EvaluatorStatus.INVALID_RESPONSE],
        abstentions=statuses[EvaluatorStatus.ABSTAINED],
        supported_findings=adjudications[AllegationStatus.VERIFIED],
        unsupported_findings=adjudications[AllegationStatus.CONTRADICTED],
        unverified_findings=adjudications[AllegationStatus.UNVERIFIED],
        model_errors=adjudications[AllegationStatus.CONTRADICTED],
        verified_contradictions=len(record.proven_false_offer_ids),
        verified_contradiction_review=DecisionBasis.REVIEW_VERIFIED_CONTRADICTION in bases,
        explicit_policy_review=bool(bases & _POLICY_REVIEW_BASES),
        evaluator_failure_review=DecisionBasis.REVIEW_EVALUATOR_FAILURE in bases,
        verified_extraction_added_review=record.escalated_by_ai and any(r.startswith("verified_extraction:") for r in record.ai_review_routes),
        calls=len(calls),
        latency_ms_total=sum(c.latency_ms for c in calls),
        prompt_tokens=sum(c.prompt_tokens for c in calls) if counted else None,
        completion_tokens=sum(c.completion_tokens for c in calls) if counted else None,
        provider_cost=str(sum((c.provider_cost for c in calls), Decimal(0))) if costed else None,
        providers=tuple(sorted({c.provider or "not exposed" for c in calls})),
        returned_models=tuple(sorted({c.returned_model or "not reported" for c in calls})),
    )
