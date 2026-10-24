"""Deterministic coordination of the rule engine and the AI analysts.

Order of play: the rule engine runs first; the selected AI condition runs next; trusted code
validates and cross-checks what came back; then `resolve_final_decision` merges. An AI verdict can
raise ALLOW to REVIEW and can do nothing else. Only the rule engine can produce BLOCK.

Nothing here counts votes, uses a model's risk score, or treats a missing response as evidence
about the merchant.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from sika_guard.evaluators.contract import (
    ConcernCode,
    Evaluator,
    EvaluatorStatus,
    EvaluatorVerdict,
    EvaluatorView,
    failure_verdict,
)
from sika_guard.evaluators.crosscheck import CrossCheck, crosscheck_claims
from sika_guard.models import Case, Decision, FindingCategory
from sika_guard.policy import DEFAULT_CONFIG, PolicyConfig, evaluate_case, resolve_final_decision


class Condition(StrEnum):
    A = "A"  # deterministic baseline
    B = "B"  # baseline plus one general-purpose analyst
    C = "C"  # baseline plus three concurrent specialists


class Coverage(StrEnum):
    NOT_APPLICABLE = "not_applicable"  # no AI condition
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"  # every evaluator failed or abstained


@dataclass(frozen=True)
class CoordinatorConfig:
    policy: PolicyConfig = DEFAULT_CONFIG
    fail_closed: bool = False  # when True, an evaluator with no usable verdict escalates ALLOW to REVIEW


class Latency(BaseModel):
    model_config = ConfigDict(frozen=True)

    deterministic_ms: float
    ai_ms: float  # wall clock for the whole AI stage; for Condition C this is about the slowest specialist
    total_ms: float


class CaseRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    condition: Condition
    deterministic_decision: Decision
    authorised: bool
    deterministic_finding_codes: tuple[str, ...]
    verdicts: tuple[EvaluatorVerdict, ...]
    crosschecks: tuple[CrossCheck, ...]
    final_decision: Decision
    escalated_by_ai: bool
    coverage: Coverage
    evaluator_disagreement: bool  # usable AI verdicts differ from each other
    ai_deterministic_disagreement: bool  # a usable AI verdict differs from the rule engine's decision
    abstained_evaluators: tuple[str, ...]  # separate from disagreement: these produced no judgement
    proven_false_offer_ids: tuple[str, ...]  # confirmed by deterministic code, never by a model alone
    uncorroborated_contradictions: tuple[str, ...]  # "evaluator:offer" where a model alleged a contradiction code did not confirm
    latency: Latency


async def _run_one(evaluator: Evaluator, view: EvaluatorView) -> EvaluatorVerdict:
    try:
        verdict = await evaluator.evaluate(view)
    except Exception as exc:  # a crashing evaluator is an explicit failure, not a silent drop
        return failure_verdict(
            view, evaluator_id=evaluator.evaluator_id, role=evaluator.role, status=EvaluatorStatus.PROVIDER_ERROR,
            reason=f"evaluator raised {type(exc).__name__}",
        )
    if verdict.case_id != view.case_id or verdict.view_sha256 != view.sha256():
        return failure_verdict(
            view, evaluator_id=evaluator.evaluator_id, role=evaluator.role, status=EvaluatorStatus.INVALID_RESPONSE,
            reason="verdict does not belong to the view it was given",
        )
    return verdict


def _offers_cited(view: EvaluatorView, refs: Sequence[str]) -> set[str]:
    by_assertion = {f"assertion:{a.assertion_id}": a.offer_id for a in view.assertions}
    by_evidence = {f"evidence:{e.evidence_id}": e.offer_id for e in view.evidence}
    found = set()
    for ref in refs:
        kind, _, ident = ref.partition(":")
        if kind in ("listing", "terms"):
            found.add(ident)
        else:
            found.update(filter(None, [by_assertion.get(ref), by_evidence.get(ref)]))
    return found


async def coordinate(
    case: Case,
    condition: Condition,
    evaluators: Sequence[Evaluator] = (),
    config: CoordinatorConfig = CoordinatorConfig(),
) -> CaseRecord:
    started = time.perf_counter()
    det = evaluate_case(case, config.policy)
    after_det = time.perf_counter()

    view = EvaluatorView.from_case(case)
    # Every evaluator is started at once; one slow or failed call never delays the others' start.
    verdicts = tuple(await asyncio.gather(*(_run_one(e, view) for e in evaluators)))
    after_ai = time.perf_counter()

    checks = crosscheck_claims(view, verdicts, config.policy)
    usable = [v for v in verdicts if v.usable]

    advisory = Decision.ALLOW
    for v in usable:
        if v.verdict.severity > advisory.severity:
            advisory = v.verdict
    if any(c.contradicted for c in checks) or (config.fail_closed and len(usable) < len(verdicts)):
        advisory = max(advisory, Decision.REVIEW, key=lambda d: d.severity)
    final = resolve_final_decision(det.decision, advisory)

    proven = {f.subject_id for f in det.findings if f.category is FindingCategory.CONFIRMED_FALSE_CLAIM and f.subject_id}
    proven |= {c.offer_id for c in checks if c.contradicted}
    uncorroborated = sorted(
        f"{v.evaluator_id}:{offer}"
        for v in usable
        for finding in v.findings
        if finding.code is ConcernCode.CLAIM_CONTRADICTED_BY_EVIDENCE
        for offer in (_offers_cited(view, finding.evidence_refs) or {"?"})
        if offer not in proven
    )

    if not verdicts:
        coverage = Coverage.NOT_APPLICABLE
    elif len(usable) == len(verdicts):
        coverage = Coverage.COMPLETE
    else:
        coverage = Coverage.PARTIAL if usable else Coverage.UNAVAILABLE

    finished = time.perf_counter()
    return CaseRecord(
        case_id=case.case_id,
        condition=condition,
        deterministic_decision=det.decision,
        authorised=det.authorised,
        deterministic_finding_codes=tuple(f.code.value for f in det.findings),
        verdicts=verdicts,
        crosschecks=checks,
        final_decision=final,
        escalated_by_ai=final.severity > det.decision.severity,
        coverage=coverage,
        evaluator_disagreement=len({v.verdict for v in usable}) > 1,
        ai_deterministic_disagreement=any(v.verdict is not det.decision for v in usable),
        abstained_evaluators=tuple(v.evaluator_id for v in verdicts if not v.usable),
        proven_false_offer_ids=tuple(sorted(proven)),
        uncorroborated_contradictions=tuple(uncorroborated),
        latency=Latency(
            deterministic_ms=(after_det - started) * 1000,
            ai_ms=(after_ai - after_det) * 1000,
            total_ms=(finished - started) * 1000,
        ),
    )
