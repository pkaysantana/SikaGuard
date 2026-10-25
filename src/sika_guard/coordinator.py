"""Deterministic coordination of the rule engine and the AI analysts.

Order of play: the rule engine runs first; the selected AI condition runs next; trusted code
validates, cross-checks and adjudicates what came back; then `resolve_final_decision` merges.

The AI layer can raise ALLOW to REVIEW and can do nothing else, and only by one of three trusted routes:
a finding that trusted code verified and found material, a claim extracted by a model that trusted code
proves contradicts the lender's terms for the selected offer, or (only if configured) the fail-closed
failure policy. A model's own verdict, suspicion or confidence enforces nothing. Only the rule engine can
produce BLOCK, and nothing can lower one.

Failure policy: a timeout, refusal, schema error or abstention is an evaluator failure, never an attack
finding and never evidence about the merchant. By default the case proceeds on the rule engine's decision
with coverage marked partial or unavailable and the failure listed (ALLOW with recorded uncertainty). With
`fail_closed=True` an evaluator failure raises ALLOW to REVIEW, labelled `review_evaluator_failure`.

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

from sika_guard.evaluators.adjudicate import (
    FindingAdjudication,
    FindingEffect,
    VerdictTreatment,
    adjudicate,
    unsupported_limit_allegations,
)
from sika_guard.evaluators.contract import (
    ConcernCode,
    Evaluator,
    EvaluatorStatus,
    EvaluatorVerdict,
    EvaluatorView,
    failure_verdict,
)
from sika_guard.evaluators.crosscheck import (
    ClaimProvenance,
    CrossCheck,
    CrossCheckOutcome,
    claim_provenance,
    crosscheck_claims,
)
from sika_guard.models import Case, Decision, DecisionBasis, FindingCategory
from sika_guard.policy import DEFAULT_CONFIG, PolicyConfig, decision_bases, evaluate_case, resolve_final_decision


class Condition(StrEnum):
    A = "A"  # deterministic baseline
    A_PLUS = "A+"  # baseline plus the deterministic scanner: a non-LLM reference for what plain parsing of the same text can find
    B = "B"  # baseline plus one general-purpose analyst
    B3 = "B3"  # baseline plus three independent calls of the same general-purpose analyst (call-count matched to C; compute is not claimed to match)
    BC = "Bc"  # baseline plus one general-purpose analyst applying all three specialist rubrics (sees everything C sees, in one call)
    C = "C"  # baseline plus three concurrent specialists


# Declared before any benchmark run and shared by every AI condition, so conditions differ only in what
# they ask the model and how often. There is no vote and no use of risk scores: the most severe
# adjudicated verdict counts, as one cautious analyst's concern is enough to ask a human to look.
AGGREGATION_POLICY = "max_severity_of_adjudicated_verdicts"


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
    claim_provenance: tuple[ClaimProvenance, ...] = ()  # source and corroboration state of each model "unverified" finding
    adjudications: tuple[FindingAdjudication, ...] = ()  # every model finding, with what trusted code could establish and why
    verdict_treatments: tuple[VerdictTreatment, ...] = ()  # how each usable verdict counted towards the decision, and why
    unsupported_limit_allegations: tuple[str, ...] = ()  # "evaluator:limit_exceeded" contradicted by the deterministic limit checks
    decision_bases: tuple[DecisionBasis, ...] = ()  # why the final decision is what it is; never conflates the kinds of REVIEW
    uncertainty_reasons: tuple[str, ...] = ()  # everything left unresolved, with its reason; empty only for a genuinely clean case
    ai_review_routes: tuple[str, ...] = ()  # each route by which the AI layer was ABLE to ask for a REVIEW here, whether or not the rule engine had already


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

    adjudications, treatments = adjudicate(view, det, verdicts, config.policy)
    enforced = any(t.counted_as is Decision.REVIEW for t in treatments)  # a verified, material finding
    # The lender's terms are authoritative; a model-extracted claim that contradicts them matters if it concerns
    # the offer being bought. (Availability contradictions are the rule engine's own, with its own materiality.)
    # The one route by which an extraction reaches REVIEW (`CrossCheck.enforces`): verified, adverse to the customer, material under the
    # predeclared standard, and about the offer being bought. It is the same test for B, B3, Bc, C and the deterministic scanner.
    contradicted_terms = any(c.enforces and c.outcome is CrossCheckOutcome.CONTRADICTED_BY_TERMS for c in checks)
    source_conflict = any(c.enforces and c.outcome is CrossCheckOutcome.SOURCE_CONFLICT for c in checks)
    failed = config.fail_closed and len(usable) < len(verdicts)
    advisory = Decision.REVIEW if (enforced or contradicted_terms or source_conflict or failed) else Decision.ALLOW
    final = resolve_final_decision(det.decision, advisory)

    # Everything unresolved is listed with its reason, so no ALLOW is silently clean: a rule-engine uncertainty, a trusted
    # cross-check that could not conclude, an evaluator failure, or an unverifiable model finding. None of these accuses anyone.
    uncertainty = tuple(
        [f"rule_engine:{f.code.value}:{f.subject_id}" for f in det.findings if f.category in (FindingCategory.UNVERIFIED_CLAIM, FindingCategory.SUSPICIOUS_BEHAVIOUR)
         or (f.category in (FindingCategory.CONFIRMED_FALSE_CLAIM, FindingCategory.UNRESOLVED_CONTRADICTION, FindingCategory.INFERIOR_SELECTION) and not f.material)]
        + [f"crosscheck:{c.outcome.value}:{c.evaluator_id}:{c.offer_id}:{c.kind.value}" for c in checks
           if c.outcome in (CrossCheckOutcome.NOT_COMPARABLE, CrossCheckOutcome.EXTRACTION_DISAGREES, CrossCheckOutcome.UNGROUNDED, CrossCheckOutcome.DISCREPANT_NOT_ADVERSE)
           or (c.outcome in (CrossCheckOutcome.CONTRADICTED_BY_TERMS, CrossCheckOutcome.SOURCE_CONFLICT) and not c.enforces)]
        + [f"evaluator_failure:{v.evaluator_id}:{v.status.value}" for v in verdicts if not v.usable]
        + [f"unverified_finding:{a.evaluator_id}:{a.code.value}" for a in adjudications if a.effect is FindingEffect.UNVERIFIED_RECORDED]
    )
    ai_bases = (
        [DecisionBasis.REVIEW_VERIFIED_CONTRADICTION] * (enforced or contradicted_terms)
        + [DecisionBasis.REVIEW_UNRESOLVED_CONFLICT] * source_conflict
        + [DecisionBasis.REVIEW_EVALUATOR_FAILURE] * failed
    )
    routes = tuple(dict.fromkeys(
        (["adjudicated_finding"] if enforced else [])
        + [f"verified_extraction:{c.source_class.value}:{c.outcome.value}" for c in checks if c.enforces and c.source_class is not None]
        + (["fail_closed_evaluator_failure"] if failed else [])
    ))
    det_bases = decision_bases(det)
    if final is Decision.BLOCK:
        bases = det_bases
    elif final is Decision.REVIEW:
        bases = tuple(dict.fromkeys([b for b in det_bases if b.value.startswith("review")] + ai_bases))
    else:
        bases = (DecisionBasis.ALLOW_WITH_UNCERTAINTY if uncertainty else DecisionBasis.ALLOW,)

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
        claim_provenance=claim_provenance(view, verdicts, config.policy),
        adjudications=adjudications,
        verdict_treatments=treatments,
        unsupported_limit_allegations=unsupported_limit_allegations(adjudications),
        decision_bases=bases,
        uncertainty_reasons=uncertainty,
        ai_review_routes=routes,
    )
