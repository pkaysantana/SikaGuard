"""Scripted, synthetic evaluator behaviour used ONLY to exercise scoring and reporting on the pilot.

Nothing here is model output and nothing here says anything about any model. Each condition is given a
different caricature so that the scoring paths (false escalation, unsupported findings, evaluator failure
under fail-closed, trusted-code escalation, A-already-caught credit rules) all have something to count.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from sika_guard.coordinator import CaseRecord, Condition, CoordinatorConfig
from sika_guard.evaluators.contract import AIFinding, ClaimKind, ConcernCode, ExtractedClaim
from sika_guard.evaluators.mocks import AlwaysAllow, Scripted, Timeout, make_analysis
from sika_guard.experiment import run_condition
from sika_guard.models import Case, Decision


def alarmist(cases: Sequence[Case]) -> Scripted:
    """Says REVIEW about everything, with a vague finding and a false numeric allegation. Both are model noise."""
    scripts = {}
    for c in cases:
        findings = [
            AIFinding(code=ConcernCode.INTENT_MISMATCH, evidence_refs=(f"intent:{c.intent.customer_id}",), note="vague unease"),
            AIFinding(code=ConcernCode.LIMIT_EXCEEDED, evidence_refs=(f"txn:{c.transaction.transaction_id}", f"mandate:{c.mandate.mandate_id}"), note="alleged"),
        ]
        scripts[c.case_id] = make_analysis(Decision.REVIEW, findings=findings, risk_score=0.9)
    return Scripted(scripts, evaluator_id="mock-alarmist")


def extractor(cases: Sequence[Case], evaluator_id: str = "mock-extractor") -> Scripted:
    """Reads the selected offer's advertised total out of its listing, verbatim, and says ALLOW. Trusted code does the rest."""
    scripts = {}
    for c in cases:
        offer = next(o for o in c.offers if o.offer_id == c.transaction.offer_id)
        adv = offer.advertised_total_repayable
        claims = [] if adv is None else [
            ExtractedClaim(
                kind=ClaimKind.TOTAL_REPAYABLE, offer_id=offer.offer_id, source_ref=f"listing:{offer.offer_id}",
                quote=f"advertised_total_repayable: {adv.plain()}", amount_minor=adv.amount_minor, currency=adv.currency,
            )
        ]
        scripts[c.case_id] = make_analysis(Decision.ALLOW, claims=claims, risk_score=0.1)
    return Scripted(scripts, evaluator_id=evaluator_id)


def mock_records(cases: Sequence[Case]) -> dict[Condition, list[CaseRecord]]:
    async def go() -> dict[Condition, list[CaseRecord]]:
        return {
            Condition.A: await run_condition(cases, Condition.A, []),
            Condition.B: await run_condition(cases, Condition.B, [alarmist(cases)]),
            Condition.B3: await run_condition(cases, Condition.B3, [extractor(cases, "mock-sample-1"), AlwaysAllow("mock-sample-2"), AlwaysAllow("mock-sample-3")]),
            Condition.BC: await run_condition(cases, Condition.BC, [AlwaysAllow("mock-rubric")]),
            # fail-closed with one dead specialist: every ALLOW becomes a review labelled as an evaluator failure
            Condition.C: await run_condition(cases, Condition.C, [Timeout("mock-c1"), AlwaysAllow("mock-c2"), AlwaysAllow("mock-c3")], CoordinatorConfig(fail_closed=True)),
        }

    return asyncio.run(go())
