"""Deterministic mock evaluators for offline operation and repeatable tests.

These are not models. Anything they output is scripted, so results from them say nothing about AI
performance. They record every view they receive so tests can check what evaluators are given.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from sika_guard.evaluators.contract import (
    AIFinding,
    EvaluatorRole,
    EvaluatorStatus,
    EvaluatorVerdict,
    EvaluatorView,
    ExtractedClaim,
    InvalidResponse,
    ModelAnalysis,
    failure_verdict,
    parse_analysis,
    verdict_from_analysis,
)
from sika_guard.evaluators.payloads import PAYLOAD_BUILDERS, collect_refs
from sika_guard.models import Decision

MOCK_PROMPT_VERSION = "mock"


def make_analysis(
    verdict: Decision | None,
    *,
    findings: Iterable[AIFinding] = (),
    claims: Iterable[ExtractedClaim] = (),
    risk_score: float = 0.5,
    abstain: bool = False,
    abstain_reason: str | None = None,
    explanation: str = "scripted",
) -> ModelAnalysis:
    return ModelAnalysis(
        verdict=verdict,
        findings=list(findings),
        extracted_claims=list(claims),
        risk_score=risk_score,
        abstain=abstain,
        abstain_reason=abstain_reason,
        explanation=explanation,
    )


class _Mock:
    def __init__(self, evaluator_id: str, role: EvaluatorRole = EvaluatorRole.MOCK):
        self.evaluator_id = evaluator_id
        self.role = role
        self.seen_views: list[EvaluatorView] = []

    async def evaluate(self, view: EvaluatorView) -> EvaluatorVerdict:
        self.seen_views.append(view)
        return self._respond(view)

    def _respond(self, view: EvaluatorView) -> EvaluatorVerdict:
        raise NotImplementedError

    def _record(self, view: EvaluatorView, analysis: ModelAnalysis) -> EvaluatorVerdict:
        return verdict_from_analysis(
            analysis,
            view,
            evaluator_id=self.evaluator_id,
            role=self.role,
            allowed_refs=collect_refs(PAYLOAD_BUILDERS[self.role](view)),
            prompt_version=MOCK_PROMPT_VERSION,
        )


class AlwaysAllow(_Mock):
    def __init__(self, evaluator_id: str = "mock-allow", role: EvaluatorRole = EvaluatorRole.MOCK):
        super().__init__(evaluator_id, role)

    def _respond(self, view: EvaluatorView) -> EvaluatorVerdict:
        return self._record(view, make_analysis(Decision.ALLOW, risk_score=0.0, explanation="always allows"))


class AlwaysReview(_Mock):
    def __init__(self, evaluator_id: str = "mock-review", role: EvaluatorRole = EvaluatorRole.MOCK):
        super().__init__(evaluator_id, role)

    def _respond(self, view: EvaluatorView) -> EvaluatorVerdict:
        return self._record(view, make_analysis(Decision.REVIEW, risk_score=0.5, explanation="always reviews"))


class Scripted(_Mock):
    """Returns the analysis scripted for a case id; abstains on any case with no script."""

    def __init__(
        self,
        scripts: Mapping[str, ModelAnalysis],
        evaluator_id: str = "mock-scripted",
        role: EvaluatorRole = EvaluatorRole.MOCK,
    ):
        super().__init__(evaluator_id, role)
        self.scripts = dict(scripts)

    def _respond(self, view: EvaluatorView) -> EvaluatorVerdict:
        analysis = self.scripts.get(view.case_id)
        if analysis is None:
            return failure_verdict(
                view, evaluator_id=self.evaluator_id, role=self.role, status=EvaluatorStatus.ABSTAINED,
                reason="no script for this case", prompt_version=MOCK_PROMPT_VERSION,
            )
        return self._record(view, analysis)


class Timeout(_Mock):
    def __init__(self, evaluator_id: str = "mock-timeout", role: EvaluatorRole = EvaluatorRole.MOCK):
        super().__init__(evaluator_id, role)

    def _respond(self, view: EvaluatorView) -> EvaluatorVerdict:
        return failure_verdict(
            view, evaluator_id=self.evaluator_id, role=self.role, status=EvaluatorStatus.TIMEOUT,
            reason="simulated timeout", prompt_version=MOCK_PROMPT_VERSION,
        )


class MalformedResponse(_Mock):
    """Feeds malformed text through the real parser, so the failure path is the production one."""

    def __init__(self, raw: str = "not json at all", evaluator_id: str = "mock-invalid", role: EvaluatorRole = EvaluatorRole.MOCK):
        super().__init__(evaluator_id, role)
        self.raw = raw

    def _respond(self, view: EvaluatorView) -> EvaluatorVerdict:
        try:
            return self._record(view, parse_analysis(self.raw))
        except InvalidResponse as exc:
            return failure_verdict(
                view, evaluator_id=self.evaluator_id, role=self.role, status=EvaluatorStatus.INVALID_RESPONSE,
                reason=str(exc), prompt_version=MOCK_PROMPT_VERSION,
            )


OFFLINE_MODES = ("allow", "review", "timeout", "invalid")


def offline_evaluators(mode: str, roles: Iterable[EvaluatorRole]) -> list[_Mock]:
    """One mock per role, so offline Condition C keeps three separately identified evaluators."""
    factories = {
        "allow": lambda r: AlwaysAllow(f"mock-allow-{r.value}", r),
        "review": lambda r: AlwaysReview(f"mock-review-{r.value}", r),
        "timeout": lambda r: Timeout(f"mock-timeout-{r.value}", r),
        "invalid": lambda r: MalformedResponse(evaluator_id=f"mock-invalid-{r.value}", role=r),
    }
    return [factories[mode](role) for role in roles]
