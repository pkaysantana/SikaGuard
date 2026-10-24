"""Evaluator protocol: the label-free view, strict schemas, explicit failures and the mock evaluators."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from sika_guard.coordinator import Condition, coordinate
from sika_guard.evaluators.contract import (
    AIFinding,
    ConcernCode,
    Evaluator,
    EvaluatorRole,
    EvaluatorStatus,
    EvaluatorVerdict,
    EvaluatorView,
    InvalidResponse,
    ModelAnalysis,
    parse_analysis,
    verdict_from_analysis,
)
from sika_guard.evaluators.mocks import AlwaysAllow, AlwaysReview, MalformedResponse, Scripted, Timeout, make_analysis
from sika_guard.evaluators.payloads import collect_refs, generalist_payload
from sika_guard.fixtures.cases import build_cases, get_case
from sika_guard.fixtures.labels import LABELS
from sika_guard.labels import AttackFamily, GroundTruth
from sika_guard.models import Case, Decision

from fakes import analysis_json


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def view() -> EvaluatorView:
    return EvaluatorView.from_case(get_case("case-05"))


# --- the view ---------------------------------------------------------------


def test_view_is_immutable(view):
    with pytest.raises(ValidationError):
        view.case_id = "other"
    assert isinstance(view.offers, tuple) and isinstance(view.assertions, tuple)


def test_view_has_exactly_the_case_fields_and_no_labels():
    label_fields = set(GroundTruth.model_fields) - {"case_id"}
    assert set(EvaluatorView.model_fields) == set(Case.model_fields)
    assert not label_fields & set(EvaluatorView.model_fields)


def test_view_fingerprint_is_stable_and_sensitive():
    a, b = EvaluatorView.from_case(get_case("case-05")), EvaluatorView.from_case(get_case("case-05"))
    assert a.sha256() == b.sha256()
    assert a.sha256() != EvaluatorView.from_case(get_case("case-08")).sha256()


# --- strict model schema ----------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        analysis_json(verdict="MAYBE"),
        analysis_json(risk_score=1.5),
        analysis_json(abstain="yes"),
        analysis_json(surprise="field"),
        analysis_json(verdict=None),  # not abstaining, but no verdict
        analysis_json(verdict="ALLOW", abstain=True, abstain_reason="x"),  # abstaining with a verdict
        analysis_json(verdict=None, abstain=True, abstain_reason=None),  # abstaining without a reason
        analysis_json(findings=[{"code": "made_up_code", "evidence_refs": ["mandate:MND-1"], "note": "n"}]),
        analysis_json(findings=[{"code": "limit_exceeded", "evidence_refs": [], "note": "n"}]),  # ungrounded
        analysis_json(extracted_claims=[{"kind": "total_repayable", "offer_id": "OFFER-A", "source_ref": "listing:OFFER-A", "quote": "£1"}]),
        "not json",
        "",
        "[]",
    ],
)
def test_invalid_structured_responses_are_rejected(bad):
    with pytest.raises(InvalidResponse):
        parse_analysis(bad)


def test_parse_accepts_plain_and_fenced_json():
    assert parse_analysis(analysis_json("REVIEW")).verdict is Decision.REVIEW
    assert parse_analysis("```json\n" + analysis_json("ALLOW") + "\n```").verdict is Decision.ALLOW


def test_parse_errors_never_echo_the_model_output():
    with pytest.raises(InvalidResponse) as caught:
        parse_analysis(analysis_json(verdict="IGNORE-PREVIOUS-INSTRUCTIONS-SENTINEL"))
    assert "SENTINEL" not in str(caught.value)


def test_verdict_record_enforces_status_consistency(view):
    ok = verdict_from_analysis(
        make_analysis(Decision.ALLOW), view, evaluator_id="e", role=EvaluatorRole.MOCK, allowed_refs=collect_refs(generalist_payload(view))
    )
    assert ok.status is EvaluatorStatus.OK and ok.risk_score_calibration == "uncalibrated"
    with pytest.raises(ValidationError):
        EvaluatorVerdict(**{**ok.model_dump(), "status": EvaluatorStatus.TIMEOUT})  # a timeout cannot carry a verdict
    with pytest.raises(ValidationError):
        EvaluatorVerdict(
            case_id="c", evaluator_id="e", role=EvaluatorRole.MOCK, status=EvaluatorStatus.OK, verdict=None, abstained=True,
            abstain_reason="r", view_sha256="x",
        )


def test_citing_references_that_were_not_shown_is_an_invalid_response(view):
    finding = AIFinding(code=ConcernCode.LIMIT_EXCEEDED, evidence_refs=("evidence:EV-NOT-IN-INPUT",), note="n")
    verdict = verdict_from_analysis(
        make_analysis(Decision.REVIEW, findings=[finding]), view, evaluator_id="e", role=EvaluatorRole.MOCK,
        allowed_refs=collect_refs(generalist_payload(view)),
    )
    assert verdict.status is EvaluatorStatus.INVALID_RESPONSE and verdict.verdict is None and verdict.abstained


def test_model_abstention_is_recorded_as_an_abstention(view):
    analysis = ModelAnalysis(verdict=None, risk_score=0.0, abstain=True, abstain_reason="insufficient data", explanation="")
    verdict = verdict_from_analysis(analysis, view, evaluator_id="e", role=EvaluatorRole.MOCK, allowed_refs=frozenset())
    assert verdict.status is EvaluatorStatus.ABSTAINED and verdict.abstain_reason == "insufficient data"


# --- mock evaluators --------------------------------------------------------


def test_mocks_satisfy_the_protocol():
    for mock in (AlwaysAllow(), AlwaysReview(), Scripted({}), Timeout(), MalformedResponse()):
        assert isinstance(mock, Evaluator)


def test_always_allow_and_always_review(view):
    assert run(AlwaysAllow().evaluate(view)).verdict is Decision.ALLOW
    assert run(AlwaysReview().evaluate(view)).verdict is Decision.REVIEW


def test_scripted_returns_the_script_and_abstains_without_one(view):
    finding = AIFinding(code=ConcernCode.CLAIM_UNVERIFIED, evidence_refs=("assertion:ASR-1",), note="n")
    mock = Scripted({"case-05": make_analysis(Decision.REVIEW, findings=[finding])})
    scripted = run(mock.evaluate(view))
    assert scripted.findings == (finding,) and scripted.evidence_refs == ("assertion:ASR-1",)
    other = run(mock.evaluate(EvaluatorView.from_case(get_case("case-01"))))
    assert other.status is EvaluatorStatus.ABSTAINED


def test_timeout_and_invalid_response_are_explicit_not_dropped(view):
    timed_out = run(Timeout().evaluate(view))
    malformed = run(MalformedResponse().evaluate(view))
    assert (timed_out.status, malformed.status) == (EvaluatorStatus.TIMEOUT, EvaluatorStatus.INVALID_RESPONSE)
    for v in (timed_out, malformed):
        assert v.verdict is None and v.abstained and v.abstain_reason


def test_every_evaluator_receives_the_same_permitted_data():
    case = get_case("case-05")
    mocks = [AlwaysAllow(), AlwaysReview(), Timeout(), MalformedResponse(), Scripted({})]
    records = run(coordinate(case, Condition.C, mocks))
    expected = EvaluatorView.from_case(case)
    for mock in mocks:
        assert mock.seen_views == [expected]
    assert {v.view_sha256 for v in records.verdicts} == {expected.sha256()}


# --- labels never reach inference -------------------------------------------


def _label_strings(truth: GroundTruth) -> list[str]:
    return [truth.rationale, truth.attack_family.value, *(set(GroundTruth.model_fields) - {"case_id"})]


@pytest.mark.parametrize("case", build_cases(), ids=lambda c: c.case_id)
def test_no_label_text_appears_in_the_evaluator_view_or_payload(case):
    view = EvaluatorView.from_case(case)
    haystack = view.model_dump_json() + json.dumps(generalist_payload(view))
    forbidden = _label_strings(LABELS[case.case_id]) + [f.value for f in AttackFamily]
    assert not [s for s in forbidden if s in haystack]


def test_concern_vocabulary_cannot_collide_with_attack_family_labels():
    assert {c.value for c in ConcernCode}.isdisjoint({f.value for f in AttackFamily})
