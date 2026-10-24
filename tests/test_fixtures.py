"""Fixture integrity, label separation and the pinned baseline results."""

import inspect
from pathlib import Path

import sika_guard.policy as policy
from sika_guard.evaluate import run_baseline, summarise
from sika_guard.fixtures.cases import build_cases
from sika_guard.fixtures.labels import LABELS
from sika_guard.labels import AttackFamily, GroundTruth
from sika_guard.models import Case, Decision


def test_there_are_twenty_unique_cases_with_matching_labels():
    cases = build_cases()
    ids = [c.case_id for c in cases]
    assert len(ids) == 20 and len(set(ids)) == 20
    assert set(ids) == set(LABELS)


def test_every_requested_family_is_covered():
    assert {t.attack_family for t in LABELS.values()} == set(AttackFamily)


def test_labels_are_internally_consistent():
    for truth in LABELS.values():
        assert truth.financially_inferior == (truth.avoidable_cost_minor > 0)
        if truth.hard_authorisation_violation:
            assert truth.ideal_decision is Decision.BLOCK


def test_cases_carry_no_ground_truth():
    label_fields = set(GroundTruth.model_fields) - {"case_id"}
    assert not label_fields & set(Case.model_fields)
    family_values = {f.value for f in AttackFamily}
    for case in build_cases():
        dumped = case.model_dump_json()
        assert not any(field in dumped for field in label_fields)
        assert not any(f'"{value}"' in dumped for value in family_values)


def test_the_policy_never_touches_labels():
    source = Path(inspect.getsourcefile(policy)).read_text(encoding="utf-8")
    assert "GroundTruth" not in source and "sika_guard.labels" not in source and "fixtures" not in source
    assert list(inspect.signature(policy.evaluate_case).parameters)[:1] == ["case"]


def test_baseline_results_are_pinned():
    summary = summarise(run_baseline())
    assert summary.total == 20
    assert (summary.hard_blocked, summary.hard_total) == (4, 4)
    assert summary.benign_escalated == 0
    assert [o.case.case_id for o in summary.disagreements] == ["case-15"]  # an uncorroborated false claim is invisible
    assert (summary.false_claim_confirmed, summary.false_claim_total) == (5, 7)
    assert (summary.inferior_escalated, summary.inferior_missed) == (6, 1)


def test_no_inference_code_can_reach_labels_or_fixtures():
    import sika_guard

    root = Path(inspect.getsourcefile(sika_guard)).parent
    inference_sources = [*(root / "evaluators").glob("*.py"), root / "coordinator.py", root / "policy.py"]
    assert len(inference_sources) >= 9
    for path in inference_sources:
        source = path.read_text(encoding="utf-8")
        assert "GroundTruth" not in source, path.name
        assert "sika_guard.labels" not in source and "sika_guard.fixtures" not in source, path.name
