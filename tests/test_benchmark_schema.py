"""Benchmark schema, the hidden-label boundary, sealing and the decision log. Offline."""

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sika_guard.benchmark import decisions, hidden, seal, visible
from sika_guard.benchmark.hidden import (
    CausalLabel,
    ClaimLabel,
    Counterfactual,
    EvidenceState,
    HiddenLabels,
    dump_labels,
    open_for_validation,
    parse_labels,
)
from sika_guard.benchmark.manifest import HIDDEN_MODULES, INFERENCE_SIDE
from sika_guard.benchmark.mockrun import mock_records
from sika_guard.benchmark.pilot import build_pilot
from sika_guard.coordinator import Condition
from sika_guard.evaluators.contract import EvaluatorView
from sika_guard.models import Case, Decision, DecisionBasis

SRC = Path(__file__).resolve().parents[1] / "src" / "sika_guard"


def pilot():
    cases, labels = build_pilot()
    return cases, {x.case_id: x for x in labels}


def by_family(labels, family):
    return next(x for x in labels.values() if x.attack_family.value == family)


def with_(label: HiddenLabels, **changes) -> HiddenLabels:
    return HiddenLabels.model_validate({**label.model_dump(), **changes})


# --- the boundary: no inference-side module can reach hidden labels ----------------------------------


def imported_modules(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return found


def inference_files() -> list[Path]:
    files = [*(SRC / "evaluators").glob("*.py"), SRC / "policy.py", SRC / "coordinator.py", SRC / "models.py", SRC / "money.py"]
    files += [SRC / "benchmark" / n for n in ("visible.py", "decisions.py", "seal.py")]
    return files


def test_no_inference_side_module_imports_any_hidden_module():
    files = inference_files()
    assert len(files) >= 15
    for path in files:
        bad = imported_modules(path) & set(HIDDEN_MODULES)
        assert not bad, f"{path.name} imports hidden module(s) {bad}"


def test_the_manifest_declares_exactly_the_boundary_this_test_enforces():
    assert set(HIDDEN_MODULES) == {"sika_guard.labels", "sika_guard.fixtures.labels", "sika_guard.benchmark.hidden"}
    assert "sika_guard.benchmark.visible" in INFERENCE_SIDE and "sika_guard.evaluators.*" in INFERENCE_SIDE


def test_evaluators_policy_and_coordinator_do_not_import_the_benchmark_package_at_all():
    for path in [*(SRC / "evaluators").glob("*.py"), SRC / "policy.py", SRC / "coordinator.py"]:
        assert not [m for m in imported_modules(path) if m.startswith("sika_guard.benchmark")], path.name


def test_a_label_field_cannot_be_a_field_of_what_a_model_is_given():
    label_fields = set(HiddenLabels.model_fields) - {"case_id", "schema_id"}
    assert not label_fields & set(Case.model_fields)
    assert not label_fields & set(EvaluatorView.model_fields)


# --- the visible side is only a Case ---------------------------------------------------------------------


def test_visible_file_round_trips_and_contains_nothing_but_cases():
    cases, _ = pilot()
    text = visible.dump_visible(cases)
    assert visible.parse_visible(text) == cases
    for line in text.splitlines():
        assert set(json.loads(line)) == {"schema", "case"}


@pytest.mark.parametrize("smuggle", [{"labels": {"x": 1}}, {"expected": "REVIEW"}, {"note": "family"}])
def test_a_visible_line_with_any_extra_key_is_refused(smuggle):
    cases, _ = pilot()
    obj = json.loads(visible.dump_visible(cases[:1]))
    obj.update(smuggle)
    with pytest.raises(ValueError):
        visible.parse_visible(json.dumps(obj))


def test_a_label_smuggled_inside_the_case_object_is_refused_by_the_strict_model():
    cases, labels = pilot()
    obj = json.loads(visible.dump_visible(cases[:1]))
    obj["case"]["attack_family"] = "bm_hidden_fees"
    with pytest.raises(ValidationError):
        visible.parse_visible(json.dumps(obj))


def test_case_ids_are_opaque_deterministic_and_unrelated_to_content():
    cases, _ = pilot()
    assert visible.opaque_id("pilot", "named") == visible.opaque_id("pilot", "named") != visible.opaque_id("pilot", "irrelevant")
    assert all(c.case_id.startswith("bm-") and len(c.case_id) == 13 for c in cases)


# --- label consistency rules ---------------------------------------------------------------------------------


def test_every_pilot_label_is_internally_consistent_and_round_trips():
    _, labels = pilot()
    assert parse_labels(dump_labels(list(labels.values()))) == labels


def test_causation_is_never_assumed_from_a_false_claim_and_a_poor_choice():
    _, labels = pilot()
    unknown = [x for x in labels.values() if x.merchant_claim_false and x.selected_inferior and x.counterfactual is None]
    assert unknown and all(x.causal_selection is CausalLabel.UNKNOWN for x in unknown)


def test_caused_requires_a_counterfactual_and_unknown_forbids_one():
    _, labels = pilot()
    caused = by_family(labels, "bm_false_availability_verifiable")
    assert caused.causal_selection is CausalLabel.CAUSED and caused.counterfactual is not None
    with pytest.raises(ValidationError, match="requires a counterfactual"):
        with_(caused, counterfactual=None)
    unknown = by_family(labels, "bm_false_availability_unverifiable")
    with pytest.raises(ValidationError, match="not unknown"):
        with_(unknown, counterfactual={"paired_case_id": "bm-0000000000", "choice_without_misinformation": "X"})
    with pytest.raises(ValidationError, match="needs a causal label"):
        with_(unknown, causal_selection=CausalLabel.NOT_APPLICABLE)


def test_without_a_false_claim_causation_is_not_applicable():
    _, labels = pilot()
    benign = by_family(labels, "bm_benign_fits_preference")
    with pytest.raises(ValidationError, match="not_applicable"):
        with_(benign, causal_selection=CausalLabel.UNKNOWN)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"hard_violation_present": True}, "must coincide"),
        ({"transaction_authorised": False}, "hard violation"),
        ({"expected_bases": (DecisionBasis.REVIEW_INFERIOR_SELECTION,)}, "do not fit a policy action"),
        ({"attack_present": True}, "never an attack"),
        ({"selected_inferior": False, "avoidable_cost_minor": 500}, "no avoidable cost"),
        ({"contradictory_evidence_available": True}, "fresh_contradicts"),
    ],
)
def test_internally_inconsistent_labels_are_rejected(change, message):
    _, labels = pilot()
    with pytest.raises(ValidationError, match=message):
        with_(by_family(labels, "bm_benign_fits_preference"), **change)


def test_a_family_that_is_always_an_attack_cannot_be_labelled_as_none():
    _, labels = pilot()
    with pytest.raises(ValidationError, match="always an attack"):
        with_(by_family(labels, "bm_hidden_fees"), attack_present=False)


def test_unknown_truth_is_representable_apart_from_the_worlds_truth():
    _, labels = pilot()
    x = by_family(labels, "bm_false_availability_conflicting")
    assert x.merchant_claim_false is True and not x.claims[0].verifiable_from_inputs and x.evidence_state is EvidenceState.FRESH_CONFLICT


# --- sealing: labels only after decisions -------------------------------------------------------------------------


def test_construction_tools_open_labels_by_a_separate_named_door(tmp_path):
    _, labels = pilot()
    path = tmp_path / "labels.jsonl"
    path.write_text(dump_labels(list(labels.values())), encoding="utf-8")
    assert open_for_validation(path).keys() == labels.keys()
    assert "open_for_validation" not in (SRC / "benchmark" / "decisions.py").read_text(encoding="utf-8")


# --- the decision log ----------------------------------------------------------------------------------------------


def test_the_decision_log_round_trips_every_record(tmp_path):
    cases, _ = pilot()
    records = mock_records(cases)
    declaration = decisions.declare(mode="mock", manifest_digest="d" * 64, visible_cases_sha256="c" * 64, conditions=list(records))
    log = tmp_path / "decisions.jsonl"
    decisions.write_decision_log(log, declaration, records)
    got_declaration, got = decisions.read_decision_log(log)
    assert got_declaration.mode == "mock" and set(got) == set(records)
    assert {c: [r.model_dump() for r in rs] for c, rs in got.items()} == {c: [r.model_dump() for r in rs] for c, rs in records.items()}
    with pytest.raises(FileExistsError):
        decisions.write_decision_log(log, declaration, records)


def test_a_decision_log_contains_no_label_vocabulary(tmp_path):
    from sika_guard.benchmark.leakage import scan_text

    cases, _ = pilot()
    records = mock_records(cases)
    declaration = decisions.declare(mode="mock", manifest_digest="d" * 64, visible_cases_sha256="c" * 64, conditions=list(records))
    log = tmp_path / "decisions.jsonl"
    decisions.write_decision_log(log, declaration, records)
    allowed = {"review_evidence_requirement_unmet", "review_evaluator_failure", "review_verified_contradiction", "review_inferior_selection",
               "allow_with_recorded_uncertainty", "review_unresolved_evidence_conflict", "block_hard_violation"}  # the system's own decision bases
    assert [h for h in scan_text(log.read_text(encoding="utf-8")) if h not in allowed] == []


def test_every_decision_condition_is_recorded_for_every_case():
    cases, _ = pilot()
    records = mock_records(cases)
    assert set(records) == {Condition.A, Condition.B, Condition.B3, Condition.BC, Condition.C}
    assert all({r.case_id for r in recs} == {c.case_id for c in cases} for recs in records.values())
    assert all(r.final_decision in set(Decision) for recs in records.values() for r in recs)
