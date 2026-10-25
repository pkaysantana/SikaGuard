"""The reproducibility manifest: what it records, how it detects drift, and that the initial one still matches the frozen code."""

import copy
import json
from pathlib import Path

import pytest

from sika_guard.benchmark import manifest as mf
from sika_guard.coordinator import AGGREGATION_POLICY
from sika_guard.policy import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "manifests" / "manifest-0001-initial.json"


def fresh():
    return mf.build_manifest(ROOT, note="test")


def resealed(m):
    m["digest"] = mf.manifest_digest(m)
    return m


# --- content ----------------------------------------------------------------------------------------------------------


def test_the_manifest_records_what_the_brief_requires():
    m = fresh()
    assert m["schema"] == mf.MANIFEST_SCHEMA and m["digest"] == mf.manifest_digest(m)
    assert m["python"]["version"].count(".") == 2 and m["dependencies"]["pydantic"] and m["dependencies"]["httpx"]
    assert set(m["conditions"]) == {"A", "A+", "B", "B3", "Bc", "C"} and m["conditions"]["A"]["evaluators"] == []
    assert m["aggregation"]["policy"] == AGGREGATION_POLICY and m["aggregation"]["fail_closed_default"] is False
    assert m["decision_policy"]["deterministic_materiality_assumptions"]["values"]["material_gap_minor"] == DEFAULT_CONFIG.material_gap_minor
    assert m["decision_policy"]["document_sha256"] and m["decision_policy"]["rule_engine_sha256"]
    assert m["ground_truth_isolation"]["hidden_modules"] == list(mf.HIDDEN_MODULES) and m["ground_truth_isolation"]["requirements"]
    req = m["model_configuration_requirements"]
    assert req["retries"].startswith("none") and "SIKA_GUARD_LLM_MODEL" in req["environment_variables_required"] and "uncontrolled" in req


def test_every_prompt_is_recorded_with_its_version_and_a_hash_of_its_text():
    m = fresh()
    versions = {e["evaluator_id"]: e["prompt_version"] for c in m["conditions"].values() for e in c["evaluators"]}
    assert versions["generalist"] == "generalist-v4" and versions["rubric_generalist"] == "rubric-generalist-v4" and versions["authority"] == "authority-v4"
    assert all(len(e["system_prompt_sha256"]) == 64 for c in m["conditions"].values() for e in c["evaluators"])
    assert len(m["conditions"]["B3"]["evaluators"]) == 3 and len(m["conditions"]["C"]["evaluators"]) == 3


def test_every_source_file_is_hashed_and_grouped_and_the_frozen_core_is_all_present():
    m = fresh()
    assert all(f in m["files"] for f in mf.FROZEN_CORE + mf.DEVELOPMENT_DATA + mf.DOCUMENTS)
    groups = {e["group"] for e in m["files"].values()}
    assert {"frozen_core", "development_data", "document", "tests", "benchmark_tooling"} <= groups
    assert m["files"]["src/sika_guard/policy.py"]["sha256"] == mf.sha256_file(ROOT / "src/sika_guard/policy.py")


def test_the_working_tree_is_recorded_so_that_head_is_never_mistaken_for_the_code():
    state = fresh()["working_tree"]
    if state["git_available"]:
        assert isinstance(state["head_matches_working_tree"], bool) and "changed_or_untracked_paths" in state
        assert "does not identify the evaluated code" in state["note"]


def test_no_secret_value_can_enter_a_manifest(monkeypatch):
    monkeypatch.setenv("SIKA_GUARD_LLM_API_KEY", "sk-SENTINEL-must-not-appear")
    monkeypatch.setenv("SIKA_GUARD_LLM_BASE_URL", "https://secret-host.example/v1")
    text = json.dumps(fresh())
    assert "SENTINEL" not in text and "secret-host" not in text


# --- tamper evidence and drift ------------------------------------------------------------------------------------------


def test_an_edited_manifest_is_refused_on_load(tmp_path):
    path = tmp_path / "m.json"
    mf.write_manifest(fresh(), path)
    assert mf.load_manifest(path)["note"] == "test"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["note"] = "changed"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="edited"):
        mf.load_manifest(path)


def test_a_recorded_manifest_is_never_overwritten(tmp_path):
    path = tmp_path / "m.json"
    mf.write_manifest(fresh(), path)
    with pytest.raises(FileExistsError):
        mf.write_manifest(fresh(), path)


def test_an_unchanged_tree_shows_no_drift_against_a_manifest_taken_a_moment_ago():
    assert mf.verify_manifest(fresh(), ROOT).clean


def test_a_changed_removed_or_added_file_is_reported_with_its_group():
    base = fresh()
    changed = copy.deepcopy(base)
    changed["files"]["src/sika_guard/policy.py"]["sha256"] = "0" * 64
    drift = mf.verify_manifest(resealed(changed), ROOT)
    assert drift.changed == {"src/sika_guard/policy.py": "frozen_core"} and drift.frozen_core_drifted

    ghost = copy.deepcopy(base)
    ghost["files"]["src/sika_guard/ghost.py"] = {"sha256": "1" * 64, "group": "source_other"}
    del ghost["files"]["src/sika_guard/money.py"]
    drift = mf.verify_manifest(resealed(ghost), ROOT)
    assert "src/sika_guard/ghost.py" in drift.removed and "src/sika_guard/money.py" in drift.added


def test_a_changed_prompt_a_changed_policy_assumption_and_a_changed_aggregation_are_each_reported():
    base = fresh()
    prompt = copy.deepcopy(base)
    prompt["conditions"]["B"]["evaluators"][0]["system_prompt_sha256"] = "2" * 64
    drift = mf.verify_manifest(resealed(prompt), ROOT)
    assert drift.prompt_changes and drift.frozen_core_drifted

    policy = copy.deepcopy(base)
    policy["decision_policy"]["deterministic_materiality_assumptions"]["values"]["material_gap_minor"] = 999
    assert mf.verify_manifest(resealed(policy), ROOT).policy_changes

    aggregation = copy.deepcopy(base)
    aggregation["aggregation"]["policy"] = "majority_vote"
    assert any("aggregation" in c for c in mf.verify_manifest(resealed(aggregation), ROOT).other_changes)


def test_benchmark_tooling_changes_are_visible_but_do_not_count_as_frozen_core_drift():
    base = fresh()
    base["files"]["src/sika_guard/benchmark/ghost_tool.py"] = {"sha256": "3" * 64, "group": "benchmark_tooling"}
    drift = mf.verify_manifest(resealed(base), ROOT)
    assert "src/sika_guard/benchmark/ghost_tool.py" in drift.removed and not drift.frozen_core_drifted


# --- the recorded manifests: a tripwire against silent change ---------------------------------------------------------------------

PRE_AUDIT = ROOT / "manifests" / "manifest-0002-pre-remediation.json"
FIRST_POST_AUDIT = ROOT / "manifests" / "manifest-0003-post-remediation.json"  # superseded by 0004, kept as history
POST_AUDIT = ROOT / "manifests" / "manifest-0004-post-remediation.json"
PRE_DIRECTION_B = ROOT / "manifests" / "manifest-0005-pre-direction-b.json"
DIRECTION_B_BASELINE = ROOT / "manifests" / "manifest-0006-direction-b.json"  # superseded by 0007, kept as history
CURRENT = ROOT / "manifests" / "manifest-0007-hackathon-runner.json"

# Every frozen-core file Direction B changed or added since manifest-0005, and nothing else. Each is explained in DIRECTION_B_TRIAGE.md.
DIRECTION_B_AMENDED_FROZEN_FILES = {
    "src/sika_guard/models.py",  # trusted context, offer scope, claim subjects, source documents, new finding/basis vocabulary
    "src/sika_guard/policy.py",  # trusted clock, scope, agent, committed payments, authority hierarchy, direction/materiality, unopen offers
    "src/sika_guard/coordinator.py",  # one AI-to-REVIEW test for every condition; Condition A+; routes recorded
    "src/sika_guard/experiment.py",  # description and report mode for Condition A+
    "src/sika_guard/evaluate.py",  # Condition A+ runs without credentials
    "src/sika_guard/evaluators/adjudicate.py",  # committed payments are a limit code
    "src/sika_guard/evaluators/amounts.py",  # conditional/optional amounts; Unicode-folded reading
    "src/sika_guard/evaluators/contract.py",  # trusted context and documents in the view; rejected claims preserved; DETERMINISTIC role
    "src/sika_guard/evaluators/crosscheck.py",  # grounded facts, source classes, direction, materiality, enforcement
    "src/sika_guard/evaluators/grounding.py",  # Unicode folds, reading hazards, document sources
    "src/sika_guard/evaluators/llm.py",  # evaluators_for("A+")
    "src/sika_guard/evaluators/payloads.py",  # documents, subjects, trusted context
    "src/sika_guard/evaluators/prompts.py",  # prompts v4 (the reason is recorded beside the version strings)
    "src/sika_guard/evaluators/scanner.py",  # new: the deterministic scanner
}
V4_PROMPT_EVALUATORS = {
    "B/generalist", "B3/generalist_1", "B3/generalist_2", "B3/generalist_3", "Bc/rubric_generalist", "C/merchant_claim", "C/credit_terms", "C/authority",
}


def _touched(drift, group):
    return {p for p, g in {**drift.changed, **drift.added, **drift.removed}.items() if g == group}


def test_the_historical_manifests_are_intact_and_none_was_overwritten():
    manifests = [mf.load_manifest(p) for p in (INITIAL, PRE_AUDIT, FIRST_POST_AUDIT, POST_AUDIT, PRE_DIRECTION_B, DIRECTION_B_BASELINE, CURRENT)]  # load_manifest refuses an edited one
    first, pre, post1, post, start, _direction_b, now = manifests
    assert "INITIAL" in first["note"] and "AUDIT BASELINE" in pre["note"] and "not a freeze" in post1["note"] and "Supersedes manifest-0003" in post["note"]
    assert "STARTING STATE" in start["note"] and "not a freeze" in now["note"].lower()
    assert len({m["digest"] for m in manifests}) == 7
    assert all(m["working_tree"]["head_matches_working_tree"] is False for m in manifests[1:])  # none of them is a committed tree
    assert all(m.get("status", "development_baseline") == "development_baseline" for m in manifests)  # none was ever approved as a freeze


def test_the_initial_manifest_exists_was_taken_before_the_tooling_and_is_intact():
    m = mf.load_manifest(INITIAL)
    assert "INITIAL" in m["note"] and m["working_tree"]["head_matches_working_tree"] is False
    assert len(m["working_tree"]["changed_or_untracked_paths"]) > 10  # HEAD was not the code that ran
    assert [p for p in m["files"] if p.startswith("src/sika_guard/benchmark/")] == ["src/sika_guard/benchmark/__init__.py", "src/sika_guard/benchmark/manifest.py"]


def test_the_frozen_core_prompts_and_decision_policy_are_unchanged_since_the_current_baseline():
    """If this fails, frozen behaviour changed after the baseline was recorded. Do not edit a manifest to make it pass: record a NEW
    manifest with a stated reason (python -m sika_guard.benchmark manifest --out manifests/... --note ...) and treat results from before
    and after as not comparable."""
    drift = mf.verify_manifest(mf.load_manifest(CURRENT), ROOT)
    assert not drift.frozen_core_drifted, "frozen core drifted:\n" + drift.describe()
    assert drift.prompt_changes == [] and drift.policy_changes == []
    assert _touched(drift, "development_data") == set()


@pytest.mark.parametrize("baseline", [POST_AUDIT, PRE_DIRECTION_B], ids=["post-audit", "pre-direction-b"])
def test_direction_b_changed_exactly_the_documented_frozen_files_and_only_the_prompts_it_says(baseline):
    drift = mf.verify_manifest(mf.load_manifest(baseline), ROOT)
    assert _touched(drift, "frozen_core") == DIRECTION_B_AMENDED_FROZEN_FILES, sorted(_touched(drift, "frozen_core") ^ DIRECTION_B_AMENDED_FROZEN_FILES)
    assert {c.split(":")[0] for c in drift.prompt_changes} == V4_PROMPT_EVALUATORS  # every role moved v3 -> v4 together, because they share one common text
    assert all("-v3@" in c and "-v4@" in c for c in drift.prompt_changes)
    assert _touched(drift, "development_data") == {"src/sika_guard/fixtures/builders.py"}  # the fixture builders gained the trusted fields; no fixture was edited
    assert drift.policy_changes  # the decision-policy document and rule engine changed on purpose, and say so


def test_manifest_0005_is_the_state_the_audit_remediation_left_it_in():
    """The starting state of Direction B differs from the post-audit baseline in tests and benchmark tooling only, never in the frozen core."""
    drift = mf.verify_manifest(mf.load_manifest(POST_AUDIT), ROOT)
    start = {p for p, e in mf.load_manifest(PRE_DIRECTION_B)["files"].items() if e["group"] == "frozen_core"}
    post = {p for p, e in mf.load_manifest(POST_AUDIT)["files"].items() if e["group"] == "frozen_core"}
    assert start == post and all(mf.load_manifest(PRE_DIRECTION_B)["files"][p]["sha256"] == mf.load_manifest(POST_AUDIT)["files"][p]["sha256"] for p in start)
    assert drift.frozen_core_drifted  # and the tree has since moved on, which is what the Direction B work is


def test_the_hackathon_runner_work_changed_only_three_frozen_files_and_no_prompt():
    """Since the Direction B baseline (0006): the amount reader (customer-dependent conditions), the credit-terms payload (plan titles) and the
    scanner (attribution across plans). No prompt changed, and neither did the decision policy's rules."""
    drift = mf.verify_manifest(mf.load_manifest(DIRECTION_B_BASELINE), ROOT)
    assert _touched(drift, "frozen_core") == {"src/sika_guard/evaluators/amounts.py", "src/sika_guard/evaluators/payloads.py", "src/sika_guard/evaluators/scanner.py"}
    assert drift.prompt_changes == [d for d in drift.prompt_changes if d.startswith("A+/")]  # only the scanner's own definition (its source hash) moved
    assert _touched(drift, "development_data") == set()
