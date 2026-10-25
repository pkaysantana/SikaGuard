"""Direction B, Phase 7 (Opus R10): an experiment run and its seal are bound to a manifest that describes the code that ran.

Offline. Each test is a way an experiment could be mislabelled, edited or left unfinished and still have its labels opened; each must now stay
closed. Completed-run integrity is what is tested. Protection against an operator who rewrites the manifest directory, the log and the
seal together is not provided, and test_the_limit_is_stated checks that the code says so.
"""

import asyncio
import json
from pathlib import Path

import pytest

from manifest_helper import record_manifest
from sika_guard.benchmark import binding, manifest as mf, runner
from sika_guard.benchmark.hidden import dump_labels, load_labels
from sika_guard.benchmark.mockrun import mock_records
from sika_guard.benchmark.pilot import build_pilot
from sika_guard.benchmark.seal import LabelsLockedError, seal_path_for, validate_log, verify_gate, write_seal_file
from sika_guard.benchmark.visible import dump_visible
from sika_guard.coordinator import Condition

ROOT = Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


def written(tmp_path, *, seal=True, digest=None, directory=None, **kw):
    cases, labels = build_pilot()
    (tmp_path / "cases.jsonl").write_text(dump_visible(cases), encoding="utf-8")
    (tmp_path / "labels.jsonl").write_text(dump_labels(labels), encoding="utf-8")
    if digest is None:
        digest, directory = record_manifest(tmp_path)
    log = tmp_path / "decisions.jsonl"
    runner.write_run_from_records(log, tmp_path / "cases.jsonl", cases, mock_records(cases), mode="mock", manifest_digest=digest, seal=seal, manifests_dir=directory, **kw)
    return tmp_path / "cases.jsonl", tmp_path / "labels.jsonl", log, directory


def rewrite_declaration(log: Path, **changes):
    lines = log.read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    head.update(changes)
    log.write_text("\n".join([json.dumps(head, sort_keys=True), *lines[1:]]) + "\n", encoding="utf-8")


def test_r10_a_run_bound_to_a_recorded_manifest_that_describes_the_code_seals_and_opens_its_labels(tmp_path):
    cases, labels, log, directory = written(tmp_path)
    seal = json.loads(seal_path_for(log).read_text(encoding="utf-8"))
    assert seal["manifest_status"] == "development_baseline" and seal["mode"] == "mock" and len(seal["frozen_core_sha256"]) == 64
    assert len(load_labels(labels, log, cases, manifests_dir=directory)) == 12


def test_r10_a_digest_that_names_no_manifest_binds_nothing_and_no_call_is_made(tmp_path):
    cases, _ = build_pilot()
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(dump_visible(cases), encoding="utf-8")
    (tmp_path / "manifests").mkdir()
    spec = runner.ExperimentSpec(repeats=1, conditions=(Condition.B,))
    with pytest.raises(binding.ManifestBindingError, match="names no recorded"):
        run(runner.run_experiment(cases, cases_path, spec, tmp_path / "run.jsonl", mode="fake", config=runner.fake_config("quiet"),
                                  inner_transport=runner.offline_transport("quiet"), manifest_digest="0" * 64, manifests_dir=tmp_path / "manifests"))
    assert not (tmp_path / "run.jsonl").exists()  # refused BEFORE a log was opened, let alone a request sent


def test_r10_a_manifest_of_different_code_cannot_bind_a_run(tmp_path):
    stale = json.loads((ROOT / "manifests" / "manifest-0001-initial.json").read_text(encoding="utf-8"))
    directory = tmp_path / "manifests"
    directory.mkdir()
    (directory / "stale.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(binding.ManifestBindingError, match="does not describe the code that is running"):
        binding.bind(stale["digest"], mode="mock", conditions=["A"], manifests_dir=directory)


def test_r10_an_edited_manifest_file_is_not_a_recorded_manifest(tmp_path):
    digest, directory = record_manifest(tmp_path)
    path = next(directory.glob("*.json"))
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["status"] = "frozen"  # promote a development baseline to an approved freeze by hand
    path.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(binding.ManifestBindingError, match="names no recorded, untampered"):
        binding.bind(digest, mode="live", conditions=["A"], manifests_dir=directory)


def test_r10_a_live_run_needs_a_frozen_approved_manifest_and_a_development_baseline_never_qualifies(tmp_path):
    digest, directory = record_manifest(tmp_path)
    with pytest.raises(binding.ManifestBindingError, match="live run needs a frozen"):
        binding.bind(digest, mode="live", conditions=["A", "B"], manifests_dir=directory)
    assert binding.bind(digest, mode="mock", conditions=["A", "B"], manifests_dir=directory).status == "development_baseline"


def test_r10_a_manifest_can_be_frozen_only_by_name_and_only_from_a_clean_committed_tree(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="clean, committed Git tree"):
        mf.build_manifest(status="frozen", approved_by="A. Reviewer")  # this working tree has uncommitted changes
    with pytest.raises(ValueError, match="name of whoever approved"):
        mf.build_manifest(status="frozen")
    clean = {"git_available": True, "head": "abc", "head_matches_working_tree": True, "changed_or_untracked_paths": [], "note": ""}
    monkeypatch.setattr(mf, "working_tree_state", lambda root: clean)
    frozen = mf.build_manifest(status="frozen", approved_by="A. Reviewer")
    assert frozen["status"] == "frozen" and frozen["approval"]["approved_by"] == "A. Reviewer" and "not an authentication" in frozen["approval"]["statement"]
    directory = tmp_path / "manifests"
    mf.write_manifest(frozen, directory / "frozen.json")
    assert binding.bind(frozen["digest"], mode="live", conditions=["A", "B", "B3", "Bc", "C"], manifests_dir=directory).status == "frozen"


def test_r10_the_declared_conditions_must_be_defined_as_the_manifest_defines_them(tmp_path):
    cases, labels, log, directory = written(tmp_path, seal=False)
    rewrite_declaration(log, conditions_sha256="0" * 64)
    with pytest.raises(LabelsLockedError, match="condition definitions"):
        write_seal_file(log, cases, manifests_dir=directory)


def test_r10_the_declared_frozen_core_must_be_the_manifests(tmp_path):
    cases, labels, log, directory = written(tmp_path, seal=False)
    rewrite_declaration(log, frozen_core_sha256="1" * 64)
    with pytest.raises(LabelsLockedError, match="frozen-core hash"):
        write_seal_file(log, cases, manifests_dir=directory)


def test_r10_a_log_that_declares_no_binding_cannot_be_sealed(tmp_path):
    cases, labels, log, directory = written(tmp_path, seal=False)
    rewrite_declaration(log, manifest_status="", frozen_core_sha256="", conditions_sha256="")
    with pytest.raises(LabelsLockedError, match="not bound to a manifest"):
        write_seal_file(log, cases, manifests_dir=directory)


def test_r10_a_log_whose_status_overstates_its_manifest_cannot_be_sealed(tmp_path):
    cases, labels, log, directory = written(tmp_path, seal=False)
    rewrite_declaration(log, manifest_status="frozen")
    with pytest.raises(LabelsLockedError, match="status"):
        write_seal_file(log, cases, manifests_dir=directory)


def test_r10_a_live_declaration_bound_to_a_development_baseline_cannot_be_sealed(tmp_path):
    cases, labels, log, directory = written(tmp_path, seal=False)
    rewrite_declaration(log, mode="live")
    with pytest.raises(LabelsLockedError, match="frozen manifest only"):
        write_seal_file(log, cases, manifests_dir=directory)


def test_r10_source_edited_while_the_run_was_in_progress_cannot_be_sealed(tmp_path, monkeypatch):
    cases, labels, log, directory = written(tmp_path, seal=False)
    monkeypatch.setattr(binding, "current_frozen_core_digest", lambda root=None: "2" * 64)  # a frozen-core file changed after the run began
    with pytest.raises(LabelsLockedError, match="changed while the run was in progress"):
        write_seal_file(log, cases, manifests_dir=directory)
    assert not seal_path_for(log).exists()


def test_r10_a_seal_that_disagrees_with_its_declaration_keeps_the_labels_closed(tmp_path):
    cases, labels, log, directory = written(tmp_path)
    seal_file = seal_path_for(log)
    for key, value in (("manifest_status", "frozen"), ("manifest_digest", "3" * 64), ("frozen_core_sha256", "4" * 64), ("mode", "live")):
        original = seal_file.read_text(encoding="utf-8")
        obj = json.loads(original)
        obj[key] = value
        seal_file.write_text(json.dumps(obj), encoding="utf-8")
        with pytest.raises(LabelsLockedError):
            verify_gate(log, cases, manifests_dir=directory)
        seal_file.write_text(original, encoding="utf-8")
    assert verify_gate(log, cases, manifests_dir=directory)[1].summary.complete


def test_r10_a_deleted_manifest_keeps_the_labels_closed_at_scoring_time(tmp_path):
    cases, labels, log, directory = written(tmp_path)
    for path in directory.glob("*.json"):
        path.unlink()
    with pytest.raises(LabelsLockedError, match="not bound to a manifest that describes it"):
        load_labels(labels, log, cases, manifests_dir=directory)


def test_r10_scoring_after_the_tree_has_legitimately_moved_on_still_works_because_the_seal_recorded_the_run_time_binding(tmp_path, monkeypatch):
    cases, labels, log, directory = written(tmp_path)
    monkeypatch.setattr(binding, "current_frozen_core_digest", lambda root=None: "5" * 64)  # the code was edited after the run
    assert len(load_labels(labels, log, cases, manifests_dir=directory)) == 12


def test_r10_the_scorer_still_cannot_make_a_substitute_seal_after_labels_are_wanted(tmp_path):
    cases, labels, log, directory = written(tmp_path, seal=False)  # complete, bound, but never sealed by the run
    with pytest.raises(LabelsLockedError, match="no persisted seal"):
        load_labels(labels, log, cases, manifests_dir=directory)
    import sika_guard.benchmark.hidden as hidden

    assert not any(name in vars(hidden) for name in ("write_seal_file", "seal_file")), "the label module must not be able to seal"


def test_r10_the_exact_case_set_and_repeats_are_still_enforced(tmp_path):
    cases, labels, log, directory = written(tmp_path, seal=False)
    rewrite_declaration(log, repeats=3)
    with pytest.raises(LabelsLockedError, match="missing"):
        write_seal_file(log, cases, manifests_dir=directory)


def test_r10_the_runner_records_the_binding_and_seals_a_complete_run_only(tmp_path):
    cases, _ = build_pilot()
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(dump_visible(cases), encoding="utf-8")
    digest, directory = record_manifest(tmp_path)
    spec = runner.ExperimentSpec(repeats=2, conditions=(Condition.A, Condition.A_PLUS, Condition.B))
    done = run(runner.run_experiment(cases, cases_path, spec, tmp_path / "ok.jsonl", mode="fake", config=runner.fake_config("quiet"),
                                     inner_transport=runner.offline_transport("quiet"), manifest_digest=digest, manifests_dir=directory))
    log = validate_log(tmp_path / "ok.jsonl", cases_path, manifests_dir=directory)
    assert done.complete and done.seal is not None and log.declaration.manifest_status == "development_baseline"
    assert log.declaration.conditions == ("A", "A+", "B") and done.calls_made == 12 * 2  # A and A+ make no call and run once per case; B runs twice
    assert {r.condition.value for r in log.records} == {"A", "A+", "B"}


def test_r10_condition_a_plus_is_defined_in_the_manifest_with_the_scanners_own_source_hash():
    manifest = mf.build_manifest()
    scanner = manifest["conditions"]["A+"]["evaluators"][0]
    assert scanner["role"] == "deterministic" and scanner["prompt_version"] == "deterministic-scanner-v1" and len(scanner["system_prompt_sha256"]) == 64
    assert manifest["files"]["src/sika_guard/evaluators/scanner.py"]["group"] == "frozen_core"


def test_the_limit_is_stated():
    assert "malicious operator" in (binding.__doc__ or "") + (__import__("sika_guard.benchmark.seal", fromlist=["x"]).__doc__ or "")
    assert "not authentication" in (binding.__doc__ or "") + (__import__("sika_guard.benchmark.seal", fromlist=["x"]).__doc__ or "")
