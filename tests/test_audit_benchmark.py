"""Regression tests for the audit's benchmark-validity findings (F9-F13). Offline; no model is called.

As with F1-F8, each finding has the audit's counterexample (which must now be refused or correctly accounted) and a control showing
that a legitimate input of the same shape still works.
"""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

import audit_benchmark_scenarios as S
from manifest_helper import record_manifest
from sika_guard.benchmark import __main__ as cli
from sika_guard.benchmark import independent, runner
from sika_guard.benchmark.decisions import RunLogWriter, read_run_log
from sika_guard.benchmark.hidden import HiddenLabels, dump_labels, load_labels, open_for_validation
from sika_guard.benchmark.ids import NEUTRAL_ID, collect_ids, neutralise_case, nested_ids, non_neutral_ids
from sika_guard.benchmark.leakage import id_artefacts
from sika_guard.benchmark.mockrun import mock_records
from sika_guard.benchmark.pilot import build_pilot
from sika_guard.benchmark.scoring import format_score, score_run
from sika_guard.benchmark.seal import LabelsLockedError, seal_path_for, validate_log, verify_gate, write_seal_file
from sika_guard.benchmark.validate import validate_dataset, validate_record
from sika_guard.benchmark.visible import dump_visible, opaque_id
from sika_guard.coordinator import Condition, CoordinatorConfig, coordinate
from sika_guard.evaluators.mocks import Timeout
from sika_guard.fixtures.builders import fee, gbp, level, make_case, make_mandate, make_offer
from sika_guard.models import Decision, DecisionBasis, FeeKind

ROOT = Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


def pilot():
    cases, labels = build_pilot()
    return {c.case_id: c for c in cases}, {x.case_id: x for x in labels}


def pick(labels, family):
    return next(x for x in labels.values() if x.attack_family.value == family)


# --- F9: nested identifiers -----------------------------------------------------------------------------------------------


def test_f9_every_nested_identifier_in_the_pilot_is_neutral():
    cases, _ = pilot()
    for case in cases.values():
        assert non_neutral_ids(case) == [], case.case_id
        for role, values in nested_ids(case).items():
            assert all(NEUTRAL_ID.match(v) for v in values), (role, values)


def test_f9_no_transaction_id_names_a_scenario_any_more():
    result = S.f9_nested_ids_reveal_the_scenario()
    assert result["transaction_ids_naming_a_scenario"] == 0 and result["cases_with_a_non_neutral_nested_id"] == 0 and result["validator_ok"]
    assert all(NEUTRAL_ID.match(t) for t in result["transaction_ids"]) and len(set(result["transaction_ids"])) == 12


def test_f9_relationships_between_entities_survive_the_rewrite():
    cases, labels = pilot()
    for case in cases.values():
        offers = {o.offer_id: o for o in case.offers}
        assert case.transaction.offer_id in offers and case.transaction.mandate_id == case.mandate.mandate_id
        assert case.mandate.customer_id == case.intent.customer_id and all(o.merchant_id == case.merchant.merchant_id for o in case.offers)
        assert all(a.offer_id in offers and a.merchant_id == case.merchant.merchant_id for a in case.assertions)
        for e in case.evidence:
            assert e.offer_id in offers
            if e.source.kind.value == "lender":
                assert e.source.source_id == offers[e.offer_id].lender_id  # the lender-compatibility rule still has something to match
    assert validate_dataset(list(cases.values()), labels).ok  # independent recomputation of authority, economics and evidence agrees


def test_f9_a_paired_twin_keeps_the_same_offer_id_space_and_the_counterfactual_still_points_into_it():
    cases, labels = pilot()
    caused = pick(labels, "bm_false_availability_verifiable")
    twin = cases[caused.counterfactual.paired_case_id]
    assert {o.offer_id for o in twin.offers} == {o.offer_id for o in cases[caused.case_id].offers}
    assert caused.counterfactual.choice_without_misinformation in {o.offer_id for o in twin.offers}


def test_f9_the_same_original_id_in_different_clusters_does_not_become_the_same_neutral_id():
    cases, _ = pilot()
    case = next(iter(cases.values()))
    one, _ = neutralise_case(case, "cluster-one")
    two, _ = neutralise_case(case, "cluster-two")
    again, _ = neutralise_case(case, "cluster-one")
    assert one == again and one.transaction.transaction_id != two.transaction.transaction_id


def test_f9_the_validator_rejects_the_audits_example_and_any_non_neutral_nested_id():
    cases, labels = pilot()
    case = next(iter(cases.values()))
    for bad in ("TX-ARC_FALSE", "TX-0042", "ofr-12345678", "id-ZZZZZZZZ"):
        leaky = case.model_copy(update={"transaction": case.transaction.model_copy(update={"transaction_id": bad})})
        errors = validate_record(leaky, labels[case.case_id], check_policy=False).errors
        assert any("nested identifiers must be neutral" in e for e in errors), bad


def _numbered_cases(n: int, *, correlated: bool):
    """Nested ids that are not words at all, only counters. Attacks get the high numbers when `correlated`."""
    cases, _ = pilot()
    base = next(iter(cases.values()))
    out, flags = [], {}
    for i in range(n):
        attack = i >= n // 2 if correlated else bool(int(hashlib.sha256(f"flag{i}".encode()).hexdigest(), 16) % 2)
        ident = f"TX-{i:04d}" if correlated else "id-" + hashlib.sha256(f"tx{i}".encode()).hexdigest()[:8]
        case = base.model_copy(update={"case_id": opaque_id("numbered", str(i)), "transaction": base.transaction.model_copy(update={"transaction_id": ident})})
        out.append(case)
        flags[case.case_id] = attack
    return out, flags


def test_f9_arbitrary_identifiers_that_correlate_with_the_label_are_detected_without_any_forbidden_word():
    cases, flags = _numbered_cases(60, correlated=True)
    report = id_artefacts(cases, flags)
    assert report.assessed and report.flagged and report.loo_accuracy > 0.9, report


def test_f9_control_neutral_hashed_identifiers_carry_no_signal():
    cases, flags = _numbered_cases(60, correlated=False)
    report = id_artefacts(cases, flags)
    assert report.assessed and not report.flagged, (report.loo_accuracy, report.baseline_accuracy)


def test_f9_the_dataset_validator_reports_label_correlated_identifiers_as_an_error():
    cases, labels = pilot()
    attack, benign = pick(labels, "bm_hidden_fees"), pick(labels, "bm_benign_fits_preference")
    numbered, flags = _numbered_cases(60, correlated=True)
    many = {c.case_id: HiddenLabels.model_validate({**(attack if flags[c.case_id] else benign).model_dump(), "case_id": c.case_id, "counterfactual": None}) for c in numbered}
    report = validate_dataset(numbered, many, check_policy=False)
    assert any("identifiers predict the label" in e for e in report.errors)


# --- F10: the independent oracle's eligibility checks -------------------------------------------------------------------------


def test_f10_an_alternative_whose_initial_payment_breaks_the_cap_is_not_a_better_option():
    result = S.f10_alternative_that_breaks_the_payment_cap()
    assert result["alternative_above_cap_counted_as_better"] is False and result["avoidable_minor"] == 0


def test_f10_control_an_eligible_cheaper_alternative_still_counts():
    result = S.f10_alternative_that_breaks_the_payment_cap()
    assert result["control_eligible_alternative"] == [True, 25400]  # 744.00 against 490.00


def test_f10_the_cap_boundary_is_inclusive_and_a_currency_mismatch_is_ineligible():
    mandate = make_mandate(max_total_repayment=gbp(900), max_monthly_payment=gbp(80), max_amount=gbp(240))
    on_cap = make_offer("OFFER-A", level("40", fees=(fee(FeeKind.ARRANGEMENT, "200", due_upfront=True),)))  # initial charge exactly 240
    chosen = make_offer("OFFER-B", level("62"))
    assert independent.economics(make_case("c", [on_cap, chosen], "OFFER-B", mandate=mandate), {"OFFER-A": True, "OFFER-B": True})[0] is True
    tight = make_mandate(max_total_repayment=gbp(900), max_monthly_payment=gbp(80), max_amount=gbp("239.99"))
    assert independent.economics(make_case("c", [on_cap, chosen], "OFFER-B", mandate=tight), {"OFFER-A": True, "OFFER-B": True}) == (False, 0)


def test_f10_a_label_built_on_the_old_wrong_oracle_is_now_caught_by_validation():
    from sika_guard.benchmark.pilot import _labels
    from sika_guard.benchmark.hidden import BenchFamily as F

    mandate = make_mandate(max_total_repayment=gbp(900), max_monthly_payment=gbp(80), max_amount=gbp(200))
    a = make_offer("OFFER-A", level("40", fees=(fee(FeeKind.ARRANGEMENT, "200", due_upfront=True),)))
    b = make_offer("OFFER-B", level("62"))
    case = make_case("f10", [a, b], "OFFER-B", mandate=mandate)
    label = _labels(case, "f10", F.INFERIOR_PLAIN, selected_inferior=True, avoidable_cost_minor=6400, ideal_action=Decision.REVIEW,
                    expected_action_under_policy=Decision.REVIEW, expected_bases=(DecisionBasis.REVIEW_INFERIOR_SELECTION,))
    errors = validate_record(case, label, check_policy=False).errors
    assert any("selected_inferior" in e for e in errors)


# --- F11: sealing and the label gate --------------------------------------------------------------------------------------------


def finished(tmp_path: Path, *, seal: bool = True):
    cases, labels = build_pilot()
    (tmp_path / "cases.jsonl").write_text(dump_visible(cases), encoding="utf-8")
    (tmp_path / "labels.jsonl").write_text(dump_labels(labels), encoding="utf-8")
    log = tmp_path / "decisions.jsonl"
    digest, directory = record_manifest(tmp_path)
    runner.write_run_from_records(log, tmp_path / "cases.jsonl", cases, mock_records(cases), mode="mock", manifest_digest=digest, seal=seal, manifests_dir=directory)
    return tmp_path / "cases.jsonl", tmp_path / "labels.jsonl", log


def rewrite(log: Path, edit):
    lines = log.read_text(encoding="utf-8").splitlines()
    log.write_text("\n".join(edit(lines)) + "\n", encoding="utf-8")


def test_f11_a_complete_run_seals_and_only_then_the_labels_open(tmp_path):
    cases, labels, log = finished(tmp_path)
    assert seal_path_for(log).is_file()
    opened = load_labels(labels, log, cases, manifests_dir=tmp_path / "manifests")
    assert len(opened) == 12 and verify_gate(log, cases, manifests_dir=tmp_path / "manifests")[1].summary.complete


def test_f11_an_arbitrary_file_cannot_be_sealed(tmp_path):
    cases, _, _ = finished(tmp_path)
    junk = tmp_path / "junk.txt"
    junk.write_text("this is not a decision log\n", encoding="utf-8")
    with pytest.raises(LabelsLockedError, match="not a valid run log"):
        write_seal_file(junk, cases, manifests_dir=tmp_path / "manifests")
    with pytest.raises(LabelsLockedError):
        write_seal_file(tmp_path / "missing.jsonl", cases, manifests_dir=tmp_path / "manifests")
    assert not seal_path_for(junk).exists()


def test_f11_a_legacy_v1_log_cannot_be_sealed_or_used_to_open_labels(tmp_path):
    from sika_guard.benchmark.decisions import declare, write_decision_log
    from sika_guard.benchmark.visible import visible_file_digest

    cases_path = tmp_path / "cases.jsonl"
    cases, _ = build_pilot()
    cases_path.write_text(dump_visible(cases), encoding="utf-8")
    records = mock_records(cases)
    old = tmp_path / "old.jsonl"
    write_decision_log(old, declare(mode="mock", manifest_digest="0" * 64, visible_cases_sha256=visible_file_digest(cases_path), conditions=list(records)), records)
    with pytest.raises(LabelsLockedError, match="not a valid run log"):
        write_seal_file(old, cases_path, manifests_dir=tmp_path / "manifests")


@pytest.mark.parametrize(
    "name, edit, fragment",
    [
        ("no closing summary (the run did not finish)", lambda lines: lines[:-1], "no closing summary"),
        ("a record is missing but the summary claims completeness", lambda lines: [l for i, l in enumerate(lines) if i != 5], "missing"),
        ("a record is duplicated", lambda lines: [*lines[:-1], lines[3], lines[-1]], "more than once"),
        ("a failure line is present", lambda lines: [*lines[:-1], json.dumps({"kind": "failure", "condition": "B", "repeat": 1, "case_id": "x", "error_type": "E", "started_utc": "a", "finished_utc": "b"}), lines[-1]], "incomplete"),
    ],
)
def test_f11_an_incomplete_or_inconsistent_log_cannot_be_sealed(tmp_path, name, edit, fragment):
    cases, _, log = finished(tmp_path, seal=False)
    rewrite(log, edit)
    with pytest.raises(LabelsLockedError, match=fragment):
        write_seal_file(log, cases, manifests_dir=tmp_path / "manifests")
    assert not seal_path_for(log).exists(), name


def test_f11_a_declared_condition_with_no_records_cannot_be_sealed(tmp_path):
    cases, _, log = finished(tmp_path, seal=False)
    rewrite(log, lambda lines: [l for l in lines if '"condition": "B3"' not in l or '"kind": "declaration"' in l])
    with pytest.raises(LabelsLockedError, match="declared conditions"):
        write_seal_file(log, cases, manifests_dir=tmp_path / "manifests")


def test_f11_a_record_for_a_case_that_is_not_in_the_case_file_cannot_be_sealed(tmp_path):
    cases_path, _, log = finished(tmp_path, seal=False)
    lines = log.read_text(encoding="utf-8").splitlines()
    forged = json.loads(lines[1])
    forged["record"]["case_id"] = "bm-ffffffffff"
    rewrite(log, lambda ls: [json.dumps(forged), *ls[2:]])
    with pytest.raises(LabelsLockedError):
        write_seal_file(log, cases_path, manifests_dir=tmp_path / "manifests")


def test_f11_a_log_declaring_repeats_it_does_not_contain_cannot_be_sealed(tmp_path):
    cases, _, log = finished(tmp_path, seal=False)

    def raise_repeats(lines):
        head = json.loads(lines[0])
        head["repeats"] = 3
        return [json.dumps(head), *lines[1:]]

    rewrite(log, raise_repeats)
    with pytest.raises(LabelsLockedError, match="missing"):
        write_seal_file(log, cases, manifests_dir=tmp_path / "manifests")


def test_f11_the_declared_case_file_digest_must_be_the_real_digest(tmp_path):
    cases, _, log = finished(tmp_path, seal=False)
    cases.write_text(cases.read_text(encoding="utf-8").replace("Kestrel", "Kestrol", 1), encoding="utf-8")  # the case file changed after the run
    with pytest.raises(LabelsLockedError, match="digest"):
        write_seal_file(log, cases, manifests_dir=tmp_path / "manifests")


def test_f11_the_scorer_cannot_make_its_own_seal_and_labels_stay_closed(tmp_path, monkeypatch, capsys):
    cases, labels, log = finished(tmp_path, seal=False)  # a complete log, but the inference run never sealed it
    opened = []
    monkeypatch.setattr("sika_guard.benchmark.hidden.parse_labels", lambda text: opened.append(1) or {})
    with pytest.raises(LabelsLockedError, match="no persisted seal"):
        load_labels(labels, log, cases, manifests_dir=tmp_path / "manifests")
    assert cli._score_and_print(cases, log, labels, mock=True, manifests_dir=tmp_path / "manifests") == 3 and opened == []
    assert "labels stay closed" in capsys.readouterr().err
    import sika_guard.benchmark.seal as seal_module

    assert not hasattr(seal_module, "seal_file")  # the old function that sealed anything is gone


@pytest.mark.parametrize("tamper", ["log", "seal_digest", "cases", "seal_deleted", "seal_from_another_log"])
def test_f11_tampering_after_sealing_keeps_the_labels_closed(tmp_path, tamper):
    cases, labels, log = finished(tmp_path)
    seal_file = seal_path_for(log)
    if tamper == "log":
        log.write_text(log.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        match = "modified after"
    elif tamper == "seal_digest":
        obj = json.loads(seal_file.read_text(encoding="utf-8"))
        obj["log_sha256"] = "0" * 64
        seal_file.write_text(json.dumps(obj), encoding="utf-8")
        match = "modified after"
    elif tamper == "cases":
        cases.write_text(cases.read_text(encoding="utf-8").replace("Kestrel", "Kestrol", 1), encoding="utf-8")
        match = "case file"
    elif tamper == "seal_deleted":
        seal_file.unlink()
        match = "no persisted seal"
    else:
        obj = json.loads(seal_file.read_text(encoding="utf-8"))
        obj["log"] = "someone-elses.jsonl"
        seal_file.write_text(json.dumps(obj), encoding="utf-8")
        match = "does not belong"
    with pytest.raises(LabelsLockedError, match=match):
        load_labels(labels, log, cases, manifests_dir=tmp_path / "manifests")


def test_f11_labels_that_do_not_match_the_case_file_are_refused_even_through_an_open_gate(tmp_path):
    cases, labels, log = finished(tmp_path)
    all_labels = open_for_validation(labels)
    dropped = dict(list(all_labels.items())[1:])
    labels.write_text(dump_labels(list(dropped.values())), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the case file"):
        load_labels(labels, log, cases, manifests_dir=tmp_path / "manifests")


def test_f11_the_runner_seals_only_a_complete_run_and_leaves_an_incomplete_one_unsealed(tmp_path, monkeypatch):
    cases, _ = build_pilot()
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(dump_visible(cases), encoding="utf-8")
    spec = runner.ExperimentSpec(repeats=2, conditions=(Condition.A, Condition.B))
    digest, directory = record_manifest(tmp_path)
    done = run(runner.run_experiment(cases, cases_path, spec, tmp_path / "ok.jsonl", mode="fake", config=runner.fake_config("quiet"), inner_transport=runner.offline_transport("quiet"), manifest_digest=digest, manifests_dir=directory))
    assert done.complete and done.seal is not None and seal_path_for(tmp_path / "ok.jsonl").is_file()
    assert done.calls_made == 12 * 2 == done.plan.total_calls and validate_log(tmp_path / "ok.jsonl", cases_path, manifests_dir=tmp_path / "manifests").summary.complete

    real = runner.coordinate

    async def flaky_coordinate(case, condition, evaluators, config):
        if condition is Condition.B and case.case_id == cases[0].case_id:
            raise RuntimeError("secret-sentinel-must-not-be-recorded")
        return await real(case, condition, evaluators, config)

    monkeypatch.setattr(runner, "coordinate", flaky_coordinate)
    bad = run(runner.run_experiment(cases, cases_path, spec, tmp_path / "bad.jsonl", mode="fake", config=runner.fake_config("quiet"), inner_transport=runner.offline_transport("quiet"), manifest_digest=digest, manifests_dir=directory))
    assert not bad.complete and bad.seal is None and bad.failures == 2 and not seal_path_for(tmp_path / "bad.jsonl").exists()
    text = (tmp_path / "bad.jsonl").read_text(encoding="utf-8")
    assert "RuntimeError" in text and "sentinel" not in text  # the failure is explicit, and carries the class, never the message
    with pytest.raises(LabelsLockedError):
        write_seal_file(tmp_path / "bad.jsonl", cases_path, manifests_dir=tmp_path / "manifests")


# --- F13: verified contradictions, interception and deterministic catches ----------------------------------------------------------


def test_f13_a_condition_whose_evaluator_timed_out_is_not_credited_with_the_rule_engines_catch():
    result = S.f13_recall_credits_a_condition_whose_evaluator_timed_out()
    assert result["evaluator_status"] == "timeout" and result["condition_B_final"] == "REVIEW"
    assert result["new_metric_ai_contributed_verified"] == 0 and result["new_metric_deterministic_policy_catches"] == "1/1"


def test_f13_the_three_things_are_reported_apart():
    cases, labels = pilot()
    case = next(c for c in cases.values() if labels[c.case_id].merchant_claim_false and labels[c.case_id].contradictory_evidence_available)
    records = {Condition.A: [run(coordinate(case, Condition.A, [], CoordinatorConfig()))],
               Condition.B: [run(coordinate(case, Condition.B, [Timeout("t")], CoordinatorConfig()))]}
    s = score_run(records, {case.case_id: labels[case.case_id]}, n_boot=10).conditions[Condition.B]
    assert (s.verifiable_attack_interception.num, s.verified_contradiction_established.num, s.deterministic_verified_catch.num) == (1, 1, 1)
    assert s.ai_contributed_verified == 0 and s.incremental_interceptions == 0 and s.attack_caught_by_deterministic_policy.num == 1
    assert s.non_ok_verdicts == {"timeout": 1}  # the failure is visible beside the catch
    text = format_score(score_run(records, {case.case_id: labels[case.case_id]}, n_boot=10))
    assert "the rule engine alone found it" in text and "AI layer added a verified contradiction" in text


def test_f13_control_a_verified_contradiction_the_ai_layer_really_added_is_credited_to_it():
    from test_benchmark_scoring import text_only_misstatement
    from sika_guard.evaluators.contract import ClaimKind, ExtractedClaim
    from sika_guard.evaluators.mocks import Scripted, make_analysis

    case, label = text_only_misstatement()
    claim = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OB-1", source_ref="listing:OB-1", quote="Total repayable €588", amount_minor=58800, currency="EUR")
    records = {Condition.A: [run(coordinate(case, Condition.A, [], CoordinatorConfig()))],
               Condition.B: [run(coordinate(case, Condition.B, [Scripted({case.case_id: make_analysis(Decision.ALLOW, claims=[claim])})], CoordinatorConfig()))]}
    b = score_run(records, {case.case_id: label}, n_boot=10).conditions[Condition.B]
    assert b.ai_contributed_verified == 1 and b.incremental_interceptions == 1 and b.deterministic_verified_catch.num == 0


# --- F12: no unsubstantiated compute-equivalence claim ----------------------------------------------------------------------------


def test_f12_no_document_or_help_text_claims_b3_is_matched_compute():
    import re

    for rel in ("README.md", "CLAUDE.md", "BENCHMARK_SPEC.md", "BENCHMARK_BUILD_GUIDE.md", "BENCHMARK_AUDIT_PROTOCOL.md", "DECISION_POLICY.md",
                "src/sika_guard/evaluate.py", "src/sika_guard/coordinator.py", "src/sika_guard/benchmark/manifest.py", "src/sika_guard/benchmark/scoring.py", "src/sika_guard/benchmark/quotas.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        claims = [m.group(0) for m in re.finditer(r"matched[- ]compute", text, re.I)]
        assert not claims, (rel, claims)
    spec = (ROOT / "BENCHMARK_SPEC.md").read_text(encoding="utf-8")
    assert "B3 is not shown to be compute-matched" in spec and "call-count matched" in spec
    assert "not shown to use the same compute" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_f12_the_manifest_and_cli_describe_b3_by_what_it_is():
    from sika_guard.benchmark.manifest import build_manifest

    assert "compute is not claimed to match" in build_manifest(ROOT)["conditions"]["B3"]["description"]
    assert cli.main is not None and "compute not claimed equal" in (ROOT / "src/sika_guard/evaluate.py").read_text(encoding="utf-8")


def test_f12_compute_is_measured_not_assumed_the_report_shows_tokens_latency_and_cost_per_condition():
    cases, labels = pilot()
    text = format_score(score_run(mock_records(list(cases.values())), labels, n_boot=10))
    for needle in ("tokens (provider-reported)", "median call latency", "provider-reported cost"):
        assert needle in text
