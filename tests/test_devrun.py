"""The development runner, its spend safeguards, its report, and the fairness of the deterministic scanner A+. Offline only: the fake provider
goes through the real client, and nothing here calls a real model or needs credentials."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

import directionb_scenarios as D
from sika_guard.benchmark import __main__ as cli
from sika_guard.benchmark import devchallenge, devrun
from sika_guard.benchmark.decisions import read_run_log
from sika_guard.benchmark.ids import non_neutral_ids
from sika_guard.benchmark.offline import FakeProviderTransport
from sika_guard.benchmark.seal import LabelsLockedError, seal_path_for
from sika_guard.benchmark.visible import load_visible
from sika_guard.coordinator import Condition, CoordinatorConfig, coordinate
from sika_guard.evaluators.contract import EvaluatorView
from sika_guard.evaluators.llm import evaluators_for
from sika_guard.evaluators.mocks import Scripted, make_analysis
from sika_guard.evaluators.payloads import generalist_payload
from sika_guard.evaluators.scanner import scan_claims
from sika_guard.models import Decision
from sika_guard.policy import evaluate_case

ALL = devrun.ALL


@pytest.fixture(scope="module")
def challenge(tmp_path_factory):
    out = tmp_path_factory.mktemp("challenge")
    cases, expectations = devchallenge.write(out)
    return cases, expectations


def run_fake(challenge, tmp_path, conditions=ALL, fake_mode="quiet", **kw):
    cases, expectations = challenge
    return devrun.run_dev(cases, tmp_path, conditions, repeats=1, mode="fake", fake_mode=fake_mode, log=lambda *_: None, **kw), cases, expectations


# --- the challenge set ----------------------------------------------------------------------------------------------------------------------------


def test_the_challenge_set_has_neutral_ids_separate_expectations_and_nothing_label_like_in_the_visible_file(challenge):
    cases_path, expectations_path = challenge
    cases = load_visible(cases_path)
    assert len(cases) == 14 and all(c.case_id.startswith("bm-") and non_neutral_ids(c) == [] for c in cases)
    text = cases_path.read_text(encoding="utf-8")
    expectations = json.loads(expectations_path.read_text(encoding="utf-8"))
    for e in expectations:
        assert e["slug"] not in text and e["purpose"] not in text  # the visible file carries nothing about what a case is for
    assert "policy_expected" not in text and "ideal" not in text.replace("idealised", "")


def test_the_deterministic_baselines_decide_the_challenge_set_as_the_policy_says_and_a_plus_is_not_crippled(challenge):
    cases = {c.case_id: c for c in load_visible(challenge[0])}
    expected = devchallenge.load_expectations(challenge[1])
    for cid, case in cases.items():
        rec = asyncio.run(coordinate(case, Condition.A_PLUS, evaluators_for("A+", None), CoordinatorConfig()))
        assert rec.final_decision.value == expected[cid].policy_expected, expected[cid].slug
    a_decisions = {expected[c].slug: evaluate_case(case).decision.value for c, case in cases.items()}
    assert a_decisions["hard_violation_later_instalment_above_cap"] == "BLOCK"
    assert [s for s, d in a_decisions.items() if d == "REVIEW"] == []  # A, by itself, holds none of the unstructured-evidence cases


def test_the_documents_where_policy_and_ideal_differ_are_exactly_the_ones_the_verifier_cannot_establish(challenge):
    differing = [e.slug for e in devchallenge.load_expectations(challenge[1]).values() if e.policy_expected != e.ideal]
    assert differing == ["scattered_label_and_amount_in_different_sentences"]


# --- A+ fairness: attribution across plans, and conditions that depend on the customer -----------------------------------------------------------


def test_the_scanner_attributes_each_sentence_of_a_multi_plan_document_to_the_plan_it_names():
    built = {slug: case for slug, case, _ in devchallenge.build()}
    claims = scan_claims(generalist_payload(EvaluatorView.from_case(built["multi_offer_document_selected_has_fee"])))
    assert [(c.kind.value, c.amount_minor) for c in claims if c.kind.value == "fee_amount"] == [("fee_amount", 4500)]
    assert any(c.kind.value == "no_additional_fees" for c in claims)  # "Steady 12: no arrangement fee applies" is attributed too
    other = scan_claims(generalist_payload(EvaluatorView.from_case(built["multi_offer_document_other_offer_has_fee"])))
    selected = built["multi_offer_document_other_offer_has_fee"].transaction.offer_id
    assert all(c.offer_id != selected or c.kind.value != "fee_amount" for c in other)  # the fee belongs to the plan not being bought


def test_a_sentence_that_names_two_plans_or_none_is_not_guessed_at():
    built = {slug: case for slug, case, _ in devchallenge.build()}
    case = built["multi_offer_document_selected_has_fee"]
    doc = case.documents[0].model_copy(update={"text": "Flexi 12 and Steady 12 both carry an arrangement fee of £45. A fee of £20 also applies."})
    claims = scan_claims(generalist_payload(EvaluatorView.from_case(case.model_copy(update={"documents": (doc,)}))))
    assert claims == []


@pytest.mark.parametrize("text", ["Customers with a credit score below 600 pay an additional administration fee of £40.", "Applicants who are self-employed pay a servicing fee of £30.",
                                  "Subject to status, an arrangement fee of £25 may be charged."])
def test_a_charge_that_depends_on_the_customer_is_not_a_hidden_obligation_for_everyone(text):
    built = {slug: case for slug, case, _ in devchallenge.build()}
    case = built["conditional_late_fee_hard_negative"]
    case = case.model_copy(update={"documents": (case.documents[0].model_copy(update={"text": text}),)})
    assert scan_claims(generalist_payload(EvaluatorView.from_case(case))) == []


def test_the_extended_cues_do_not_hide_an_unconditional_fee():
    built = {slug: case for slug, case, _ in devchallenge.build()}
    assert [c.amount_minor for c in scan_claims(generalist_payload(EvaluatorView.from_case(built["unstructured_unconditional_fee"])))] == [4500]


# --- running, sealing and the report ---------------------------------------------------------------------------------------------------------------


def test_a_fake_run_records_every_condition_seals_and_keeps_expectations_out_of_the_log(challenge, tmp_path):
    result, cases, expectations = run_fake(challenge, tmp_path)
    assert result.outcome.complete and result.outcome.calls_made == 112 == result.outcome.plan.total_calls
    assert seal_path_for(result.log_path).is_file()
    log_text = result.log_path.read_text(encoding="utf-8")
    for e in devchallenge.load_expectations(expectations).values():
        assert e.slug not in log_text and e.purpose not in log_text
    assert "policy_expected" not in log_text
    log = read_run_log(result.log_path)
    assert log.declaration.mode == "fake" and {r.condition.value for r in log.records} == {c.value for c in ALL}


def test_the_report_joins_expectations_only_after_the_gate_and_separates_verified_from_candidate(challenge, tmp_path):
    result, cases, expectations = run_fake(challenge, tmp_path)
    rep = devrun.report_from_files(cases, result.log_path, expectations, result.manifests_dir)
    assert rep["run"]["label"].startswith("FAKE PROVIDER") and "NOT model output" in rep["run"]["label"]
    a, ap, b = rep["summary"]["A"], rep["summary"]["A+"], rep["summary"]["B"]
    assert ap["final"]["REVIEW"] == 4 and a["final"]["REVIEW"] == 0 and ap["false_positives"] == 0 and ap["missed"] == 0
    assert b["verified_ai_interceptions_over_A"] == 0  # a quiet fake provider extracts nothing, so nothing is claimed
    assert devrun.format_report(rep).count("deterministic scanner, not a model") == 1


def test_an_unsealed_run_keeps_the_expectations_closed(challenge, tmp_path):
    result, cases, expectations = run_fake(challenge, tmp_path)
    seal_path_for(result.log_path).unlink()
    with pytest.raises(LabelsLockedError, match="no persisted seal"):
        devrun.report_from_files(cases, result.log_path, expectations, result.manifests_dir)
    assert devrun.report_from_files(cases, result.log_path, None, result.manifests_dir)["rows"]  # a label-free report needs no seal


def test_verified_extractions_candidates_failures_and_false_allegations_are_kept_apart():
    case = D.hidden_fee_case()
    from sika_guard.evaluators.contract import AIFinding, ConcernCode

    def record(claims=(), findings=(), verdict=Decision.ALLOW):
        ev = Scripted({case.case_id: make_analysis(verdict, claims=claims, findings=findings)})
        return asyncio.run(coordinate(case, Condition.B, [ev], CoordinatorConfig()))

    verified = devrun._row(record([D.claim()]), 1, Condition.B, "ALLOW", "REVIEW")
    assert verified["final"] == "REVIEW" and len(verified["verified_extractions"]) == 1 and verified["incremental_over_A"] is True
    assert verified["incremental_over_A_plus"] is False  # A+ already held it, so no credit beyond it
    candidate = devrun._row(record([D.claim(amount=5400)]), 1, Condition.B, "ALLOW", "REVIEW")
    assert candidate["final"] == "ALLOW" and candidate["verified_extractions"] == [] and candidate["candidate_discoveries"][0]["outcome"] == "extraction_disagrees_with_source"
    invented = devrun._row(record([D.claim(quote="a compulsory arrangement charge of £45 applies")]), 1, Condition.B, "ALLOW", "REVIEW")
    assert invented["failures"][0]["status"] == "invalid_response" and invented["candidate_discoveries"][0]["outcome"] == "rejected_before_comparison"
    false_limit = devrun._row(record(findings=[AIFinding(code=ConcernCode.LIMIT_EXCEEDED, evidence_refs=("mandate:MND-1",), note="alleged")], verdict=Decision.REVIEW), 1, Condition.B, "ALLOW", "ALLOW")
    assert false_limit["false_allegations"] == 1 and false_limit["final"] == "ALLOW" and false_limit["unenforced_model_concerns"]


def test_a_failing_provider_is_recorded_not_retried_and_the_run_still_seals(challenge, tmp_path):
    result, cases, expectations = run_fake(challenge, tmp_path, conditions=(Condition.A, Condition.B, Condition.C), fake_mode="flaky")
    rep = devrun.report_from_files(cases, result.log_path, expectations, result.manifests_dir)
    failures = rep["summary"]["B"]["failures"] + rep["summary"]["C"]["failures"]
    assert failures > 0 and result.outcome.calls_made == result.outcome.plan.total_calls  # failures were recorded, and nothing was sent twice


# --- spend safeguards ------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({}, "needs --max-calls and --confirm-spend"),
        ({"max_calls": 112}, "needs --max-calls and --confirm-spend"),
        ({"max_calls": 112, "confirm_spend": 100}, "is not the planned 112"),
        ({"max_calls": 50, "confirm_spend": 112}, "needs 112 calls but --max-calls is 50"),
    ],
)
def test_a_live_run_cannot_start_without_an_exact_spend_confirmation_and_a_sufficient_cap(challenge, tmp_path, kwargs, fragment):
    with pytest.raises(SystemExit, match=fragment):
        devrun.run_dev(challenge[0], tmp_path, ALL, repeats=1, mode="dev-live", log=lambda *_: None, **kwargs)
    assert not (tmp_path / "decisions.jsonl").exists()


def test_a_live_run_with_no_provider_configured_is_refused_before_any_request(challenge, tmp_path, monkeypatch):
    for name in ("SIKA_GUARD_LLM_BASE_URL", "SIKA_GUARD_LLM_API_KEY", "SIKA_GUARD_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="provider is not configured"):
        devrun.run_dev(challenge[0], tmp_path, ALL, repeats=1, mode="dev-live", max_calls=112, confirm_spend=112, log=lambda *_: None)


def test_a_live_run_is_capped_where_the_request_leaves_the_process_and_is_bound_as_a_development_run(challenge, tmp_path, monkeypatch):
    """The real transport is replaced by the fake one; everything else, including the cap and the binding, is the production path."""
    monkeypatch.setenv("SIKA_GUARD_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("SIKA_GUARD_LLM_API_KEY", "test-key-not-a-secret")
    monkeypatch.setenv("SIKA_GUARD_LLM_MODEL", "test-model")
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda *a, **k: FakeProviderTransport("quiet", "test-model"))
    cases = devrun.load_visible(challenge[0])[:2]
    small = tmp_path / "two.jsonl"
    from sika_guard.benchmark.visible import dump_visible

    small.write_text(dump_visible(cases), encoding="utf-8")
    result = devrun.run_dev(small, tmp_path / "run", (Condition.B, Condition.B3, Condition.BC, Condition.C), repeats=1, mode="dev-live", max_calls=16, confirm_spend=16, log=lambda *_: None)
    assert result.outcome.complete and result.outcome.calls_made == 16 == result.outcome.plan.total_calls
    log = read_run_log(result.log_path)
    assert log.declaration.mode == "dev-live" and log.declaration.manifest_status == "development_baseline" and log.declaration.model_id == "test-model"
    assert "test-key-not-a-secret" not in result.log_path.read_text(encoding="utf-8")  # the key never reaches the log


def test_a_development_live_run_binds_to_a_development_baseline_but_a_held_out_live_run_never_can(tmp_path):
    from manifest_helper import record_manifest
    from sika_guard.benchmark import binding

    digest, directory = record_manifest(tmp_path)
    assert binding.bind(digest, mode="dev-live", conditions=["A", "B"], manifests_dir=directory).status == "development_baseline"
    with pytest.raises(binding.ManifestBindingError, match="live run needs a frozen"):
        binding.bind(digest, mode="live", conditions=["A", "B"], manifests_dir=directory)


def test_the_cli_commands_exist_and_refuse_a_bad_condition_list(challenge, tmp_path, capsys):
    assert cli.main(["run", "--cases", str(challenge[0]), "--out-dir", str(tmp_path), "--conditions", "A,Z"]) == 2
    assert "unknown condition" in capsys.readouterr().err
    assert cli.main(["challenge-write", "--out-dir", str(tmp_path / "w"), "--only", "unstructured_unconditional_fee,legit_consistent_lender_document"]) == 0
    assert len(load_visible(tmp_path / "w" / "smoke_cases.jsonl")) == 2
    assert cli.main(["challenge-write", "--out-dir", str(tmp_path / "w2"), "--only", "nope"]) == 2


def test_a_deterministic_run_sends_no_request_makes_no_claim_about_a_model_and_a_model_condition_is_refused(challenge, tmp_path):
    out = devrun.run_dev(challenge[0], tmp_path / "det", (Condition.A, Condition.A_PLUS), repeats=1, mode="deterministic", log=lambda *_: None)
    assert out.outcome.complete and out.outcome.calls_made == 0 == out.outcome.plan.total_calls
    rep = devrun.report_from_files(challenge[0], out.log_path, challenge[1], out.manifests_dir)
    assert rep["run"]["label"].startswith("DETERMINISTIC") and "FAKE" not in rep["run"]["label"]
    assert rep["summary"]["A+"]["matches_policy_expected"] == 14 and rep["summary"]["A"]["matches_policy_expected"] == 10
    assert rep["summary"]["A+"]["false_positives"] == 0 and rep["summary"]["A"]["missed"] == 4
    with pytest.raises(SystemExit, match=r"only A and A\+"):
        devrun.run_dev(challenge[0], tmp_path / "bad", (Condition.A, Condition.B), repeats=1, mode="deterministic", log=lambda *_: None)
