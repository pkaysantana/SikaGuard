"""Scoring, statistics, reporting and the CLI."""

import asyncio
import json

import pytest

from sika_guard import evaluate
from sika_guard.coordinator import Condition
from sika_guard.evaluators.mocks import AlwaysReview, Timeout
from sika_guard.experiment import (
    P95_MIN_SAMPLES,
    RunMetadata,
    describe_ms,
    export_json,
    format_experiment_report,
    percentile,
    run_condition,
    score,
)
from sika_guard.fixtures.cases import build_cases
from sika_guard.fixtures.labels import LABELS
from sika_guard.models import Decision

from fakes import SECRET, FakeProvider, analysis_json, completion

META = RunMetadata(condition=Condition.B, mode="offline-mock:review", evaluators=(), llm=None, fail_closed=False, policy={})


def run(coro):
    return asyncio.run(coro)


def scored_with(evaluators, condition=Condition.B):
    return score(run(run_condition(build_cases(), condition, evaluators)), LABELS)


# --- statistics -------------------------------------------------------------


def test_percentile_is_nearest_rank():
    assert percentile(list(range(1, 21)), 0.95) == 19
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([3, 1, 2], 0.5) == 2


def test_small_samples_are_labelled_descriptive_and_show_no_p95():
    text = describe_ms([10.0, 20.0, 30.0])
    assert "descriptive only, n=3" in text and "p95" not in text


def test_p95_appears_only_once_the_sample_is_large_enough():
    text = describe_ms([float(i) for i in range(P95_MIN_SAMPLES)])
    assert "p95" in text and "descriptive" not in text
    assert describe_ms([]) == "n/a (no samples)"


# --- scoring ----------------------------------------------------------------


def test_a_model_that_always_says_review_changes_nothing_because_a_verdict_alone_enforces_nothing():
    scored = scored_with([AlwaysReview()])
    harmful = [s for s in scored if s.harmful]
    benign = [s for s in scored if s.benign]
    assert (len(harmful), len(benign)) == (12, 8)
    assert [s.intercepted for s in scored] == [s.intercepted_by_baseline for s in scored]  # nothing is newly intercepted
    assert [s.falsely_escalated for s in scored] == [s.falsely_escalated_by_baseline for s in scored]  # and nothing newly escalated
    assert not any(s.record.escalated_by_ai for s in scored)
    assert all(s.record.final_decision is s.record.deterministic_decision for s in scored)


def test_scoring_joins_labels_only_after_inference():
    records = run(run_condition(build_cases(), Condition.B, [AlwaysReview()]))
    assert "ideal_decision" not in records[0].model_dump_json()
    assert all(s.truth.case_id == s.record.case_id for s in score(records, LABELS))


def test_deterministic_blocks_are_reported_as_preserved():
    report = format_experiment_report(META, scored_with([AlwaysReview(), Timeout()]))
    assert "Deterministic BLOCKs preserved: 4 of 4" in report


# --- report honesty ---------------------------------------------------------


def test_offline_report_says_it_is_not_model_output_and_makes_no_superiority_claim():
    report = format_experiment_report(META, scored_with([AlwaysReview()]))
    assert "NOT model output" in report and "OFFLINE MOCK" in report
    assert "model calls: none were made" in report and "no model calls, so no usage" in report
    assert "not a representative benchmark".lower() in report.lower()
    assert "No comparison with a matched-compute single-model baseline" in report


def test_failures_and_abstentions_are_reported_separately_from_disagreement():
    report = format_experiment_report(META, scored_with([Timeout()]))
    assert "timeout=20" in report and "unavailable=20" in report
    assert "cases with at least one abstaining or failed evaluator: 20" in report
    assert "disagree with each other: 0" in report


def _usage_on_every_other_call():
    calls = iter(range(10_000))

    def responder(role, body):
        usage = {"prompt_tokens": 100, "completion_tokens": 20} if next(calls) % 2 == 0 else None
        return completion(analysis_json("ALLOW"), usage=usage)

    return responder


def test_usage_is_reported_only_where_the_provider_returned_it():
    provider = FakeProvider(_usage_on_every_other_call())

    async def go():
        async with provider.client() as client:
            from sika_guard.evaluators.llm import generalist

            return await run_condition(build_cases(), Condition.B, generalist(client))

    meta = RunMetadata(Condition.B, "live", (), {"model": "test-model"}, False, {})
    report = format_experiment_report(meta, score(run(go()), LABELS))
    assert "tokens reported on 10 of 20 calls: prompt 1000, completion 200" in report
    assert "cost: not reported" in report  # no prices configured, so nothing is invented
    assert "model call generalist: median" in report and "Mode: LIVE" in report


def test_export_contains_scoring_flags_but_no_credentials(tmp_path):
    provider = FakeProvider()

    async def go():
        async with provider.client() as client:
            from sika_guard.evaluators.llm import generalist

            return await run_condition(build_cases()[:3], Condition.B, generalist(client))

    path = tmp_path / "out.json"
    export_json(path, RunMetadata(Condition.B, "live", (), provider.client().config.describe(), False, {}), score(run(go()), LABELS))
    text = path.read_text(encoding="utf-8")
    assert SECRET not in text
    data = json.loads(text)
    assert len(data["records"]) == 3 and "scoring" in data["records"][0] and data["meta"]["llm"]["model"] == "test-model"


# --- CLI --------------------------------------------------------------------


def test_default_cli_output_is_still_the_original_baseline_report(capsys):
    assert evaluate.main([]) == 0
    assert capsys.readouterr().out.strip() == evaluate.format_report(evaluate.run_baseline()).strip()


def test_offline_cli_runs_both_conditions_without_credentials(capsys, monkeypatch):
    for name in ("SIKA_GUARD_LLM_BASE_URL", "SIKA_GUARD_LLM_API_KEY", "SIKA_GUARD_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for condition in ("B", "C"):
        assert evaluate.main(["--condition", condition, "--offline", "review", "--limit", "4", "--cases"]) == 0
    out = capsys.readouterr().out
    assert "Condition B" in out and "Condition C" in out and "mock-review-merchant_claim" in out


def test_live_cli_without_credentials_names_the_missing_variables(capsys, monkeypatch):
    for name in ("SIKA_GUARD_LLM_BASE_URL", "SIKA_GUARD_LLM_API_KEY", "SIKA_GUARD_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert evaluate.main(["--condition", "C"]) == 2
    err = capsys.readouterr().err
    for name in ("SIKA_GUARD_LLM_BASE_URL", "SIKA_GUARD_LLM_API_KEY", "SIKA_GUARD_LLM_MODEL"):
        assert name in err


def test_cli_rejects_nonsensical_combinations(capsys):
    with pytest.raises(SystemExit):
        evaluate.main(["--offline", "allow"])  # Condition A takes no evaluators
    with pytest.raises(SystemExit):
        evaluate.main(["--condition", "B", "--offline", "allow", "--case", "no-such-case"])


def test_live_cli_path_end_to_end_with_a_fake_provider_and_no_secret_in_output(capsys, monkeypatch, tmp_path):
    provider = FakeProvider(lambda role, body: completion(analysis_json("ALLOW"), usage={"prompt_tokens": 1_000_000, "completion_tokens": 500_000}))
    monkeypatch.setenv("SIKA_GUARD_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("SIKA_GUARD_LLM_API_KEY", SECRET)
    monkeypatch.setenv("SIKA_GUARD_LLM_MODEL", "test-model")
    monkeypatch.setenv("SIKA_GUARD_LLM_PRICE_INPUT_PER_MTOK", "2")
    monkeypatch.setenv("SIKA_GUARD_LLM_PRICE_OUTPUT_PER_MTOK", "8")
    monkeypatch.setattr(evaluate, "LLMClient", provider.client_for)
    out_file = tmp_path / "records.json"
    assert evaluate.main(["--condition", "C", "--limit", "2", "--cases", "--output-json", str(out_file)]) == 0
    captured = capsys.readouterr()
    assert len(provider.requests) == 6
    assert SECRET not in captured.out + captured.err + out_file.read_text(encoding="utf-8")
    assert "Mode: LIVE" in captured.out and "model test-model" in captured.out
    assert "tokens reported on 6 of 6 calls" in captured.out
    assert "cost on 6 of 6 calls" in captured.out and "36.000000" in captured.out  # 6 calls x (1M x 2 + 0.5M x 8) per million
