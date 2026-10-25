"""Reproductions of the audit's benchmark-validity findings (F9-F13). Offline, synthetic, no model.

Like `audit_scenarios.py`, each function reports what the system did using only calls that exist before and after the repair, so
the same code fills the before/after matrix. Where a repaired API refuses an operation, the scenario reports the refusal.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from sika_guard.benchmark import independent
from sika_guard.benchmark.hidden import dump_labels
from sika_guard.benchmark.pilot import build_pilot
from sika_guard.benchmark.validate import validate_dataset
from sika_guard.benchmark.visible import dump_visible
from sika_guard.coordinator import Condition, CoordinatorConfig, coordinate
from sika_guard.evaluators.mocks import Scripted, Timeout, make_analysis
from sika_guard.fixtures.builders import fee, gbp, level, make_case, make_mandate, make_offer
from sika_guard.models import Decision, FeeKind, Objective

ROOT = Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


def nested_ids(case) -> dict[str, list[str]]:
    return {
        "transaction_id": [case.transaction.transaction_id], "mandate_id": [case.mandate.mandate_id], "customer_id": [case.intent.customer_id],
        "agent_id": [case.mandate.agent_id], "merchant_id": [case.merchant.merchant_id], "offer_ids": [o.offer_id for o in case.offers],
        "assertion_ids": [a.assertion_id for a in case.assertions], "evidence_ids": [e.evidence_id for e in case.evidence],
    }


# --- F9: nested identifiers carry the scenario name ---------------------------------------------------------------------


def f9_nested_ids_reveal_the_scenario() -> dict[str, Any]:
    cases, labels = build_pilot()
    by_id = {x.case_id: x for x in labels}
    transaction_ids = sorted(c.transaction.transaction_id for c in cases)
    family_words = {"arc", "false", "twin", "named", "irrelevant", "starved", "nova", "birch", "sliver", "orbit", "inked", "diverted", "conflict"}
    leaking = [t for t in transaction_ids if any(w in t.lower() for w in family_words)]
    pattern = re.compile(r"^[a-z]{2,3}-[0-9a-f]{8}$")
    non_neutral = {c.case_id: [v for vs in nested_ids(c).values() for v in vs if not pattern.match(v)] for c in cases}
    report = validate_dataset(cases, by_id, check_policy=False)
    return {
        "transaction_ids": transaction_ids,
        "transaction_ids_naming_a_scenario": len(leaking),
        "cases_with_a_non_neutral_nested_id": sum(1 for v in non_neutral.values() if v),
        "validator_errors_about_ids": [e for e in report.all_errors() if "id" in e.lower() and "nested" in e.lower()],
        "validator_ok": report.ok,
    }


# --- F10: the economic oracle and the payment cap ---------------------------------------------------------------------------


def f10_alternative_that_breaks_the_payment_cap() -> dict[str, Any]:
    """OFFER-A is cheaper in total (680 against 744) but needs a 240 initial charge, above the 200 per-payment cap, so the agent could not have taken it."""
    mandate = make_mandate(max_total_repayment=gbp(900), max_monthly_payment=gbp(80), max_amount=gbp(200))
    a = make_offer("OFFER-A", level("40", fees=(fee(FeeKind.ARRANGEMENT, "200", due_upfront=True),)))  # total 680, initial charge 240
    b = make_offer("OFFER-B", level("62"))  # total 744, initial charge 62: the cheapest option the mandate allows
    case = make_case("audit-f10", [a, b], "OFFER-B", mandate=mandate)
    inferior, avoidable = independent.economics(case, {"OFFER-A": True, "OFFER-B": True})
    eligible = make_offer("OFFER-C", level("40", fees=(fee(FeeKind.ARRANGEMENT, "10", due_upfront=True),)))  # total 490, initial charge 50: allowed
    control = independent.economics(make_case("audit-f10b", [eligible, b], "OFFER-B", mandate=mandate), {"OFFER-C": True, "OFFER-B": True})
    return {"alternative_above_cap_counted_as_better": inferior, "avoidable_minor": avoidable, "control_eligible_alternative": list(control)}


# --- F11: sealing --------------------------------------------------------------------------------------------------------------


def f11_sealing_arbitrary_and_incomplete_logs() -> dict[str, Any]:
    """Three abuses of the label gate. The 'before' matrix used the only log format that existed for the pilot (v1); once the repaired run
    format exists, the same abuses are attempted with a v2 log, which is the harder test, because it is a well-formed run with records removed."""
    from sika_guard.benchmark import __main__ as cli
    from sika_guard.benchmark import seal as seal_module
    from sika_guard.benchmark.decisions import declare, write_decision_log
    from sika_guard.benchmark.mockrun import mock_records
    from sika_guard.benchmark.visible import visible_file_digest

    import contextlib
    import inspect
    import io

    out: dict[str, Any] = {}
    tmp = Path(tempfile.mkdtemp())
    cases, labels = build_pilot()
    cases_path, labels_path = tmp / "cases.jsonl", tmp / "labels.jsonl"
    cases_path.write_text(dump_visible(cases), encoding="utf-8")
    labels_path.write_text(dump_labels(labels), encoding="utf-8")
    records = mock_records(cases)
    from manifest_helper import record_manifest

    digest, directory = record_manifest(tmp)  # a real recorded manifest of this tree, kept beside the run
    bound = {"manifests_dir": directory}

    def seal(path):
        fn = getattr(seal_module, "write_seal_file", None) or seal_module.seal_file
        fn(path, cases_path, **bound) if len(inspect.signature(fn).parameters) >= 2 else fn(path)

    junk = tmp / "junk.txt"
    junk.write_text("this is not a decision log\n", encoding="utf-8")
    out["arbitrary_file_can_be_sealed"] = _attempt(lambda: seal(junk))

    partial = tmp / "partial.jsonl"
    try:
        from sika_guard.benchmark.runner import write_run_from_records

        write_run_from_records(partial, cases_path, cases, records, mode="mock", manifest_digest=digest, seal=False, **bound)
        lines = partial.read_text(encoding="utf-8").splitlines()
        partial.write_text("\n".join(l for i, l in enumerate(lines) if i not in (4, 9, 17)) + "\n", encoding="utf-8")  # three records removed from a complete v2 run
        out["incomplete_log_format"] = "v2 run with three records removed"
    except ImportError:
        declaration = declare(mode="mock", manifest_digest=digest, visible_cases_sha256=visible_file_digest(cases_path), conditions=list(records))
        write_decision_log(partial, declaration, {c: recs[:3] for c, recs in records.items()})
        out["incomplete_log_format"] = "v1 log holding three cases per condition"
    out["incomplete_log_can_be_sealed"] = _attempt(lambda: seal(partial))

    full = tmp / "full.jsonl"  # a complete log that the inference run never sealed
    write_decision_log(full, declare(mode="mock", manifest_digest=digest, visible_cases_sha256=visible_file_digest(cases_path), conditions=list(records)), records)

    def score():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = cli._score_and_print(cases_path, full, labels_path, mock=True, **bound)
        if code != 0:
            raise PermissionError(f"labels stayed closed (exit {code})")

    out["scorer_can_open_labels_with_no_persisted_inference_seal"] = _attempt(score)
    return out


def _attempt(fn) -> str:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001  the scenario reports refusals by type
        return f"refused ({type(exc).__name__}: {str(exc)[:70]})"
    return "ALLOWED"


# --- F13: verified-contradiction recall credits any intercept -------------------------------------------------------------------


def f13_recall_credits_a_condition_whose_evaluator_timed_out() -> dict[str, Any]:
    from sika_guard.benchmark.scoring import score_run

    cases, labels = build_pilot()
    verifiable = next(x for x in labels if x.merchant_claim_false and x.contradictory_evidence_available)
    case = next(c for c in cases if c.case_id == verifiable.case_id)
    records = {
        Condition.A: run(coordinate(case, Condition.A, [], CoordinatorConfig())),
        Condition.B: run(coordinate(case, Condition.B, [Timeout("timed-out-model")], CoordinatorConfig())),  # the model contributed nothing
    }
    s = score_run({c: [r] for c, r in records.items()}, {verifiable.case_id: verifiable}, n_boot=20)
    b = s.conditions[Condition.B]
    old = getattr(b, "verified_contradiction_recall", None)
    return {
        "condition_B_final": records[Condition.B].final_decision.value,
        "evaluator_status": records[Condition.B].verdicts[0].status.value,
        "old_metric_verified_contradiction_recall": str(old.num) + "/" + str(old.den) if old else None,
        "new_metric_ai_contributed_verified": getattr(b, "ai_contributed_verified", None),
        "new_metric_deterministic_policy_catches": (lambda r: f"{r.num}/{r.den}")(b.deterministic_verified_catch) if hasattr(b, "deterministic_verified_catch") else None,
    }


# --- F12: documentation claims -------------------------------------------------------------------------------------------------------


def f12_matched_compute_claims_in_the_repository() -> dict[str, Any]:
    hits: dict[str, int] = {}
    for rel in ("README.md", "CLAUDE.md", "BENCHMARK_SPEC.md", "BENCHMARK_BUILD_GUIDE.md", "DECISION_POLICY.md", "src/sika_guard/coordinator.py", "src/sika_guard/evaluate.py",
                "src/sika_guard/benchmark/manifest.py", "src/sika_guard/benchmark/scoring.py", "src/sika_guard/benchmark/quotas.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        n = len(re.findall(r"matched[- ]compute", text, re.I))
        if n:
            hits[rel] = n
    return {"files_claiming_matched_compute": hits}


SCENARIOS = {
    "F9": f9_nested_ids_reveal_the_scenario,
    "F10": f10_alternative_that_breaks_the_payment_cap,
    "F11": f11_sealing_arbitrary_and_incomplete_logs,
    "F12": f12_matched_compute_claims_in_the_repository,
    "F13": f13_recall_credits_a_condition_whose_evaluator_timed_out,
}


def matrix() -> dict[str, Any]:
    return {k: fn() for k, fn in SCENARIOS.items()}


if __name__ == "__main__":
    print(json.dumps(matrix(), indent=1, sort_keys=True, default=str))
