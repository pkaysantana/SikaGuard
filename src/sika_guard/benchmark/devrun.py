"""The development-case runner and its report.

A thin layer over `runner.run_experiment`: it adds nothing to how a case is decided (the coordinator, the evaluators, the prompts and the
financial policy are exactly those every other path uses), only the command-line plumbing, the spend safeguards and a report.

  * Cases are read from a model-visible file. Hidden labels and expectations are never an argument of a run; the report reads an
    expectations file only after the run has been sealed complete, through the same gate the benchmark uses.
  * A fake provider (`--mode fake`) goes through the real client and is plumbing: its replies are scripted and say nothing about any model.
  * `--mode dev-live` calls the configured provider. It needs the LLM environment variables, an explicit `--max-calls`, and
    `--confirm-spend N` equal to the exact planned number of calls. The count is also enforced where the request leaves the process.
    Nothing retries. A dev-live run is bound to a development-baseline manifest and can never support a held-out claim.
  * The report keeps four things apart: what a condition decided; what was a VERIFIED extraction (the only route by which an AI condition
    can raise a REVIEW); what was merely a CANDIDATE (a claim or concern that trusted code could not establish: reported, never counted as an
    interception); and what failed (malformed replies, provider failures, abstentions, false allegations).

This module must not import the hidden-label module.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from sika_guard.benchmark import manifest as mf
from sika_guard.benchmark import runner
from sika_guard.benchmark.decisions import RunLog, read_run_log
from sika_guard.benchmark.seal import LabelsLockedError, verify_gate
from sika_guard.benchmark.visible import load_visible
from sika_guard.coordinator import CaseRecord, Condition
from sika_guard.evaluators.adjudicate import FindingEffect
from sika_guard.evaluators.crosscheck import CrossCheckOutcome
from sika_guard.evaluators.llm import ConfigError, LLMConfig
from sika_guard.models import Decision

ALL = (Condition.A, Condition.A_PLUS, Condition.B, Condition.B3, Condition.BC, Condition.C)
_CANDIDATE_OUTCOMES = {CrossCheckOutcome.EXTRACTION_DISAGREES, CrossCheckOutcome.NOT_COMPARABLE, CrossCheckOutcome.UNGROUNDED, CrossCheckOutcome.DISCREPANT_NOT_ADVERSE}


def parse_conditions(text: str) -> tuple[Condition, ...]:
    by_value = {c.value.lower(): c for c in ALL}
    try:
        return tuple(by_value[part.strip().lower()] for part in text.split(",") if part.strip())
    except KeyError as exc:
        raise ValueError(f"unknown condition {exc.args[0]!r}; expected some of {', '.join(c.value for c in ALL)}") from None


# --- running ----------------------------------------------------------------------------------------------------------------------------


@dataclass
class DevRunResult:
    outcome: runner.RunOutcome
    log_path: Path
    manifests_dir: Path
    cases_path: Path


def run_dev(
    cases_path: Path, out_dir: Path, conditions: tuple[Condition, ...], *, repeats: int, mode: str, fake_mode: str = "quiet",
    max_calls: int | None = None, confirm_spend: int | None = None, fail_closed: bool = False, log: Any = print,
) -> DevRunResult:
    """Run development cases through the chosen conditions and seal the log. Raises SystemExit with a reason rather than ever spending
    when a safeguard is not met."""
    cases = load_visible(cases_path)
    spec = runner.ExperimentSpec(conditions=conditions, repeats=repeats, fail_closed=fail_closed, max_calls=max_calls, experiment_name=f"dev-{mode}")
    plan = runner.plan_calls(len(cases), spec)
    log(runner.format_plan(plan, spec))
    if mode == "dev-live":
        if max_calls is None or confirm_spend is None:
            raise SystemExit("refused: a dev-live run needs --max-calls and --confirm-spend N (the exact planned number of calls).")
        if confirm_spend != plan.total_calls:
            raise SystemExit(f"refused: --confirm-spend {confirm_spend} is not the planned {plan.total_calls} calls.")
        if plan.total_calls > max_calls:
            raise SystemExit(f"refused: the plan needs {plan.total_calls} calls but --max-calls is {max_calls}.")
        try:
            config = LLMConfig.from_env()
        except ConfigError as exc:
            raise SystemExit("refused: the provider is not configured (missing: " + ", ".join(exc.missing or ("a value",)) + ").") from None
        transport: httpx.AsyncBaseTransport = httpx.AsyncHTTPTransport()
        log(f"Provider: {config.describe()['base_url']}  model: {config.model}  (key not shown)")
    elif mode == "deterministic":
        if any(c not in (Condition.A, Condition.A_PLUS) for c in conditions):
            raise SystemExit("refused: --mode deterministic runs only A and A+, which make no model call.")
        config, transport = runner.fake_config("quiet"), runner.offline_transport("quiet")  # never used: A and A+ send no request
        log("Mode: DETERMINISTIC. No model and no provider are involved; these are real results of the rule engine and the scanner.")
    elif mode == "fake":
        config, transport = runner.fake_config(fake_mode), runner.offline_transport(fake_mode)
        log(f"Mode: FAKE provider '{fake_mode}': scripted replies through the real client. Plumbing only; not model output.")
    else:
        raise SystemExit(f"unknown mode {mode!r}; use deterministic, fake or dev-live")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = out_dir / "manifests"
    manifest = mf.build_manifest(note=f"development run ({mode}); development baseline, never a freeze")
    manifest_path = manifests_dir / f"manifest-{manifest['digest'][:12]}.json"
    if not manifest_path.exists():
        mf.write_manifest(manifest, manifest_path)
    log_path = out_dir / "decisions.jsonl"
    outcome = asyncio.run(runner.run_experiment(
        cases, cases_path, spec, log_path, mode=mode, config=config, inner_transport=transport, manifest_digest=manifest["digest"], manifests_dir=manifests_dir,
    ))
    return DevRunResult(outcome, log_path, manifests_dir, cases_path)


# --- reporting --------------------------------------------------------------------------------------------------------------------------


def _sev(decision: str | Decision | None) -> int:
    return -1 if decision is None else Decision(decision).severity


def _row(rec: CaseRecord, repeat: int, condition: Condition, a: str | None, aplus: str | None) -> dict[str, Any]:
    final = rec.final_decision.value
    checks = [
        {"evaluator": c.evaluator_id, "kind": c.kind.value, "offer": c.offer_id, "outcome": c.outcome.value, "source": c.source_class.value if c.source_class else None,
         "direction": c.direction.value if c.direction else None, "material": c.material, "enforces": c.enforces, "detail": c.detail}
        for c in rec.crosschecks
    ]
    verified = [c for c in checks if c["enforces"]]
    claims = sum(len(v.extracted_claims) for v in rec.verdicts)
    candidates = [c for c in checks if c["outcome"] in {o.value for o in _CANDIDATE_OUTCOMES}]
    candidates += [{"evaluator": v.evaluator_id, "kind": r.claim.kind.value, "offer": r.claim.offer_id, "outcome": "rejected_before_comparison", "detail": r.reason}
                   for v in rec.verdicts for r in v.rejected_claims]
    # A concern a model raised about a passage it cited ("doc:..."), in a finding rather than as an extracted claim. Trusted code cannot establish
    # it (nothing was extracted to verify), so it is a candidate discovery and never an interception. Findings are adjudicated against
    # structured data only, so one that cites a document is not counted as a false allegation even if it also cites something the structured data
    # contradicts.
    doc_concerns = [a_ for a_ in rec.adjudications if any(ref.startswith("doc:") for ref in a_.evidence_refs) and a_.effect is not FindingEffect.ENFORCING]
    candidates += [{"evaluator": a_.evaluator_id, "kind": a_.code.value, "offer": None, "outcome": "concern_about_a_cited_document_not_established", "detail": a_.reason[:110]} for a_ in doc_concerns]
    false_allegations = [a_ for a_ in rec.adjudications if a_.effect is FindingEffect.MODEL_ERROR and not any(ref.startswith("doc:") for ref in a_.evidence_refs)]
    unenforced_concerns = [
        {"evaluator": t.evaluator_id, "model_verdict": t.model_verdict.value, "counted_as": t.counted_as.value}
        for t in rec.verdict_treatments if t.model_verdict is not Decision.ALLOW and t.counted_as is Decision.ALLOW
    ]
    calls = [c for v in rec.verdicts for c in v.calls]
    counted = [c for c in calls if c.prompt_tokens is not None and c.completion_tokens is not None]
    costed = [c.provider_cost for c in calls if c.provider_cost is not None]
    failures = [{"evaluator": v.evaluator_id, "status": v.status.value, "reason": v.abstain_reason} for v in rec.verdicts if not v.usable]
    return {
        "condition": condition.value, "repeat": repeat, "case_id": rec.case_id, "final": final, "deterministic_A": rec.deterministic_decision.value,
        "A": a, "A_plus": aplus, "bases": [b.value for b in rec.decision_bases], "routes": list(rec.ai_review_routes),
        "escalated_above_rule_engine": rec.escalated_by_ai,
        "incremental_over_A": a is not None and _sev(final) > _sev(a),
        "incremental_over_A_plus": aplus is not None and _sev(final) > _sev(aplus),
        "verified_extractions": verified, "claims_extracted": claims, "candidate_discoveries": candidates, "unenforced_model_concerns": unenforced_concerns,
        "findings": len(rec.adjudications), "false_allegations": len(false_allegations), "document_concerns": len(doc_concerns),
        "unverified_findings": sum(a_.effect is FindingEffect.UNVERIFIED_RECORDED for a_ in rec.adjudications),
        "verdict_statuses": {s: sum(v.status.value == s for v in rec.verdicts) for s in sorted({v.status.value for v in rec.verdicts})},
        "schema_valid_verdicts": sum(v.usable or v.status.value == "abstained" for v in rec.verdicts), "verdicts": len(rec.verdicts), "failures": failures,
        "coverage": rec.coverage.value, "uncertainty": list(rec.uncertainty_reasons),
        "calls": len(calls), "call_latency_ms": [round(c.latency_ms, 1) for c in calls], "wall_ms": round(rec.latency.total_ms, 1), "ai_stage_ms": round(rec.latency.ai_ms, 1),
        "prompt_tokens": sum(c.prompt_tokens for c in counted) if calls and len(counted) == len(calls) else None,
        "completion_tokens": sum(c.completion_tokens for c in counted) if calls and len(counted) == len(calls) else None,
        "provider_cost_unit_unknown": str(sum(costed)) if costed and len(costed) == len(calls) else None,
        "providers": sorted({c.provider or "not exposed" for c in calls}), "returned_models": sorted({c.returned_model or "not reported" for c in calls}),
        "models_requested": sorted({c.model for c in calls}),
    }


def build_report(log: RunLog, expectations: dict[str, Any] | None = None) -> dict[str, Any]:
    by_key = {(r.condition, r.repeat, r.record.case_id): r.record for r in log.records}
    rows = []
    for r in log.records:
        a = (by_key.get((Condition.A, r.repeat, r.record.case_id)) or by_key.get((Condition.A, 1, r.record.case_id)))
        ap = (by_key.get((Condition.A_PLUS, r.repeat, r.record.case_id)) or by_key.get((Condition.A_PLUS, 1, r.record.case_id)))
        row = _row(r.record, r.repeat, r.condition, a.final_decision.value if a else None, ap.final_decision.value if ap else None)
        if expectations is not None and r.record.case_id in expectations:
            e = expectations[r.record.case_id]
            row.update(slug=e.slug, policy_expected=e.policy_expected, ideal=e.ideal,
                       matches_policy_expected=row["final"] == e.policy_expected,
                       false_positive=_sev(row["final"]) > _sev(e.policy_expected), missed=_sev(row["final"]) < _sev(e.policy_expected),
                       ideal_not_reached=_sev(row["final"]) < _sev(e.ideal))
        rows.append(row)

    summary: dict[str, Any] = {}
    for cond in sorted({r["condition"] for r in rows}, key=[c.value for c in ALL].index):
        mine = [r for r in rows if r["condition"] == cond]
        lat = [x for r in mine for x in r["call_latency_ms"]]
        s: dict[str, Any] = {
            "cases": len(mine), "final": {d: sum(r["final"] == d for r in mine) for d in ("ALLOW", "REVIEW", "BLOCK")},
            "verified_ai_interceptions_over_A": sum(1 for r in mine if r["incremental_over_A"] and r["verified_extractions"] and r["escalated_above_rule_engine"]),
            "verified_ai_interceptions_over_A_and_A_plus": sum(1 for r in mine if r["incremental_over_A"] and r["incremental_over_A_plus"] and r["verified_extractions"]),
            "interceptions_over_A_not_attributable_to_a_verified_extraction": sum(1 for r in mine if r["incremental_over_A"] and not r["verified_extractions"]),
            "candidate_discoveries": sum(len(r["candidate_discoveries"]) for r in mine), "document_concerns": sum(r["document_concerns"] for r in mine), "ignored_model_alarms": sum(len(r["unenforced_model_concerns"]) for r in mine), "claims_extracted": sum(r["claims_extracted"] for r in mine),
            "false_allegations": sum(r["false_allegations"] for r in mine), "unverified_findings": sum(r["unverified_findings"] for r in mine),
            "verdicts": sum(r["verdicts"] for r in mine), "failures": sum(len(r["failures"]) for r in mine),
            "schema_failures": sum(f["status"] == "invalid_response" for r in mine for f in r["failures"]),
            "provider_failures": sum(f["status"] in ("timeout", "provider_error", "refusal") for r in mine for f in r["failures"]),
            "abstentions": sum(f["status"] == "abstained" for r in mine for f in r["failures"]),
            "calls": sum(r["calls"] for r in mine), "median_call_latency_ms": round(statistics.median(lat), 1) if lat else None,
            "median_wall_ms_per_case": round(statistics.median(r["wall_ms"] for r in mine), 1),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in mine) if any(r["prompt_tokens"] is not None for r in mine) else None,
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in mine) if any(r["completion_tokens"] is not None for r in mine) else None,
            "provider_cost_unit_unknown": str(sum((Decimal(r["provider_cost_unit_unknown"]) for r in mine if r["provider_cost_unit_unknown"]), Decimal(0))) if any(r["provider_cost_unit_unknown"] for r in mine) else None,
        }
        if expectations is not None:
            s.update(candidates_where_ideal_is_not_allow=sum(len(r["candidate_discoveries"]) for r in mine if r.get("ideal") not in (None, "ALLOW")),
                     ignored_alarms_on_cases_to_allow=sum(len(r["unenforced_model_concerns"]) for r in mine if r.get("ideal") == "ALLOW"),
                     candidates_where_ideal_is_allow=sum(len(r["candidate_discoveries"]) for r in mine if r.get("ideal") == "ALLOW"),
                     matches_policy_expected=sum(bool(r.get("matches_policy_expected")) for r in mine), false_positives=sum(bool(r.get("false_positive")) for r in mine),
                     missed=sum(bool(r.get("missed")) for r in mine), ideal_not_reached=sum(bool(r.get("ideal_not_reached")) for r in mine))
        summary[cond] = s
    d = log.declaration
    return {
        "run": {"mode": d.mode, "conditions": list(d.conditions), "repeats": d.repeats, "cases": d.cases, "planned_calls": d.planned_calls, "model_id": d.model_id,
                "manifest_digest": d.manifest_digest[:16], "manifest_status": d.manifest_status, "complete": bool(log.summary and log.summary.complete),
                "calls_recorded": log.summary.calls_recorded if log.summary else None,
                "label": {"deterministic": "DETERMINISTIC conditions only (A and A+): no model involved; real results of the rule engine and the scanner", "fake": "FAKE PROVIDER: scripted replies, plumbing only; NOT model output", "mock": "MOCK: scripted, NOT model output",
                          "dev-live": "LIVE model calls on DEVELOPMENT cases; not a held-out result", "live": "LIVE"}.get(d.mode, d.mode)},
        "summary": summary, "rows": rows,
    }


def format_report(rep: dict[str, Any]) -> str:
    run = rep["run"]
    out = [f"Sika Guard development run: {run['label']}", f"  conditions {', '.join(run['conditions'])}; {run['cases']} case(s) x {run['repeats']} repeat(s); planned calls {run['planned_calls']}, recorded {run['calls_recorded']}; model {run['model_id']}; manifest {run['manifest_digest']} ({run['manifest_status']})"]
    has_exp = any("slug" in r for r in rep["rows"])
    out += ["", "By condition (all figures are counts over the cases above; descriptive only)"]
    head = f"  {'cond':<5}{'ALLOW':>6}{'REVIEW':>7}{'BLOCK':>6}{'verified>A':>12}{'also >A+':>9}{'candidates':>11}{'false alleg.':>13}{'schema fail':>12}{'prov fail':>10}{'calls':>6}{'med call ms':>12}"
    if has_exp:
        head += f"{'cand@hold':>10}{'cand@ok':>8}{'ign.alarm':>10}{'match':>7}{'false+':>7}{'missed':>7}"
    out.append(head)
    for cond, s in rep["summary"].items():
        line = f"  {cond:<5}{s['final']['ALLOW']:>6}{s['final']['REVIEW']:>7}{s['final']['BLOCK']:>6}{s['verified_ai_interceptions_over_A']:>12}{s['verified_ai_interceptions_over_A_and_A_plus']:>9}{s['candidate_discoveries']:>11}{s['false_allegations']:>13}{s['schema_failures']:>12}{s['provider_failures']:>10}{s['calls']:>6}{str(s['median_call_latency_ms'] or '-'):>12}"
        if has_exp:
            line += f"{s['candidates_where_ideal_is_not_allow']:>10}{s['candidates_where_ideal_is_allow']:>8}{s['ignored_alarms_on_cases_to_allow']:>10}{s['matches_policy_expected']:>7}{s['false_positives']:>7}{s['missed']:>7}"
        out.append(line)
    out.append("  'verified>A' counts cases raised above Condition A by a VERIFIED extraction (for A+ the extractor is the deterministic scanner, not a model); 'also >A+' requires that A+ had not already held them.")
    out.append("  'candidates' are claims or concerns trusted code could not establish: reported, never counted as an interception.")
    if has_exp:
        out.append("  'cand@hold' = candidates on cases a person would want held (possible discoveries); 'cand@ok' = candidates on cases that should be allowed (false alarms, recorded and not enforced).")
        out.append("  'ign.alarm' = model REVIEW/BLOCK verdicts on cases that should be allowed that the policy did NOT enforce: false positives this design prevented (a condition that obeyed its model would have made them).")
    conds = list(rep["summary"])
    out += ["", "Per case: final decision by condition" + (" (expected = what the policy should decide; ideal = what a person would want)" if has_exp else "")]
    ids = sorted({r["case_id"] for r in rep["rows"]}, key=lambda i: next((r.get("slug", i) for r in rep["rows"] if r["case_id"] == i), i))
    out.append(f"  {'case':<50}" + (f"{'expect':>7}{'ideal':>7}" if has_exp else "") + "".join(f"{c:>8}" for c in conds))
    for cid in ids:
        mine = {r["condition"]: r for r in rep["rows"] if r["case_id"] == cid and r["repeat"] == 1}
        first = next(iter(mine.values()))
        cells = "".join(f"{(mine[c]['final'] if c in mine else '-'):>8}" for c in conds)
        out.append(f"  {first.get('slug', cid)[:49]:<50}" + (f"{first['policy_expected']:>7}{first['ideal']:>7}" if has_exp else "") + cells)
    notable = [r for r in rep["rows"] if r["verified_extractions"] or r["candidate_discoveries"] or r["failures"] or r["false_allegations"]]
    if notable:
        out += ["", "Details (verified extractions, candidate discoveries, failures and false allegations)"]
        for r in notable:
            out.append(f"  {r.get('slug', r['case_id'])} [{r['condition']}] final {r['final']} (A {r['A']}, A+ {r['A_plus']})")
            for v in r["verified_extractions"]:
                out.append(f"      VERIFIED  {v['evaluator']}: {v['kind']} {v['outcome']} {v['source']} material={v['material']}")
            for c in r["candidate_discoveries"]:
                out.append(f"      candidate {c['evaluator']}: {c['kind']} -> {c['outcome']}: {c['detail'][:110]}")
            for f in r["failures"]:
                out.append(f"      failure   {f['evaluator']}: {f['status']}: {(f['reason'] or '')[:110]}")
            if r["false_allegations"]:
                out.append(f"      false allegations contradicted by structured data: {r['false_allegations']}")
    out += ["", "What this does not show: nothing here is a result about model quality beyond these cases; the cases are development data written alongside the rules,"]
    out.append("and a fake or mock run says nothing about any model.")
    return "\n".join(out)


def report_from_files(cases_path: Path, log_path: Path, expectations_path: Path | None, manifests_dir: Path | None) -> dict[str, Any]:
    """Build a report. Expectations are read only AFTER the gate has confirmed the run was sealed complete and unmodified."""
    log = read_run_log(log_path)
    expectations = None
    if expectations_path is not None:
        from sika_guard.benchmark.devchallenge import load_expectations

        verify_gate(log_path, cases_path, manifests_dir=manifests_dir)  # raises LabelsLockedError; nothing below is read if the run is not sealed
        expectations = load_expectations(expectations_path)
    return build_report(log, expectations)


def write_outputs(rep: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    (out_dir / "results.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (out_dir / "results.txt").write_text(format_report(rep) + "\n", encoding="utf-8")
    return out_dir / "results.json", out_dir / "results.txt"


__all__ = ["LabelsLockedError", "ALL", "parse_conditions", "run_dev", "build_report", "format_report", "report_from_files", "write_outputs"]
if __name__ == "__main__":  # pragma: no cover
    sys.exit("use: python -m sika_guard.benchmark run|report")
