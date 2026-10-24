"""Run a condition over cases, then score it against ground truth.

Inference and scoring are separate steps. `run_condition` takes cases and evaluators and never sees
a label; `score` joins labels to finished records afterwards. Every figure is descriptive: the
fixtures are 20 hand-authored cases, not a representative benchmark.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from sika_guard.coordinator import CaseRecord, Condition, CoordinatorConfig, Coverage, coordinate
from sika_guard.evaluators.contract import CallRecord, Evaluator, EvaluatorStatus
from sika_guard.evaluators.crosscheck import CrossCheckOutcome
from sika_guard.labels import GroundTruth
from sika_guard.models import Case, Decision

P95_MIN_SAMPLES = 20  # below this a p95 is just the maximum, so only descriptive figures are shown


@dataclass(frozen=True)
class RunMetadata:
    condition: Condition
    mode: str  # "live", or "offline-mock:<mode>" for scripted evaluators
    evaluators: tuple[Mapping[str, str | None], ...]  # evaluator_id, prompt_version, model_id
    llm: Mapping[str, object] | None  # LLMConfig.describe(): never contains the key
    fail_closed: bool
    policy: Mapping[str, object]

    @property
    def is_live(self) -> bool:
        return self.mode == "live"


async def run_condition(
    cases: Sequence[Case], condition: Condition, evaluators: Sequence[Evaluator], config: CoordinatorConfig = CoordinatorConfig()
) -> list[CaseRecord]:
    """Cases run one after another so per-case latency is not distorted by other cases; within a
    case the evaluators run concurrently."""
    return [await coordinate(case, condition, evaluators, config) for case in cases]


# --- Scoring (labels enter only here) ---------------------------------------


@dataclass(frozen=True)
class Scored:
    record: CaseRecord
    truth: GroundTruth

    @property
    def harmful(self) -> bool:
        return self.truth.ideal_decision is not Decision.ALLOW

    @property
    def intercepted(self) -> bool:
        return self.harmful and self.record.final_decision is not Decision.ALLOW

    @property
    def intercepted_by_baseline(self) -> bool:
        return self.harmful and self.record.deterministic_decision is not Decision.ALLOW

    @property
    def benign(self) -> bool:
        return self.truth.ideal_decision is Decision.ALLOW

    @property
    def falsely_escalated(self) -> bool:
        return self.benign and self.record.final_decision is not Decision.ALLOW

    @property
    def falsely_escalated_by_baseline(self) -> bool:
        return self.benign and self.record.deterministic_decision is not Decision.ALLOW

    @property
    def authorised_but_inferior(self) -> bool:
        return self.truth.financially_inferior and not self.truth.hard_authorisation_violation


def score(records: Sequence[CaseRecord], labels: Mapping[str, GroundTruth]) -> list[Scored]:
    return [Scored(r, labels[r.case_id]) for r in records]


# --- Statistics -------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def describe_ms(values: Sequence[float]) -> str:
    n = len(values)
    if n == 0:
        return "n/a (no samples)"
    median = statistics.median(values)
    if n < P95_MIN_SAMPLES:
        return f"median {median:.1f} ms, range {min(values):.1f}-{max(values):.1f} ms (descriptive only, n={n})"
    return f"median {median:.1f} ms, p95 {percentile(values, 0.95):.1f} ms (n={n})"


def _calls(records: Sequence[CaseRecord]) -> list[CallRecord]:
    return [c for r in records for v in r.verdicts for c in v.calls]


def _tally(c: Counter) -> str:
    return " ".join(f"{d.value}={c[d]}" for d in Decision)


# --- Report -----------------------------------------------------------------


def format_experiment_report(meta: RunMetadata, scored: Sequence[Scored], *, show_cases: bool = False) -> str:
    records = [s.record for s in scored]
    n = len(scored)
    out = [f"Sika Guard - Condition {meta.condition.value} ({_CONDITION_NAMES[meta.condition]})"]
    if meta.is_live:
        out.append("Mode: LIVE model inference")
    else:
        out.append(f"Mode: OFFLINE MOCK ({meta.mode.split(':', 1)[-1]}). Scripted evaluators, NOT model output; this checks the plumbing only.")
    for e in meta.evaluators:
        out.append(f"  evaluator {e['evaluator_id']}: prompt {e['prompt_version']}, model {e['model_id']}")
    if meta.llm:
        out.append("  provider config: " + ", ".join(f"{k}={v}" for k, v in meta.llm.items()))
    out.append(f"  policy: {', '.join(f'{k}={v}' for k, v in meta.policy.items())}; fail_closed={meta.fail_closed}")
    out += [
        f"Cases: {n} hand-authored synthetic fixtures. Not a representative benchmark; every figure below is descriptive.",
        "No comparison with a matched-compute single-model baseline has been made.",
        "",
        "Final verdicts",
        f"  expected (fixture labels)  {_tally(Counter(s.truth.ideal_decision for s in scored))}",
        f"  A: deterministic alone     {_tally(Counter(r.deterministic_decision for r in records))}",
        f"  {meta.condition.value}: with AI, final         {_tally(Counter(r.final_decision for r in records))}",
        "",
    ]

    harmful = [s for s in scored if s.harmful]
    benign = [s for s in scored if s.benign]
    inferior = [s for s in scored if s.authorised_but_inferior]
    out += [
        f"Harmful cases (fixture ideal is not ALLOW): {len(harmful)}",
        f"  intercepted: {sum(s.intercepted for s in harmful)} of {len(harmful)}  (deterministic alone: {sum(s.intercepted_by_baseline for s in harmful)})",
        f"  newly intercepted because of the AI layer: {sum(s.intercepted and not s.intercepted_by_baseline for s in harmful)}",
        f"Benign cases (fixture ideal is ALLOW): {len(benign)}",
        f"  falsely escalated: {sum(s.falsely_escalated for s in benign)} of {len(benign)}  (deterministic alone: {sum(s.falsely_escalated_by_baseline for s in benign)})",
        f"  newly escalated because of the AI layer: {sum(s.falsely_escalated and not s.falsely_escalated_by_baseline for s in benign)}",
        f"Authorised but financially inferior: {len(inferior)}",
        f"  escalated: {sum(s.record.final_decision is not Decision.ALLOW for s in inferior)}  (deterministic alone: {sum(s.record.deterministic_decision is not Decision.ALLOW for s in inferior)})",
        f"  missed (final ALLOW): {sum(s.record.final_decision is Decision.ALLOW for s in inferior)}",
    ]
    blocks = [r for r in records if r.deterministic_decision is Decision.BLOCK]
    out += [f"Deterministic BLOCKs preserved: {sum(r.final_decision is Decision.BLOCK for r in blocks)} of {len(blocks)}", ""]

    out.append("Evaluator outcomes (status is explicit; nothing is dropped)")
    for evaluator_id in dict.fromkeys(v.evaluator_id for r in records for v in r.verdicts):
        verdicts = [v for r in records for v in r.verdicts if v.evaluator_id == evaluator_id]
        verdict_counts = Counter(v.verdict for v in verdicts if v.usable)
        statuses = Counter(v.status for v in verdicts)
        failures = " ".join(f"{s.value}={statuses[s]}" for s in EvaluatorStatus if s is not EvaluatorStatus.OK)
        out.append(f"  {evaluator_id}: ok={statuses[EvaluatorStatus.OK]} ({_tally(verdict_counts)})  {failures}")
    coverage = Counter(r.coverage for r in records)
    out += [
        "  coverage per case: " + " ".join(f"{c.value}={coverage[c]}" for c in Coverage),
        "",
        "Disagreement and abstention (separate outcomes; no majority vote is used)",
        f"  cases where usable AI verdicts disagree with each other: {sum(r.evaluator_disagreement for r in records)}",
        f"  cases where an AI verdict differs from the deterministic decision: {sum(r.ai_deterministic_disagreement for r in records)}",
        f"  cases with at least one abstaining or failed evaluator: {sum(bool(r.abstained_evaluators) for r in records)}",
    ]
    checks = Counter(c.outcome for r in records for c in r.crosschecks)
    out += [
        "Trusted cross-checks of model-extracted claims: " + (" ".join(f"{o.value}={checks[o]}" for o in CrossCheckOutcome) if checks else "none extracted"),
        f"Model contradiction allegations not confirmed by deterministic code: {sum(len(r.uncorroborated_contradictions) for r in records)}",
        f"Cases escalated by the AI layer (final stricter than deterministic): {sum(r.escalated_by_ai for r in records)}",
        "",
        "Latency",
        f"  total evaluation per case: {describe_ms([r.latency.total_ms for r in records])}",
        f"  AI stage wall time per case: {describe_ms([r.latency.ai_ms for r in records])}",
    ]
    calls = _calls(records)
    if not calls:
        out.append("  model calls: none were made")
    else:
        per_case_sum = [sum(c.latency_ms for v in r.verdicts for c in v.calls) for r in records if r.verdicts]
        out.append(f"  sum of call latencies per case: {describe_ms(per_case_sum)}")
        for evaluator_id in dict.fromkeys(c.evaluator_id for c in calls):
            out.append(f"  model call {evaluator_id}: {describe_ms([c.latency_ms for c in calls if c.evaluator_id == evaluator_id])}")
    out += ["", "Usage (only what the provider returned; nothing is estimated)"]
    if not calls:
        out.append("  no model calls, so no usage")
    else:
        with_tokens = [c for c in calls if c.prompt_tokens is not None and c.completion_tokens is not None]
        priced = [c for c in calls if c.cost is not None]
        out.append(
            f"  tokens reported on {len(with_tokens)} of {len(calls)} calls: "
            f"prompt {sum(c.prompt_tokens for c in with_tokens)}, completion {sum(c.completion_tokens for c in with_tokens)}"
        )
        if priced:
            out.append(f"  cost on {len(priced)} of {len(calls)} calls, in the units of the configured prices: {sum((c.cost for c in priced), Decimal(0)):.6f}")
        else:
            out.append("  cost: not reported (pricing not configured, or the provider returned no token counts)")

    if show_cases:
        out += ["", "Per case (* = final differs from the fixture's ideal decision)"]
        for s in scored:
            r = s.record
            mark = "*" if r.final_decision is not s.truth.ideal_decision else " "
            verdicts = " ".join(f"{v.evaluator_id}={v.verdict.value if v.usable else v.status.value}" for v in r.verdicts) or "-"
            codes = ", ".join(dict.fromkeys(f.code.value for v in r.verdicts for f in v.findings)) or "-"
            out.append(
                f" {mark}{r.case_id} expected {s.truth.ideal_decision.value:<6} deterministic {r.deterministic_decision.value:<6} "
                f"final {r.final_decision.value:<6} | {verdicts} | ai codes: {codes}"
            )
    return "\n".join(out)


_CONDITION_NAMES = {
    Condition.A: "deterministic baseline",
    Condition.B: "one general-purpose analyst plus deterministic baseline",
    Condition.C: "three concurrent specialists plus deterministic baseline",
}


def export_json(path: Path, meta: RunMetadata, scored: Sequence[Scored]) -> None:
    """Raw records for later analysis. Contains no credentials; ground truth appears only as scoring flags."""
    payload = {
        "meta": {
            "condition": meta.condition.value,
            "mode": meta.mode,
            "evaluators": list(meta.evaluators),
            "llm": meta.llm,
            "fail_closed": meta.fail_closed,
            "policy": meta.policy,
        },
        "records": [
            {
                **s.record.model_dump(mode="json"),
                "scoring": {
                    "harmful": s.harmful,
                    "intercepted": s.intercepted,
                    "falsely_escalated": s.falsely_escalated,
                    "authorised_but_inferior": s.authorised_but_inferior,
                },
            }
            for s in scored
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
