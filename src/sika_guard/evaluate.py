"""Baseline evaluator: run the deterministic policy over the fixtures and score it.

"Expected" always means the fixture author's ideal decision (ground truth).
"Detected" always means what the deterministic policy actually returned. The
policy only ever sees the `Case`; labels are joined afterwards for scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path

from sika_guard.coordinator import AGGREGATION_POLICY, Condition, CoordinatorConfig
from sika_guard.evaluators.llm import (
    ENV_JSON_MODE,
    ENV_PRICE_IN,
    ENV_PRICE_OUT,
    ENV_TEMPERATURE,
    ENV_TIMEOUT,
    REQUIRED_ENV,
    ConfigError,
    LLMClient,
    LLMConfig,
    evaluators_for,
)
from sika_guard.evaluators.mocks import OFFLINE_MODES, offline_evaluators
from sika_guard.evaluators.prompts import specs_for
from sika_guard.experiment import RunMetadata, export_json, format_experiment_report, run_condition, score
from sika_guard.fixtures.cases import build_cases
from sika_guard.fixtures.labels import LABELS
from sika_guard.labels import AttackFamily, GroundTruth
from sika_guard.models import Case, Decision, DecisionBasis, FindingCategory, SecurityDecision
from sika_guard.policy import DEFAULT_CONFIG, PolicyConfig, decision_bases, evaluate_case


@dataclass(frozen=True)
class CaseOutcome:
    case: Case
    truth: GroundTruth
    result: SecurityDecision


def run_baseline(config: PolicyConfig = DEFAULT_CONFIG) -> list[CaseOutcome]:
    outcomes = []
    for case in build_cases():
        result = evaluate_case(case, config)  # the policy receives the case only
        outcomes.append(CaseOutcome(case, LABELS[case.case_id], result))
    return outcomes


@dataclass
class FamilyRow:
    total: int = 0
    expected: Counter = field(default_factory=Counter)
    detected: Counter = field(default_factory=Counter)
    agree: int = 0


@dataclass
class BaselineSummary:
    total: int = 0
    expected: Counter = field(default_factory=Counter)
    detected: Counter = field(default_factory=Counter)
    agree: int = 0
    hard_total: int = 0
    hard_blocked: int = 0
    blocked_without_hard_label: int = 0
    false_claim_total: int = 0
    false_claim_confirmed: int = 0
    false_claim_escalated_unconfirmed: int = 0
    false_claim_undetected: int = 0
    inferior_total: int = 0  # authorised (no hard violation) but inferior
    inferior_escalated: int = 0
    inferior_missed: int = 0
    benign_total: int = 0  # fixture ideal decision is ALLOW
    benign_escalated: int = 0
    finding_categories: Counter = field(default_factory=Counter)
    by_family: dict[AttackFamily, FamilyRow] = field(default_factory=dict)
    disagreements: list[CaseOutcome] = field(default_factory=list)


def summarise(outcomes: list[CaseOutcome]) -> BaselineSummary:
    s = BaselineSummary()
    for o in outcomes:
        truth, result = o.truth, o.result
        decision = result.decision
        s.total += 1
        s.expected[truth.ideal_decision] += 1
        s.detected[decision] += 1
        row = s.by_family.setdefault(truth.attack_family, FamilyRow())
        row.total += 1
        row.expected[truth.ideal_decision] += 1
        row.detected[decision] += 1
        if decision is truth.ideal_decision:
            s.agree += 1
            row.agree += 1
        else:
            s.disagreements.append(o)
        s.finding_categories.update(f.category for f in result.findings)

        if truth.hard_authorisation_violation:
            s.hard_total += 1
            s.hard_blocked += decision is Decision.BLOCK
        elif decision is Decision.BLOCK:
            s.blocked_without_hard_label += 1

        if truth.false_merchant_claim:
            s.false_claim_total += 1
            if result.has(FindingCategory.CONFIRMED_FALSE_CLAIM):
                s.false_claim_confirmed += 1
            elif decision is not Decision.ALLOW:
                s.false_claim_escalated_unconfirmed += 1
            else:
                s.false_claim_undetected += 1

        if truth.financially_inferior and not truth.hard_authorisation_violation:
            s.inferior_total += 1
            if decision is Decision.ALLOW:
                s.inferior_missed += 1
            else:
                s.inferior_escalated += 1

        if truth.ideal_decision is Decision.ALLOW:
            s.benign_total += 1
            s.benign_escalated += decision is not Decision.ALLOW
    return s


def _counts(c: Counter) -> str:
    return "  ".join(f"{d.value}={c[d]}" for d in Decision)


def format_report(outcomes: list[CaseOutcome], *, show_cases: bool = False) -> str:
    s = summarise(outcomes)
    out = [
        "Sika Guard - deterministic baseline",
        f"Fixtures: {s.total} synthetic cases. No AI model was called.",
        "Expected = fixture label (ideal decision). Detected = deterministic policy output.",
        "The fixtures were written alongside the policy, so agreement is not a performance estimate.",
        "",
        "Decisions",
        f"  expected  {_counts(s.expected)}",
        f"  detected  {_counts(s.detected)}",
        f"  detected == expected in {s.agree} of {s.total} cases",
        "",
        "Hard authorisation violations (labelled)",
        f"  caught (BLOCK): {s.hard_blocked} of {s.hard_total}",
        f"  BLOCKed without a hard-violation label: {s.blocked_without_hard_label}",
        "",
        "Merchant claims or figures that are actually false (labelled)",
        f"  confirmed by a deterministic contradiction: {s.false_claim_confirmed} of {s.false_claim_total}",
        f"  escalated without confirmation (unresolved evidence): {s.false_claim_escalated_unconfirmed}",
        f"  undetected (ALLOW): {s.false_claim_undetected}",
        "",
        "Authorised but financially inferior (labelled, no hard violation)",
        f"  escalated to REVIEW: {s.inferior_escalated} of {s.inferior_total}",
        f"  missed (ALLOW): {s.inferior_missed}",
        "",
        "Benign cases (expected ALLOW)",
        f"  unnecessarily reviewed or blocked: {s.benign_escalated} of {s.benign_total}",
        "",
        "Decision bases (why each case ended where it did)",
        *(f"  {b.value}: {n}" for b in DecisionBasis if (n := sum(b in decision_bases(o.result) for o in outcomes))),
        "  An evidence-requirement review means a decision-critical claim lacks corroboration. It does not mean anyone lied.",
        "",
        "Findings emitted, by category (all cases)",
    ]
    out += [f"  {cat.value}: {s.finding_categories[cat]}" for cat in FindingCategory]
    out += ["", "By attack family (expected | detected)"]
    width = max(len(f.value) for f in s.by_family)
    for fam, row in sorted(s.by_family.items(), key=lambda kv: kv[0].value):
        out.append(f"  {fam.value:<{width}}  n={row.total}  {_counts(row.expected)}  |  {_counts(row.detected)}  agree={row.agree}")

    out += ["", "Disagreements between detected and expected"]
    if not s.disagreements:
        out.append("  none")
    for o in s.disagreements:
        out.append(f"  {o.case.case_id} ({o.truth.attack_family.value}): expected {o.truth.ideal_decision.value}, detected {o.result.decision.value}")
        out.append(f"    why: {o.truth.rationale}")

    if show_cases:
        out += ["", "Per case"]
        for o in outcomes:
            mark = " " if o.result.decision is o.truth.ideal_decision else "*"
            codes = ", ".join(f.code.value for f in o.result.findings) or "no findings"
            out.append(
                f" {mark}{o.case.case_id}  {o.truth.attack_family.value:<30} expected {o.truth.ideal_decision.value:<6} "
                f"detected {o.result.decision.value:<6} authorised={o.result.authorised!s:<5}  {codes}"
            )
    return "\n".join(out)


SENSITIVITY_GAPS_MINOR = (0, 500, 1000, 5000, 7500, 10000, 20000)


def format_sensitivity() -> str:
    """How the provisional evidence-requirement assumptions change Condition A on the 20 development fixtures.

    The labels appear here only to count unnecessary reviews and misses. The fixtures are development data, so this
    is a description of how sensitive the rule is, not a performance estimate and not a basis for choosing a threshold."""
    cases = build_cases()
    rows = []

    def row(label: str, config: PolicyConfig) -> str:
        results = [(c, evaluate_case(c, config)) for c in cases]
        required = [c.case_id for c, r in results if r.has(FindingCategory.EVIDENCE_REQUIREMENT_UNMET, material_only=True)]
        reviewed = [c.case_id for c, r in results if r.decision is Decision.REVIEW]
        unnecessary = [c.case_id for c, r in results if r.decision is Decision.REVIEW and LABELS[c.case_id].ideal_decision is Decision.ALLOW]
        missed = [c.case_id for c, r in results if r.decision is Decision.ALLOW and LABELS[c.case_id].ideal_decision is Decision.REVIEW]
        return f"  {label:<40} reviews={len(reviewed):<2} evidence-requirement={','.join(x[5:] for x in required) or '-':<9} unnecessary={','.join(x[5:] for x in unnecessary) or '-':<9} missed={','.join(x[5:] for x in missed) or '-'}"

    rows.append(row("requirement OFF", PolicyConfig(require_corroboration=False)))
    for stale in (True, False):
        for gap in SENSITIVITY_GAPS_MINOR:
            label = f"gap >= {gap / 100:>6.2f} GBP, stale {'counts' if stale else 'ignored'}"
            rows.append(row(label, PolicyConfig(evidence_requirement_gap_minor=gap, stale_evidence_counts_as_uncorroborated=stale)))
    return "\n".join(
        [
            "Sika Guard - sensitivity of the evidence-requirement rule (Condition A, 20 development fixtures)",
            "Provisional policy assumptions: the customer asked to minimise cost; the claim says an authorised offer is unavailable;",
            "no fresh independent evidence confirms it (stale-only evidence counts or not); the cheaper offer is cheaper by at least the gap.",
            "These fixtures are development data and were written alongside the rules. This table shows sensitivity, not accuracy,",
            "and must not be used to pick the threshold that scores best.",
            "",
            *rows,
            "",
            "Case numbers are shown without the 'case-' prefix. 'unnecessary' = reviewed although the fixture label is ALLOW;",
            "'missed' = allowed although the fixture label is REVIEW.",
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sika-guard-eval",
        description=(
            "Evaluate the deterministic baseline (A), or add one general-purpose analyst (B), three samples of it (B3), "
            "one analyst applying all three specialist rubrics (Bc), or three concurrent specialists (C)."
        ),
    )
    parser.add_argument(
        "--condition", choices=[c.value for c in Condition], default="A",
        help="A: deterministic baseline (default); B: plus one analyst; B3: plus three samples of that analyst (same number of calls as C; compute not claimed equal); "
        "Bc: plus one analyst using all three specialist rubrics (same information as C, one call); C: plus three specialists",
    )
    parser.add_argument("--offline", choices=OFFLINE_MODES, help="B/C only: use scripted mock evaluators instead of a model API (checks plumbing, not AI)")
    parser.add_argument("--cases", action="store_true", help="also print one line per case")
    parser.add_argument("--sensitivity", action="store_true", help="Condition A only: show how the evidence-requirement assumptions change the baseline")
    parser.add_argument("--case", action="append", dest="case_ids", metavar="ID", help="evaluate only this case id (repeatable)")
    parser.add_argument("--limit", type=int, metavar="N", help="evaluate only the first N cases")
    parser.add_argument("--fail-closed", action="store_true", help="B/C only: escalate ALLOW to REVIEW when an evaluator yields no usable verdict")
    parser.add_argument("--output-json", type=Path, metavar="PATH", help="B/C only: also write raw records to this file")
    return parser


def _select_cases(args: argparse.Namespace) -> list[Case]:
    cases = build_cases()
    if args.case_ids:
        unknown = sorted(set(args.case_ids) - {c.case_id for c in cases})
        if unknown:
            raise SystemExit(f"unknown case id(s): {', '.join(unknown)}")
        cases = [c for c in cases if c.case_id in args.case_ids]
    return cases[: args.limit] if args.limit else cases


def _missing_config_message(missing: tuple[str, ...]) -> str:
    return "\n".join(
        [
            "Cannot run Condition B/C against a live model: missing environment variables: " + ", ".join(missing or REQUIRED_ENV),
            f"Required: {', '.join(REQUIRED_ENV)}.",
            f"Optional: {ENV_TIMEOUT} (seconds, default 60), {ENV_TEMPERATURE}, {ENV_JSON_MODE} (0 to omit response_format), "
            f"{ENV_PRICE_IN} and {ENV_PRICE_OUT} (both needed to report cost).",
            "Use --offline allow|review|timeout|invalid to exercise the pipeline with mock evaluators instead.",
        ]
    )


async def _run_ai(args: argparse.Namespace) -> int:
    condition = Condition(args.condition)
    specs = list(specs_for(condition.value))
    cases = _select_cases(args)
    config = CoordinatorConfig(fail_closed=args.fail_closed)

    async with AsyncExitStack() as stack:
        llm_description = None
        if condition is Condition.A_PLUS:
            mode, evaluators = "deterministic-scanner (no model, no network)", evaluators_for("A+", None)
        elif args.offline:
            mode = f"offline-mock:{args.offline}"
            evaluators = offline_evaluators(args.offline, [spec.role for spec in specs], [spec.evaluator_id for spec in specs])
        else:
            try:
                llm_config = LLMConfig.from_env()
            except ConfigError as exc:
                print(_missing_config_message(exc.missing) if exc.missing else f"Configuration error: {exc}", file=sys.stderr)
                return 2
            client = await stack.enter_async_context(LLMClient(llm_config))
            evaluators = evaluators_for(condition.value, client)
            llm_description, mode = llm_config.describe(), "live"
        records = await run_condition(cases, condition, evaluators, config)

    seen = {v.evaluator_id: {"evaluator_id": v.evaluator_id, "prompt_version": v.prompt_version, "model_id": v.model_id} for r in records for v in r.verdicts}
    meta = RunMetadata(
        condition=condition,
        mode=mode,
        evaluators=tuple(seen.values()),
        llm=llm_description,
        fail_closed=args.fail_closed,
        policy={
            "max_evidence_age_minutes": config.policy.max_evidence_age.total_seconds() / 60,
            "material_gap_minor": config.policy.material_gap_minor,
            "aggregation": AGGREGATION_POLICY,
        },
    )
    scored = score(records, LABELS)  # labels are joined only now, after inference
    print(format_experiment_report(meta, scored, show_cases=args.cases))
    if args.output_json:
        export_json(args.output_json, meta, scored)
        print(f"\nRaw records written to {args.output_json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.sensitivity:
        if args.condition != "A":
            parser.error("--sensitivity describes the deterministic rule engine and applies to Condition A only")
        print(format_sensitivity())
        return 0
    if args.condition == "A":
        if args.offline or args.fail_closed or args.output_json:
            parser.error("--offline, --fail-closed and --output-json apply to Conditions B and C only")
        print(format_report(run_baseline(), show_cases=args.cases))
        return 0
    return asyncio.run(_run_ai(args))


if __name__ == "__main__":
    raise SystemExit(main())
