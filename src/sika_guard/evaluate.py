"""Baseline evaluator: run the deterministic policy over the fixtures and score it.

"Expected" always means the fixture author's ideal decision (ground truth).
"Detected" always means what the deterministic policy actually returned. The
policy only ever sees the `Case`; labels are joined afterwards for scoring.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field

from sika_guard.fixtures.cases import build_cases
from sika_guard.fixtures.labels import LABELS
from sika_guard.labels import AttackFamily, GroundTruth
from sika_guard.models import Case, Decision, FindingCategory, SecurityDecision
from sika_guard.policy import DEFAULT_CONFIG, PolicyConfig, evaluate_case


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sika-guard-eval", description="Evaluate the deterministic baseline on the synthetic fixtures.")
    parser.add_argument("--cases", action="store_true", help="also print one line per case")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(format_report(run_baseline(), show_cases=args.cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
