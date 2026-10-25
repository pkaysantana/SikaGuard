"""Sampling for the human label audit (see BENCHMARK_AUDIT_PROTOCOL.md). Construction-phase tooling.

Auditors are people, not models, so the packet shows the proposed labels next to the case. It is built to
spend human attention where labels are most likely to be wrong or contestable: everything flagged by
validation, every disagreement between the ideal and the policy's action, every case whose truth the
evaluator cannot establish, every case near the materiality gap, every causation label, plus a random
slice of the rest.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence

from sika_guard.benchmark.hidden import CausalLabel, HiddenLabels
from sika_guard.benchmark.quotas import band_of
from sika_guard.benchmark.validate import DatasetReport
from sika_guard.models import Case, Decision


def audit_packet(cases: Sequence[Case], labels: Mapping[str, HiddenLabels], report: DatasetReport, *, fraction: float = 0.10, seed: int = 7) -> list[dict]:
    reasons: dict[str, list[str]] = {}

    def flag(case_id: str, why: str) -> None:
        reasons.setdefault(case_id, []).append(why)

    for item in report.audit_items:
        case_id, _, why = item.partition(":")
        flag(case_id, why.strip())
    for cid, x in labels.items():
        if x.ideal_action is not x.expected_action_under_policy:
            flag(cid, "ideal action differs from the policy's action")
        if any(not c.verifiable_from_inputs for c in x.claims):
            flag(cid, "a claim the evaluator cannot verify from its inputs")
        if x.selected_inferior and x.avoidable_cost_minor and band_of(x.avoidable_cost_minor) == "around the gap":
            flag(cid, "avoidable cost is near the materiality gap")
        if x.causal_selection is not CausalLabel.NOT_APPLICABLE:
            flag(cid, f"causal label {x.causal_selection.value}")
        if x.counterfactual is not None:
            flag(cid, "paired-twin construction")
        if not x.attack_present and x.ideal_action is Decision.ALLOW and x.claims:
            flag(cid, "benign case that contains a disputed claim: confirm it is realistic and genuinely legitimate")
    rest = sorted(set(labels) - set(reasons))
    rng = random.Random(seed)
    for cid in rng.sample(rest, k=min(len(rest), max(0, round(fraction * len(labels))))):
        flag(cid, "random draw")
    by_id = {c.case_id: c for c in cases}
    return [
        {"case_id": cid, "reasons": why, "visible_case": json.loads(by_id[cid].model_dump_json()), "proposed_labels": json.loads(labels[cid].model_dump_json())}
        for cid, why in sorted(reasons.items())
        if cid in by_id
    ]
