"""Descriptive uncertainty for paired comparisons, clustered by scenario.

Cases that share a template (a scenario and its paraphrases) are not independent experiments, so the unit
of resampling is the cluster. The bootstrap is deterministic (fixed seed) and percentile-based. It describes
how much the observed difference could move if different clusters had been drawn; it is not a power
calculation and not a test of significance. The guards below (`MIN_CLUSTERS`, `MIN_DISCORDANT`) are blunt
rules to avoid over-reading tiny samples, not statistically derived thresholds.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

MIN_CLUSTERS = 10
MIN_DISCORDANT = 5
DEFAULT_BOOT = 2000
DEFAULT_SEED = 20_240_611


@dataclass(frozen=True)
class Interval:
    point: float | None
    low: float | None
    high: float | None
    cases: int
    clusters: int

    @property
    def defined(self) -> bool:
        return self.low is not None

    def __str__(self) -> str:
        if self.point is None:
            return "n/a (no cases)"
        if self.low is None:
            return f"{self.point:.3f} (no interval: {self.clusters} cluster(s))"
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] ({self.cases} cases, {self.clusters} clusters)"


def cluster_bootstrap(
    values: Mapping[str, float], clusters: Mapping[str, str], *, n_boot: int = DEFAULT_BOOT, seed: int = DEFAULT_SEED, alpha: float = 0.05
) -> Interval:
    """Mean of `values` with a percentile interval from resampling whole clusters with replacement."""
    if not values:
        return Interval(None, None, None, 0, 0)
    groups: dict[str, list[float]] = {}
    for case_id, v in values.items():
        groups.setdefault(clusters[case_id], []).append(v)
    sums = [(sum(g), len(g)) for g in groups.values()]
    total, n = sum(s for s, _ in sums), sum(c for _, c in sums)
    point = total / n
    if len(sums) < 2:
        return Interval(point, None, None, n, len(sums))
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_boot):
        picks = [sums[rng.randrange(len(sums))] for _ in range(len(sums))]
        estimates.append(sum(s for s, _ in picks) / sum(c for _, c in picks))
    estimates.sort()
    low = estimates[int((alpha / 2) * n_boot)]
    high = estimates[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return Interval(point, low, high, n, len(sums))


class Judgement(StrEnum):
    IMPROVEMENT_DEMONSTRATED = "improvement demonstrated"
    WORSE_DEMONSTRATED = "worse performance demonstrated"
    NO_DEMONSTRATED_DIFFERENCE = "no demonstrated difference"  # not the same as demonstrated equivalence
    INSUFFICIENT_EVIDENCE = "evidence insufficient to decide"


def judge(difference: Interval, discordant: int, *, higher_is_better: bool) -> Judgement:
    """Read a paired difference (first condition minus second) conservatively.

    "No demonstrated difference" means the interval includes zero. It does not mean the conditions are equal."""
    if not difference.defined or difference.clusters < MIN_CLUSTERS or discordant < MIN_DISCORDANT:
        return Judgement.INSUFFICIENT_EVIDENCE
    assert difference.low is not None and difference.high is not None
    if difference.low > 0 or difference.high < 0:
        first_is_higher = difference.low > 0
        return Judgement.IMPROVEMENT_DEMONSTRATED if first_is_higher == higher_is_better else Judgement.WORSE_DEMONSTRATED
    return Judgement.NO_DEMONSTRATED_DIFFERENCE


# --- confirmatory testing -------------------------------------------------------------------------------------------------

DEFAULT_PERMUTATIONS = 20_000
EXACT_ENUMERATION_MAX_CLUSTERS = 16
ALPHA = 0.05


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm's step-down adjusted p-values (family-wise error control, no independence assumption).

    The family size is len(pvalues). A hypothesis that could not be tested must be passed as 1.0 rather than dropped,
    so that untestable comparisons never make the others look more significant."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def signflip_pvalue(cluster_sums: list[float], n_cases: int, *, n_perm: int = DEFAULT_PERMUTATIONS, seed: int = DEFAULT_SEED) -> float:
    """Two-sided sign-flip randomisation test of a paired mean difference, resampling at the cluster level.

    Under the null that the two conditions are interchangeable, the sign of each scenario cluster's total paired difference is
    arbitrary. The statistic is |mean difference|; clusters, not cases or repeats, are the units that are flipped. Exact for up
    to 16 clusters, Monte Carlo (with the +1 correction) beyond that. A test of "the conditions differ": a large p-value is not
    evidence that they are the same."""
    if n_cases <= 0 or not cluster_sums:
        return 1.0
    observed = abs(sum(cluster_sums)) / n_cases
    eps = 1e-12
    if len(cluster_sums) <= EXACT_ENUMERATION_MAX_CLUSTERS:
        totals = [0.0]
        for s in cluster_sums:
            totals = [t + s for t in totals] + [t - s for t in totals]
        return sum(abs(t) / n_cases >= observed - eps for t in totals) / len(totals)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        hits += abs(sum(s if rng.random() < 0.5 else -s for s in cluster_sums)) / n_cases >= observed - eps
    return (hits + 1) / (n_perm + 1)
