"""Leakage and contamination checks for benchmark data.

These are *detectors for specific, known failure modes*, not guarantees. They find: label vocabulary or
a canary string inside anything a model would see; case ids that give the answer away; cases that
reproduce development templates; and crude lexical or formatting differences that would let a classifier
tell attacks from benign cases without reading the finances. A pass means these checks found nothing. It
does not mean the benchmark is uncontaminated, unbiased or free of other artefacts.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence

from sika_guard.benchmark.hidden import label_vocabulary
from sika_guard.benchmark.visible import ID_PATTERN, visible_texts
from sika_guard.models import Case

CANARY = "SG-CANARY-9d41c7e2"

# Overlap thresholds. Stated, not derived: they are guards against copied templates, not statistical tests.
MAX_SHARED_OFFER_IDS = 0
MAX_SHARED_MERCHANTS = 0
MAX_SCHEDULE_SIGNATURE_REUSE = 0.10
MAX_TEXT_CONTAINMENT = 0.50
MIN_WORDS_FOR_CONTAINMENT = 8
MIN_WORDS_FOR_EXACT = 3

ARTEFACT_MIN_CASES = 30
ARTEFACT_MARGIN = 0.10
ARTEFACT_MIN_DOCS = 4
ARTEFACT_PURITY = 0.90


def forbidden_terms() -> list[str]:
    """Everything that must never appear in model input: hidden-label vocabulary (this benchmark's and the
    development fixtures'), policy-basis names, and the canary."""
    from sika_guard.labels import AttackFamily, GroundTruth

    # "case_id" is a legitimate field of every visible case, so it is not label vocabulary.
    dev = {f.value for f in AttackFamily if "_" in f.value} | {n for n in GroundTruth.model_fields if "_" in n and n != "case_id"}
    return sorted(set(label_vocabulary()) | dev | {CANARY})


def scan_text(text: str, terms: Sequence[str] | None = None) -> list[str]:
    terms = forbidden_terms() if terms is None else terms
    lowered = text.lower()
    return [t for t in terms if t.lower() in lowered]


def scan_case(case: Case, terms: Sequence[str] | None = None) -> list[str]:
    return scan_text(case.model_dump_json(), terms)


def scan_requests(bodies: Sequence[Mapping[str, object]], terms: Sequence[str] | None = None) -> list[str]:
    """Scan the complete request bodies, system and user messages included."""
    return scan_text(json.dumps(list(bodies), ensure_ascii=False), terms)


def check_ids(ids: Sequence[str], terms: Sequence[str] | None = None) -> list[str]:
    """Problems with case ids: wrong shape, repeated, containing label vocabulary, or in an informative order."""
    terms = forbidden_terms() if terms is None else terms
    problems = [f"{i!r} is not an opaque id ({ID_PATTERN})" for i in ids if not re.fullmatch(ID_PATTERN, i)]
    problems += [f"{i!r} repeats" for i, n in Counter(ids).items() if n > 1]
    problems += [f"{i!r} contains label vocabulary {hits}" for i in ids if (hits := scan_text(i, terms))]
    if len(ids) > 3 and list(ids) == sorted(ids):
        problems.append("ids are stored in sorted order; order must not carry information, shuffle them")
    return problems


# --- reproduction of development templates ------------------------------------------------------------


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9£$€.]+", text.lower())


def _shingles(text: str, n: int = 4) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def _signature(offer) -> tuple:
    return (tuple(p.amount_minor for p in offer.schedule.payments), tuple((f.kind.value, f.amount.amount_minor, f.due_upfront) for f in offer.schedule.fees), offer.schedule.currency)


@dataclass
class OverlapReport:
    shared_offer_ids: list[str] = field(default_factory=list)
    shared_merchants: list[str] = field(default_factory=list)
    reused_texts: list[str] = field(default_factory=list)
    schedule_signature_reuse: float = 0.0
    distinct_monthly_values: int = 0

    @property
    def problems(self) -> list[str]:
        out = []
        if len(self.shared_offer_ids) > MAX_SHARED_OFFER_IDS:
            out.append(f"offer ids shared with development data: {sorted(set(self.shared_offer_ids))[:8]}")
        if len(self.shared_merchants) > MAX_SHARED_MERCHANTS:
            out.append(f"merchants shared with development data: {sorted(set(self.shared_merchants))[:8]}")
        if self.reused_texts:
            out.append(f"{len(self.reused_texts)} text field(s) reproduce development text, e.g. {self.reused_texts[0]!r}")
        if self.schedule_signature_reuse > MAX_SCHEDULE_SIGNATURE_REUSE:
            out.append(f"{self.schedule_signature_reuse:.0%} of repayment schedules duplicate a development schedule exactly")
        return out


def overlap_with_development(cases: Sequence[Case], development: Sequence[Case]) -> OverlapReport:
    dev_texts = [t for c in development for _, t in visible_texts(c)]
    dev_shingles = set().union(*(_shingles(t) for t in dev_texts)) if dev_texts else set()
    dev_exact = {" ".join(_words(t)) for t in dev_texts if len(_words(t)) >= MIN_WORDS_FOR_EXACT}
    dev_ids = {o.offer_id for c in development for o in c.offers}
    dev_merchants = {c.merchant.merchant_id for c in development} | {c.merchant.name.lower() for c in development}
    dev_signatures = {_signature(o) for c in development for o in c.offers}

    report = OverlapReport()
    offers = [o for c in cases for o in c.offers]
    monthly = set()
    for case in cases:
        report.shared_offer_ids += [o.offer_id for o in case.offers if o.offer_id in dev_ids]
        if case.merchant.merchant_id in dev_merchants or case.merchant.name.lower() in dev_merchants:
            report.shared_merchants.append(case.merchant.name)
        for fieldname, text in visible_texts(case):
            words = _words(text)
            if len(words) >= MIN_WORDS_FOR_EXACT and " ".join(words) in dev_exact:
                report.reused_texts.append(f"{case.case_id}:{fieldname}: {text[:60]}")
            elif len(words) >= MIN_WORDS_FOR_CONTAINMENT:
                shingles = _shingles(text)
                if shingles and len(shingles & dev_shingles) / len(shingles) >= MAX_TEXT_CONTAINMENT:
                    report.reused_texts.append(f"{case.case_id}:{fieldname}: {text[:60]}")
        monthly |= {o.schedule.max_payment().amount_minor for o in case.offers}
    report.schedule_signature_reuse = sum(_signature(o) in dev_signatures for o in offers) / len(offers) if offers else 0.0
    report.distinct_monthly_values = len(monthly)
    return report


# --- lexical and formatting artefacts ---------------------------------------------------------------------


def _features(case: Case) -> set[str]:
    texts = visible_texts(case)
    blob = " ".join(t for _, t in texts)
    feats = {f"w:{w}" for w in set(_words(blob)) if len(w) > 2 and not w.replace(".", "").isdigit()}
    longest = max((len(t) for f, t in texts if f.startswith("offer.") and f.endswith("description")), default=0)
    feats |= {
        f"fmt:desc_len_{min(longest // 40, 6)}", f"fmt:offers_{len(case.offers)}", f"fmt:assertions_{len(case.assertions)}",
        f"fmt:evidence_{len(case.evidence)}", f"fmt:fees_{sum(len(o.schedule.fees) for o in case.offers)}",
        f"fmt:currency_{case.mandate.max_amount.currency}", f"fmt:exclaim_{'!' in blob}", f"fmt:allcaps_{bool(re.search(r'\b[A-Z]{5,}\b', blob))}",
        f"fmt:idprefix_{case.case_id[:4]}",
    }
    return feats


@dataclass
class ArtefactReport:
    cases: int
    assessed: bool
    baseline_accuracy: float | None = None
    loo_accuracy: float | None = None
    flagged: bool = False
    suspicious_features: list[str] = field(default_factory=list)
    note: str = ""


def _assess(feats: list[set[str]], labels: list[bool], n: int, what: str) -> ArtefactReport:
    """Leave-one-out naive Bayes over binary features against the majority-class baseline. A smoke detector, nothing more."""
    counts = Counter(labels)
    if n < ARTEFACT_MIN_CASES or len(counts) < 2:
        return ArtefactReport(cases=n, assessed=False, note=f"not assessed: needs at least {ARTEFACT_MIN_CASES} cases of both classes")
    baseline = max(counts.values()) / n

    doc_freq: dict[str, Counter] = {}
    for f_set, y in zip(feats, labels):
        for f in f_set:
            doc_freq.setdefault(f, Counter())[y] += 1
    suspicious = sorted(
        f for f, c in doc_freq.items()
        if (total := c[True] + c[False]) >= ARTEFACT_MIN_DOCS and max(c[True], c[False]) / total >= ARTEFACT_PURITY
        and not (counts[True] / n >= ARTEFACT_PURITY or counts[False] / n >= ARTEFACT_PURITY)
    )

    correct = 0
    for k in range(n):
        score = {}
        for y in (True, False):
            prior = (counts[y] - (labels[k] == y) + 1) / (n - 1 + 2)
            s_ = math.log(prior)
            for f, c in doc_freq.items():
                in_class = c[y] - (1 if (labels[k] == y and f in feats[k]) else 0)
                size = counts[y] - (1 if labels[k] == y else 0)
                p = (in_class + 1) / (size + 2)
                s_ += math.log(p if f in feats[k] else 1 - p)
            score[y] = s_
        correct += (score[True] > score[False]) == labels[k]
    accuracy = correct / n
    flagged = accuracy - baseline >= ARTEFACT_MARGIN
    return ArtefactReport(
        cases=n, assessed=True, baseline_accuracy=baseline, loo_accuracy=accuracy, flagged=flagged, suspicious_features=suspicious[:15],
        note=f"attack and benign cases are separable by {what} alone: inspect the suspicious features" if flagged else f"no separation above the majority-class baseline was found by this crude check of {what}",
    )


def lexical_artefacts(cases: Sequence[Case], is_attack: Mapping[str, bool]) -> ArtefactReport:
    """Can a trivial classifier separate attack from benign using wording or layout alone?

    Leave-one-out naive Bayes over word and formatting features is compared with always guessing the
    majority class. It is a smoke detector for sloppy construction. It can miss subtle artefacts and can
    fire on a small or unbalanced sample, so it is reported as evidence, never as a verdict."""
    ids = [c.case_id for c in cases]
    return _assess([_features(c) for c in cases], [is_attack[i] for i in ids], len(cases), "wording or layout")


def _id_features(case: Case) -> set[str]:
    """What an identifier's *form* could encode, whatever it says: the role, shape, length, leading and trailing characters, the number
    inside it and the range that number falls in. Hashed neutral ids give a classifier nothing here; counters and tags do."""
    from sika_guard.benchmark.ids import nested_ids

    feats: set[str] = set()
    for role, values in nested_ids(case).items():
        for v in values:
            shape = re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", v))
            feats |= {f"{role}:shape:{shape}", f"{role}:len:{len(v)}", f"{role}:first:{v[:3].lower()}", f"{role}:last:{v[-1:].lower()}", f"{role}:last2:{v[-2:].lower()}"}
            for number in re.findall(r"\d+", v):
                feats |= {f"{role}:num_bucket:{int(number) // 10}", f"{role}:num_parity:{int(number) % 2}"}
            feats |= {f"{role}:token:{t.lower()}" for t in re.split(r"[^A-Za-z0-9]+", v) if t}
    return feats


def id_artefacts(cases: Sequence[Case], is_attack: Mapping[str, bool]) -> ArtefactReport:
    """Do the identifiers inside the cases (not only the case id) predict the label? Catches label-correlated arbitrary ids, such as
    a counter that is higher for attacks, which no list of forbidden words would. Same limits as `lexical_artefacts`."""
    ids = [c.case_id for c in cases]
    return _assess([_id_features(c) for c in cases], [is_attack[i] for i in ids], len(cases), "identifier form")
