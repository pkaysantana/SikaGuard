"""Model-visible benchmark data: exactly the existing `Case`, and nothing else.

A visible case file contains one JSON object per line, `{"schema": ..., "case": {...}}`. `Case` forbids
extra fields, so a label cannot ride along inside it, and this loader additionally refuses any line
with another top-level key. This module must not import the hidden-label module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sika_guard.evaluators.contract import EvaluatorView
from sika_guard.models import Case

VISIBLE_SCHEMA = "sika-guard-benchmark-visible/1"
ID_PATTERN = r"^bm-[0-9a-f]{10}$"


def opaque_id(*material: str) -> str:
    """A case id derived from construction material by hashing. It carries no meaning; ids are shuffled in files."""
    return "bm-" + hashlib.sha256("\x1f".join(material).encode("utf-8")).hexdigest()[:10]


def dump_visible(cases: list[Case]) -> str:
    return "".join(json.dumps({"schema": VISIBLE_SCHEMA, "case": json.loads(c.model_dump_json())}, sort_keys=True) + "\n" for c in cases)


def parse_visible(text: str) -> list[Case]:
    cases = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if set(obj) != {"schema", "case"} or obj["schema"] != VISIBLE_SCHEMA:
            raise ValueError(f"line {number}: a visible case line carries exactly 'schema' and 'case', nothing else")
        cases.append(Case.model_validate(obj["case"]))
    return cases


def load_visible(path: Path) -> list[Case]:
    return parse_visible(path.read_text(encoding="utf-8"))


def visible_file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_view(case: Case) -> EvaluatorView:
    """What an evaluator is given: the existing label-free view."""
    return EvaluatorView.from_case(case)


def visible_texts(case: Case) -> list[tuple[str, str]]:
    """Every free-text or name field a person or model could read, as (field, text). Used by overlap and artefact checks."""
    texts = [("merchant.name", case.merchant.name), ("intent.instruction", case.intent.instruction)]
    for offer in case.offers:
        texts += [(f"offer.{offer.offer_id}.title", offer.title), (f"offer.{offer.offer_id}.description", offer.description)]
    texts += [(f"assertion.{a.assertion_id}", a.statement) for a in case.assertions]
    texts += [(f"evidence_source.{e.evidence_id}", e.source.name) for e in case.evidence]
    return [(f, t) for f, t in texts if t.strip()]
