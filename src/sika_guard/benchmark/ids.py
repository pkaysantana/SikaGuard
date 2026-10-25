"""Neutral identifiers for every entity in a case, applied consistently.

An identifier a model reads is part of its input. `TX-ARC_FALSE` tells it the scenario; `OFFER-X7` versus `OFFER-A` can encode a
label just as well. So every nested identifier (transaction, mandate, customer, agent, acting agent, merchant, offer, payee, lender, assertion,
evidence, document and source ids) is replaced by `id-<8 hex>`, a hash of a salt and the original. The mapping is a function of the
*string*, so every relationship between entities survives (a transaction still names its offer, a lender-kind evidence source
still carries the offer's lender id). The salt is per cluster, not per case, so a paired twin built from the same world keeps
the same offer ids and the counterfactual's recorded choice still refers to the same offer.

This module is builder-side tooling: it touches labels as well as cases, so inference code never imports it.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sika_guard.benchmark.hidden import HiddenLabels
from sika_guard.models import Case

NEUTRAL_ID = re.compile(r"^id-[0-9a-f]{8}$")
ID_FIELDS = frozenset(
    {"customer_id", "requested_offer_id", "mandate_id", "agent_id", "merchant_id", "offer_id", "lender_id", "payee_id", "transaction_id", "assertion_id", "evidence_id", "source_id", "acting_agent_id", "document_id"}
)
ID_LIST_FIELDS = frozenset({"allowed_payee_ids"})
REF_KINDS = ("assertion", "listing", "terms", "evidence", "merchant", "txn", "mandate", "intent", "doc")


def collect_ids(node: Any, found: set[str] | None = None) -> set[str]:
    found = set() if found is None else found
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ID_FIELDS and isinstance(value, str):
                found.add(value)
            elif key in ID_LIST_FIELDS and isinstance(value, list):
                found.update(v for v in value if isinstance(v, str))
            else:
                collect_ids(value, found)
    elif isinstance(node, list):
        for item in node:
            collect_ids(item, found)
    return found


def make_mapping(originals: set[str], salt: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for old in sorted(originals):
        width = 8
        while True:
            new = "id-" + hashlib.sha256(f"{salt}|{old}".encode("utf-8")).hexdigest()[:width]
            if new not in mapping.values():
                break
            width += 1  # a collision within one case; vanishingly rare, and resolved rather than ignored
        mapping[old] = new
    return mapping


def _rewrite(node: Any, mapping: dict[str, str]) -> Any:
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in ID_FIELDS and isinstance(value, str):
                out[key] = mapping.get(value, value)
            elif key in ID_LIST_FIELDS and isinstance(value, list):
                out[key] = [mapping.get(v, v) for v in value]
            else:
                out[key] = _rewrite(value, mapping)
        return out
    if isinstance(node, list):
        return [_rewrite(item, mapping) for item in node]
    return node


def neutralise_case(case: Case, salt: str) -> tuple[Case, dict[str, str]]:
    """The same case with every nested identifier replaced. Returns the case and the mapping used (original -> neutral)."""
    data = json.loads(case.model_dump_json())
    mapping = make_mapping(collect_ids({k: v for k, v in data.items() if k != "case_id"}), salt)
    return Case.model_validate(_rewrite(data, mapping)), mapping


def _map_ref(ref: str, mapping: dict[str, str]) -> str:
    kind, sep, ident = ref.partition(":")
    return f"{kind}{sep}{mapping.get(ident, ident)}" if kind in REF_KINDS and sep else ref


def neutralise_labels(labels: HiddenLabels, mapping: dict[str, str]) -> HiddenLabels:
    """Re-point a label's references (claims, provenance, world state, counterfactual choice) at the neutral ids."""
    data = json.loads(labels.model_dump_json())
    for claim in data["claims"]:
        claim["claim_ref"] = _map_ref(claim["claim_ref"], mapping)
    for entry in data["provenance"]:
        entry["ref"] = _map_ref(entry["ref"], mapping)
    for offer in data["world_state"]:
        offer["offer_id"] = mapping.get(offer["offer_id"], offer["offer_id"])
    if data.get("counterfactual"):
        choice = data["counterfactual"]["choice_without_misinformation"]
        data["counterfactual"]["choice_without_misinformation"] = mapping.get(choice, choice)
    return HiddenLabels.model_validate(data)


def neutralise_pair(case: Case, labels: HiddenLabels, salt: str) -> tuple[Case, HiddenLabels]:
    new_case, mapping = neutralise_case(case, salt)
    return new_case, neutralise_labels(labels, mapping)


def nested_ids(case: Case) -> dict[str, list[str]]:
    """Every identifier in a case other than its own id, by role. These are what a model reads."""
    d = json.loads(case.model_dump_json())
    d.pop("case_id", None)
    found: dict[str, list[str]] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ID_FIELDS and isinstance(value, str):
                    found.setdefault(key, []).append(value)
                elif key in ID_LIST_FIELDS and isinstance(value, list):
                    found.setdefault(key, []).extend(v for v in value if isinstance(v, str))
                else:
                    walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    walk(d, "")
    return found


def non_neutral_ids(case: Case) -> list[str]:
    return sorted({v for values in nested_ids(case).values() for v in values if not NEUTRAL_ID.match(v)})
