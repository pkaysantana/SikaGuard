"""Is a model's extracted claim grounded in what that evaluator was actually shown?

A claim is trusted code's input only if: it cites a source item that is in *this evaluator's* request; that item concerns the offer the
claim is about (or, for an unbound document, the quoted words themselves name the offer); and the quote appears in one of that item's text
fields as the evaluator saw it. Checking against the full case instead would let a specialist "quote" a field it never received, and
checking only the reference would let a truthful quotation about one offer be attributed to another.

Source items come from two places in a request: `untrusted_merchant_content` (everything the merchant controls) and
`authenticated_source_documents` (passages whose origin is independent of the merchant: a lender's disclosure, an aggregator's notice).
Being in the second section says who produced a passage, never that what it says is true; `crosscheck` decides what it establishes.

Formatting is handled without trusting it. Matching folds Unicode compatibility forms (full-width digits and letters, no-break and
thin spaces, ligatures), removes invisible characters, and unifies quotation marks and dashes, so a quotation that differs from the
text only in formatting still grounds. The original text is never altered or discarded: what was folded is reported so a reviewer can
see it. Text whose reading is ambiguous in a way a fold could exploit (bidirectional controls that reorder displayed text, digits
from another script, superscript or circled digits that a fold would turn into ordinary ones) is flagged as a reading hazard, and
nothing is concluded from the amounts in it.

Pure functions over the evaluator's payload; no import of the contract types, so both the contract and the cross-check can use it.
"""

from __future__ import annotations

import unicodedata
from typing import Any

SOURCE_SECTIONS = ("untrusted_merchant_content", "authenticated_source_documents")

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­᠎"), None)
_PUNCTUATION = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"',
                              "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-"})
_BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩‎‏؜")
_FULLWIDTH_DIGITS = frozenset(chr(c) for c in range(0xFF10, 0xFF1A))


def canonical(text: str) -> tuple[str, tuple[str, ...]]:
    """The text in the form it is matched and read in, and the names of the folds that actually changed it. The original is untouched."""
    steps: list[str] = []
    out = text
    folded = unicodedata.normalize("NFKC", out)
    if folded != out:
        steps.append("compatibility_forms_folded")
        out = folded
    cleaned = out.translate(_INVISIBLE)
    if cleaned != out:
        steps.append("invisible_characters_removed")
        out = cleaned
    unified = out.translate(_PUNCTUATION)
    if unified != out:
        steps.append("quotes_and_dashes_unified")
        out = unified
    return out, tuple(steps)


def reading_hazards(text: str) -> tuple[str, ...]:
    """Reasons the amounts in `text` cannot be safely read, from the ORIGINAL text. Empty when the text is plain enough to trust."""
    found: list[str] = []
    if any(c in _BIDI_CONTROLS for c in text):
        found.append("bidirectional_controls")
    # A digit that is not ASCII and is not a plain full-width digit: another script's digits, or superscript/circled/fraction forms that
    # NFKC would turn into ordinary digits ("£68²" is a footnote marker, not "£682").
    for c in text:
        if c.isascii() or c in _FULLWIDTH_DIGITS:
            continue
        category = unicodedata.category(c)
        if category == "Nd" or (category == "No" and any(ch.isdigit() for ch in unicodedata.normalize("NFKC", c))):
            found.append("non_ascii_digits")
            break
    return tuple(found)


def walk_items(block: Any) -> list[dict[str, Any]]:
    """Every dict item in a payload section, whether the section is a mapping of lists, a list, or a single item."""
    if isinstance(block, dict) and isinstance(block.get("ref"), str):
        return [block]
    values = list(block.values()) if isinstance(block, dict) else block if isinstance(block, list) else []
    out: list[dict[str, Any]] = []
    for value in values:
        out.extend(walk_items(value))
    return out


def source_items(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every source item in the request, by ref, from both source sections. Lender terms and customer facts are not source items."""
    return {item["ref"]: item for section in SOURCE_SECTIONS for item in walk_items(payload.get(section))}


def merchant_items(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Only the items the merchant controls."""
    return {item["ref"]: item for item in walk_items(payload.get("untrusted_merchant_content"))}


def _norm(text: str) -> str:
    folded, _ = canonical(text)
    return " ".join(folded.replace('"', "").replace("'", "").split()).casefold()


def _plain(text: str) -> str:
    return " ".join(text.replace('"', "").replace("'", "").split()).casefold()


def locate_quote_detail(item: dict[str, Any], quote: str) -> tuple[str, tuple[str, ...]] | None:
    """(the text field of `item` that contains `quote`, the folds that were needed to match it), or None. Matching is whitespace- and
    case-insensitive and ignores quote marks, so a quotation of a JSON value or of `name: value` both count. If the quote only matches
    once Unicode formatting is folded, the folds are returned so a reviewer can see them."""
    if not _norm(quote):
        return None
    for key, value in item.items():
        if key == "ref" or not isinstance(value, str):
            continue
        candidates = (value, f"{key}: {value}")
        if any(_plain(quote) in _plain(c) for c in candidates):
            return key, ()
        for c in candidates:
            if _norm(quote) in _norm(c):
                return key, tuple(sorted(set(canonical(c)[1]) | set(canonical(quote)[1]))) or ("whitespace_or_case",)
    return None


def locate_quote(item: dict[str, Any], quote: str) -> str | None:
    found = locate_quote_detail(item, quote)
    return None if found is None else found[0]


def _offer_names(payload: dict[str, Any], offer_id: str) -> list[str]:
    """How this evaluator's own request refers to an offer: its id and any title shown for it."""
    names = [offer_id]
    for section in SOURCE_SECTIONS + ("trusted",):
        block = payload.get(section)
        for value in block.values() if isinstance(block, dict) else []:
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, dict) and item.get("offer_id") == offer_id and isinstance(item.get("title"), str) and item["title"].strip():
                    names.append(item["title"])
    return names


def claim_problem(payload: dict[str, Any], *, source_ref: str, offer_id: str, quote: str) -> str | None:
    """Why the claim is not grounded, or None if it is. The reason names references, never the model's quoted text."""
    item = source_items(payload).get(source_ref)
    if item is None:
        return f"{source_ref} is not a source in this evaluator's input"
    bound = item.get("offer_id")
    if bound is None and source_ref.startswith("doc:"):
        if not any(_norm(name) in _norm(quote) for name in _offer_names(payload, offer_id)):
            return f"{source_ref} is not bound to an offer, and the quoted words do not name {offer_id}"
    elif bound != offer_id:
        return f"{source_ref} concerns {bound}, but the claim is about {offer_id}"
    if locate_quote_detail(item, quote) is None:
        return f"the quote is not in {source_ref} as this evaluator was shown it"
    return None
