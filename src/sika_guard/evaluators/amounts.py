"""Deterministic reading of monetary amounts out of text, with the quantity each amount states.

Used to check that a figure a model extracted is the figure its quote states, *as the kind of figure the claim says it is*.
Three rules keep it honest:

  * An amount expression is consumed and validated as a whole. "£680k", "£0.68m", "£680,50", "£680.123" and "£6,80" are
    recognised as money-like and reported as malformed; they are never partially parsed into "£680" or "£0.68".
  * Each amount is labelled by the words next to it: a total repayable, a monthly payment, a fee, a stated total of fees,
    or unknown. A monthly payment is not a total and a fee is not a repayment.
  * Other amounts in the same quote do not hide an explicit one: each mention is read in its own clause.

It reads exact decimals only. Text it cannot read gives no amount; it never guesses. This is a lexical aid, not natural-language
understanding: wording it does not recognise yields `UNKNOWN`, which downstream is uncertainty, never a verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from pydantic import ValidationError

from sika_guard.evaluators.grounding import canonical
from sika_guard.money import CURRENCY_SYMBOLS, Money

_SYMBOLS = "".join(re.escape(s) for s in CURRENCY_SYMBOLS.values())
_BY_SYMBOL = {symbol: code for code, symbol in CURRENCY_SYMBOLS.items()}
_CODES = "|".join(CURRENCY_SYMBOLS)
_SCALE_WORDS = {"k", "m", "mn", "bn", "b", "thousand", "million", "billion", "hundred", "grand"}
_NUMBER_OK = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?$")

_SYMBOL_FORM = re.compile(rf"(?P<sym>[{_SYMBOLS}])\s?(?P<num>\d[\d.,]*)(?P<suffix>[A-Za-z]+)?")
_CODE_BEFORE = re.compile(rf"\b(?P<code>{_CODES})\s?(?P<num>\d[\d.,]*)(?P<suffix>[A-Za-z]+)?")
_CODE_AFTER = re.compile(rf"(?<![\w.,{_SYMBOLS}])(?P<num>\d[\d.,]*)(?P<scale>\s?(?:k|m|mn|bn|thousand|million|billion))?\s?(?P<code>{_CODES})\b", re.I)


class Quantity(StrEnum):
    TOTAL = "total_repayable"  # the whole sum to be repaid
    MONTHLY = "monthly_payment"  # one instalment
    FEE = "fee"  # a single fee or charge
    FEE_TOTAL = "fee_total"  # an explicitly stated total of fees
    UNKNOWN = "unknown"  # the words around it do not say


@dataclass(frozen=True)
class AmountMention:
    text: str
    money: Money | None  # None when the expression is malformed
    quantity: Quantity
    problem: str | None = None
    conditional: bool = False  # the clause makes it depend on something ("if", "optional", "late", "may"): not an obligation on everyone


# Clauses are separated at punctuation and at "and / but / plus", so one amount's words do not label another's. A colon is not a break:
# it attaches a label to its amount ("Fee: £20 total").
# ("including" and "with" do NOT separate clauses: "total cost including fees £680" must stay one phrase.)
_CLAUSE_BREAK = re.compile(r"[;]|,\s|\.\s|\s[-\u2013\u2014]\s|\s(?:and|but|plus)\s", re.I)

# Words that name a quantity, used only when no decisive pattern applies and exactly one quantity is in play.
_TOTAL_KW = re.compile(r"\b(?:total|in total|in all|overall|altogether|repayable|to repay|all[- ]in|full cost|cost of credit)\b", re.I)
_MONTHLY_KW = re.compile(r"\ba month\b|\bper month\b|\beach month\b|\bevery month\b|\bmonthly\b|/\s?mo(?:nth)?\b|\binstal?ments?\b|\bpayments? of\b|\ba week\b|\bper week\b", re.I)
_FEE_KW = re.compile(r"\b(?:fees?|charges?|admin(?:istration)?|arrangement|servicing|set-?up)\b", re.I)

# Decisive wording, checked first. "After" patterns read the text directly following the amount, "before" patterns the text
# directly preceding it. A unit right after the amount ("per month") decides it, even beside a "total" before it.
_CONNECT = r"\s*(?:of|is|are|=|:)?\s*$"
_MONTHLY_AFTER = re.compile(r"^\s?(?:/\s?mo(?:nth)?\b|per month\b|a month\b|each month\b|every month\b|monthly\b|a week\b|per week\b)", re.I)
_MONTHLY_BEFORE = re.compile(r"(?:\bmonthly(?: payments?| instal?ments?)?|\bpayments? of|\binstal?ments? of)" + _CONNECT, re.I)
_FEE_TOTAL_BEFORE = re.compile(r"\btotal (?:\w+ )?(?:fees?|charges?)" + _CONNECT, re.I)
_FEE_TOTAL_AFTER = re.compile(r"^\s?(?:in (?:total )?(?:fees|charges)\b|total fees\b)", re.I)
_TOTAL_BEFORE = re.compile(r"(?:\btotal(?: repayable| cost| amount| payable)?|\brepayable|\bto repay|\boverall(?: cost)?)" + _CONNECT, re.I)
_TOTAL_AFTER = re.compile(r"^\s?(?:in total\b|in all\b|overall\b|altogether\b|total\b|repayable\b|to repay\b)", re.I)
_FEE_BEFORE = re.compile(r"(?:\b\w+ )?(?:fees?|charges?)\s*(?:of|is|=|:)\s*$", re.I)  # a connector is required: "fees £680" is not decisive
_FEE_AFTER = re.compile(r"^\s?(?:fees?\b|charges?\b|admin(?:istration)? fee\b|arrangement fee\b|servicing fee\b|set-?up fee\b)", re.I)

# A fee that depends on something the customer does or chooses is not a mandatory obligation of the agreement.
_CONDITIONAL = re.compile(
    r"\b(?:optional|optionally|if|unless|should you|in the event|on request|only when|when you|whenever|late|early|missed|defaults?|arrears|may|might|can be|could"
    r"|(?:customers?|applicants?|borrowers?|people|those|anyone|anybody) (?:with|who|whose|under|over|aged|earning|living|without)|for (?:customers|applicants|borrowers)|depending on|subject to|where)\b", re.I
)

_NO_FEE = [
    re.compile(p, re.I)
    for p in (
        r"\bno\s+(?:(?:additional|extra|hidden|further|other|upfront|admin(?:istration)?|arrangement|servicing|set-?up|booking)\s+){0,3}(?:fees?|charges?)\b",
        r"\bfee[- ]free\b",
        r"\bzero\s+(?:\w+\s+)?(?:fees?|charges?)\b",
        r"\bwithout\s+(?:any\s+)?(?:(?:additional|extra|hidden|further|other)\s+)?(?:fees?|charges?)\b",
    )
]


def no_fee_expression(text: str) -> bool:
    """Does the text actually say there are no fees (or no additional fees)? Unrelated text does not."""
    return any(p.search(text) for p in _NO_FEE)


def _scale_follows(text: str, end: int) -> bool:
    rest = text[end:].lstrip(" ")
    word = re.match(r"[A-Za-z]+", rest)
    return bool(word) and word.group(0).lower() in _SCALE_WORDS


def _candidates(text: str) -> list[tuple[int, int, str, str, str | None, str | None]]:
    """(start, end, currency, number, attached_suffix_or_scale, raw) for every money-like expression, overlaps removed."""
    found = []
    for m in _SYMBOL_FORM.finditer(text):
        found.append((m.start(), m.end(), _BY_SYMBOL[m.group("sym")], m.group("num"), m.group("suffix"), m.group(0)))
    for m in _CODE_BEFORE.finditer(text):
        found.append((m.start(), m.end(), m.group("code").upper(), m.group("num"), m.group("suffix"), m.group(0)))
    for m in _CODE_AFTER.finditer(text):
        found.append((m.start(), m.end(), m.group("code").upper(), m.group("num"), (m.group("scale") or "").strip() or None, m.group(0)))
    found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    kept: list = []
    for item in found:
        if not kept or item[0] >= kept[-1][1]:
            kept.append(item)
    return kept


def _clause_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _CLAUSE_BREAK.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def _quantity(text: str, start: int, end: int, clause: tuple[int, int], prev_end: int, next_start: int) -> Quantity:
    """The quantity this amount states, or UNKNOWN when the words around it do not settle it.

    Decisive wording decides. Otherwise, if exactly one quantity's keywords appear around the amount in its clause, that is the
    answer; if several do, it is UNKNOWN. Unknown is uncertainty downstream, never a verdict, which is the safe failure."""
    lo, hi = max(clause[0], prev_end), min(clause[1], next_start)
    before, after = text[lo:start], text[end:hi]
    if _MONTHLY_AFTER.search(after):
        return Quantity.MONTHLY
    if _FEE_TOTAL_BEFORE.search(before) or _FEE_TOTAL_AFTER.search(after):
        return Quantity.FEE_TOTAL
    fee_decisive = bool(_FEE_BEFORE.search(before) or _FEE_AFTER.search(after))
    total_decisive = bool(_TOTAL_BEFORE.search(before) or _TOTAL_AFTER.search(after))
    if fee_decisive and total_decisive:
        return Quantity.FEE_TOTAL  # "Fee: £20 total"
    if _MONTHLY_BEFORE.search(before):
        return Quantity.MONTHLY
    if total_decisive:
        return Quantity.TOTAL
    if fee_decisive:
        return Quantity.FEE
    present = {q for q, pattern in ((Quantity.TOTAL, _TOTAL_KW), (Quantity.MONTHLY, _MONTHLY_KW), (Quantity.FEE, _FEE_KW)) if pattern.search(before + " " + after)}
    return next(iter(present)) if len(present) == 1 else Quantity.UNKNOWN


def scan_amounts(text: str) -> list[AmountMention]:
    """Every money-like expression in `text`, in order, each either a valid amount with a quantity or marked malformed.
    The text is read in its Unicode-folded form (see `grounding.canonical`); the caller's original is never changed."""
    text, _ = canonical(text)
    candidates = _candidates(text)
    spans = _clause_spans(text)
    mentions = []
    for i, (start, end, currency, number, suffix, raw) in enumerate(candidates):
        number = number.rstrip(".,")  # sentence punctuation after the number, not part of it
        problem = None
        if suffix:
            problem = f"scaled or suffixed amount ({suffix!r})"
        elif _scale_follows(text, end):
            problem = "scaled amount (a scale word follows)"
        elif not _NUMBER_OK.match(number):
            problem = "malformed number (grouping or decimal places)"
        money = None
        if problem is None:
            try:
                money = Money.of(Decimal(number.replace(",", "")), currency)
            except (ValidationError, ValueError, ArithmeticError):
                problem = "amount cannot be represented exactly"
        clause = next((s for s in spans if s[0] <= start <= s[1]), (0, len(text)))
        prev_end = candidates[i - 1][1] if i else 0
        next_start = candidates[i + 1][0] if i + 1 < len(candidates) else len(text)
        quantity = _quantity(text, start, end, clause, prev_end, next_start) if money is not None else Quantity.UNKNOWN
        conditional = bool(_CONDITIONAL.search(text[clause[0]:clause[1]]))
        mentions.append(AmountMention(text=raw, money=money, quantity=quantity, problem=problem, conditional=conditional))
    return mentions


def parse_amounts(text: str) -> list[Money]:
    """Every *valid* amount in order, without duplicates. Malformed expressions contribute nothing (see `scan_amounts`)."""
    out: list[Money] = []
    for mention in scan_amounts(text):
        if mention.money is not None and mention.money not in out:
            out.append(mention.money)
    return out


def mentions_money(text: str) -> bool:
    """Is there any money-like expression at all, valid or not?"""
    return bool(_candidates(text))
