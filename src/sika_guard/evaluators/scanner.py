"""The deterministic scanner: a non-LLM baseline that extracts the same kind of claims a model is asked to extract.

If a plain parser can pull an important financial statement out of unstructured text reliably, then a model that finds the same
statement has not shown that it adds anything: the question Direction B asks is what AI finds that this cannot. So the scanner exists to
be measured against, as Condition "A+". It sees exactly what the generalist sees, produces `ExtractedClaim`s, and those go through the
identical grounding, comparison, materiality and enforcement path as any model's (`crosscheck`). It has no route to a decision that a
model lacks, it is not a model, and it never emits a verdict of its own: its verdict is always ALLOW and only its verified extractions can
matter.

It is deliberately simple. It reads each sentence of each source passage with the same amount reader the verifier uses (`amounts`), so
whatever it extracts the verifier can check, and wording the reader cannot place is left alone rather than guessed at.
"""

from __future__ import annotations

import re

from sika_guard.evaluators.amounts import Quantity, no_fee_expression, scan_amounts
from sika_guard.evaluators.contract import (
    ClaimKind,
    EvaluatorRole,
    EvaluatorVerdict,
    EvaluatorView,
    ExtractedClaim,
)
from sika_guard.evaluators.grounding import SOURCE_SECTIONS, walk_items
from sika_guard.evaluators.mocks import make_analysis
from sika_guard.evaluators.contract import verdict_from_analysis
from sika_guard.evaluators.payloads import PAYLOAD_BUILDERS, collect_refs
from sika_guard.models import Decision

SCANNER_VERSION = "deterministic-scanner-v1"
_TEXT_FIELDS = ("title", "description", "statement", "text")
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_MAX_CLAIMS = 12
_MAX_QUOTE = 300


class DeterministicScanner:
    """Condition A+. Satisfies the `Evaluator` protocol; makes no network call and consults no model."""

    evaluator_id = "deterministic-scanner"
    role = EvaluatorRole.DETERMINISTIC

    async def evaluate(self, view: EvaluatorView) -> EvaluatorVerdict:
        payload = PAYLOAD_BUILDERS[self.role](view)
        analysis = make_analysis(
            Decision.ALLOW, claims=scan_claims(payload), risk_score=0.0,
            explanation="deterministic scan of source passages; it extracts statements and makes no judgement",
        )
        return verdict_from_analysis(
            analysis, view, evaluator_id=self.evaluator_id, role=self.role, allowed_refs=collect_refs(payload),
            prompt_version=SCANNER_VERSION, payload=payload,
        )


def scan_claims(payload: dict) -> list[ExtractedClaim]:
    """Every total, fee or no-fees statement the amount reader can place in a sentence of a source passage, bound to the passage's offer."""
    claims: list[ExtractedClaim] = []
    for section in SOURCE_SECTIONS:
        for item in walk_items(payload.get(section)):
            offer_id, ref = item.get("offer_id"), item.get("ref")
            if not isinstance(offer_id, str) or not isinstance(ref, str):
                continue
            for field in _TEXT_FIELDS:
                text = item.get(field)
                if not isinstance(text, str) or not text.strip():
                    continue
                for sentence in (s.strip() for s in _SENTENCES.split(text)):
                    if len(sentence) < 3:
                        continue
                    quote = sentence[:_MAX_QUOTE]
                    mentions = scan_amounts(sentence)
                    if any(m.problem for m in mentions):
                        continue
                    for m in mentions:
                        if m.money is None or m.conditional:
                            continue
                        if m.quantity is Quantity.TOTAL:
                            claims.append(ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id=offer_id, source_ref=ref, quote=quote,
                                                         amount_minor=m.money.amount_minor, currency=m.money.currency))
                        elif m.quantity in (Quantity.FEE, Quantity.FEE_TOTAL) and m.money.amount_minor > 0:
                            claims.append(ExtractedClaim(kind=ClaimKind.FEE_AMOUNT, offer_id=offer_id, source_ref=ref, quote=quote,
                                                         amount_minor=m.money.amount_minor, currency=m.money.currency))
                    if no_fee_expression(sentence) and not any(m.money is not None and m.quantity in (Quantity.FEE, Quantity.FEE_TOTAL) for m in mentions):
                        claims.append(ExtractedClaim(kind=ClaimKind.NO_ADDITIONAL_FEES, offer_id=offer_id, source_ref=ref, quote=quote))
    return claims[:_MAX_CLAIMS]
