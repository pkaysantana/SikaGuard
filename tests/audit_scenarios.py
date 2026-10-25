"""Reproductions of the independent audit's findings (F1-F13), as synthetic offline scenarios.

Each scenario returns a plain dict of what the system actually did, using only APIs that exist both before and after
the repairs, so the same functions produce the before/after counterexample matrix (`audit/matrix_*.json`) and drive the
regression tests (`tests/test_audit_regressions.py`). Nothing here calls a model: live-path scenarios use the fake
provider through the real client. These are investigation inputs, not authority: each scenario states what it constructs.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from pydantic import ValidationError

from sika_guard.coordinator import CaseRecord, Condition, CoordinatorConfig, coordinate
from sika_guard.evaluators.contract import AIFinding, ClaimKind, ConcernCode, ExtractedClaim
from sika_guard.evaluators.llm import evaluators_for
from sika_guard.evaluators.mocks import Scripted, make_analysis
from sika_guard.fixtures.builders import fee, gbp, level, make_assertion, make_case, make_evidence, make_offer
from sika_guard.models import Decision, EvidenceSource, FeeKind, Money, SourceKind
from sika_guard.models import Case

from fakes import FakeProvider, analysis_json, completion

AI = (Condition.B, Condition.B3, Condition.BC, Condition.C)


def run(coro):
    return asyncio.run(coro)


def observe(rec: CaseRecord) -> dict[str, Any]:
    """The facts a reviewer needs: what was decided, why, what trusted code concluded, and how every evaluator fared."""
    return {
        "final": rec.final_decision.value,
        "deterministic": rec.deterministic_decision.value,
        "bases": [b.value for b in rec.decision_bases],
        "crosschecks": [f"{c.kind.value}:{c.outcome.value}" for c in rec.crosschecks],
        "proven_false": list(rec.proven_false_offer_ids),
        "verdict_statuses": [v.status.value for v in rec.verdicts],
        "coverage": rec.coverage.value,
        "uncertainty": list(getattr(rec, "uncertainty_reasons", ())),
    }


def scripted(case: Case, claims=(), findings=(), verdict=Decision.ALLOW, fail_closed=False) -> dict[str, Any]:
    ev = Scripted({case.case_id: make_analysis(verdict, claims=claims, findings=findings)})
    return observe(run(coordinate(case, Condition.B, [ev], CoordinatorConfig(fail_closed=fail_closed))))


def claim_json(kind: str, offer_id: str, source_ref: str, quote: str, amount_minor=None, currency=None, available=None) -> dict:
    return {"kind": kind, "offer_id": offer_id, "source_ref": source_ref, "quote": quote, "asserted_available": available, "amount_minor": amount_minor, "currency": currency}


def live(case: Case, condition: Condition, replies: dict[str, dict] | dict, fail_closed=False) -> dict[str, Any]:
    """Run a condition through the real client and a fake provider. `replies` maps a role (or "*") to an analysis dict."""
    def responder(role, body):
        spec = replies.get(role, replies.get("*", {}))
        return completion(analysis_json(**spec) if spec else analysis_json("ALLOW"))

    provider = FakeProvider(responder)

    async def go():
        async with provider.client() as client:
            return await coordinate(case, condition, evaluators_for(condition.value, client), CoordinatorConfig(fail_closed=fail_closed))

    return observe(run(go()))


def across_conditions(case: Case, replies, fail_closed=False) -> dict[str, Any]:
    return {c.value: live(case, c, replies, fail_closed) for c in AI}


# --- constructions ---------------------------------------------------------------------------------------------


def fee_case(description="", advertised=None, a_fees=("20",), a_monthly="55") -> Case:
    """A is the cheapest authorised offer (real total 680 with a 20 fee), so the rule engine alone says ALLOW."""
    fees = tuple(fee(FeeKind.ADMINISTRATION, f) for f in a_fees)
    a = make_offer("OFFER-A", level(a_monthly, fees=fees), advertised=advertised, description=description)
    b = make_offer("OFFER-B", level("62"))
    return make_case("audit-fees", [a, b], "OFFER-A")


# --- F1: a claim whose quote does not express the proposition -----------------------------------------------------------


def f1a_no_fees_claim_from_unrelated_text() -> dict:
    case = fee_case(description="Includes accident cover for the term.")
    claim = ExtractedClaim(kind=ClaimKind.NO_ADDITIONAL_FEES, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="Includes accident cover")
    return {"scripted": scripted(case, [claim]), "live": across_conditions(case, {"*": {"verdict": "ALLOW", "extracted_claims": [claim_json("no_additional_fees", "OFFER-A", "listing:OFFER-A", "Includes accident cover")]}})}


def f1b_zero_fee_amount_against_a_quote_stating_20() -> dict:
    case = fee_case(description="An administration fee of £20 applies.")
    claim = ExtractedClaim(kind=ClaimKind.FEE_AMOUNT, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="An administration fee of £20 applies", amount_minor=0, currency="GBP")
    return {"scripted": scripted(case, [claim])}


# --- F2: subject and visibility of the quoted source -------------------------------------------------------------------------


def f2a_quote_from_offer_y_attributed_to_offer_x() -> dict:
    x = make_offer("OFFER-X", level("55", fees=(fee(FeeKind.ADMINISTRATION, "20"),)), advertised=None)
    y = make_offer("OFFER-Y", level("58"), advertised=None, description="Total repayable £696.")  # truthful: Y really totals 696
    case = make_case("audit-misattribution", [x, y], "OFFER-X")
    claim = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-X", source_ref="listing:OFFER-Y", quote="Total repayable £696", amount_minor=69600, currency="GBP")
    return {"scripted": scripted(case, [claim])}


def f2b_specialist_quotes_a_field_it_was_never_shown() -> dict:
    case = fee_case(description="Total repayable £660.")  # lender total is 680; the description is a genuine understatement
    claim = claim_json("total_repayable", "OFFER-A", "listing:OFFER-A", "Total repayable £660", 66000, "GBP")
    only_credit_terms = {"credit_terms": {"verdict": "ALLOW", "extracted_claims": [claim]}, "*": {}}
    return {
        "credit_terms_specialist_quotes_the_description": live(case, Condition.C, only_credit_terms),
        "control_generalist_sees_the_description": live(case, Condition.B, {"*": {"verdict": "ALLOW", "extracted_claims": [claim]}}),
        "control_merchant_claim_specialist_sees_it": live(case, Condition.C, {"merchant_claim": {"verdict": "ALLOW", "extracted_claims": [claim]}, "*": {}}),
    }


# --- F3: merchant-channel evidence flagged independent -----------------------------------------------------------------------


def _merchant_source(independent: bool = True) -> EvidenceSource:
    return EvidenceSource(source_id="MER-1", name="Sunrise portal export", kind=SourceKind.MERCHANT_CHANNEL, independent_of_merchant=independent)


def _withdrawal_case(evidence) -> Case:
    a, b = make_offer("OFFER-A", level("55")), make_offer("OFFER-B", level("62"))
    return make_case("audit-launder", [a, b], "OFFER-B", assertions=(make_assertion("OFFER-A", False, "Offer A has been withdrawn."),), evidence=evidence)


def f3_merchant_channel_declaring_itself_independent() -> dict:
    out: dict[str, Any] = {}
    for name, available in (("corroborates_the_merchants_own_claim", False), ("contradicts_the_merchants_own_claim", True)):
        try:
            case = _withdrawal_case((make_evidence("OFFER-A", available, source=_merchant_source(True)),))
        except ValidationError as exc:
            out[name] = {"rejected_at_construction": type(exc).__name__}
            continue
        out[name] = observe(run(coordinate(case, Condition.A, [], CoordinatorConfig())))
    out["control_no_evidence_at_all"] = observe(run(coordinate(_withdrawal_case(()), Condition.A, [], CoordinatorConfig())))
    out["control_honest_merchant_channel_flag"] = observe(run(coordinate(_withdrawal_case((make_evidence("OFFER-A", False, source=_merchant_source(False)),)), Condition.A, [], CoordinatorConfig())))
    return out


# --- F4: other amounts in the quote hide an explicit total ---------------------------------------------------------------------


def f4a_total_claim_whose_quote_also_states_an_instalment() -> dict:
    case = fee_case(description="£55 a month, total repayable £660.")  # real total 680
    claim = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="£55 a month, total repayable £660", amount_minor=66000, currency="GBP")
    return {"scripted": scripted(case, [claim])}


def f4b_fee_claim_whose_quote_also_states_an_instalment() -> dict:
    case = fee_case(description="A £55 monthly payment and total fees of £5 apply.")  # real fees 20
    claim = ExtractedClaim(kind=ClaimKind.FEE_AMOUNT, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="A £55 monthly payment and total fees of £5 apply", amount_minor=500, currency="GBP")
    return {"scripted": scripted(case, [claim])}


# --- F5: the wrong quantity, and partial parsing of malformed or scaled amounts --------------------------------------------------


def f5a_monthly_payment_filed_as_total_repayment() -> dict:
    case = fee_case(description="Pay just £55 a month.")
    claim = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="Pay just £55 a month", amount_minor=5500, currency="GBP")
    return {"scripted": scripted(case, [claim])}


def f5b_malformed_and_scaled_amounts() -> dict:
    out = {}
    # Each extracted amount is exactly what a partial parse of the string would yield (the leading digits), which is the counterexample.
    for label, text, minor in (("scaled_k", "Total repayable £680k", 68000), ("scaled_m", "Total repayable £0.68m", 68), ("decimal_comma", "Total repayable £680,50", 68000),
                               ("three_decimals", "Total repayable £680.123", 68000), ("grouping_slip", "Total repayable £6,80", 600)):
        case = fee_case(description=text + ".")
        claim = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote=text, amount_minor=minor, currency="GBP")
        out[label] = scripted(case, [claim])
    return out


# --- F6: disagreement and incompatibility must not vanish into a clean ALLOW ----------------------------------------------------


def f6a_advertised_total_in_another_currency() -> dict:
    a = make_offer("OFFER-A", level("55"), advertised=Money.of("660", "USD"))
    b = make_offer("OFFER-B", level("62"))
    case = make_case("audit-currency", [a, b], "OFFER-A")
    return {"rule_engine_only": observe(run(coordinate(case, Condition.A, [], CoordinatorConfig())))}


def f6b_model_extraction_disagrees_with_the_structured_figure() -> dict:
    case = fee_case(advertised=gbp("680"))
    claim = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="advertised_total_repayable: 680.00 GBP", amount_minor=6800, currency="GBP")
    return {"scripted": scripted(case, [claim])}


# --- F8: an invented quotation must count against coverage and fail-closed handling ----------------------------------------------


def f8_invented_quotation() -> dict:
    case = fee_case(description="Standard instalment plan.")
    invented = claim_json("no_additional_fees", "OFFER-A", "listing:OFFER-A", "guaranteed zero fees, honestly")
    reply = {"*": {"verdict": "ALLOW", "extracted_claims": [invented]}}
    return {"open": across_conditions(case, reply), "fail_closed": {c.value: live(case, c, reply, fail_closed=True) for c in (Condition.B, Condition.BC)}}


# --- F7: what max_amount covers (a product decision; see AUDIT_TRIAGE.md) ------------------------------------------------------------


def f7_later_instalment_above_max_amount() -> dict:
    """A 250 instalment follows a 30 initial payment under a 200 per-payment ceiling. No monthly cap is set, so only max_amount could object."""
    from sika_guard.fixtures.builders import make_intent, make_mandate
    from sika_guard.models import RepaymentSchedule

    sched = RepaymentSchedule(payments=(gbp("30"),) + (gbp("250"),) * 2 + (gbp("30"),) * 9)
    a = make_offer("OFFER-A", sched)
    case = make_case("audit-cap", [a], "OFFER-A", intent=make_intent(max_monthly_payment=None, max_total_repayment=gbp(2000)),
                     mandate=make_mandate(max_monthly_payment=None, max_total_repayment=gbp(2000), max_amount=gbp(200)))
    return {"rule_engine_only": observe(run(coordinate(case, Condition.A, [], CoordinatorConfig())))}


SCENARIOS: dict[str, Callable[[], dict]] = {
    "F1a": f1a_no_fees_claim_from_unrelated_text,
    "F1b": f1b_zero_fee_amount_against_a_quote_stating_20,
    "F2a": f2a_quote_from_offer_y_attributed_to_offer_x,
    "F2b": f2b_specialist_quotes_a_field_it_was_never_shown,
    "F3": f3_merchant_channel_declaring_itself_independent,
    "F4a": f4a_total_claim_whose_quote_also_states_an_instalment,
    "F4b": f4b_fee_claim_whose_quote_also_states_an_instalment,
    "F5a": f5a_monthly_payment_filed_as_total_repayment,
    "F5b": f5b_malformed_and_scaled_amounts,
    "F6a": f6a_advertised_total_in_another_currency,
    "F6b": f6b_model_extraction_disagrees_with_the_structured_figure,
    "F7": f7_later_instalment_above_max_amount,
    "F8": f8_invented_quotation,
}


def matrix() -> dict[str, Any]:
    return {k: fn() for k, fn in SCENARIOS.items()}


if __name__ == "__main__":
    print(json.dumps(matrix(), indent=1, sort_keys=True))
