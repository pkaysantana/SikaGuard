"""DEVELOPMENT-ONLY scenarios for the Direction B information-to-enforcement boundary.

Every scenario is synthetic and offline: model output is scripted, or comes from the fake provider through the real client. No labels are
involved and nothing here may be copied into a held-out benchmark. They exist to show, case by case, that:

  1. A detects a structured mandate violation (BLOCK).
  2. A detects a structured financial contradiction.
  3. A correctly ALLOWs a legitimate purchase (including one with a consistent lender document and one with a conditional fee).
  4. A ALLOWs a case whose material fact exists only in an unstructured lender document its declared interface cannot read.
  5. A model extracts that fact and trusted code validates it independently, producing a justified REVIEW.
  6. An unsupported model allegation cannot produce that REVIEW.
  7. An incorrect extraction cannot produce a false verified contradiction.
  8. A verified extraction does not take credit for a risk A already detected.

and that the same opportunity exists, under the same rules, for B, B3, Bc, C and the deterministic scanner A+.
"""

from __future__ import annotations

from typing import Any

import audit_scenarios as S
from sika_guard.coordinator import CaseRecord, Condition, CoordinatorConfig, coordinate
from sika_guard.evaluators.contract import AIFinding, ClaimKind, ConcernCode, ExtractedClaim
from sika_guard.evaluators.llm import evaluators_for
from sika_guard.evaluators.mocks import Scripted, make_analysis
from sika_guard.fixtures.builders import LENDER_SOURCE, MERCHANT_SOURCE, fee, gbp, level, make_case, make_document, make_intent, make_mandate, make_offer
from sika_guard.models import Case, Decision, EvidenceSource, FeeKind, RepaymentSchedule, SourceKind
from sika_guard.policy import evaluate_case

DOC_ID = "LD-1"
DOC_REF = f"doc:{DOC_ID}"
HIDDEN_FEE_TEXT = (
    "Agreement summary. Please note that a one-off arrangement charge of £45 is added to your first monthly statement. "
    "This charge is not included in the instalments shown above."
)
HIDDEN_FEE_QUOTE = "a one-off arrangement charge of £45 is added to your first monthly statement"
OTHER_LENDER = EvidenceSource(source_id="LND-OTHER", name="Unrelated Credit", kind=SourceKind.LENDER, independent_of_merchant=True)


def observe(rec: CaseRecord) -> dict[str, Any]:
    out = S.observe(rec)
    out.update(
        escalated_by_ai=rec.escalated_by_ai,
        routes=list(rec.ai_review_routes),
        checks=[(c.source_class.value if c.source_class else None, c.outcome.value, c.direction.value if c.direction else None, c.material) for c in rec.crosschecks],
        rejected=sum(len(v.rejected_claims) for v in rec.verdicts),
    )
    return out


def claim(*, kind: str = "fee_amount", quote: str = HIDDEN_FEE_QUOTE, amount: int | None = 4500, ref: str = DOC_REF, offer: str = "OFFER-A", currency: str | None = "GBP") -> ExtractedClaim:
    return ExtractedClaim(
        kind=ClaimKind(kind), offer_id=offer, source_ref=ref, quote=quote,
        amount_minor=amount if kind != "no_additional_fees" else None, currency=currency if kind != "no_additional_fees" else None,
    )


def claim_json(**kw) -> dict:
    return claim(**kw).model_dump(mode="json")


def hidden_fee_case(text: str = HIDDEN_FEE_TEXT, *, source: EvidenceSource = LENDER_SOURCE, doc_offer: str | None = "OFFER-A", description: str = "", **kw) -> Case:
    """A (selected) is the cheapest authorised offer and its structured terms list no fees, so the rule engine alone says ALLOW."""
    a = make_offer("OFFER-A", level("55"), description=description)
    b = make_offer("OFFER-B", level("62"))
    return make_case("db-hidden-fee", [a, b], "OFFER-A", documents=(make_document(DOC_ID, text, offer_id=doc_offer, source=source),), **kw)


def run_scripted(case: Case, claims=(), findings=(), verdict=Decision.ALLOW, condition=Condition.B, fail_closed=False) -> dict[str, Any]:
    ev = Scripted({case.case_id: make_analysis(verdict, claims=claims, findings=findings)})
    return observe(S.run(coordinate(case, condition, [ev], CoordinatorConfig(fail_closed=fail_closed))))


def run_condition(case: Case, condition: Condition, replies: dict | None = None) -> dict[str, Any]:
    """Through the real client and the fake provider (or, for A+, the deterministic scanner). `replies` maps a role to analysis fields."""
    from fakes import FakeProvider, analysis_json, completion

    replies = replies or {}

    def responder(role, body):
        spec = replies.get(role, replies.get("*", {}))
        return completion(analysis_json("ALLOW", **spec))

    provider = FakeProvider(responder)

    async def go():
        async with provider.client() as client:
            return await coordinate(case, condition, evaluators_for(condition.value, client), CoordinatorConfig())

    return observe(S.run(go()))


# --- 1-3: what A does by itself -------------------------------------------------------------------------------------------------------------------


def a_detects_a_structured_mandate_violation() -> dict:
    sched = RepaymentSchedule(payments=(gbp("30"),) + (gbp("250"),) * 2 + (gbp("30"),) * 9)
    case = make_case("db-cap", [make_offer("OFFER-A", sched)], "OFFER-A", intent=make_intent(max_monthly_payment=None, max_total_repayment=gbp(2000)),
                     mandate=make_mandate(max_monthly_payment=None, max_total_repayment=gbp(2000), max_amount=gbp(200)))
    d = evaluate_case(case)
    return {"decision": d.decision.value, "authorised": d.authorised, "findings": sorted(f.code.value for f in d.findings)}


def a_detects_a_structured_financial_contradiction() -> dict:
    a = make_offer("OFFER-A", level("55"), advertised=gbp("600"))  # real total 660
    d = evaluate_case(make_case("db-adv", [a, make_offer("OFFER-B", level("62"))], "OFFER-A"))
    return {"decision": d.decision.value, "findings": sorted(f.code.value for f in d.findings)}


def a_allows_a_legitimate_purchase() -> dict:
    plain = make_case("db-ok", [make_offer("OFFER-A", level("55")), make_offer("OFFER-B", level("62"))], "OFFER-A")
    # The lender's own document states a fee the structured terms already list: consistent, nothing to review.
    a = make_offer("OFFER-A", level("55", fees=(fee(FeeKind.ARRANGEMENT, "20"),)))
    consistent = make_case("db-ok-doc", [a, make_offer("OFFER-B", level("70"))], "OFFER-A",
                           documents=(make_document(DOC_ID, "An arrangement fee of £20 applies to this plan.", offer_id="OFFER-A"),))
    out = {"plain": evaluate_case(plain).decision.value, "consistent_lender_document_alone": evaluate_case(consistent).decision.value}
    claim_consistent = claim(quote="An arrangement fee of £20 applies to this plan", amount=2000)
    out["consistent_lender_document_extracted"] = run_scripted(consistent, [claim_consistent])
    return out


# --- 4-5: the incremental channel --------------------------------------------------------------------------------------------------------------------


def a_allows_the_hidden_obligation() -> dict:
    d = evaluate_case(hidden_fee_case())
    return {"decision": d.decision.value, "findings": sorted(f.code.value for f in d.findings)}


def a_verified_extraction_justifies_a_review() -> dict:
    return run_scripted(hidden_fee_case(), [claim()])


def every_condition_has_the_same_opportunity() -> dict:
    case = hidden_fee_case()
    fee_claim = {"extracted_claims": [claim_json()]}
    # The credit-terms specialist is the one C member shown the lender's document; the others have nothing to extract from it.
    replies = {"generalist": fee_claim, "rubric_generalist": fee_claim, "credit_terms": fee_claim}
    return {c.value: run_condition(case, c, replies) for c in (Condition.A_PLUS, Condition.B, Condition.B3, Condition.BC, Condition.C)}


# --- 6: an unsupported allegation is not enough ---------------------------------------------------------------------------------------------------------


def an_unsupported_allegation_cannot_produce_the_review() -> dict:
    case = hidden_fee_case()
    findings = [AIFinding(code=ConcernCode.HIDDEN_OBLIGATION, evidence_refs=(DOC_REF,), note="the lender document mentions an extra charge"),
                AIFinding(code=ConcernCode.TERMS_INCONSISTENT_WITH_REPRESENTATION, evidence_refs=(DOC_REF, "terms:OFFER-A"), note="terms and document differ")]
    return {
        "allegations_only_with_a_REVIEW_verdict": run_scripted(case, findings=findings, verdict=Decision.REVIEW),
        "no_claim_extracted_at_all": run_scripted(case, verdict=Decision.REVIEW),
    }


# --- 7: an incorrect extraction cannot create a verified contradiction ---------------------------------------------------------------------------


def incorrect_extractions_cannot_create_a_contradiction() -> dict:
    case = hidden_fee_case()
    out = {
        "transposed_amount": run_scripted(case, [claim(amount=5400)]),
        "invented_quote": run_scripted(case, [claim(quote="a compulsory arrangement charge of £45 applies")]),
        "wrong_quantity_stated_as_an_instalment": run_scripted(
            hidden_fee_case("Your instalment of £45 is due each month. Please keep it paid."),
            [claim(quote="Your instalment of £45 is due each month", amount=4500)]),
    }
    conditional = hidden_fee_case("A late payment fee of £45 may apply if an instalment is missed. There is no other charge to you.")
    out["a_conditional_fee_is_not_mandatory"] = run_scripted(conditional, [claim(quote="A late payment fee of £45 may apply if an instalment is missed")])
    out["a_document_of_another_lenders_is_not_this_lenders"] = run_scripted(hidden_fee_case(source=OTHER_LENDER), [claim()])
    out["a_document_bound_to_a_different_offer"] = run_scripted(hidden_fee_case(doc_offer="OFFER-B"), [claim()])
    out["the_merchant_states_the_same_fee_in_its_own_document"] = run_scripted(
        hidden_fee_case(source=MERCHANT_SOURCE), [claim()])  # a merchant's document is a representation: it cannot make the lender's terms conflict
    out["a_mislabelled_kind_total_for_a_fee"] = run_scripted(case, [claim(kind="total_repayable", amount=70500)])
    out["amount_in_a_script_the_reader_will_not_trust"] = run_scripted(
        hidden_fee_case("A one-off arrangement charge of £٤٥ is added to your first monthly statement."),
        [claim(quote="A one-off arrangement charge of £٤٥ is added to your first monthly statement")])
    return out


# --- 8: no double credit ---------------------------------------------------------------------------------------------------------------------------


def a_verified_extraction_takes_no_credit_for_what_a_already_found() -> dict:
    a = make_offer("OFFER-A", level("55"), advertised=gbp("600"), description="Total repayable £600.")
    case = make_case("db-double", [a, make_offer("OFFER-B", level("62"))], "OFFER-A")
    extracted = claim(kind="total_repayable", quote="Total repayable £600", amount=60000, ref="listing:OFFER-A")
    return {"rule_engine": evaluate_case(case).decision.value, "with_the_same_fact_extracted": run_scripted(case, [extracted])}


def matrix() -> dict[str, Any]:
    return {
        "1_structured_mandate_violation": a_detects_a_structured_mandate_violation(),
        "2_structured_financial_contradiction": a_detects_a_structured_financial_contradiction(),
        "3_legitimate_purchase": a_allows_a_legitimate_purchase(),
        "4_A_allows_the_hidden_obligation": a_allows_the_hidden_obligation(),
        "5_verified_extraction": a_verified_extraction_justifies_a_review(),
        "5b_every_condition": every_condition_has_the_same_opportunity(),
        "6_unsupported_allegation": an_unsupported_allegation_cannot_produce_the_review(),
        "7_incorrect_extractions": incorrect_extractions_cannot_create_a_contradiction(),
        "8_no_double_credit": a_verified_extraction_takes_no_credit_for_what_a_already_found(),
    }
