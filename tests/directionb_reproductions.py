"""Reproductions of Opus's findings R1-R10, and of F6/F7, as synthetic offline scenarios.

Like `audit_scenarios.py`, each scenario uses only APIs that exist both before and after the Direction B changes, so the same code
produces `audit/matrix_before_directionb.json` (run on the unrepaired tree) and `audit/matrix_after_directionb.json`, and the regression
tests reuse the constructions. Nothing calls a model; claims are scripted. These are investigation inputs, not authority.

A scenario that cannot even be expressed on the old tree (it needs a field that did not exist) records that fact instead of failing.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import audit_scenarios as S
from sika_guard.evaluators.contract import ClaimKind, ExtractedClaim
from sika_guard.fixtures.builders import (
    NOW,
    T0,
    gbp,
    level,
    make_assertion,
    make_case,
    make_evidence,
    make_mandate,
    make_offer,
)
from sika_guard.models import Evidence, EvidenceSource, MerchantAssertion, Scope, SourceKind
from sika_guard.policy import evaluate_case

AGGREGATOR = EvidenceSource(source_id="AGG-1", name="CompareCredit feed", kind=SourceKind.AGGREGATOR, independent_of_merchant=True)


import inspect

# On the repaired tree the decision clock is a trusted input; the reproductions pin it to NOW so a back-dated proposal is the only thing
# that varies. The unrepaired tree has no such input, which is exactly finding R1.
_CLOCK = {"evaluated_at": NOW} if "evaluated_at" in inspect.signature(make_case).parameters else {}


def _verdict(case) -> dict[str, Any]:
    d = evaluate_case(case)
    return {"decision": d.decision.value, "authorised": d.authorised, "findings": sorted({f.code.value for f in d.findings})}


def _has_field(model, name: str) -> bool:
    return name in model.model_fields


def _pair(**selected_kwargs):
    return make_offer("OFFER-A", level("55"), **selected_kwargs), make_offer("OFFER-B", level("62"))


def _unavailable_selected_case(**kw):
    """The merchant says the selected offer is available; fresh lender evidence (observed before the proposal) says it is not."""
    a, b = _pair()
    return make_case("rb", [a, b], "OFFER-A",
                     assertions=(make_assertion("OFFER-A", True, "In stock, available now", at=T0),),
                     evidence=(make_evidence("OFFER-A", False),), **{**_CLOCK, **kw})


# --- R1: the decision clock is chosen by the proposing agent -------------------------------------------------------------------


def r1_agent_chooses_the_clock() -> dict:
    a, b = _pair()
    expired = make_mandate(issued_at=NOW - timedelta(days=60), expires_at=NOW - timedelta(days=30))
    backdated = make_case("r1a", [a, b], "OFFER-A", mandate=expired, proposed_at=NOW - timedelta(days=45), **_CLOCK)
    honest_clock_evidence = _unavailable_selected_case(proposed_at=T0 + timedelta(minutes=1))  # back-dated to before the evidence
    return {
        "expired_mandate_but_the_agent_back_dates_the_proposal_into_its_window": _verdict(backdated),
        "back_dated_proposal_makes_fresh_contradicting_evidence_irrelevant": _verdict(honest_clock_evidence),
        "same_case_with_an_honest_proposal_time": _verdict(_unavailable_selected_case()),
    }


# --- R2: the transaction declares its own scope -------------------------------------------------------------------------------


def r2_scope_is_whatever_the_proposal_says() -> dict:
    a, b = _pair()
    card_only = make_mandate(permitted_scopes=frozenset({Scope.CARD_PURCHASE}))
    return {"credit_agreement_declared_as_card_purchase_under_a_card_only_mandate": _verdict(make_case("r2", [a, b], "OFFER-A", mandate=card_only, scope=Scope.CARD_PURCHASE))}


# --- R3: the mandate's agent is never compared with anyone ---------------------------------------------------------------------


def r3_mandate_for_another_agent() -> dict:
    a, b = _pair()
    return {"mandate_issued_to_a_different_agent": _verdict(make_case("r3", [a, b], "OFFER-A", mandate=make_mandate(agent_id="AGENT-SOMEONE-ELSE")))}


# --- R4: direction and materiality of a financial comparison --------------------------------------------------------------------


def r4_overstated_advertised_total() -> dict:
    case = S.fee_case(advertised=gbp("700"))  # lender 680; the merchant advertises MORE than the customer will pay
    claim = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote="700.00", amount_minor=70000, currency="GBP")
    return {"rule_engine": _verdict(case), "with_a_model_extracted_claim_of_the_advertised_figure": S.scripted(case, [claim])}


# --- R5: Unicode and formatting break grounding ---------------------------------------------------------------------------------


def r5_unicode_and_formatting() -> dict:
    out = {}
    variants = {
        "zero_width_space_inside_a_word": ("Total repay​able £600.", "Total repayable £600"),
        "fullwidth_digits": ("Total repayable £６００.", "Total repayable £600"),
        "no_break_space": ("Total repayable £600.", "Total repayable £600"),
        "curly_quotation_marks": ("“Total repayable”: £600.", "Total repayable: £600"),
    }
    for name, (text, quote) in variants.items():
        case = S.fee_case(description=text)  # lender 680, so £600 is an understatement
        claim = ExtractedClaim(kind=ClaimKind.TOTAL_REPAYABLE, offer_id="OFFER-A", source_ref="listing:OFFER-A", quote=quote, amount_minor=60000, currency="GBP")
        out[name] = S.scripted(case, [claim])
    return out


# --- R7: product availability is not customer eligibility -----------------------------------------------------------------------


def r7_eligibility_is_not_availability() -> dict:
    a = make_offer("OFFER-A", level("55"))
    b = make_offer("OFFER-B", level("62"))
    kw = {"subject": "customer_eligibility"} if _has_field(MerchantAssertion, "subject") else {}
    claim = MerchantAssertion(assertion_id="ASR-1", merchant_id="MER-1", offer_id="OFFER-A", claims_available=False,
                              statement="Plan A is not available to you.", made_at=T0, **kw)
    ev = make_evidence("OFFER-A", True)  # product-level: the lender still sells plan A
    case = make_case("r7", [a, b], "OFFER-B", assertions=(claim,), evidence=(ev,))
    return {"merchant_says_the_customer_cannot_have_A_lender_says_the_product_exists": _verdict(case), "assertion_carries_a_subject_field": bool(kw)}


# --- R8: impossible merchant timestamps -----------------------------------------------------------------------------------------


def r8_future_dated_claim() -> dict:
    honest = _unavailable_selected_case()
    future = honest.model_copy(update={"assertions": (make_assertion("OFFER-A", True, "In stock, available now", at=NOW + timedelta(days=1)),)})
    try:
        return {"honest_claim_time": _verdict(honest), "claim_dated_one_day_after_the_decision": _verdict(future)}
    except Exception as exc:  # noqa: BLE001  the repaired tree may reject the case outright
        return {"honest_claim_time": _verdict(honest), "claim_dated_one_day_after_the_decision": f"rejected ({type(exc).__name__})"}


# --- R9: source authority ------------------------------------------------------------------------------------------------------


def r9_authority_and_time() -> dict:
    a, b = _pair()
    asr = (make_assertion("OFFER-A", False, "Plan A is withdrawn.", at=T0),)

    def case(lender_at, agg_at):
        return make_case("r9", [a, b], "OFFER-B", assertions=asr, evidence=(
            make_evidence("OFFER-A", True, at=lender_at, evidence_id="EV-L"),
            make_evidence("OFFER-A", False, source=AGGREGATOR, at=agg_at, evidence_id="EV-G"),
        ))

    return {
        "lender_newer_than_the_aggregator_who_disagrees": _verdict(case(T0 + timedelta(minutes=4), T0 + timedelta(minutes=2))),
        "aggregator_newer_than_the_lender_who_disagrees": _verdict(case(T0 + timedelta(minutes=2), T0 + timedelta(minutes=4))),
    }


# --- F6 / F7 ---------------------------------------------------------------------------------------------------------------------


def f6_selected_offer_advertised_in_another_currency() -> dict:
    a = make_offer("OFFER-A", level("55"), advertised=S.Money(amount_minor=70000, currency="USD"))
    b = make_offer("OFFER-B", level("62"))
    return {"advertised_total_in_USD_on_a_GBP_schedule": _verdict(make_case("f6", [a, b], "OFFER-A"))}


def f7_later_instalment_above_max_amount() -> dict:
    return S.f7_later_instalment_above_max_amount()


# --- R10: a seal that is not bound to an approved manifest -------------------------------------------------------------------------


def r10_seal_not_bound_to_a_manifest() -> dict:
    import json
    import tempfile
    from pathlib import Path

    from sika_guard.benchmark.hidden import dump_labels
    from sika_guard.benchmark.mockrun import mock_records
    from sika_guard.benchmark.pilot import build_pilot
    from sika_guard.benchmark.runner import write_run_from_records
    from sika_guard.benchmark.visible import dump_visible

    root = Path(__file__).resolve().parents[1]
    stale = json.loads((root / "manifests" / "manifest-0001-initial.json").read_text(encoding="utf-8"))["digest"]
    cases, _ = build_pilot()
    records = mock_records(cases)
    out = {}
    for name, digest in (("a_manifest_digest_that_names_no_manifest", "0" * 64), ("the_initial_manifest_whose_frozen_files_no_longer_match_the_tree", stale)):
        tmp = Path(tempfile.mkdtemp())
        cases_path = tmp / "cases.jsonl"
        cases_path.write_text(dump_visible(cases), encoding="utf-8")
        try:
            write_run_from_records(tmp / "log.jsonl", cases_path, cases, records, mode="mock", manifest_digest=digest, seal=True)
            out[name] = "SEALED"
        except Exception as exc:  # noqa: BLE001  the scenario reports refusals by type
            out[name] = f"refused ({type(exc).__name__}: {str(exc)[:90]})"
    return out


def matrix() -> dict[str, Any]:
    return {
        "R1": r1_agent_chooses_the_clock(),
        "R2": r2_scope_is_whatever_the_proposal_says(),
        "R3": r3_mandate_for_another_agent(),
        "R4": r4_overstated_advertised_total(),
        "R5": r5_unicode_and_formatting(),
        "R7": r7_eligibility_is_not_availability(),
        "R8": r8_future_dated_claim(),
        "R9": r9_authority_and_time(),
        "R10": r10_seal_not_bound_to_a_manifest(),
        "F6": f6_selected_offer_advertised_in_another_currency(),
        "F7": f7_later_instalment_above_max_amount(),
    }
