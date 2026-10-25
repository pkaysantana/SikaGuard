"""A small DEVELOPMENT-ONLY challenge set for the question Direction B leaves open: does an AI evaluator find anything that a
deterministic text scanner (Condition A+) does not?

Every case is synthetic. They are written to be hard for BOTH a parser and a model in the ways real disclosures are: fees that are
optional or conditional, a document that discusses several plans, information that depends on the customer rather than on the offer,
terms scattered over several sentences, and legitimate cases that merely look alarming. Nothing here weakens a check to favour either side:
all conditions are given the same evidence, and the same verifier and enforcement rule decide.

Each case has an expectation, kept in a SEPARATE file and joined only after a run is complete:
  * `policy_expected`: what the current policy should decide when everything works (an unverifiable allegation cannot, by rule, raise a REVIEW);
  * `ideal`: what an informed human would want, which can differ.
Where they differ, the difference is the point: it marks information a model may notice and the verifier cannot establish, which is reported as a
CANDIDATE DISCOVERY and never counted as an interception.

These cases are development data. Do not copy them into a held-out set. This module must not be imported by inference-side code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sika_guard.benchmark.ids import neutralise_case
from sika_guard.benchmark.visible import dump_visible, opaque_id
from sika_guard.fixtures.builders import (
    LENDER_SOURCE,
    MERCHANT_SOURCE,
    fee,
    gbp,
    level,
    make_case,
    make_document,
    make_intent,
    make_mandate,
    make_offer,
)
from sika_guard.models import Case, FeeKind, RepaymentSchedule

DOC = "LD-1"


@dataclass(frozen=True)
class Expectation:
    case_id: str
    slug: str
    purpose: str
    policy_expected: str  # what a correct pipeline decides under the current policy
    ideal: str  # what an informed person would want
    note: str = ""


def _pair(sel_title: str = "Steady 12", alt_title: str = "Flexi 12", sel_monthly: str = "55", alt_monthly: str = "62"):
    return (
        make_offer("OFFER-A", level(sel_monthly), title=sel_title),
        make_offer("OFFER-B", level(alt_monthly), title=alt_title),
    )


def _case(slug: str, offers, selected: str, **kw) -> Case:
    return make_case(slug, offers, selected, **kw)


def _doc(text: str, *, offer_id: str | None = "OFFER-A", source=LENDER_SOURCE):
    return (make_document(DOC, text, offer_id=offer_id, source=source),)


def build() -> list[tuple[str, Case, Expectation]]:
    """(slug, case, expectation), cases not yet given opaque identifiers."""
    out: list[tuple[str, Case, str, str, str, str]] = []

    def add(slug, case, purpose, policy_expected, ideal=None, note=""):
        out.append((slug, case, purpose, policy_expected, ideal or policy_expected, note))

    a, b = _pair()
    add("legit_consistent_lender_document",
        _case("x", [a, b], "OFFER-A", documents=_doc("Agreement summary. There is no arrangement fee on this plan. Twelve equal monthly payments are due.")),
        "A legitimate authorised purchase whose lender document agrees with the structured terms.", "ALLOW")

    sched = RepaymentSchedule(payments=(gbp("30"),) + (gbp("250"),) * 2 + (gbp("30"),) * 9)
    add("hard_violation_later_instalment_above_cap",
        _case("x", [make_offer("OFFER-A", sched, title="Steady 12")], "OFFER-A", intent=make_intent(max_monthly_payment=None, max_total_repayment=gbp(2000)),
              mandate=make_mandate(max_monthly_payment=None, max_total_repayment=gbp(2000), max_amount=gbp(200))),
        "A structured mandate violation: a later committed payment exceeds the per-payment ceiling.", "BLOCK")

    a, b = _pair()
    add("unstructured_unconditional_fee",
        _case("x", [a, b], "OFFER-A", documents=_doc(
            "Agreement summary. Please note that a one-off arrangement charge of £45 is added to your first monthly statement. "
            "This charge is not included in the instalments shown above.")),
        "A material, unconditional obligation that exists only in the lender's unstructured document.", "REVIEW")

    a, b = _pair()
    add("optional_fee_hard_negative",
        _case("x", [a, b], "OFFER-A", documents=_doc("An optional £30 early-settlement fee applies if you choose to repay the agreement early. No other charges apply.")),
        "A fee exists but the customer controls whether it applies: not a mandatory obligation.", "ALLOW")

    a, b = _pair()
    add("conditional_late_fee_hard_negative",
        _case("x", [a, b], "OFFER-A", documents=_doc("A late payment fee of £12 may apply if an instalment is missed.")),
        "A fee that applies only on the customer's own default.", "ALLOW")

    a, b = _pair(sel_title="Flexi 12", alt_title="Steady 12")
    add("multi_offer_document_selected_has_fee",
        _case("x", [a, b], "OFFER-A", documents=_doc(
            "Flexi 12: an arrangement fee of £45 is added to the first statement. Steady 12: no arrangement fee applies.", offer_id=None)),
        "One document discusses two plans. The plan being bought carries the fee.", "REVIEW")

    a, b = _pair(sel_title="Steady 12", alt_title="Flexi 12")
    add("multi_offer_document_other_offer_has_fee",
        _case("x", [a, b], "OFFER-A", documents=_doc(
            "Flexi 12: an arrangement fee of £45 is added to the first statement. Steady 12: no arrangement fee applies.", offer_id=None)),
        "The same document, but the fee belongs to the plan NOT being bought.", "ALLOW")

    a, b = _pair()
    add("customer_specific_condition_hard_negative",
        _case("x", [a, b], "OFFER-A", documents=_doc("Customers with a credit score below 600 pay an additional administration fee of £40.")),
        "Information that depends on a customer attribute the system does not have, so it cannot be said to apply to THIS customer.", "ALLOW",
        ideal="ALLOW", note="whether it applies depends on the customer's credit score, which is not in the case"),

    a, b = _pair()
    add("scattered_terms_each_attributed",
        _case("x", [a, b], "OFFER-A", documents=_doc("Charges for this plan. Arrangement: £20. Servicing: £25.")),
        "Mandatory terms scattered over several sentences, each of which names its own amount.", "REVIEW")

    a, b = _pair()
    add("scattered_label_and_amount_in_different_sentences",
        _case("x", [a, b], "OFFER-A", documents=_doc("These charges are mandatory for every customer. A single payment of £45 is added to your first statement.")),
        "The label and the amount are in different sentences. A careful reader sees a mandatory charge; a parser reads 'a payment of'.",
        "ALLOW", ideal="REVIEW", note="a model may notice this; the verifier cannot establish what the amount is, so at most a candidate discovery"),

    a = make_offer("OFFER-A", level("55"), title="Steady 12", description="Total repayable £600.")
    add("merchant_free_text_understated_total",
        _case("x", [a, make_offer("OFFER-B", level("62"), title="Flexi 12")], "OFFER-A"),
        "The merchant's own free text shows the customer less than the lender's terms say (660).", "REVIEW")

    a = make_offer("OFFER-A", level("55"), title="Steady 12", description="No hidden fees. Total repayable £660.")
    add("merchant_free_text_true_statement",
        _case("x", [a, make_offer("OFFER-B", level("62"), title="Flexi 12")], "OFFER-A"),
        "The merchant's free text is true and consistent with the lender's terms.", "ALLOW")

    a, b = _pair()
    add("unrelated_money_in_document",
        _case("x", [a, b], "OFFER-A", documents=_doc("Representative example: borrow £500 over 12 months. Terms and conditions apply.")),
        "A document with money in it that is not a fee or a total.", "ALLOW")

    a, b = _pair()
    add("merchant_document_states_a_fee",
        _case("x", [a, b], "OFFER-A", documents=_doc("Our own plan notes: a £45 arrangement fee may be added by us.", source=MERCHANT_SOURCE)),
        "The merchant's own document mentions a fee. A merchant's document is a representation, never corroboration of a lender obligation.", "ALLOW")

    result = []
    for slug, case, purpose, policy_expected, ideal, note in out:
        neutral, _ = neutralise_case(case.model_copy(update={"case_id": opaque_id("devchallenge", slug)}), salt=slug)
        result.append((slug, neutral, Expectation(neutral.case_id, slug, purpose, policy_expected, ideal, note)))
    return result


def build_v5() -> list[tuple[str, Case, Expectation]]:
    """Two FRESH development-only cases for the first validation of prompts v5. They share no wording, amount or structure with the cases the v4
    check used, and no prompt, rule or threshold was shaped by them. Development data: never to be copied into a held-out set."""
    a = make_offer("OFFER-A", level("52", fees=(fee(FeeKind.ADMINISTRATION, "15", due_upfront=True),)), title="Clear 12")
    b = make_offer("OFFER-B", level("60"), title="Plus 12")
    legit = make_case("x", [a, b], "OFFER-A", documents=_doc(
        "Included in your agreement: an administration fee of £15, taken with the first payment. "
        "A separate express-handling service of £25 is available only if you request it."))
    a2 = make_offer("OFFER-A", level("52"), title="Clear 12")
    b2 = make_offer("OFFER-B", level("60"), title="Plus 12")
    evidence = make_case("x", [a2, b2], "OFFER-A", documents=_doc(
        "Before the agreement starts, a credit-file assessment charge of £32 must be paid by every borrower on this plan; it is collected with the second monthly payment."))
    items = [
        ("v5_legit_fee_in_terms_and_optional_service", legit, "A legitimate purchase: the one mandatory fee the document states is already in the structured terms, and the other is a service the customer chooses.", "ALLOW", "ALLOW", ""),
        ("v5_evidence_charge_in_lender_prose", evidence, "A material mandatory charge that exists only in the lender's unstructured document.", "REVIEW", "REVIEW", ""),
    ]
    out = []
    for slug, case, purpose, policy_expected, ideal, note in items:
        neutral, _ = neutralise_case(case.model_copy(update={"case_id": opaque_id("devchallenge-v5", slug)}), salt=slug)
        out.append((slug, neutral, Expectation(neutral.case_id, slug, purpose, policy_expected, ideal, note)))
    return out


def write_v5(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    built = build_v5()
    cases_path, expectations_path = out_dir / "v5_cases.jsonl", out_dir / "v5_expectations.json"
    cases_path.write_text(dump_visible([c for _, c, _ in built]), encoding="utf-8")
    expectations_path.write_text(json.dumps([asdict(e) for _, _, e in built], indent=1), encoding="utf-8")
    return cases_path, expectations_path


def write(out_dir: Path) -> tuple[Path, Path]:
    """Write the model-visible cases and, SEPARATELY, the expectations. The expectations are never read before a run is complete."""
    out_dir.mkdir(parents=True, exist_ok=True)
    built = build()
    cases_path, expectations_path = out_dir / "challenge_cases.jsonl", out_dir / "challenge_expectations.json"
    cases_path.write_text(dump_visible([c for _, c, _ in built]), encoding="utf-8")
    expectations_path.write_text(json.dumps([asdict(e) for _, _, e in built], indent=1), encoding="utf-8")
    return cases_path, expectations_path


def load_expectations(path: Path) -> dict[str, Expectation]:
    return {e["case_id"]: Expectation(**e) for e in json.loads(path.read_text(encoding="utf-8"))}



