"""Small constructors that keep fixtures and tests short. All data is fictional."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sika_guard.models import (
    AvailabilitySubject,
    Case,
    CustomerIntent,
    Evidence,
    EvidenceSource,
    Fee,
    FinancingOffer,
    Mandate,
    Merchant,
    MerchantAssertion,
    Objective,
    ProposedTransaction,
    RepaymentSchedule,
    Scope,
    SourceDocument,
    SourceKind,
    TrustedContext,
)
from sika_guard.money import Money

UTC = timezone.utc
T0 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)  # when the merchant makes its claim
NOW = T0 + timedelta(minutes=5)  # when the agent proposes the payment

MERCHANT = Merchant(merchant_id="MER-1", name="Sunrise Electronics")
ACTING_AGENT_ID = "AGENT-1"  # the agent session the trusted context says submitted the proposal (the default mandate is issued to it)
LENDER_ID = "LND-1"
LENDER_PAYEE = "PAY-LND-1"

LENDER_SOURCE = EvidenceSource(source_id=LENDER_ID, name="Northbridge Lending API", kind=SourceKind.LENDER, independent_of_merchant=True)
AGGREGATOR_SOURCE = EvidenceSource(source_id="AGG-1", name="CompareCredit feed", kind=SourceKind.AGGREGATOR, independent_of_merchant=True)
MERCHANT_SOURCE = EvidenceSource(source_id="MER-1", name="Sunrise Electronics portal", kind=SourceKind.MERCHANT_CHANNEL, independent_of_merchant=False)

FLAGSHIP_INSTRUCTION = (
    "Select the cheapest suitable 12-month financing option. "
    "Total repayment must not exceed £750 and monthly payments must not exceed £65."
)

_AUTO = object()  # advertised total defaults to the true total


def gbp(amount: Decimal | str | int) -> Money:
    return Money.of(amount, "GBP")


def level(monthly: Decimal | str | int, months: int = 12, fees: tuple[Fee, ...] = ()) -> RepaymentSchedule:
    return RepaymentSchedule.level(gbp(monthly), months, fees)


def make_offer(
    offer_id: str,
    schedule: RepaymentSchedule,
    *,
    advertised: Money | None | object = _AUTO,
    title: str | None = None,
    description: str = "",
    payee_id: str = LENDER_PAYEE,
    lender_id: str = LENDER_ID,
    scope: Scope = Scope.CREDIT_AGREEMENT,
) -> FinancingOffer:
    return FinancingOffer(
        offer_id=offer_id,
        merchant_id=MERCHANT.merchant_id,
        lender_id=lender_id,
        payee_id=payee_id,
        title=title or f"Financing plan {offer_id}",
        description=description,
        schedule=schedule,
        advertised_total_repayable=schedule.total() if advertised is _AUTO else advertised,
        scope=scope,
    )


def make_intent(**overrides) -> CustomerIntent:
    fields = dict(
        customer_id="CUST-1",
        instruction=FLAGSHIP_INSTRUCTION,
        objective=Objective.MINIMISE_TOTAL_REPAYMENT,
        term_months=12,
        max_total_repayment=gbp(750),
        max_monthly_payment=gbp(65),
    )
    return CustomerIntent(**{**fields, **overrides})


def make_mandate(**overrides) -> Mandate:
    fields = dict(
        mandate_id="MND-1",
        customer_id="CUST-1",
        agent_id="AGENT-1",
        allowed_payee_ids=frozenset({LENDER_PAYEE}),
        permitted_scopes=frozenset({Scope.CREDIT_AGREEMENT}),
        max_amount=gbp(200),
        max_total_repayment=gbp(750),
        max_monthly_payment=gbp(65),
        issued_at=T0 - timedelta(hours=1),
        expires_at=T0 + timedelta(hours=24),
    )
    return Mandate(**{**fields, **overrides})


def make_transaction(offer: FinancingOffer, **overrides) -> ProposedTransaction:
    fields = dict(
        transaction_id="TXN-1",
        mandate_id="MND-1",
        offer_id=offer.offer_id,
        payee_id=offer.payee_id,
        amount=offer.schedule.initial_charge(),
        scope=Scope.CREDIT_AGREEMENT,
        proposed_at=NOW,
    )
    return ProposedTransaction(**{**fields, **overrides})


def make_assertion(
    offer_id: str,
    claims_available: bool,
    statement: str,
    *,
    at: datetime = T0,
    assertion_id: str = "ASR-1",
    subject: AvailabilitySubject = AvailabilitySubject.PRODUCT_AVAILABILITY,
) -> MerchantAssertion:
    return MerchantAssertion(
        assertion_id=assertion_id,
        merchant_id=MERCHANT.merchant_id,
        offer_id=offer_id,
        claims_available=claims_available,
        statement=statement,
        made_at=at,
        subject=subject,
    )


def make_evidence(
    offer_id: str,
    offer_available: bool,
    *,
    source: EvidenceSource = LENDER_SOURCE,
    at: datetime = T0 + timedelta(minutes=3),
    evidence_id: str = "EV-1",
    subject: AvailabilitySubject = AvailabilitySubject.PRODUCT_AVAILABILITY,
    customer_id: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id, offer_id=offer_id, offer_available=offer_available, source=source, observed_at=at,
        subject=subject, customer_id=customer_id,
    )


def make_document(
    document_id: str,
    text: str,
    *,
    offer_id: str | None = None,
    source: EvidenceSource = LENDER_SOURCE,
    at: datetime = T0 + timedelta(minutes=2),
) -> SourceDocument:
    return SourceDocument(document_id=document_id, offer_id=offer_id, source=source, published_at=at, text=text)


def make_case(
    case_id: str,
    offers: list[FinancingOffer],
    selected_id: str,
    *,
    intent: CustomerIntent | None = None,
    mandate: Mandate | None = None,
    assertions: tuple[MerchantAssertion, ...] = (),
    evidence: tuple[Evidence, ...] = (),
    documents: tuple[SourceDocument, ...] = (),
    evaluated_at: datetime | None = None,
    acting_agent_id: str = ACTING_AGENT_ID,
    **txn_overrides,
) -> Case:
    selected = next(o for o in offers if o.offer_id == selected_id)
    return Case(
        case_id=case_id,
        intent=intent or make_intent(),
        mandate=mandate or make_mandate(),
        merchant=MERCHANT,
        offers=tuple(offers),
        transaction=make_transaction(selected, **txn_overrides),
        trusted=TrustedContext(evaluated_at=evaluated_at or txn_overrides.get("proposed_at") or NOW, acting_agent_id=acting_agent_id),
        assertions=assertions,
        evidence=evidence,
        documents=documents,
    )


def fee(kind, amount: Decimal | str | int, *, due_upfront: bool = False) -> Fee:
    return Fee(kind=kind, amount=gbp(amount), due_upfront=due_upfront)
