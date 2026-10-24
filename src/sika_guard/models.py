"""Validated domain models.

Everything in this module may be shown to an evaluator, including a future LLM.
Synthetic ground-truth labels live in `labels.py` on purpose; detector code must
never import them.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from sika_guard.money import Money, sum_money


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Customer intent and delegated mandate ---------------------------------


class Scope(StrEnum):
    CREDIT_AGREEMENT = "credit_agreement"
    CARD_PURCHASE = "card_purchase"


class Objective(StrEnum):
    MINIMISE_TOTAL_REPAYMENT = "minimise_total_repayment"
    MINIMISE_MONTHLY_PAYMENT = "minimise_monthly_payment"
    SPECIFIC_OFFER = "specific_offer"


class CustomerIntent(_Model):
    """What the authenticated customer asked the agent to do."""

    customer_id: str
    instruction: str
    objective: Objective
    term_months: int | None = Field(default=None, gt=0)
    max_total_repayment: Money | None = None
    max_monthly_payment: Money | None = None
    requested_offer_id: str | None = None

    @model_validator(mode="after")
    def _specific_offer_needs_an_offer(self) -> CustomerIntent:
        if (self.objective is Objective.SPECIFIC_OFFER) != (self.requested_offer_id is not None):
            raise ValueError("requested_offer_id must be set if and only if objective is specific_offer")
        return self


class Mandate(_Model):
    """Spending authority delegated to the agent. Signature checking is out of scope."""

    mandate_id: str
    customer_id: str
    agent_id: str
    allowed_payee_ids: frozenset[str] = Field(min_length=1)
    permitted_scopes: frozenset[Scope] = Field(min_length=1)
    max_amount: Money  # ceiling for any single payment
    max_total_repayment: Money | None = None
    max_monthly_payment: Money | None = None
    issued_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def _window_is_ordered(self) -> Mandate:
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")
        return self


# --- Merchant, offers and repayment schedules ------------------------------


class Merchant(_Model):
    merchant_id: str
    name: str


class FeeKind(StrEnum):
    ARRANGEMENT = "arrangement"
    ADMINISTRATION = "administration"
    SERVICING = "servicing"


class Fee(_Model):
    kind: FeeKind
    amount: Money
    due_upfront: bool = False


class RepaymentSchedule(_Model):
    """Monthly payments in order, plus any mandatory fees."""

    payments: tuple[Money, ...] = Field(min_length=1)
    fees: tuple[Fee, ...] = ()

    @model_validator(mode="after")
    def _single_currency(self) -> RepaymentSchedule:
        currencies = {p.currency for p in self.payments} | {f.amount.currency for f in self.fees}
        if len(currencies) > 1:
            raise ValueError(f"schedule mixes currencies {sorted(currencies)}")
        return self

    @classmethod
    def level(cls, payment: Money, months: int, fees: tuple[Fee, ...] = ()) -> RepaymentSchedule:
        return cls(payments=(payment,) * months, fees=fees)

    @property
    def currency(self) -> str:
        return self.payments[0].currency

    @property
    def term_months(self) -> int:
        return len(self.payments)

    def total_payments(self) -> Money:
        return sum_money(self.payments, self.currency)

    def total_fees(self) -> Money:
        return sum_money((f.amount for f in self.fees), self.currency)

    def total(self) -> Money:
        """Everything the customer must repay: all payments plus all mandatory fees."""
        return self.total_payments() + self.total_fees()

    def max_payment(self) -> Money:
        return max(self.payments, key=lambda p: p.amount_minor)

    def initial_charge(self) -> Money:
        """What is collected when the agreement starts: first payment plus upfront fees."""
        upfront = sum_money((f.amount for f in self.fees if f.due_upfront), self.currency)
        return self.payments[0] + upfront


class FinancingOffer(_Model):
    offer_id: str
    merchant_id: str
    lender_id: str
    payee_id: str
    title: str
    description: str = ""  # merchant-controlled free text
    schedule: RepaymentSchedule
    advertised_total_repayable: Money | None = None  # the merchant's headline figure

    @property
    def total_repayable(self) -> Money:
        return self.schedule.total()


# --- Transaction, merchant assertions, independent evidence ----------------


class ProposedTransaction(_Model):
    transaction_id: str
    mandate_id: str
    offer_id: str
    payee_id: str
    amount: Money  # collected now: first payment plus any upfront fees
    scope: Scope
    proposed_at: AwareDatetime

    @model_validator(mode="after")
    def _amount_is_positive(self) -> ProposedTransaction:
        if self.amount.amount_minor == 0:
            raise ValueError("transaction amount must be greater than zero")
        return self


class MerchantAssertion(_Model):
    """A merchant statement about whether an offer is currently available."""

    assertion_id: str
    merchant_id: str
    offer_id: str
    claims_available: bool
    statement: str
    made_at: AwareDatetime


class SourceKind(StrEnum):
    LENDER = "lender"
    AGGREGATOR = "aggregator"
    MERCHANT_CHANNEL = "merchant_channel"


class EvidenceSource(_Model):
    source_id: str  # for a LENDER source, the lender's id
    name: str
    kind: SourceKind
    independent_of_merchant: bool


class Evidence(_Model):
    """An observation about an offer's availability from outside the merchant."""

    evidence_id: str
    offer_id: str
    offer_available: bool
    source: EvidenceSource
    observed_at: AwareDatetime


class Case(_Model):
    """The complete evaluator-visible input. Carries no ground-truth labels."""

    case_id: str
    intent: CustomerIntent
    mandate: Mandate
    merchant: Merchant
    offers: tuple[FinancingOffer, ...] = Field(min_length=1)
    transaction: ProposedTransaction
    assertions: tuple[MerchantAssertion, ...] = ()
    evidence: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def _offer_ids_are_unique(self) -> Case:
        ids = [o.offer_id for o in self.offers]
        if len(ids) != len(set(ids)):
            raise ValueError("offer_id values must be unique within a case")
        return self


# --- Detector output --------------------------------------------------------


class Decision(StrEnum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"

    @property
    def severity(self) -> int:
        return _SEVERITY[self]


_SEVERITY = {Decision.ALLOW: 0, Decision.REVIEW: 1, Decision.BLOCK: 2}


class FindingCategory(StrEnum):
    HARD_VIOLATION = "hard_violation"  # breaks a mandate or an explicit customer restriction
    CONFIRMED_FALSE_CLAIM = "confirmed_false_claim"  # contradicted by fresh independent evidence or arithmetic
    UNRESOLVED_CONTRADICTION = "unresolved_contradiction"  # fresh independent evidence disagrees with itself
    UNVERIFIED_CLAIM = "unverified_claim"  # no usable independent evidence either way
    INFERIOR_SELECTION = "inferior_selection"  # authorised, but worse than another usable offer
    SUSPICIOUS_BEHAVIOUR = "suspicious_behaviour"  # a pattern worth noting; not proof of an attack


class FindingCode(StrEnum):
    MANDATE_MISMATCH = "mandate_mismatch"
    CUSTOMER_MISMATCH = "customer_mismatch"
    MANDATE_EXPIRED = "mandate_expired"
    MANDATE_NOT_YET_VALID = "mandate_not_yet_valid"
    SCOPE_NOT_PERMITTED = "scope_not_permitted"
    PAYEE_NOT_ALLOWED = "payee_not_allowed"
    PAYEE_OFFER_MISMATCH = "payee_offer_mismatch"
    CURRENCY_MISMATCH = "currency_mismatch"
    AMOUNT_EXCEEDS_MANDATE = "amount_exceeds_mandate"
    AMOUNT_OFFER_MISMATCH = "amount_offer_mismatch"
    OFFER_NOT_FOUND = "offer_not_found"
    MANDATE_TOTAL_EXCEEDED = "mandate_total_exceeded"
    MANDATE_MONTHLY_EXCEEDED = "mandate_monthly_exceeded"
    INTENT_TOTAL_EXCEEDED = "intent_total_exceeded"
    INTENT_MONTHLY_EXCEEDED = "intent_monthly_exceeded"
    INTENT_TERM_MISMATCH = "intent_term_mismatch"
    INTENT_OFFER_MISMATCH = "intent_offer_mismatch"
    AVAILABILITY_CONTRADICTED = "availability_contradicted"
    ADVERTISED_TOTAL_MISMATCH = "advertised_total_mismatch"
    EVIDENCE_CONFLICT = "evidence_conflict"
    EVIDENCE_STALE = "evidence_stale"
    NO_INDEPENDENT_EVIDENCE = "no_independent_evidence"
    COSTLIER_THAN_USABLE_OFFER = "costlier_than_usable_offer"
    INSTRUCTION_LIKE_TEXT = "instruction_like_text"


class Finding(_Model):
    category: FindingCategory
    code: FindingCode
    message: str
    subject_id: str | None = None
    material: bool = True  # whether it bears on this transaction's outcome


class SelectionAssessment(_Model):
    """Decision quality, kept apart from authorisation."""

    selected_offer_id: str
    better_offer_id: str | None = None
    objective_gap: Money | None = None  # selected cost minus best usable cost, when positive
    inferior: bool = False  # the gap is at least the policy's materiality threshold


class SecurityDecision(_Model):
    decision: Decision
    authorised: bool  # no hard violation; says nothing about decision quality
    findings: tuple[Finding, ...] = ()
    selection: SelectionAssessment | None = None

    def has(self, category: FindingCategory, *, material_only: bool = False) -> bool:
        return any(f.category is category and (f.material or not material_only) for f in self.findings)
