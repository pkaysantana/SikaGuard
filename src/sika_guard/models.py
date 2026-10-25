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
    max_amount: Money  # ceiling for any single payment; enforced today against the transaction's initial charge only (AUDIT_TRIAGE.md F7)
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
    # What kind of authority paying for this offer exercises. It is a property of the lender's product, like the schedule, and is
    # trusted for the same reason the schedule is (lender-originated). The proposing agent does NOT choose it: a transaction that
    # declares a different scope is a mismatch, never a re-classification (AUDIT R2).
    scope: Scope = Scope.CREDIT_AGREEMENT

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


class AvailabilitySubject(StrEnum):
    """What a statement or an observation about "availability" is actually about. The two are different facts and neither can test the other:
    that a lender sells a product does not show that this customer may have it, and a decline for one customer does not withdraw the
    product (AUDIT R7)."""

    PRODUCT_AVAILABILITY = "product_availability"  # the lender currently offers this product
    CUSTOMER_ELIGIBILITY = "customer_eligibility"  # this customer may (or may not) take it


class MerchantAssertion(_Model):
    """A merchant statement about whether an offer is currently available."""

    assertion_id: str
    merchant_id: str
    offer_id: str
    claims_available: bool
    statement: str
    made_at: AwareDatetime
    subject: AvailabilitySubject = AvailabilitySubject.PRODUCT_AVAILABILITY


class SourceKind(StrEnum):
    LENDER = "lender"
    AGGREGATOR = "aggregator"
    MERCHANT_CHANNEL = "merchant_channel"


class EvidenceSource(_Model):
    source_id: str  # for a LENDER source, the lender's id
    name: str
    kind: SourceKind
    independent_of_merchant: bool  # a claim the source's provider makes about itself; see `is_independent_of`

    @model_validator(mode="after")
    def _a_merchant_channel_is_never_independent(self) -> EvidenceSource:
        if self.kind is SourceKind.MERCHANT_CHANNEL and self.independent_of_merchant:
            raise ValueError("a merchant-channel source cannot be independent of the merchant")
        return self

    def is_independent_of(self, merchant_id: str) -> bool:
        """Independence is derived from what the source is, not only from what it declares. A merchant-channel source is never
        independent, nor is a source that carries the merchant's own identity, whatever the flag says. (The sandbox still assumes
        the flag is truthful for other kinds; it does not authenticate sources.)"""
        return self.independent_of_merchant and self.kind is not SourceKind.MERCHANT_CHANNEL and self.source_id != merchant_id


class Evidence(_Model):
    """An observation about an offer from outside the merchant. `subject` says which fact: that the product is on sale
    (`offer_available`), or that this customer may or may not have it (a decision, which must name the customer it is about)."""

    evidence_id: str
    offer_id: str
    offer_available: bool  # for CUSTOMER_ELIGIBILITY: True means eligible, False means declined
    source: EvidenceSource
    observed_at: AwareDatetime
    subject: AvailabilitySubject = AvailabilitySubject.PRODUCT_AVAILABILITY
    customer_id: str | None = None  # required for, and only for, a customer-eligibility decision

    @model_validator(mode="after")
    def _eligibility_names_its_customer(self) -> Evidence:
        if (self.subject is AvailabilitySubject.CUSTOMER_ELIGIBILITY) != (self.customer_id is not None):
            raise ValueError("customer_id must be set if and only if the evidence is about customer eligibility")
        return self


class SourceDocument(_Model):
    """A passage of unstructured text with a known origin: a lender's disclosure, an aggregator's notice, or the merchant's own page.

    The origin says WHO produced it, not that everything in it is true or that it concerns the right agreement. `offer_id` binds it to
    one offer (None: it discusses several, and a claim drawn from it must name the offer in the quoted words). A document whose source is
    the merchant's channel can never corroborate the merchant (AUDIT R9). Models read these; trusted code decides what they establish."""

    document_id: str
    offer_id: str | None = None
    source: EvidenceSource
    published_at: AwareDatetime
    text: str = Field(min_length=1, max_length=6000)


class TrustedContext(_Model):
    """Facts established by the trusted case-assembly context. Not by the proposing agent, the merchant or any model.

    In this sandbox the fixture asserts them, and that is an assumption, not authentication: a production deployment would take the time
    from the platform's own clock and the agent identity from the authenticated session that submitted the proposal. What matters here is
    that the policy never reads them from the agent's own proposal (AUDIT R1, R3)."""

    evaluated_at: AwareDatetime  # the decision clock: mandate validity, evidence freshness and claim chronology are judged at this instant
    acting_agent_id: str  # the agent that actually submitted this proposal


class Case(_Model):
    """The complete evaluator-visible input. Carries no ground-truth labels."""

    case_id: str
    intent: CustomerIntent
    mandate: Mandate
    merchant: Merchant
    offers: tuple[FinancingOffer, ...] = Field(min_length=1)
    transaction: ProposedTransaction
    trusted: TrustedContext
    assertions: tuple[MerchantAssertion, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    documents: tuple[SourceDocument, ...] = ()

    @model_validator(mode="after")
    def _documents_are_well_formed(self) -> Case:
        ids = [d.document_id for d in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document_id values must be unique within a case")
        offer_ids = {o.offer_id for o in self.offers}
        for d in self.documents:
            if d.offer_id is not None and d.offer_id not in offer_ids:
                raise ValueError(f"document {d.document_id} is bound to an offer that is not in the case")
            if d.source.independent_of_merchant and d.source.source_id == self.merchant.merchant_id:
                raise ValueError(f"document {d.document_id} claims to be independent but its source carries the merchant's own identity")
        return self

    @model_validator(mode="after")
    def _offer_ids_are_unique(self) -> Case:
        ids = [o.offer_id for o in self.offers]
        if len(ids) != len(set(ids)):
            raise ValueError("offer_id values must be unique within a case")
        return self

    @model_validator(mode="after")
    def _no_evidence_independent_under_the_merchants_own_identity(self) -> Case:
        for e in self.evidence:
            if e.source.independent_of_merchant and e.source.source_id == self.merchant.merchant_id:
                raise ValueError(f"evidence {e.evidence_id} claims to be independent but its source carries the merchant's own identity")
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
    EVIDENCE_REQUIREMENT_UNMET = "evidence_requirement_unmet"  # an uncorroborated claim is decision-critical under the customer's objective
    SUSPICIOUS_BEHAVIOUR = "suspicious_behaviour"  # a pattern worth noting; not proof of an attack
    INCOMPARABLE_REPRESENTATION = "incomparable_representation"  # a figure shown for the selected offer cannot be compared with the lender's terms
    SELECTED_OFFER_NOT_OPEN = "selected_offer_not_open"  # fresh independent evidence says the offer being bought cannot be taken (withdrawn, or this customer declined)


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
    ADVERTISED_CURRENCY_INCOMPATIBLE = "advertised_currency_incompatible"
    ADVERTISED_TOTAL_OVERSTATED = "advertised_total_overstated"  # the headline figure is higher than the lender's total: recorded, not a harm
    EVIDENCE_CONFLICT = "evidence_conflict"
    EVIDENCE_STALE = "evidence_stale"
    NO_INDEPENDENT_EVIDENCE = "no_independent_evidence"
    COSTLIER_THAN_USABLE_OFFER = "costlier_than_usable_offer"
    DECISION_CRITICAL_CLAIM_UNCORROBORATED = "decision_critical_claim_uncorroborated"
    INSTRUCTION_LIKE_TEXT = "instruction_like_text"
    PROPOSAL_FROM_THE_FUTURE = "proposal_from_the_future"  # the agent dated its proposal after the trusted decision clock
    SCOPE_OFFER_MISMATCH = "scope_offer_mismatch"  # the proposal declares a scope other than the offer's own
    AGENT_MISMATCH = "agent_mismatch"  # the mandate was issued to a different agent than the one acting
    COMMITTED_PAYMENT_EXCEEDS_MANDATE = "committed_payment_exceeds_mandate"  # a later committed payment exceeds the per-payment ceiling
    IMPOSSIBLE_TIMESTAMP = "impossible_timestamp"  # a merchant statement dated after the decision clock; ignored, never trusted
    ELIGIBILITY_DECLINED = "eligibility_declined"  # a lender decision shows this customer cannot take the offer
    OFFER_REPORTED_UNAVAILABLE = "offer_reported_unavailable"  # fresh independent evidence says the product itself is not on sale, whatever anyone claimed
    ELIGIBILITY_UNRESOLVED = "eligibility_unresolved"  # the merchant says the customer is ineligible and no lender decision shows it


class DecisionBasis(StrEnum):
    """Why a case ended where it did. Kept apart so that a review for an unmet evidence requirement is
    never confused with a review for a verified contradiction, or with a hard block.

    Nothing here describes a merchant as dishonest: only a verified contradiction names a false claim."""

    ALLOW = "allow"
    ALLOW_WITH_UNCERTAINTY = "allow_with_recorded_uncertainty"  # unverified claims, model concerns or evaluator failures were recorded; none was enforced
    REVIEW_EVIDENCE_REQUIREMENT = "review_evidence_requirement_unmet"  # a decision-critical claim lacks the corroboration the policy requires
    REVIEW_VERIFIED_CONTRADICTION = "review_verified_contradiction"  # a material claim is contradicted by trustworthy evidence or lender terms
    REVIEW_UNRESOLVED_CONFLICT = "review_unresolved_evidence_conflict"  # fresh independent sources disagree about a material claim
    REVIEW_INFERIOR_SELECTION = "review_inferior_selection"  # authorised, but materially worse than a usable alternative
    REVIEW_EVALUATOR_FAILURE = "review_evaluator_failure"  # fail-closed mode only: an evaluator produced no usable verdict; not an attack finding
    REVIEW_INCOMPARABLE_REPRESENTATION = "review_incomparable_representation"  # the selected offer's headline figure cannot be compared with the lender's terms
    REVIEW_SELECTED_OFFER_NOT_OPEN = "review_selected_offer_not_open"  # fresh independent evidence says the selected offer cannot be taken
    BLOCK_HARD_VIOLATION = "block_hard_violation"  # the transaction breaks the mandate or an explicit customer restriction


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
    excluded_as_not_open: tuple[str, ...] = ()  # offers left out of the comparison because fresh independent evidence says they cannot be taken


class SecurityDecision(_Model):
    decision: Decision
    authorised: bool  # no hard violation; says nothing about decision quality
    findings: tuple[Finding, ...] = ()
    selection: SelectionAssessment | None = None

    def has(self, category: FindingCategory, *, material_only: bool = False) -> bool:
        return any(f.category is category and (f.material or not material_only) for f in self.findings)
