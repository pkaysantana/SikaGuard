"""Evaluator contract.

Input is an immutable, label-free `EvaluatorView`. Output is an `EvaluatorVerdict`
in which every outcome, including timeouts, refusals and malformed responses, is an
explicit status and never a silent drop.

Two schemas are kept apart on purpose:
  * `ModelAnalysis` is the strict JSON a model is asked to return.
  * `EvaluatorVerdict` is the record trusted code builds around it, adding identity,
    status, timing and usage that a model must never be able to set.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Protocol, Self, runtime_checkable

from sika_guard.evaluators.grounding import claim_problem
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, ValidationError, model_validator

from sika_guard.models import (
    Case,
    CustomerIntent,
    Decision,
    Evidence,
    FinancingOffer,
    Mandate,
    Merchant,
    MerchantAssertion,
    ProposedTransaction,
    SourceDocument,
    TrustedContext,
)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Input ------------------------------------------------------------------


class EvaluatorView(_Model):
    """What an evaluator may see about a case. Carries no labels, no expected decision and no
    deterministic findings, so each evaluator forms its own view."""

    case_id: str
    intent: CustomerIntent
    mandate: Mandate
    merchant: Merchant
    offers: tuple[FinancingOffer, ...]
    transaction: ProposedTransaction
    trusted: TrustedContext
    assertions: tuple[MerchantAssertion, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    documents: tuple[SourceDocument, ...] = ()

    @classmethod
    def from_case(cls, case: Case) -> Self:
        return cls(**{name: getattr(case, name) for name in cls.model_fields})

    @property
    def as_of(self) -> datetime:
        """The decision clock, from the trusted context. Never the proposal's own timestamp."""
        return self.trusted.evaluated_at

    def sha256(self) -> str:
        """Stable fingerprint, used to show that every evaluator was given the same data."""
        return hashlib.sha256(_canonical_json(self.model_dump()).encode()).hexdigest()


def _canonical_json(obj: Any) -> str:
    def default(value: Any) -> Any:
        if isinstance(value, frozenset | set):
            return sorted(value, key=str)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Decimal):
            return str(value)
        raise TypeError(f"cannot serialise {type(value).__name__}")

    return json.dumps(obj, sort_keys=True, default=default, separators=(",", ":"))


# --- Vocabulary -------------------------------------------------------------


class EvaluatorRole(StrEnum):
    MOCK = "mock"
    GENERALIST = "generalist"
    RUBRIC_GENERALIST = "rubric_generalist"  # one generalist applying all three specialist rubrics (Condition Bc)
    MERCHANT_CLAIM = "merchant_claim"
    CREDIT_TERMS = "credit_terms"
    AUTHORITY = "authority"
    DETERMINISTIC = "deterministic"  # the non-LLM scanner of Condition A+: not a model, sees what the generalist sees


class EvaluatorStatus(StrEnum):
    OK = "ok"
    ABSTAINED = "abstained"  # the evaluator chose not to judge
    REFUSAL = "refusal"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"  # unparseable, schema-invalid or citing unknown evidence
    PROVIDER_ERROR = "provider_error"


class ConcernCode(StrEnum):
    """Closed vocabulary for AI findings. Deliberately distinct from the deterministic FindingCode."""

    CLAIM_CONTRADICTED_BY_EVIDENCE = "claim_contradicted_by_evidence"
    CLAIM_UNVERIFIED = "claim_unverified"
    CLAIM_EVIDENCE_OUT_OF_DATE = "claim_evidence_out_of_date"
    INSTRUCTION_IN_UNTRUSTED_CONTENT = "instruction_in_untrusted_content"
    SELECTION_WORSE_THAN_ALTERNATIVE = "selection_worse_than_alternative"
    INTENT_MISMATCH = "intent_mismatch"
    HIDDEN_OBLIGATION = "hidden_obligation"
    TERMS_INCONSISTENT_WITH_REPRESENTATION = "terms_inconsistent_with_representation"
    LIMIT_EXCEEDED = "limit_exceeded"
    PAYEE_OR_IDENTITY_CONCERN = "payee_or_identity_concern"
    SCOPE_OR_VALIDITY_CONCERN = "scope_or_validity_concern"
    CURRENCY_OR_AMOUNT_CONCERN = "currency_or_amount_concern"
    OTHER_CONCERN = "other_concern"


class ClaimKind(StrEnum):
    AVAILABILITY = "availability"
    TOTAL_REPAYABLE = "total_repayable"
    NO_ADDITIONAL_FEES = "no_additional_fees"
    FEE_AMOUNT = "fee_amount"  # a stated total of mandatory fees


# --- What a model must return -----------------------------------------------


class AIFinding(_Model):
    code: ConcernCode
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    note: str = Field(max_length=300)


class ExtractedClaim(_Model):
    """A merchant claim a model pulled out of untrusted content. Trusted code, not the model,
    later decides whether it holds."""

    kind: ClaimKind
    offer_id: str
    source_ref: str
    quote: str = Field(min_length=3, max_length=300)  # must appear verbatim in the cited source
    asserted_available: bool | None = None
    amount_minor: int | None = Field(default=None, ge=0)
    currency: str | None = None

    @model_validator(mode="after")
    def _fields_match_kind(self) -> Self:
        # A stray asserted_available on a non-availability claim is tolerated: trusted code ignores it.
        has_amount = self.amount_minor is not None or self.currency is not None
        if self.kind is ClaimKind.AVAILABILITY and self.asserted_available is None:
            raise ValueError("availability claims need asserted_available")
        if self.kind is ClaimKind.AVAILABILITY and has_amount:
            raise ValueError("an availability claim carries no amount; it may have been mislabelled")
        if self.kind in (ClaimKind.TOTAL_REPAYABLE, ClaimKind.FEE_AMOUNT) and (self.amount_minor is None or self.currency is None):
            raise ValueError(f"{self.kind.value} claims need amount_minor and currency")
        if self.kind is ClaimKind.NO_ADDITIONAL_FEES and has_amount:
            raise ValueError("a no-additional-fees claim carries no amount")
        return self


class ModelAnalysis(_Model):
    verdict: Decision | None
    findings: list[AIFinding] = Field(default_factory=list, max_length=8)
    extracted_claims: list[ExtractedClaim] = Field(default_factory=list, max_length=12)
    risk_score: float = Field(ge=0, le=1)  # uncalibrated; never used to decide anything
    abstain: StrictBool
    abstain_reason: str | None = Field(default=None, max_length=300)
    explanation: str = Field(max_length=800)

    @model_validator(mode="after")
    def _abstention_is_consistent(self) -> Self:
        if self.abstain and (self.verdict is not None or not self.abstain_reason):
            raise ValueError("an abstention needs a reason and no verdict")
        if not self.abstain and self.verdict is None:
            raise ValueError("a non-abstaining analysis needs a verdict")
        return self


# --- What trusted code records ----------------------------------------------


class CallStatus(StrEnum):
    OK = "ok"
    TIMEOUT = "timeout"
    REFUSAL = "refusal"
    PROVIDER_ERROR = "provider_error"


class CallRecord(_Model):
    """One model API call. Token counts and costs are None unless the provider returned them
    (or, for `cost`, prices were configured).

    `latency_ms` comes from a monotonic clock. `started_at` and `finished_at` are UTC wall-clock
    timestamps for ordering and overlap analysis only; do not derive latency from them.

    `cost` is computed locally from configured prices. `provider_cost` is the provider's own
    `usage.cost`, kept exactly as returned. Its unit is not documented in the response, so it is
    labelled UNKNOWN and is never added to `cost` or treated as a currency amount."""

    evaluator_id: str
    model: str
    status: CallStatus
    latency_ms: float
    http_status: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: Decimal | None = None
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None
    provider: str | None = None  # the `provider` field of the response body, when it is a string
    endpoint_host: str | None = None  # host of the configured endpoint; never a path, query or credential
    provider_cost: Decimal | None = None
    provider_cost_unit: Literal["UNKNOWN"] | None = None
    returned_model: str | None = None  # the `model` the response body names; may differ from the requested `model`
    raw_usage: dict[str, Any] | None = None  # the provider's `usage` object exactly as returned (JSON-safe, size-capped)


class RejectedClaim(_Model):
    """A claim a model extracted that trusted code refused, kept with the reason so a rejection can be studied rather than lost.
    A rejection is never evidence about the merchant."""

    claim: ExtractedClaim
    stage: Literal["reply_references", "reply_grounding"] = "reply_grounding"  # which check refused it: an unknown reference or offer, or a quote not in the source
    reason: str


class Normalisation(_Model):
    """A presentation-only repair made to a model reply, with the original kept for audit.

    Only free-text display fields (finding notes, the explanation, an abstention reason) that exceed
    their length cap are ever repaired, by truncation. Identifiers, amounts, currencies, codes,
    claim kinds and quotes are never repaired: an error in any of them rejects the whole reply."""

    path: str  # e.g. "findings.0.note"
    kind: Literal["truncated_presentation_text"] = "truncated_presentation_text"
    original_length: int
    kept_length: int
    original_text: str


class EvaluatorVerdict(_Model):
    case_id: str
    evaluator_id: str
    role: EvaluatorRole
    status: EvaluatorStatus
    verdict: Decision | None
    findings: tuple[AIFinding, ...] = ()
    extracted_claims: tuple[ExtractedClaim, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    risk_score: float | None = None
    risk_score_calibration: Literal["uncalibrated"] = "uncalibrated"
    abstained: bool
    abstain_reason: str | None = None
    explanation: str = ""
    prompt_version: str | None = None
    model_id: str | None = None
    view_sha256: str
    calls: tuple[CallRecord, ...] = ()
    raw_response: str | None = None  # the model's reply text exactly as returned, kept for audit
    normalisations: tuple[Normalisation, ...] = ()  # empty means the reply needed no repair
    rejected_claims: tuple[RejectedClaim, ...] = ()  # extracted claims trusted code refused, with why; they never reach the decision

    @property
    def normalised(self) -> bool:
        return bool(self.normalisations)

    @model_validator(mode="after")
    def _status_matches_fields(self) -> Self:
        if self.status is EvaluatorStatus.OK:
            if self.verdict is None or self.abstained:
                raise ValueError("an OK verdict needs a verdict and must not abstain")
        elif self.verdict is not None or not self.abstained or not self.abstain_reason:
            raise ValueError("any non-OK status is an abstention with a reason and no verdict")
        return self

    @property
    def usable(self) -> bool:
        return self.status is EvaluatorStatus.OK


@runtime_checkable
class Evaluator(Protocol):
    evaluator_id: str
    role: EvaluatorRole

    async def evaluate(self, view: EvaluatorView) -> EvaluatorVerdict: ...


# --- Building verdicts ------------------------------------------------------


class InvalidResponse(ValueError):
    """The model's output was unparseable, schema-invalid or ungrounded."""


def _strip_fence(raw: str | None) -> str:
    if raw is None or not raw.strip():
        raise InvalidResponse("empty response")
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text


def _problems(errors: list[Any]) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'response'}: {e['type']}" for e in errors[:5])


def parse_analysis(raw: str | None) -> ModelAnalysis:
    """Parse model text into a ModelAnalysis, strictly. Error text never echoes the model's content."""
    text = _strip_fence(raw)
    try:
        return ModelAnalysis.model_validate_json(text)
    except ValidationError as exc:
        raise InvalidResponse(f"schema validation failed ({_problems(exc.errors())})") from None


# Free-text display fields whose only possible defect is being too long. Everything else is
# security-critical and is never repaired.
_PRESENTATION_FIELDS = {("findings", "note"), ("explanation",), ("abstain_reason",)}


def _is_presentation_overflow(error: Any) -> bool:
    loc = tuple(error["loc"])
    shape = (loc[0], loc[-1]) if len(loc) == 3 and isinstance(loc[1], int) else loc
    return error["type"] == "string_too_long" and shape in _PRESENTATION_FIELDS


def parse_analysis_normalised(raw: str | None) -> tuple[ModelAnalysis, tuple[Normalisation, ...]]:
    """Like `parse_analysis`, except that a reply whose only defects are over-long display text is
    accepted with that text truncated and the originals recorded in `Normalisation`s.

    A defect in any other field (a code, reference, offer id, claim kind, amount, currency or quote)
    rejects the whole reply, even when an over-long note is also present, and nothing is repaired."""
    text = _strip_fence(raw)
    try:
        return ModelAnalysis.model_validate_json(text), ()
    except ValidationError as exc:
        errors = exc.errors()
    if not errors or not all(_is_presentation_overflow(e) for e in errors):
        raise InvalidResponse(f"schema validation failed ({_problems(errors)})") from None

    data = json.loads(text)  # valid JSON: every error was a length error, not a syntax error
    norms = []
    for error in errors:
        *parents, leaf = error["loc"]
        node = data
        for key in parents:
            node = node[key]
        original = node[leaf]
        limit = int(error["ctx"]["max_length"])
        node[leaf] = original[: limit - 1] + "…"
        norms.append(
            Normalisation(
                path=".".join(str(p) for p in error["loc"]), original_length=len(original), kept_length=limit, original_text=original
            )
        )
    try:
        return ModelAnalysis.model_validate(data), tuple(norms)
    except ValidationError as exc:
        raise InvalidResponse(f"schema validation failed ({_problems(exc.errors())})") from None


RAW_RESPONSE_CAP = 100_000


def failure_verdict(
    view: EvaluatorView,
    *,
    evaluator_id: str,
    role: EvaluatorRole,
    status: EvaluatorStatus,
    reason: str,
    prompt_version: str | None = None,
    model_id: str | None = None,
    calls: tuple[CallRecord, ...] = (),
    raw_response: str | None = None,
    normalisations: tuple[Normalisation, ...] = (),
    rejected_claims: tuple[RejectedClaim, ...] = (),
) -> EvaluatorVerdict:
    return EvaluatorVerdict(
        case_id=view.case_id,
        evaluator_id=evaluator_id,
        role=role,
        status=status,
        verdict=None,
        abstained=True,
        abstain_reason=reason,
        prompt_version=prompt_version,
        model_id=model_id,
        view_sha256=view.sha256(),
        calls=calls,
        raw_response=raw_response[:RAW_RESPONSE_CAP] if raw_response is not None else None,
        normalisations=normalisations,
        rejected_claims=rejected_claims,
    )


def verdict_from_analysis(
    analysis: ModelAnalysis,
    view: EvaluatorView,
    *,
    evaluator_id: str,
    role: EvaluatorRole,
    allowed_refs: frozenset[str],
    prompt_version: str | None = None,
    model_id: str | None = None,
    calls: tuple[CallRecord, ...] = (),
    raw_response: str | None = None,
    normalisations: tuple[Normalisation, ...] = (),
    payload: dict[str, Any] | None = None,
) -> EvaluatorVerdict:
    """Ground a parsed analysis in what the evaluator was actually shown, then record it.

    With `payload` (the request this evaluator received), every extracted claim must also be grounded in it: the source must be a
    merchant item in that request, concern the claim's offer, and contain the quote. An invented, misattributed or unseen quotation
    makes the whole reply an INVALID_RESPONSE, which is an evaluator failure under the documented failure policy. The production
    paths always pass it; `None` skips this check (the trusted cross-check still repeats it)."""
    common = dict(
        evaluator_id=evaluator_id, role=role, prompt_version=prompt_version, model_id=model_id, calls=calls,
        raw_response=raw_response, normalisations=normalisations,
    )
    cited = {r for f in analysis.findings for r in f.evidence_refs} | {c.source_ref for c in analysis.extracted_claims}
    offer_ids = {o.offer_id for o in view.offers}
    unknown = sorted(cited - allowed_refs)
    bad_offers = sorted({c.offer_id for c in analysis.extracted_claims} - offer_ids)
    if unknown or bad_offers:
        refused = tuple(
            RejectedClaim(claim=c, stage="reply_references", reason="cites a reference or an offer that is not in this evaluator's input")
            for c in analysis.extracted_claims
            if c.source_ref not in allowed_refs or c.offer_id not in offer_ids
        )
        return failure_verdict(
            view, status=EvaluatorStatus.INVALID_RESPONSE, rejected_claims=refused, reason=f"cited references not present in the input: {(unknown + bad_offers)[:5]}", **common
        )
    if payload is not None:
        rejected = tuple(
            RejectedClaim(claim=claim, reason=problem)
            for claim in analysis.extracted_claims
            if (problem := claim_problem(payload, source_ref=claim.source_ref, offer_id=claim.offer_id, quote=claim.quote)) is not None
        )
        if rejected:
            return failure_verdict(
                view, status=EvaluatorStatus.INVALID_RESPONSE, rejected_claims=rejected,
                reason=f"extracted claim not grounded in what this evaluator was shown: {rejected[0].reason}", **common,
            )
    if analysis.abstain:
        return failure_verdict(view, status=EvaluatorStatus.ABSTAINED, reason=analysis.abstain_reason or "abstained", **common)
    return EvaluatorVerdict(
        case_id=view.case_id,
        status=EvaluatorStatus.OK,
        verdict=analysis.verdict,
        findings=tuple(analysis.findings),
        extracted_claims=tuple(analysis.extracted_claims),
        evidence_refs=tuple(sorted(cited)),
        risk_score=analysis.risk_score,
        abstained=False,
        explanation=analysis.explanation,
        view_sha256=view.sha256(),
        **{**common, "raw_response": raw_response[:RAW_RESPONSE_CAP] if raw_response is not None else None},
    )
