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

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, model_validator

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
    assertions: tuple[MerchantAssertion, ...] = ()
    evidence: tuple[Evidence, ...] = ()

    @classmethod
    def from_case(cls, case: Case) -> Self:
        return cls(**{name: getattr(case, name) for name in cls.model_fields})

    @property
    def as_of(self) -> datetime:
        return self.transaction.proposed_at

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
    MERCHANT_CLAIM = "merchant_claim"
    CREDIT_TERMS = "credit_terms"
    AUTHORITY = "authority"


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
        if self.kind is ClaimKind.AVAILABILITY and self.asserted_available is None:
            raise ValueError("availability claims need asserted_available")
        if self.kind is ClaimKind.TOTAL_REPAYABLE and (self.amount_minor is None or self.currency is None):
            raise ValueError("total_repayable claims need amount_minor and currency")
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
    """One model API call. Token counts and cost are None unless the provider returned them."""

    evaluator_id: str
    model: str
    status: CallStatus
    latency_ms: float
    http_status: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: Decimal | None = None


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


def parse_analysis(raw: str | None) -> ModelAnalysis:
    """Parse model text into a ModelAnalysis. Error text never echoes the model's content."""
    if raw is None or not raw.strip():
        raise InvalidResponse("empty response")
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    try:
        return ModelAnalysis.model_validate_json(text)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'response'}: {e['type']}" for e in exc.errors()[:5])
        raise InvalidResponse(f"schema validation failed ({problems})") from None


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
) -> EvaluatorVerdict:
    """Ground a parsed analysis in what the evaluator was actually shown, then record it."""
    cited = {r for f in analysis.findings for r in f.evidence_refs} | {c.source_ref for c in analysis.extracted_claims}
    offer_ids = {o.offer_id for o in view.offers}
    unknown = sorted(cited - allowed_refs)
    bad_offers = sorted({c.offer_id for c in analysis.extracted_claims} - offer_ids)
    if unknown or bad_offers:
        return failure_verdict(
            view,
            evaluator_id=evaluator_id,
            role=role,
            status=EvaluatorStatus.INVALID_RESPONSE,
            reason=f"cited references not present in the input: {(unknown + bad_offers)[:5]}",
            prompt_version=prompt_version,
            model_id=model_id,
            calls=calls,
        )
    if analysis.abstain:
        return failure_verdict(
            view,
            evaluator_id=evaluator_id,
            role=role,
            status=EvaluatorStatus.ABSTAINED,
            reason=analysis.abstain_reason or "abstained",
            prompt_version=prompt_version,
            model_id=model_id,
            calls=calls,
        )
    return EvaluatorVerdict(
        case_id=view.case_id,
        evaluator_id=evaluator_id,
        role=role,
        status=EvaluatorStatus.OK,
        verdict=analysis.verdict,
        findings=tuple(analysis.findings),
        extracted_claims=tuple(analysis.extracted_claims),
        evidence_refs=tuple(sorted(cited)),
        risk_score=analysis.risk_score,
        abstained=False,
        explanation=analysis.explanation,
        prompt_version=prompt_version,
        model_id=model_id,
        view_sha256=view.sha256(),
        calls=calls,
    )
