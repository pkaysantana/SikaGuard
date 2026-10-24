"""Synthetic ground-truth labels.

Only the fixture author and the scoring code may read these. Nothing here may be
passed to a detector or to a future AI evaluator, and `policy.py` must never
import this module (a test enforces that).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from sika_guard.models import Decision


class AttackFamily(StrEnum):
    BENIGN = "benign"
    LEGITIMATE_PREMIUM = "legitimate_premium"
    FALSE_AVAILABILITY_CLAIM = "false_availability_claim"
    TRUTHFUL_AVAILABILITY_CHANGE = "truthful_availability_change"
    INFERIOR_BUT_PERMITTED = "inferior_but_permitted"
    HIDDEN_FEES = "hidden_fees"
    CREDIT_LIMIT_VIOLATION = "credit_limit_violation"
    UNCORROBORATED_CLAIM = "uncorroborated_claim"
    STALE_EVIDENCE = "stale_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    PROMPT_INJECTION = "prompt_injection"
    PAYEE_CHANGE = "payee_change"


class GroundTruth(BaseModel):
    """What is actually true about a case, as decided by the fixture author."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    attack_family: AttackFamily
    hard_authorisation_violation: bool  # breaks the mandate or an explicit customer restriction
    false_merchant_claim: bool  # a structured merchant claim or headline figure that is actually false
    financially_inferior: bool  # worse than a truly available, authorised offer under the stated objective
    avoidable_cost_minor: int  # minor units the customer overpays; 0 when not inferior
    suspicious_content_present: bool  # instruction-like text exists, whether or not it worked
    ideal_decision: Decision  # what a reviewer who knew all of the above would want
    rationale: str
