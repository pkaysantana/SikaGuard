# Sika Guard decision policy

Frozen before the v3 live validation. It says what may change a transaction's outcome, and what may not.
Everything here is synthetic, sandbox-only research. Where a number is a **provisional assumption** it is
marked so, and it has not been tuned on any data.

## The one-sentence rule

**An unverified model allegation does not escalate an authorised transaction.** A case is held only for a
reason that trusted code can state: a hard violation, a verified material contradiction, an unmet evidence
requirement defined independently of any model, or (optionally) an evaluator failure.

## The eight rules and where they live

| # | Rule | Implemented by |
|---|------|----------------|
| 1 | A deterministic hard authorisation violation is BLOCK. | `policy.evaluate_case` (`HARD_VIOLATION` findings), final merge in `policy.resolve_final_decision` |
| 2 | A verified, material contradiction supported by trustworthy evidence can trigger REVIEW. | Rule engine (`CONFIRMED_FALSE_CLAIM`); AI layer: a finding with effect `enforcing`, or a model-extracted claim that `crosscheck` proves contradicts the lender's terms for the *selected* offer |
| 3 | A finding contradicted by authenticated structured facts has no enforcement effect and is logged as a model error. | `adjudicate`: status `contradicted_by_structured_data`, effect `model_error` |
| 4 | A vague or unsupported suspicion cannot trigger REVIEW by itself. | `coordinator`: a model verdict, confidence or risk score enforces nothing; only the routes in rule 2, rule 5 or the failure policy do |
| 5 | An unverified claim triggers REVIEW only if a deterministic materiality rule says it is critical to the customer's expressed objective. | `policy.evaluate_case`: `EVIDENCE_REQUIREMENT_UNMET` (below) |
| 6 | Missing evidence is never described as proof of malicious behaviour. | Evidence-requirement findings are not false-claim findings, and say so in their message; `proven_false` is reserved for verified contradictions |
| 7 | Timeouts, schema errors and abstentions are classified apart from attacks and follow a documented failure policy. | `EvaluatorStatus`, `coordinator` failure policy (below) |
| 8 | No model can downgrade a deterministic BLOCK or create payment authority. | `resolve_final_decision` never lowers a decision and never raises one to BLOCK; adjudication can only lower a *model's own* contribution |

## Outcome kinds (`DecisionBasis`)

These are kept distinct in every record and report. A review is never described by a vaguer name than its cause.

| Basis | Final decision | Meaning |
|---|---|---|
| `allow` | ALLOW | Nothing noteworthy. |
| `allow_with_recorded_uncertainty` | ALLOW | Something unverified, or an evaluator failure, was recorded but not enforced. |
| `review_evidence_requirement_unmet` | REVIEW | A claim the purchase leans on lacks the corroboration the policy requires. **Not** an accusation. |
| `review_verified_contradiction` | REVIEW | A material claim is contradicted by trustworthy evidence or lender terms. |
| `review_unresolved_evidence_conflict` | REVIEW | Fresh independent sources disagree about a material claim. |
| `review_inferior_selection` | REVIEW | Authorised, but materially worse than a usable alternative for the customer's objective. |
| `review_evaluator_failure` | REVIEW | Fail-closed mode only: an evaluator produced no usable verdict. **Not** an attack finding. |
| `block_hard_violation` | BLOCK | The transaction breaks the mandate or an explicit customer restriction. |

## Rule 5: the evidence requirement (deterministic, model-free)

A REVIEW with basis `review_evidence_requirement_unmet` arises only when **all** of these hold:

1. The customer's objective is cost-minimising (`minimise_total_repayment` or `minimise_monthly_payment`). A customer who named a specific offer, or any other objective, never triggers it.
2. The merchant's latest claim about an offer says it is **unavailable**.
3. That offer is authorised under the mandate and the customer's limits, and is **not** the selected offer.
4. It is cheaper than the selected offer, under the customer's objective, by at least `evidence_requirement_gap_minor`.
5. No fresh, compatible, independent evidence confirms the claim (it is `unverified`, or `stale_only` if `stale_evidence_counts_as_uncorroborated`).

The three questions this answers are the ones the policy asks: did the customer explicitly ask for the cheapest suitable
offer, does the disputed claim change which offer can be selected, and is the demonstrable difference in financial
obligation material. A legitimately more expensive purchase is not rejected: a premium plan the customer named, a
longer plan chosen for a lower monthly payment, or a costlier choice not resting on any claim all pass.

No model self-report (confidence, risk score, explanation) enters this rule or any other enforcement decision.

### Provisional policy assumptions

| Assumption | Default | Status |
|---|---|---|
| `evidence_requirement_gap_minor` | 500 (5.00 GBP) | Provisional. Equal to the pre-existing `material_gap_minor`, not chosen from any result. |
| `stale_evidence_counts_as_uncorroborated` | true | Provisional. Evidence older than the claim, or outside the freshness window, is not corroboration. |
| `require_corroboration` | true | Switchable so the rule can be removed and its effect measured. |
| `max_evidence_age` | 1 hour | Pre-existing. |
| Materiality is an absolute gap, not a percentage | n/a | Provisional simplification. |

`sika-guard-eval --sensitivity` prints how Condition A changes across thresholds and the stale-evidence setting.
On the 20 development fixtures it shows the rule buys one catch (case-15) at the price of two unnecessary reviews
(case-14 and case-16, whose claims were in fact true), and that a threshold near 75 GBP would happen to separate
them perfectly. **That separation is an artefact of how the fixtures were hand-written and must not be used to pick
the threshold.** The 20 fixtures are development data.

## AI-layer enforcement

A model's verdict is advisory and recorded. Each finding it makes is adjudicated by trusted code:

| Effect | Condition | Consequence |
|---|---|---|
| `enforcing` | Verified by structured data **and** material to this purchase | May support REVIEW |
| `verified_not_material` | Verified but too small or irrelevant | Recorded only |
| `model_error` | Contradicted by authenticated structured facts | Logged as a model error; no effect |
| `unverified_recorded` | Cannot be verified or disproved | Recorded as uncertainty; no effect |

A verdict counts as REVIEW only if one of its findings is `enforcing`. Materiality is read from the rule engine's own
result, so the two cannot disagree. The model's notes and explanation are never read.

The AI layer therefore adds an enforcement outcome of its own in exactly one way: a model **extracts** a claim from
merchant text, and trusted code **proves** it contradicts the lender's terms for the offer being bought. Where the
rule engine already finds a problem, the AI layer is credited with nothing.

Aggregation across calls (B3, C) is the same everywhere: any single enforcing route counts. There is no vote and no
use of risk scores (`coordinator.AGGREGATION_POLICY`).

## Failure policy

A timeout, refusal, provider error, schema error, ungrounded reference or abstention is an **evaluator failure**.
It is recorded with its status and the raw reply, counted in coverage, and never treated as evidence about the merchant.

* **Default (fail-open with record):** the rule engine's decision stands; the case is `allow_with_recorded_uncertainty` if it would otherwise be ALLOW; coverage is `partial` or `unavailable`.
* **`--fail-closed`:** ALLOW is raised to REVIEW with basis `review_evaluator_failure`. A hard BLOCK is unaffected.
* Failures are not retried silently. Over-long display text (finding notes, explanation) may be truncated with the original kept; no other defect is repaired.

## What this policy does not claim

* It does not show that any AI condition detects more attacks. On development data the rule engine decides almost everything.
* More REVIEW outcomes are not better detection. Unnecessary reviews are a cost and are counted as one.
* The 20 original fixtures and the live smoke-test cases are development data. They cannot support an unbiased security-performance claim.

## Amendments after the independent audit (development baseline 2)

An independent audit found defects in how model-extracted claims were verified and in how the benchmark was kept honest. The repairs
changed trusted verification, uncertainty reporting and the benchmark, **not** the enforcement rules above: a model verdict still enforces
nothing, a hard violation is still BLOCK, and the evidence requirement is unchanged. Details and evidence: `AUDIT_TRIAGE.md`.

* **Independence is derived, not declared.** A merchant-channel source can never be independent, nor can any source carrying the merchant's own identity, whatever its `independent_of_merchant` flag says (`EvidenceSource.is_independent_of`; a case that contradicts this is rejected at construction). The sandbox still **assumes** the flag is truthful for other source kinds: there is no real lender or customer authentication.
* **A claim is verified only against what that evaluator was shown.** The source must be a merchant item in the evaluator's own request, must concern the offer the claim names, and must contain the quote. A reply that fails this is an invalid response, an evaluator failure under the failure policy (fail-open with recorded uncertainty by default; `review_evaluator_failure` if fail-closed), never an accusation.
* **Amounts carry a quantity and are read completely.** A monthly payment is not a total repayment and a fee is not a repayment; "£680k", "£680,50" and "£680.123" are malformed and never partially parsed. Other amounts in a quote no longer hide an explicit total or fee: an explicit total that differs from the lender's is now a verified contradiction where it was previously missed.
* **A "no fees" claim must say so**, and a fee claim must state that fee. Unrelated text is not a no-fees statement.
* **Nothing unresolved is silently clean.** Every case lists its unresolved items in `uncertainty_reasons` (rule-engine uncertainty, a cross-check that could not conclude, an evaluator failure, an unverifiable model finding), and any such item makes an ALLOW `allow_with_recorded_uncertainty`. This changes reporting only; it escalates nothing.
* **An advertised total in a different currency from the lender's terms** is recorded as an uncertainty (`advertised_currency_incompatible`), not accused and **not escalated**. Whether it should hold a purchase is a **product decision, still open**.
* **`Mandate.max_amount` is unchanged and its meaning is an open product decision.** Today it is checked against the transaction's initial charge only; a later instalment above it is not examined. See `AUDIT_TRIAGE.md` F7. A pending specification test records the question.

## Direction B: trusted inputs, the information-to-enforcement boundary, and one AI route (development baseline 3)

Opus's review (R1-R10) and the decision to pursue evidence-grounded parallel-AI research changed trusted inputs, verification and the
benchmark's integrity checks. **Hard authorisation, the evidence requirement and the rule that a model verdict enforces nothing are unchanged.**
Details, reproductions and before/after results: `DIRECTION_B_TRIAGE.md`.

### Trusted inputs (R1, R2, R3)

* **The decision clock is `Case.trusted.evaluated_at`**, not the proposal's `proposed_at`. Mandate validity, evidence freshness and claim chronology are judged at it, and so is every evaluator's `as_of`. A proposal dated after it is a hard violation.
* **Scope comes from the lender's product** (`FinancingOffer.scope`). The mandate must permit that scope, and a proposal that declares another is a hard violation (`scope_offer_mismatch`).
* **The acting agent is `Case.trusted.acting_agent_id`** and must equal `mandate.agent_id`. The proposal carries no identity.
* The sandbox **asserts** these three inputs; it does not authenticate them. Production would take them from the platform clock, the authenticated session and the lender's signed product data.

### F7 and F6, decided

* `Mandate.max_amount` bounds **every individually committed payment**: the initial charge, each later instalment, and each fee that is not due upfront (`committed_payment_exceeds_mandate`, BLOCK). `max_total_repayment` is a separate limit and unchanged. Cumulative spending across agreements is not modelled.
* An advertised total in another currency on the **selected** offer is a REVIEW under its own basis (`review_incomparable_representation`). Nothing is converted and nobody is accused. On an offer that is not being bought it is only recorded.

### Direction and materiality (R4)

A figure someone showed the customer is compared with the authoritative one in a direction. Showing the customer **less** than they will pay is adverse; **more** is recorded and never escalated. Materiality is predeclared in `PolicyConfig` and untuned: `understatement_material_minor = 1` (any exact shortfall, unchanged for the structured advertised total), `hidden_obligation_material_minor = 500` (the existing 5.00 gap), and a hidden obligation of any size is material if it would carry the total over a mandate or customer limit. `max_document_age = 365 days` bounds how old a lender document may be and still be compared with current terms.

### Availability, eligibility, authority and time (R7, R8, R9)

* A statement or observation has a **subject**: product availability or customer eligibility. Each can test only its own kind. An eligibility decision must name the customer and come from the offer's own lender. A verified decline excludes an alternative from the comparison; a decline for the offer being bought holds the purchase (`review_selected_offer_not_open`); an eligibility statement nobody can verify is `eligibility_unresolved`, never an accusation.
* **Source authority:** the offer's own lender outranks an independent third party; the merchant's own channel has none. Disagreeing fresh observations resolve to the higher authority unless a lower-authority observation is newer than the winner's latest, in which case they stay in conflict; equal ranks that disagree stay in conflict; one source's later observation supersedes its own earlier one.
* A merchant statement, an observation or a document **dated after the decision clock is impossible**: it is ignored and recorded, never trusted. Fresh independent evidence that the selected offer is not on sale holds the purchase whether or not anyone claimed anything.

### The boundary between information and enforcement

Six levels are kept apart, and a failure at any level is recorded, never an accusation:

1. a model-generated **allegation** (`AIFinding`): recorded; verified only against structured data; enforces nothing of its own;
2. a model-extracted **candidate claim** (`ExtractedClaim`): untrusted; a claim the evaluator was not shown, or whose quote is not in the cited source, is rejected and preserved on the verdict (`rejected_claims`);
3. a verifiably **grounded fact** (`crosscheck.GroundedFact`): the source was in that evaluator's request, concerns the claimed offer, is dated no later than the clock, and contains the quote; its class (the merchant's own representation, the offer's own lender, another independent party) is decided from the case's data; amounts are read with quantity, currency and scope (an amount the passage makes conditional or optional is not mandatory), and text with an ambiguous reading is refused;
4. an **independently supported contradiction** (`crosscheck.CrossCheck`): the grounded figure compared with the lender's structured terms, with direction and no conversion. A merchant's document can never corroborate the merchant. A lender's own document that states more than its structured terms is a **conflict between two lender sources** to be resolved by a person, not proof that either is false;
5. a **deterministically material finding**, under the predeclared standard;
6. the **final decision**: an advisory can raise ALLOW to REVIEW and do nothing else; only the rule engine produces BLOCK.

### The one AI-to-REVIEW route

`CrossCheck.enforces`: an adverse discrepancy (`contradicted_by_terms` or `source_conflict`), material under the predeclared standard, about the offer being bought. It is the **same test for B, B3, Bc, C and the deterministic scanner A+**; no condition has a route the others lack, and it is recorded in `CaseRecord.ai_review_routes`. The deterministic scanner (Condition A+) extracts statements from the same passages with the same amount reader and goes through the same verification: it exists so that an AI condition can be measured against what plain parsing finds. A verified extraction takes no credit for a risk the rule engine already held (`escalated_by_ai` is false).

### Binding an experiment to a manifest (R10)

A run is bound, before any inference, to a recorded untampered manifest that describes the frozen core, prompts and policy now running, with the declared conditions defined as the manifest defines them; the frozen core must not change during the run, and the seal records all of it. A live run needs a manifest with status `frozen` (a clean committed tree and a named approver). This is the integrity of a **completed** run, not protection against someone who can rewrite the manifest directory, the log and the seal together.

### Hackathon amendment: a fair deterministic baseline and development runs

Writing the development challenge set (`benchmark/devchallenge.py`) showed two weaknesses in the deterministic scanner A+ and the shared amount reader. Neither was fixed in order to favour any condition; both made A+ weaker than a plain parser should be, and both are recorded by `audit/challenge_baseline_before_scanner_fix.json`, which keeps the results from before the change.

* The scanner skipped a lender document that discusses several plans. It now attributes each sentence of such a document to the one plan the sentence names (by title or id), and still does not guess when a sentence names none or several.
* The amount reader treated "Customers with a credit score below 600 pay a fee of £40" as an obligation on everyone. A charge that depends on a customer attribute ("customers with/who", "applicants", "borrowers", "subject to", "depending on", "where") is now conditional, so it is neither an extracted fee nor a hidden obligation. This applies equally to a model's extraction and to the scanner, because both are verified by the same reader.
* The credit-terms specialist is now shown each plan's title (and nothing else the merchant wrote), so that the one specialist shown a lender document can name the plan a multi-plan document is about. The prompts are unchanged.

Both changes were made after seeing the development challenge set, which is development data written alongside the rules. They are not evidence of anything about held-out performance.

**Run modes.** `deterministic` (A and A+ only; no model), `fake` (a fake provider through the real client; plumbing, not model output) and `dev-live` (the configured provider, on development cases, bound to a development-baseline manifest, with a mandatory `--max-calls` and an exact `--confirm-spend`). Only a `live` run bound to a frozen manifest can support a held-out claim, and none exists.

### Prompts v5: the extraction contract (no change to verification or enforcement)

The first live check of prompts v4 (16 calls, two development cases; kept unchanged in `results/smoke-v4-2026-10-25/`) showed that models reported a concern about an amount stated in a lender document as a **finding** and returned **no extracted claim** for it, so trusted code had nothing to verify. v5 repairs the interface, not the policy:

* A finding is a concern, which trusted code can only record. An extracted claim is a concrete statement a source makes, copied with its amount: the only thing trusted code can check against the structured terms. A concern resting on a stated amount needs the statement extracted as well, whether or not the model thinks it is a problem; a claim is never invented, and what is uncertain is said in a `claim_unverified` finding.
* A claim's `source_ref` is defined by where its item sits (inside `untrusted_merchant_content` or `authenticated_source_documents`, never under `trusted`), a ref is only the exact text of a `ref` field, and each specialist is told which of its own items can be a claim source (the authority analyst has none).
* `fee_amount` covers one stated mandatory fee or a stated sum, not only a stated total.

**Unchanged:** grounding, source classes, direction, predeclared materiality, the single AI-to-REVIEW route, mandate checks, deterministic BLOCK precedence. A concern without a grounded extraction still enforces nothing; an invalid reference still voids the whole reply (now with the refused claims preserved even when the reference was unknown). v4 results are v4 results and are not re-labelled.
