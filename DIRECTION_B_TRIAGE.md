# Direction B: Opus findings R1-R10, the information-to-enforcement boundary, and what remains

Status: **implemented on a development baseline. Not frozen.** The frozen core, the prompts (now v4) and the decision policy changed, so
anything measured before this work is not comparable with anything measured after it. A new freeze is needed before any held-out run.

## 0. How to read this

* **Source of the findings.** Opus's analysis reached me as ten one-line statements (R1-R10) and an instruction to reproduce each against
  the current revision. Its full text was not in my context, so every finding below was reproduced independently from its one-line
  description, offline, and where my construction differs from Opus's the difference is mine. Compare this against Opus's full risk
  register when you have it.
* **Evidence in the repository.** `tests/directionb_reproductions.py` produced `audit/matrix_before_directionb.json` on the unrepaired tree and
  `audit/matrix_after_directionb.json` on the repaired tree (the same code, run twice). `tests/directionb_scenarios.py` produced
  `audit/directionb_dev_scenarios.json`. The regression tests are `tests/test_direction_b_*.py`. The starting state is recorded, unedited, as
  `manifests/manifest-0005-pre-direction-b.json`.
* **Not tuned.** No prompt was edited to improve a result, no threshold was moved to make a condition look better, and no fixture was
  changed. The 20 development fixtures decide exactly as before under Condition A (18 of 20 agree with their labels; the same two
  disagreements, case-14 and case-16).
* **Nothing here is a result about a model.** Every AI output in these tests is scripted or comes from the fake provider.

## 1. Dispositions

| # | Finding | Reproduced at this revision | Disposition |
|---|---|---|---|
| R1 | Agent-controlled decision clock | yes | **Fixed.** `Case.trusted.evaluated_at` is the only clock; the proposal's timestamp decides nothing. |
| R2 | Agent-controlled transaction scope | yes | **Fixed.** Scope is derived from the lender's product (`FinancingOffer.scope`); a declared scope that differs is a hard violation. |
| R3 | Unenforced mandate agent | yes | **Fixed** against a trusted input (`Case.trusted.acting_agent_id`). The sandbox asserts that input; it does not authenticate it. |
| R4 | Comparison without direction or materiality | yes | **Fixed.** Direction (understates / overstates / equal) and a predeclared materiality standard in `PolicyConfig`. |
| R5 | Unicode/formatting grounding failures | yes | **Fixed.** Matching folds formatting, reads folded text, reports the folds, keeps the original, and refuses ambiguous digits. |
| R6 | Too few AI-to-enforcement routes for any incremental detection | yes (earlier memo) | **Addressed.** One new route: a verified extraction. See sections 3 and 4, and the honest limit in section 5. |
| R7 | Product availability confused with customer eligibility | yes | **Fixed.** Separate subjects; a verified decline is a lender decision about this customer. |
| R8 | Future-dated merchant claims | yes | **Fixed.** Ignored and recorded; and a rule that acts on the lender's own evidence even if no claim exists. |
| R9 | No source-authority or conflict hierarchy | yes | **Fixed** (authority, scope, time). |
| R10 | Seal not bound to an approved manifest | yes | **Fixed** for completed-run integrity. Not protection against a malicious operator. |
| F6 | Currency-incompatible headline on the selected offer | open | **Resolved as you directed:** REVIEW under its own basis; no conversion. |
| F7 | `max_amount` and later instalments | open | **Resolved by the mandate's own wording:** it bounds every individually committed payment. |

## 2. Findings in detail

### R1: the decision clock
* **Before.** `as_of` was `transaction.proposed_at` (policy.py, evaluate_case and check_offer_authorised), so the agent chose the instant at which the mandate was valid and at which evidence was fresh. A proposal back-dated 45 days into an expired mandate gave ALLOW; a proposal dated before the evidence made fresh contradicting evidence irrelevant (REVIEW to ALLOW).
* **After.** `TrustedContext.evaluated_at` is the decision clock for the mandate window, evidence freshness, claim chronology and every evaluator (`EvaluatorView.as_of`). A proposal dated after it is a hard violation (`proposal_from_the_future`).
* **Limit.** In the sandbox a fixture asserts `evaluated_at`. A real deployment takes it from the platform's clock.

### R2: scope
* **Before.** `check_scope(mandate, txn.scope)` tested the scope the proposal declared. A credit agreement declared as a card purchase passed under a card-only mandate.
* **After.** The offer carries the scope of the lender's product; the mandate must permit THAT scope, and a proposal that declares another is `scope_offer_mismatch`. Alternatives are judged by their own scope.
* **Limit.** The scope of a product is a lender-originated classification, trusted for the same reason the schedule is.

### R3: agent identity
* **Before.** `Mandate.agent_id` was never compared with anything (the transaction has no actor field).
* **After.** `TrustedContext.acting_agent_id` (the session that submitted the proposal) must equal `mandate.agent_id`, else `agent_mismatch` (BLOCK). The proposal still carries no identity, so it cannot vouch for itself.
* **Limit.** The sandbox cannot authenticate who the acting agent is; it only refuses to take that from the proposal.

### R4: direction and materiality
* **Before.** The cross-check treated any inequality as a contradiction, in both directions and at any size; the rule engine treated only a shortfall as one. A merchant advertising MORE than the lender's total (no harm) was held by the AI route but allowed by the rule engine.
* **After.** One comparison (`policy.compare_figures`) gives a direction and never converts currencies. A merchant representation that shows the customer LESS than they will pay is adverse; MORE is recorded (`advertised_total_overstated`, `discrepant_not_adverse`) and never escalated. Materiality is predeclared in `PolicyConfig` and untuned: `understatement_material_minor = 1` (any exact shortfall, which is what the rule engine already did), `hidden_obligation_material_minor = 500` (the existing 5.00 gap), and any hidden obligation that would carry the total over a mandate or customer limit is material at any size.

### R5: Unicode and formatting
* **Before.** Zero-width characters, full-width digits and curly quotation marks made a faithful quotation "not in the source", so the whole reply became an invalid response.
* **After.** Matching folds compatibility forms, removes invisible characters and unifies quotes and dashes (`grounding.canonical`). The original text and the quote are unchanged and the folds that were needed are in `CrossCheck.folds`. Lookalike letters from another script are NOT folded, so they do not ground. Text with bidirectional controls, or digits from another script or in superscript form, is flagged (`reading_hazards`) and no amount is taken from it.

### R7: availability versus eligibility
* **Before.** Evidence that a lender still sells a product "contradicted" a merchant's statement that the customer could not have it, producing a verified accusation.
* **After.** Assertions and evidence carry a subject. Each subject can test only its own kind. An eligibility decision must name the customer and come from the offer's own lender. Four states are distinct: a product observation, an unresolved eligibility statement (`eligibility_unresolved`, not an accusation), a verified decline (the alternative is excluded from comparison), and a decline for the offer being bought (REVIEW under `review_selected_offer_not_open`).

### R8: impossible timestamps
* **Before.** A claim dated after the decision clock turned fresh contradicting evidence "stale", and the case was ALLOWed.
* **After.** Such a claim (and evidence or a document dated after the clock) is ignored and recorded as a note (`impossible_timestamp`). Separately, fresh independent evidence that the SELECTED offer is not on sale now holds the purchase whether or not anyone claimed anything (`offer_reported_unavailable`); an alternative so reported is not a usable comparison. Stale, conflicting, merchant-channel and other-lender evidence do not.

### R9: authority, scope and time
* **Before.** Any two fresh independent observations that disagreed were "conflicted", whoever they were.
* **After.** `SourceAuthority`: the offer's own lender outranks an independent third party; the merchant's own channel has none. Disagreeing fresh observations resolve to the highest authority unless a lower-authority observation is NEWER than the winner's latest (the state may have changed), in which case the conflict stays open; two sources of equal rank that disagree stay in conflict; one source's later observation supersedes its own earlier one. The outranked observation is kept in the finding. A lender document that states an obligation the lender's structured terms omit is a conflict between two lender sources, not a proof that either is false.

### R10: binding a run to a manifest
* **Before.** A run log only quoted a manifest digest. A digest naming no manifest, and the digest of a manifest of different code, both sealed.
* **After.** Before any inference, `binding.bind` requires the digest to name a recorded, untampered manifest that describes the code now running (frozen core, prompts, policy). The declaration records the manifest's status, a hash of the frozen core as it was when the run started, and a hash of the manifest's definitions of the declared conditions. Sealing re-checks all of that and that the frozen core did not change while the run was in progress. The seal records them and the gate compares seal, declaration and manifest. A live run needs a manifest with status `frozen`, which is refused unless the Git tree is clean and a person is named as approver. A completed run can still be scored after the code legitimately moves on.
* **Limit (stated in the code).** This is the integrity of a completed run. Someone who can rewrite the manifest directory, the log and the seal consistently can forge all three; that needs signatures made outside this repository.

## 3. The information-to-enforcement boundary

| Level | What it is | Type and code | What can stop it | What it can become |
|---|---|---|---|---|
| 1 | A model-generated allegation | `AIFinding` | Adjudication (`adjudicate.py`): verified only against structured data, and only enforcing if the rule engine already holds it | Recorded; never an enforcement of its own |
| 2 | A model-extracted candidate claim | `ExtractedClaim` | Reply grounding (`contract.verdict_from_analysis`); rejected claims are preserved on the verdict | A grounded fact |
| 3 | A verifiably grounded fact | `crosscheck.GroundedFact` | The claimed source was in THIS evaluator's request, concerns the claimed offer, is dated no later than the clock, contains the quote; its class comes from trusted data; amounts are read with quantity, currency, scope (conditional/optional) and hazards | An independently supported discrepancy |
| 4 | An independently supported contradiction | `crosscheck.CrossCheck` (outcome, direction, stated and reference figures) | Compared with the lender's structured terms, with direction and no conversion; a lender document against the same lender's terms is a conflict | A material finding |
| 5 | A deterministically material finding | `CrossCheck.material`, predeclared in `PolicyConfig` | Not adverse, not material, or not about the selected offer | An enforcing extraction |
| 6 | The final decision | `Decision` | `resolve_final_decision`: an advisory can raise ALLOW to REVIEW and do nothing else | ALLOW, REVIEW (never BLOCK from the AI layer) |

A model verdict, confidence or risk score is recorded at level 1 and decides nothing.

## 4. The single AI-to-REVIEW route

`CrossCheck.enforces` is true when: the outcome is `contradicted_by_terms` (a merchant representation shows the customer less than the lender's terms) or `source_conflict` (a lender's own document states an obligation the structured terms omit); AND the discrepancy is material under the predeclared standard; AND the claim is about the offer being bought. `coordinator` raises ALLOW to REVIEW on exactly that (basis `review_verified_contradiction` or `review_unresolved_evidence_conflict`) and records the route in `CaseRecord.ai_review_routes`. It is the same test for B, B3, Bc, C and the deterministic scanner A+. It cannot produce a BLOCK.

## 5. What the development scenarios show, and what they do not

They show that the pathway works and stops where it should: a case where Condition A says ALLOW because a material fact (a one-off £45 arrangement charge) exists only in an unstructured lender document; a scripted extraction that trusted code verifies and that justifies a REVIEW; an unsupported allegation that cannot produce that REVIEW; incorrect, conditional, transposed, mislabelled, wrongly bound, other-lender, merchant-authored and ambiguous-script extractions that cannot; and a verified extraction that takes no credit for a risk A already found.

They do **not** show that AI adds anything a plain parser cannot. The deterministic scanner A+ finds the same fact in the same text and reaches REVIEW through the identical path. This is structural, not an accident of the example: the verifier reads the figure out of the quote with the same amount reader a scanner uses, so a model's extraction can only be certified if a scanner could have extracted it. What a model could add is selecting the right passage and attributing it to the right offer in text a scanner cannot segment; verifying that semantic labelling deterministically is the open design question (section 7). **The flagship scenario still has no incremental opportunity** (its deception is availability, which Condition A already catches whenever the cheaper alternative and the evidence are visible); the new channel concerns a different class of deception.

## 6. Trust assumptions that remain (sandbox versus production)

* The trusted context (`evaluated_at`, `acting_agent_id`), `FinancingOffer.scope`, the lender's structured terms, mandates, customer intent, the independence flag of non-merchant source kinds and the origin of an "authenticated" document are all **asserted by the fixture**. None is authenticated. Production would take them from the platform clock, the authenticated session, the lender's signed terms and signed documents.
* The manifest directory, the run log and the seal are files; see R10.
* Nothing authenticates the person named as approver of a frozen manifest.

## 7. Open items and risks

1. **Semantic verification.** A model's real advantage is semantic reading, which a deterministic verifier cannot certify. A "label-unverified" tier that could still raise a REVIEW would trade false positives for coverage; it was deliberately NOT built. Decide whether to build it, and how to price its false positives.
2. **New thresholds, untuned:** `understatement_material_minor = 1`, `hidden_obligation_material_minor = 500`, `max_document_age = 365 days`.
3. **New false-positive risks introduced:** (a) a currency-incompatible headline on the selected offer now holds the purchase; (b) a lender document stating more than its structured terms holds it as a conflict, even if the document is the stale or wrong one (bounded by the document window); (c) fresh independent evidence that the selected offer is unavailable holds it even with no merchant claim; (d) a later instalment above `max_amount` is now BLOCK where it was ALLOW; (e) an eligibility statement that cannot be verified is held for corroboration where a product observation used to accuse.
4. **Lexical limits** of the amount reader remain (a dash between a label and an amount leaves the quantity unknown, which is recorded uncertainty, not a verdict).
5. **Eligibility conditions** stated in prose (a document that says who may apply) are not extracted: there is no structured customer profile to compare them with.
6. **Availability stated in a document** is not extracted: there is no deterministic verifier for negation or tense in prose.
7. **Not built (paused):** scoring of repeated runs and Holm-corrected comparisons; the independent held-out dataset; the runner CLI. The scorer refuses repeated logs.
8. **Freeze.** No frozen manifest can be recorded until the tree is clean and committed; that is yours to do.

## 8. Verification

* **Test suite:** 973 passed, 0 failed, 0 expected failures. The strict F7 specification test is gone because F7 is decided; it is now an ordinary regression test.
* **Mutation checks:** 46 single-defence mutations of the Direction B repairs (clock, scope, agent, committed payments, subjects, authority and time, direction, materiality, unavailability, enforcement, document authority and independence, Unicode folds and hazards, grounding at the reply, payload sections, the coordinator's test, and every manifest-binding check). The first run left three uncovered (an eligibility decision need not be about this customer; a lender document BELOW the terms treated as a conflict; a document from the future trusted by the grounding stage). Tests were added for each and all 46 are now caught. A mutation run shows only that a defence is exercised by some test, not that the tests are complete.
* **The 20 development fixtures** decide exactly as before under Condition A (18 of 20 agree; the same two disagreements). The pilot's 12 cases and labels are unchanged in meaning.
* **Not run:** no model, no network, no paid call, no live run. Prompts v4 have had no live validation.
