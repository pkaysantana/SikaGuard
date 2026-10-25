# Building the held-out benchmark: instructions for the dataset worker

For whoever constructs the held-out cases. You are deliberately kept apart from the people who wrote the
evaluator prompts and the decision rules. Read this guide, `BENCHMARK_SPEC.md` and `DECISION_POLICY.md`. Everything is
synthetic: no real merchants, customers, lenders or accounts.

## What you must not read or use

* The evaluator prompts (`src/sika_guard/evaluators/prompts.py`), the adjudication, cross-check and policy code, and `tests/data/recorded_replies.json`. Writing cases to suit what those do would defeat the purpose.
* The 20 development fixtures and the 12 pilot cases, except as **format examples**. Do not copy a merchant, an offer id, a description, a repayment schedule or an amount pattern from them. The validator measures this.
* Any model output on your cases. Do not run an evaluator on your data while building it.

## What you produce

Two separate files:

* `cases.jsonl`: one line per case, `{"schema": "sika-guard-benchmark-visible/1", "case": {...}}`, where `case` is exactly the existing `Case` model (customer intent, mandate, merchant, offers, proposed transaction, merchant assertions, evidence). Nothing else may appear: not a label, a note, a family, a hint.
* `labels.jsonl`: one line per case, `{"schema": "sika-guard-benchmark-labels/1", "labels": {...}}`, the `HiddenLabels` model in `src/sika_guard/benchmark/hidden.py`. Its docstrings define every field.

Case ids are `bm-` plus 10 hex characters, derived by hashing (`visible.opaque_id`), never counting, and the lines of `cases.jsonl` are
in shuffled order. An id must carry no information about family, label or split.

**Every nested identifier is part of what a model reads**, so it must be neutral too: transaction, mandate, customer, agent, merchant, offer, payee, lender, assertion, evidence and source ids must all be `id-<8 hex>`. Build the case with any working names, then run `benchmark/ids.neutralise_pair(case, labels, salt=cluster_id)`, which rewrites the case and re-points the labels consistently (and keeps a paired twin's offer ids shared). `validate` rejects any other form, and a separate check refuses a dataset whose identifiers predict the label even when they contain no forbidden word.

## Fields every case must now carry

* `trusted`: `evaluated_at` (the decision clock) and `acting_agent_id` (the agent that submitted the proposal). The proposal's own `proposed_at` and `scope` are only what the agent *says*; do not rely on them to make a case valid or invalid.
* Each offer's `scope` (default `credit_agreement`). A proposal that declares a different scope is a hard violation, which is a legitimate attack to build.
* An assertion or an evidence record's `subject` is `product_availability` or `customer_eligibility`. An eligibility decision names its `customer_id` and must come from the offer's own lender to count.
* `documents` (optional): unstructured passages with a known origin. A lender document binds to one offer (`offer_id`) or, if it names several, the quoted words must name the offer. The origin says who wrote it, not that it is correct. Do not write documents to suit what the verifier can read; write realistic passages, including ones a plain parser would not understand.
* `max_amount` bounds EVERY committed payment, so a schedule with a large later instalment is a hard violation.

## The target

240 cases in the families and counts in `BENCHMARK_SPEC.md` section 3 and `src/sika_guard/benchmark/quotas.py` (that file is the authority; this guide must not contradict it). Cases from one scenario template
are **not independent**, so give each scenario a `template_id` and `cluster_id`, build at least 4 clusters per family, and put
no more than 4 cases in a cluster. Do not pad a family with paraphrases of one scenario.

## How to build a case

1. **Write the world first.** Decide what is true in the simulation: which offers are really available, what the lender's schedule really is. That is `world_state`. The evaluator will see only some of it.
2. **Then decide what is visible.** Lender-originated terms (`schedule`) are authoritative. The merchant's title, description, advertised total and assertions are claims. Independent evidence (`Evidence`) exists only if you put it in the case, with a source that is genuinely independent of the merchant and, for a lender source, is the offer's own lender. Observation times decide freshness (one hour, and not before the claim).
3. **Classify every visible item** in `provenance`: `terms:` are authenticated issuer information; `listing:` are merchant marketing or merchant-structured; `assertion:` are merchant-structured or unsupported claims; `evidence:` from an independent source are an authenticated issuer (lender) or a regulatory reference (register, regulator, rate table). A merchant record may be marked `signed`, but a signature proves who wrote it and that it is unaltered. It never makes it true or independent, and the validator rejects a signed merchant record classed as independent.
4. **Label by construction, from the policy document.** Fill `expected_action_under_policy` and `expected_bases` by reasoning from `DECISION_POLICY.md`, not by running the code. Fill `ideal_action` as an informed reviewer would. Where they differ, say so in `audit_notes`: it is data, not an error.
5. **One availability assertion per case** (schema v1). Use `listing:` claims (the advertised total) for terms-related deception.
6. **Causation needs a twin.** If a case has a false claim and a poor choice, do not label the claim as the cause. Either build a paired twin (same customer, mandate and schedules, no false claim) and record the choice it makes in `counterfactual`, or leave `causal_selection` as `unknown`. At least 12 cases must have a twin.
7. **Vary the finances.** Different amounts, terms (not only 12 months), fees, currencies (GBP, EUR, USD), objectives (all three), and gaps around the materiality level: the avoidable-cost bands in the spec are multiples of the frozen gap so that you straddle it. Do not aim at any particular number.
8. **Write benign cases that are really benign.** Plausible, unremarkable, with a realistic reason for the choice. A hard negative that looks suspicious by construction is a bad hard negative. Build the benign twin of every attack family, with matching length, tone and layout, so that wording alone cannot separate them.
9. **Injection text** belongs in merchant free text only, in realistic phrasings, and only where the family calls for it. Benign instruction-like text (terms of use, "do not reply to this message") must exist too.
10. **Truth the evaluator cannot establish** is required (6 cases): the world has an answer, the visible case does not. Set `verifiable_from_inputs` false.

## Check your work

```bash
python -m sika_guard.benchmark validate --cases cases.jsonl --labels labels.jsonl --heldout
python -m sika_guard.benchmark quotas   --cases cases.jsonl --labels labels.jsonl
python -m sika_guard.benchmark leakage-scan --cases cases.jsonl
python -m sika_guard.benchmark audit-sample --cases cases.jsonl --labels labels.jsonl --out audit.json
```

* Errors must be zero. A **warning** about lexical artefacts means wording, length or layout separates attacks from benign cases: inspect the named features and rebuild, do not just reword the flagged tokens.
* **Audit items** are for people. Do not edit a label to make an audit item disappear unless the audit shows the label was wrong.
* The checks catch specific mistakes. A pass does not show the dataset is unbiased or uncontaminated.

## Hand-over

Deliver `cases.jsonl`, `labels.jsonl`, the audit packet, a short note of every judgement call, and the generator or template list (so independence from the development data can be reviewed). Do not run any evaluator.
Hash-freeze is done after the human audit, by someone else (`manifest`). If anything in the data changes afterwards, the audit and the freeze are void for the changed cases.
