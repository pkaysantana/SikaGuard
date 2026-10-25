# Sika Guard: two-minute demonstration

**What it is.** A security layer for autonomous purchasing agents. A deterministic rule engine decides whether a financing purchase is authorised (BLOCK). AI evaluators may *read unstructured evidence* and point at facts, but **trusted code verifies each fact before it can matter**, and only a verified, material, adverse fact can raise an ALLOW to a human REVIEW. A model can never create a BLOCK, and a model's opinion alone never changes a decision.

Everything is synthetic and runs in a sandbox. No real payment, merchant, lender or customer is involved.

## The result in one paragraph

On a development case where a lender's *prose* states a mandatory £32 charge that the structured terms omit, **Condition C (three concurrent specialists) produced, with a real model, a verified REVIEW**: its credit-terms specialist extracted the charge as a structured claim, and trusted code grounded the quote, compared it with the lender's terms and judged it material. Condition B (one generalist) saw the same document, called it a "hidden obligation" in a finding, and extracted nothing, so nothing was verified and it stayed at ALLOW. **But the plain text scanner A+ reaches the same REVIEW deterministically, so there is no interception beyond A+.** One case, one run, one model: it shows the pathway is reachable by a real model, not that specialist parallelism wins.

## Run it (Windows, from the repository root; nothing below spends money unless it says so)

```bash
pip install -e ".[dev]"
pytest                                              # 1028 offline tests, no network, no credentials
```

### 1. The three demonstration cases, deterministic and instant

```bash
python -m sika_guard.benchmark challenge-write --out-dir demo --only legit_consistent_lender_document,hard_violation_later_instalment_above_cap,unstructured_unconditional_fee
python -m sika_guard.benchmark run --cases demo/smoke_cases.jsonl --expectations demo/smoke_expectations.json --out-dir demo/run --conditions A,A+ --mode deterministic
```

| Case | What it is | A (rule engine) | A+ (rule engine + plain text scanner) |
|---|---|---|---|
| `legit_consistent_lender_document` | A legitimate purchase; the lender's document agrees with the structured terms | ALLOW | ALLOW |
| `hard_violation_later_instalment_above_cap` | A later instalment (£250) exceeds the mandate's £200 per-payment ceiling | **BLOCK** | **BLOCK** |
| `unstructured_unconditional_fee` | The lender's *prose* states a mandatory charge; the structured terms list none | ALLOW (cannot read prose) | **REVIEW** (verified) |

### 2. Rehearse the AI conditions with no provider and no cost

```bash
python -m sika_guard.benchmark run --cases demo/smoke_cases.jsonl --out-dir demo/fake --mode fake --fake-mode extractor
```

This goes through the real client with a **scripted fake provider**. Its replies are not model output; it checks the plumbing only.

### 3. Read the saved live results (free)

```bash
python -m sika_guard.benchmark report --cases results/v5-validation/v5_cases.jsonl --decisions results/v5-validation/live/decisions.jsonl --expectations results/v5-validation/v5_expectations.json
python -m sika_guard.benchmark report --cases results/smoke-v4-2026-10-25/smoke_cases.jsonl --decisions results/smoke-v4-2026-10-25/live/decisions.jsonl --expectations results/smoke-v4-2026-10-25/smoke_expectations.json
```

The first is the latest check (prompts **v5**, 8 calls); the second is the earlier one (prompts **v4**, 16 calls). They are different prompts on different cases and are not comparable with each other. A report only prints: it never rewrites the saved run (add `--save-to DIR` to keep a copy), and it makes no model call.

### 4. Re-run a live check (spends money; hard-capped, never retried)

```bash
python -m sika_guard.benchmark run --cases results/v5-validation/v5_cases.jsonl --expectations results/v5-validation/v5_expectations.json --out-dir demo/live --conditions A,A+,B,C --mode dev-live --max-calls 8 --confirm-spend 8
```

It refuses to start unless `--confirm-spend` equals the exact planned number of calls, and the cap is enforced where each request leaves the process. It needs `SIKA_GUARD_LLM_BASE_URL`, `SIKA_GUARD_LLM_API_KEY` and `SIKA_GUARD_LLM_MODEL` (never printed or saved). A live call can stall for tens of seconds, so present the saved results.

## What we actually measured

### Live, prompts v5: `openai/gpt-4o-mini`, two fresh development cases, 8 calls (25 October 2026)

| | A | A+ | B | C |
|---|---|---|---|---|
| Legitimate case | ALLOW | ALLOW | ALLOW | ALLOW |
| Charge stated only in the lender's prose | ALLOW | **REVIEW** | ALLOW | **REVIEW** |
| Right on both (policy) | 1 of 2 | **2 of 2** | 1 of 2 | **2 of 2** |
| Verified interception beyond A | 0 | 1 (scanner, not a model) | 0 | **1 (a real model)** |
| Beyond **both** A and A+ | 0 | 0 | 0 | **0** |
| Charge extracted as a checkable claim | n/a | yes (scanner) | **no** (a `hidden_obligation` finding only) | **yes** (credit-terms specialist) |
| Replies that failed the strict contract | n/a | n/a | 1 of 2 | 1 of 6 |
| Prompt / completion tokens | none | none | 6,871 / 914 | 16,977 / 1,876 |
| Provider-reported cost (unit not stated) | 0 | 0 | 0.00158 | 0.00295 |
| Median call latency | n/a | n/a | 4.9 s | 3.7 s |

Total: 8 paid requests, 0 retries, 23,848 prompt and 2,790 completion tokens, 0.00453 provider-reported cost (unit unknown), two upstream providers behind the one model id.

**What the verified extraction looked like.** The credit-terms specialist returned a claim of kind `fee_amount`, amount 3200 GBP, citing the lender's document by its exact reference with the quotation "a credit-file assessment charge of £32 must be paid by every borrower on this plan". Trusted code confirmed the quote is in that document, that the document is the offer's own lender's, and that the lender's document states mandatory fees of £32.00 where the structured terms say £0.00; the discrepancy is material, so the case became REVIEW (`review_unresolved_evidence_conflict`: two lender sources disagree, and a person decides). The model's own verdicts did not decide this: the merchant-claim and authority specialists said BLOCK and were not enforced.

**What went wrong, plainly.**
* **B did not extract the charge**, even though prompts v5 tell it to. One sample, so this is not a conclusion about B, but the difference with C is not evidence that specialisation helps: the credit-terms role text explicitly points its specialist at the lender's documents, and with one case and one run chance cannot be excluded.
* **2 of 8 replies were invalid, both on the legitimate case.** B added a field the schema forbids inside its findings; the credit-terms specialist copied a document reference one character short (`doc:id-4cb93a6` for `doc:id-4cb93a6f`). The v4 confusion (citing lender terms or a section name as a source) did not recur. Because both document-reading replies on the legitimate case were invalid, **this run does not test how v5 behaves when it reads a benign document**.
* **Models still make false allegations**: C made 7 that structured data contradicts, mostly limit violations (for example that a 624 total exceeds a 750 limit, and, on the other case, that a 67.00 initial charge exceeds a 65.00 monthly limit); B made one. Trusted code logged every one as a model error. None was enforced.
* **No false positive in any final decision.** The legitimate case drew REVIEW verdicts from two C specialists; the policy refused to enforce them.

### Live, prompts v4: two earlier development cases, 16 calls

| | A | A+ | B | B3 | Bc | C |
|---|---|---|---|---|---|---|
| Legitimate case | ALLOW | ALLOW | ALLOW | ALLOW | ALLOW | ALLOW |
| Charge stated only in the lender's prose | ALLOW | **REVIEW** | ALLOW | ALLOW | ALLOW | ALLOW |
| Verified AI interceptions | 0 | 1 (scanner) | 0 | 0 | 0 | 0 |
| Replies that failed the strict contract | n/a | n/a | 0 of 2 | 0 of 6 | 0 of 2 | 2 of 6 |

Under v4 no AI condition extracted the charge: the models reported it as a finding. That is the failure prompts v5 were written to repair. The two live checks documented here used **24 paid requests** and 0.0130 provider-reported cost (unit unknown). Three earlier smoke rounds on prompts v1 to v3 used 32 more (their raw replies are kept in `tests/data/recorded_replies.json`).

### Deterministic: all 14 development challenge cases (real; no model involved)

```bash
python -m sika_guard.benchmark challenge-write --out-dir results/challenge-set
python -m sika_guard.benchmark run --cases results/challenge-set/challenge_cases.jsonl --expectations results/challenge-set/challenge_expectations.json --out-dir demo/challenge --conditions A,A+ --mode deterministic
```

| | A | A+ |
|---|---|---|
| Decisions that match the policy's expectation | 10 of 14 | **14 of 14** |
| Missed (should be held, was allowed) | 4 | 0 |
| False positives | 0 | 0 |

The cases include optional and conditional fees, a customer-dependent charge, one document covering two plans, scattered terms, true merchant statements and a merchant's own document that mentions a fee. One case (`scattered_label_and_amount_in_different_sentences`) is a charge a careful person would hold that neither A nor A+ holds, because the verifier cannot establish what the amount is; an AI condition that noticed it could report only a *candidate discovery*, never an interception.

**Honesty about the baseline.** After seeing the first results on this set we made A+ and the shared amount reader fairer (they had missed a multi-plan document and mistaken a customer-dependent fee for a universal one). The earlier results are kept in `audit/challenge_baseline_before_scanner_fix.json`. The set and A+ were developed together, so A+'s 14 of 14 is development data, not a held-out score. The AI conditions were **not** run on this set (112 paid calls; not authorised).

## Specialist concurrency, with the numbers

In C, three specialists (merchant claims, credit terms, authority) are started at the same moment, each sees only the slice of the evidence its job needs, and none can read another's answer. The AI stage takes about as long as the *slowest* call, not the sum:

| Run (live, v5) | Calls | Sum of call times | Wall time of the AI stage |
|---|---|---|---|
| C, charge in prose | 3 | 10.4 s | **4.2 s** |
| C, legitimate case | 3 | 10.1 s | **4.1 s** |

Concurrency is not a speed-up over one call (B's median per-case time was 4.9 s against C's 4.1 s here, and 4.0 s against 5.1 s in the v4 run), and it exposes a case to the slowest of its calls: in the v4 run one B3 call took 37.6 s. B3 used about 25% more prompt tokens than C in that run, so "same number of calls" is not "same compute".

## Limitations (please read)

* **Two small live checks (four cases between them), one model.** An interoperability and honesty check, not a benchmark. No statistical claim is made.
* **Incremental benefit over a plain parser is unproven, and so far zero.** In every live run A+ did everything the verified AI findings did. The only place a model might add value (reading prose a parser cannot place) is exactly where the verifier cannot certify its reading, so it can only be reported as a candidate.
* **The v5 success is one extraction on one case.** The prompts were written after seeing the v4 failure; the v5 cases are fresh, but the charge is the same kind of statement. Whether v5 generalises is unknown.
* **Models hallucinate and mis-copy.** False limit allegations, an extra schema field, and a mis-copied reference all occurred; trusted code rejected or logged each, and none changed a decision.
* **The evidence is synthetic and the cases are development data**, written alongside the rules and the prompts. Nothing here is a held-out result.
* **Trust is asserted, not authenticated.** The decision clock, the acting agent, a product's scope, the lender's terms and the origin of a "lender document" are taken from the case; a real system would need signed inputs.
* **Run integrity is procedural.** Runs are bound to a recorded manifest, but nobody has approved a frozen manifest, and the manifest, log and seal are files a determined operator could rewrite together.
* **Not built:** a held-out dataset, repeated-run statistics, a frontend, any payment integration.
