# Sika Guard: HackNotts submission notes

**Repository:** `github.com/pkaysantana/SikaGuard` (branch `checkpoint/direction-b-offline`; see the push commands below, which wait for approval).
**One sentence:** a deterministic rule engine decides BLOCK, AI evaluators may read unstructured evidence, and trusted code verifies every fact they extract before it can raise anything, only ever to a human REVIEW. In our live check a real model's extraction was verified into a REVIEW; a plain text scanner reaches the same REVIEW, so we claim **no advantage over the scanner**.

## Two-minute script (about 270 words)

**0:00 The problem (20 s).** "An AI purchasing agent can be fully authorised and still be fooled, by text. The mandate says what it may spend. It cannot read the small print."

**0:20 The design (30 s).** "Sika Guard puts a deterministic rule engine in charge of BLOCK. AI evaluators may read the unstructured evidence, but they decide nothing. Every fact a model extracts is checked by trusted code: was the quote really in the document that evaluator was shown, is the document the lender's, does it conflict with the structured terms, is the difference material. Only then can a case go to a human, and a model can never block."

**0:50 The demo (30 s).** *Run the command in DEMO.md section 1 (instant, no model).* "Three cases. A legitimate purchase: allowed. A later instalment above the mandate's ceiling: blocked by the rules. And a mandatory charge that exists only in the lender's prose: the rules alone allow it; the deterministic scanner finds it, trusted code verifies it, and it goes to review."

**1:20 The live result (25 s).** *Open the saved v5 report (DEMO.md section 3).* "With a real model, eight calls: Condition C's credit-terms specialist pulled the charge out as a structured claim with an exact quote, trusted code verified it, and the case went to review. The single generalist saw the same document, called it a concern, extracted nothing, and stayed at allow. The models also made false claims, such as a 624 total exceeding a 750 limit; trusted code caught and ignored them, and no model verdict changed a decision."

**1:45 The honest finding (15 s).** "The plain scanner reaches the same review, so we found no interception beyond it, and one case does not show that specialists beat a generalist. What we built is the machinery to ask that question honestly: separated trust domains, verification before enforcement, and sealed, reproducible runs."

## 30-second fallback (no network, no model, nothing to wait for)

Run these two commands (both free and instant; the second only prints):

```bash
python -m sika_guard.benchmark run --cases results/demo/smoke_cases.jsonl --expectations results/demo/smoke_expectations.json --out-dir demo_out --conditions A,A+ --mode deterministic
python -m sika_guard.benchmark report --cases results/v5-validation/v5_cases.jsonl --decisions results/v5-validation/live/decisions.jsonl --expectations results/v5-validation/v5_expectations.json
```

Say: "Rules block a payment that breaks the mandate, allow a legitimate one, and cannot read prose. A verified extraction from the lender's own document sent the prose-only charge to review. In the saved live run, Condition C's specialist produced that verified claim; the generalist did not; and the plain scanner matches it, so no claim of superiority. All of it is saved, sealed and reproducible."

## Submission checklist

- [ ] Approve and run the push commands below.
- [ ] Open the GitHub page: `README.md` renders, `DEMO.md` is linked from it, `results/` is visible.
- [ ] From a fresh clone: `pip install -e ".[dev]"` then `pytest` reports 1028 passed (needs Python 3.12+).
- [ ] Run the 30-second fallback once on the machine you will present from.
- [ ] Have `DEMO.md` section 3 (the saved v5 report) ready in a terminal: it needs no network.
- [ ] Decide whether to add a licence file (see the optional list).
- [ ] Do not present: any claim that specialist parallelism wins, any held-out score, or any statement that the system is production-ready.

## Push commands (do not run until approved)

The remote `origin` already points at `https://github.com/pkaysantana/SikaGuard.git`, whose `main` is at the commit this branch started from, so the branch is a plain fast-forward.

```bash
# Option A (recommended): push the branch and open it on GitHub
git push -u origin checkpoint/direction-b-offline

# Option B: also make main contain the work (a fast-forward; main is an ancestor of the branch)
git checkout main
git merge --ff-only checkpoint/direction-b-offline
git push origin main
```

## Verify it from a clean clone (Windows)

```bash
git clone https://github.com/pkaysantana/SikaGuard.git sikaguard-check
cd sikaguard-check
git checkout checkpoint/direction-b-offline
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                   # 1028 tests; no network, no credentials
python -m sika_guard.benchmark run --cases results/challenge-set/challenge_cases.jsonl --out-dir demo_out --conditions A,A+ --mode deterministic
```

## What is in the repository

| Path | What |
|---|---|
| `DEMO.md` | commands, measured results, concurrency, limitations |
| `DECISION_POLICY.md`, `DIRECTION_B_TRIAGE.md`, `AUDIT_TRIAGE.md` | the policy and why it is what it is |
| `results/` | saved, sealed runs: `smoke-v4-2026-10-25/` (16 calls, prompts v4), `v5-validation/` (8 calls, prompts v5), `challenge-set/`, `demo/` |
| `manifests/` | 0001-0008, never edited; none is an approved freeze |
| `audit/` | before/after matrices and the pre-fairness-fix baseline |

## Known issues to disclose, not hide

* The two live checks documented here used 24 paid requests (earlier smoke rounds on prompts v1 to v3 used 32 more); no held-out data; no approved freeze.
* The 14-case challenge set and the scanner A+ were developed together; A+'s 14 of 14 is development data.
* The full 112-call challenge run was not made (not authorised).
* Trust inputs (clock, acting agent, scope, lender terms, document origin) are asserted by the fixtures, not authenticated.
* The saved results record the provider endpoint `https://api.telluvian.ai/v1` and the model id `openai/gpt-4o-mini`; no key, header or account identifier is recorded.

## Blockers and options

**Actual blockers:** none technical. The only gate is your approval to push.

**Optional, not blocking:**
* No `LICENSE` file: a public repository without one is "all rights reserved" by default. Add one if you want others to reuse it.
* Whether to publish the raw model replies in `results/*/live/decisions.jsonl` (synthetic cases only; they are the evidence for the results).
* The provider name in the saved logs (above) you may prefer to keep private.
* Research, not submission: a held-out dataset, repeated runs, running the AI conditions on the full challenge set, and a frontend.
