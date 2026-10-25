# Sika Guard: HackNotts submission notes

**Repository:** `C:\Users\Don\SikaHub\SikaGuard\SikaGuard`, branch `checkpoint/direction-b-offline`. Nothing has been pushed or deployed.

## What to say in two minutes

1. *The problem.* An autonomous purchasing agent can be talked into a harmful financing decision by text it reads. Rules catch what is structured; they cannot read prose.
2. *The design.* Deterministic rules decide BLOCK. AI evaluators may read unstructured evidence, but every fact they extract is verified by trusted code (grounded in what that evaluator was shown, compared with the lender's terms, checked for direction and materiality) before it can raise anything, and only to REVIEW. A model's opinion alone never changes a decision, and a model can never BLOCK.
3. *The demo.* Run section 1 of `DEMO.md`: a legitimate case (ALLOW), a hard violation (BLOCK), and a charge that exists only in the lender's prose (A allows it; the scanner A+ and, in our live check, Condition C's credit-terms specialist hold it for review after trusted verification).
4. *The honest finding.* A plain text scanner reaches the same REVIEW, so we found no interception beyond it; one real-model success on one case does not show that specialist parallelism wins. Say so.

## Verify it from a clean clone

```bash
git clone <path-to-repo> clone && cd clone && git checkout checkpoint/direction-b-offline
pip install -e ".[dev]" && pytest                    # 1026 tests; no network, no credentials
python -m sika_guard.benchmark run --cases results/challenge-set/challenge_cases.jsonl --out-dir /tmp/c --conditions A,A+ --mode deterministic
```

## What is in the repository

| Path | What |
|---|---|
| `DEMO.md` | commands, results, concurrency, limitations |
| `DECISION_POLICY.md`, `DIRECTION_B_TRIAGE.md`, `AUDIT_TRIAGE.md` | the policy and why it is what it is |
| `results/` | saved, sealed runs: `smoke-v4-2026-10-25/` (16 calls), `v5-validation/` (8 calls), `challenge-set/`, `demo/` |
| `manifests/` | 0001-0008, never edited; none is an approved freeze |
| `audit/` | before/after matrices and the pre-fairness-fix baseline |

## Known issues to disclose, not hide

* 24 paid calls in total across two live checks; no held-out data; no approved freeze.
* The 14-case challenge set and the scanner A+ were developed together.
* The full 112-call challenge run was not made (not authorised).
* Trust inputs are asserted by the fixtures, not authenticated.
