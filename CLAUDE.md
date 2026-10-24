# Sika Guard - notes for Claude sessions

Sika Guard is a hackathon security-research project. It tests whether independent AI systems can prevent financially harmful decisions by autonomous purchasing agents. See `README.md` for the scenario and the vocabulary.

## Scope and boundaries

- Work only inside this repository.
- **Do not search, read, import from or modify** sibling projects (Sika Sentinel, SikaHub, or anything else), parent directories, the user's Downloads folder or other external folders. The research behind this project lives in external ChatGPT and Claude conversations that are not stored here. Do not try to find or reconstruct it; ask the user if you need a decision it would have settled.
- Do not add dependencies, network calls, model calls, databases or payment SDKs without a concrete need the user has agreed to.

## Invariants - do not weaken these

1. **A hard violation is always BLOCK.** Any `HARD_VIOLATION` finding means `Decision.BLOCK` and `authorised=False`.
2. **Nothing downstream can lower or replace a deterministic BLOCK.** Future LLM verdicts go through `policy.resolve_final_decision`, which lets an advisory verdict escalate ALLOW to REVIEW at most. Keep that true.
3. **Authorisation and decision quality are different questions.** An in-mandate transaction that costs more than another option is REVIEW at most, never "unauthorised" or BLOCK.
4. **Missing or stale evidence is not a contradiction.** Only fresh, compatible, independent evidence can confirm a claim false. Do not report absence of evidence as deception.
5. **Evidence must be independent and compatible.** Same offer, not from the merchant's own channel, a lender source must be the offer's lender, observed after the claim, not from the future, within the freshness window.
6. **Ground-truth labels stay separate.** `labels.py` and `fixtures/labels.py` must never be imported by `policy.py` or passed to a detector or an LLM. `Case` carries no labels, and case ids stay neutral (no hints about the attack family). `tests/test_fixtures.py` enforces this.
7. **Money is exact.** Integer minor units via `Money`; never `float`. Mixed currencies raise rather than convert.
8. **Objective, checkable facts use deterministic rules**: numeric limits, repayment arithmetic, identifier equality, timestamps. Reserve model judgement for what rules cannot decide, and never for these.
9. **Do not fabricate results.** Report what the evaluator actually outputs. Never imply a live model was tested, and do not tune fixtures to make the baseline look better.
10. Do not treat every malicious-looking sentence as a payment attack. Instruction-like text is a `SUSPICIOUS_BEHAVIOUR` note unless something else escalates the case.

## Commands

```bash
pip install -e ".[dev]"
pytest
sika-guard-eval --cases
```

## Conventions

- Python 3.12+, Pydantic v2 models (frozen, `extra="forbid"`), standard library elsewhere.
- Prefer small readable functions over abstractions. Keep `policy.py` free of I/O.
- When changing a rule, add or update a test and re-run the evaluator. If a fixture's result changes, say why in the commit.
- Out of scope until the user says otherwise: model calls, Telluvian, parallel LLM orchestration, adaptive routing, the 320-case benchmark, frontend, live payments, wallets, PSPs, cross-rail budgets, production auth, deployment.
