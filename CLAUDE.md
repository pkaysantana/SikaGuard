# Sika Guard - notes for Claude sessions

Sika Guard is a hackathon security-research project. It tests whether independent AI systems can prevent financially harmful decisions by autonomous purchasing agents. See `README.md` for the scenario, the vocabulary and the three experimental conditions (A rules only, B plus one analyst, C plus three specialists).

## Scope and boundaries

- Work only inside this repository.
- **Do not search, read, import from or modify** sibling projects (Sika Sentinel, SikaHub, or anything else), parent directories, the user's Downloads folder or other external folders. The research behind this project lives in external ChatGPT and Claude conversations that are not stored here. Do not try to find or reconstruct it; ask the user if you need a decision it would have settled.
- Do not add dependencies, databases, payment SDKs or agent frameworks without a concrete need the user has agreed to. The only network dependency is `httpx`, used by the OpenAI-compatible client.
- Do not commit, push or deploy unless asked.

## Invariants - do not weaken these

1. **A hard violation is always BLOCK.** Any `HARD_VIOLATION` finding means `Decision.BLOCK` and `authorised=False`.
2. **Nothing downstream can lower or replace a deterministic BLOCK.** Model verdicts go through `policy.resolve_final_decision`, which lets an advisory verdict raise ALLOW to REVIEW and do nothing else. No model output creates a BLOCK.
3. **Authorisation and decision quality are different questions.** An in-mandate transaction that costs more than another option is REVIEW at most, never "unauthorised" or BLOCK.
4. **Missing or stale evidence is not a contradiction.** Only fresh, compatible, independent evidence can confirm a claim false. Do not report absence of evidence as deception.
5. **Evidence must be independent and compatible.** Same offer, not from the merchant's own channel, a lender source must be the offer's lender, observed after the claim, not from the future, within the freshness window.
6. **Ground-truth labels stay separate.** `labels.py` and `fixtures/labels.py` must never be imported by `policy.py` or anything under `evaluators/`, and never reach a prompt. `Case` and `EvaluatorView` carry no labels; case ids stay neutral. Labels are joined only in `experiment.score`, after inference.
7. **Money is exact.** Integer minor units via `Money`; never `float`. Mixed currencies raise rather than convert.
8. **Objective, checkable facts use deterministic rules**: numeric limits, repayment arithmetic, identifier equality, timestamps. Reserve model judgement for what rules cannot decide.
9. **Do not fabricate results.** Report what the evaluator actually outputs. Never imply a live model was tested when only mocks ran, and do not tune fixtures or prompts to make a condition look better.
10. Do not treat every malicious-looking sentence as a payment attack. Instruction-like text is a `SUSPICIOUS_BEHAVIOUR` note unless something else escalates the case.

### AI layer

11. **Models are advisory.** A model's `risk_score` is recorded and never used to decide. No majority vote. A model's allegation is not "proven false": only the rule engine, or `crosscheck` over validated extractions, proves a claim false.
12. **Trust domains.** Merchant-controlled text goes only in the user message, under `untrusted_merchant_content`, as JSON data. System prompts are static strings with no case data. Untrusted content never flows into another analyst's prompt, and no analyst reads another's output.
13. **B and C must see the same information.** Every specialist payload is a subset of the generalist's and together they cover it all. `tests/test_condition_c.py` enforces this; if you change a payload, keep that test green rather than loosening it. Do not give B or C the rule engine's findings.
14. **Comparisons across trust domains happen in trusted code** (`evaluators/crosscheck.py`), over validated structured fields, never by asking a model to compare.
15. **Failures are explicit.** Timeouts, refusals, provider errors, malformed output and ungrounded references each have an `EvaluatorStatus` and are recorded as abstentions. A missing response is never evidence about the merchant. Do not silently retry or drop them.
16. **Prompts are versioned.** Changing a prompt means changing its version string in `evaluators/prompts.py` and saying why. Never edit one to improve a result.
17. **Secrets.** Never print, log or save `SIKA_GUARD_LLM_API_KEY`, and never put provider response bodies in errors (they can echo credentials). Provider errors record the status code and error type only.
18. **Token counts and cost come only from the provider's response** and configured prices. Never estimate them.
19. **Mocks (`evaluators/mocks.py`, `--offline`) are for plumbing.** Their output is not evidence about model performance and reports must say so.

## Commands

```bash
pip install -e ".[dev]"
pytest
sika-guard-eval --cases                                  # Condition A
sika-guard-eval --condition C --offline review --limit 5 # pipeline check, no credentials
sika-guard-eval --condition C --limit 3                  # live; needs SIKA_GUARD_LLM_* variables
```

## Conventions

- Python 3.12+, Pydantic v2 models (frozen, `extra="forbid"`), standard library elsewhere.
- Prefer small readable functions over abstractions. Keep `policy.py` free of I/O; only `evaluators/llm.py` touches the network.
- When changing a rule, add or update a test and re-run the evaluator. If a fixture's result changes, say why in the commit.
- Use `tests/fakes.py` (an `httpx.MockTransport` provider) for any test involving the client; no test may need network access or credentials.
- Out of scope until the user says otherwise: the matched-compute baseline, adaptive routing, held-out evaluation, the 320-case benchmark, frontend, live payments, wallets, PSPs, cross-rail budgets, production auth, deployment.
