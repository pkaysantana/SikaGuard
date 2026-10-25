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

11. **Models are advisory.** A model's `risk_score` is recorded and never used to decide. No majority vote. A model's allegation is not "proven false": only the rule engine, or `crosscheck` over validated extractions, proves a claim false. Findings are adjudicated in trusted code (`evaluators/adjudicate.py`) and the enforcement rules are frozen in `DECISION_POLICY.md`: a model verdict, suspicion or confidence enforces nothing; only a finding that is verified and material, a model-extracted claim that code proves contradicts the lender's terms, an unmet evidence requirement in the rule engine, or the documented fail-closed option can hold a purchase. A finding contradicted by structured data is logged as a model error, an unverifiable one stays `unverified`, and notes and explanations are never read. Over-long display text may be truncated (original kept, repair recorded); nothing else in a reply is ever repaired.
12. **Trust domains.** Merchant-controlled text goes only in the user message, under `untrusted_merchant_content`, as JSON data. System prompts are static strings with no case data. Untrusted content never flows into another analyst's prompt, and no analyst reads another's output.
13. **B, B3, Bc and C must see the same information.** B, B3 and Bc get the generalist payload; Bc and B3 differ from B only in the prompt and the number of calls, and all four share one aggregation rule (`coordinator.AGGREGATION_POLICY`). Every specialist payload is a subset of the generalist's and together they cover it all. `tests/test_condition_c.py` enforces this; if you change a payload, keep that test green rather than loosening it. Do not give B or C the rule engine's findings.
14. **Comparisons across trust domains happen in trusted code** (`evaluators/crosscheck.py`), over validated structured fields, never by asking a model to compare.
15. **Failures are explicit.** Timeouts, refusals, provider errors, malformed output and ungrounded references each have an `EvaluatorStatus` and are recorded as abstentions. A missing response is never evidence about the merchant. Do not silently retry or drop them.
16. **Prompts are versioned.** Changing a prompt means changing its version string in `evaluators/prompts.py` and saying why. Never edit one to improve a result.
17. **Secrets.** Never print, log or save `SIKA_GUARD_LLM_API_KEY`, and never put provider response bodies in errors (they can echo credentials). Provider errors record the status code and error type only.
18. **Token counts and cost come only from the provider's response** and configured prices. Never estimate them.
19. **Mocks (`evaluators/mocks.py`, `--offline`) are for plumbing.** Their output is not evidence about model performance and reports must say so.

### Trusted verification (after the independent audit)

23. **A model's extracted claim is verified only against what that evaluator was shown** (`evaluators/grounding.py`): a merchant item in its own request, for the same offer, containing the quote. Otherwise the reply is an invalid response and an evaluator failure, never an accusation. Amounts are read with their quantity and consumed completely (`evaluators/amounts.py`): never partially parse, never take an instalment for a total.
24. **Independence comes from what a source is, not from a flag the data supplies** (`EvidenceSource.is_independent_of`). The sandbox assumes other source kinds are honest; it does not authenticate lenders or customers. Say so wherever it matters.
25. **Nothing unresolved is silently clean.** `CaseRecord.uncertainty_reasons` lists it; do not fix a missed detection by escalating every unverified claim.

### Direction B: trusted inputs and the information-to-enforcement boundary

26. **Authority inputs come from the trusted context, never from the proposal.** The decision clock is `Case.trusted.evaluated_at`, the acting agent is `Case.trusted.acting_agent_id`, and scope is the offer's own (`FinancingOffer.scope`). The proposal's `proposed_at` and `scope` decide nothing. The sandbox asserts these inputs; it does not authenticate them, and anything that says otherwise is wrong.
27. **`max_amount` bounds every individually committed payment**, not only the first. A statement or observation dated after the decision clock is ignored and recorded, never trusted. Evidence is judged by what it is about (product availability, or this customer's eligibility), who it comes from (source authority) and when, never by a flag.
28. **Six levels stay apart** (allegation, candidate claim, grounded fact, independently supported contradiction, material finding, final decision), and a failure at any level is recorded, never an accusation. There is exactly one AI-to-REVIEW route, `CrossCheck.enforces`, and it is the same for every condition including the deterministic scanner A+. No AI path creates a BLOCK. A merchant's document never corroborates the merchant, and the AI never supplies a source identity.
29. **Do not weaken Condition A to flatter an AI condition**, and do not delete a detection A already makes. A verified extraction takes no credit for a risk A already found. A deterministic parser that could extract the same fact is a baseline (A+) to measure against, not something to hide.
30. **A run is bound to a manifest** that describes the code that ran (`benchmark/binding.py`); a live run needs a manifest with status `frozen`. That is the integrity of a completed run, not protection against someone who can rewrite the manifest directory, the log and the seal together.

### Benchmark tooling

20. **The hidden-label boundary.** `benchmark/hidden.py`, `labels.py` and `fixtures/labels.py` may not be imported by `policy.py`, anything under `evaluators/`, `coordinator.py`, or `benchmark/{visible,decisions,seal}.py` (a test enforces it). Labels open only through `hidden.load_labels(labels, log, cases)`, which needs the persisted seal the inference run wrote over a complete, validated v2 log (`seal.verify_gate`). Nothing seals an arbitrary file and the scorer cannot seal; construction tools use `open_for_validation`. Every nested id in a case is neutral (`benchmark/ids.py`).
21. **Frozen means frozen.** `manifests/manifest-0001-initial.json` records the files, prompts and policy that define the evaluation. A test fails if the frozen core drifts. Never edit a manifest to make it pass: record a new one with a reason.
22. **Development data stays development data.** The 20 fixtures, the live smoke cases, the recorded replies and the benchmark pilot can never be reported as held-out results. Do not tune a prompt, rule or threshold on them or on any held-out data.

## Commands

```bash
pip install -e ".[dev]"
pytest
sika-guard-eval --cases                                  # Condition A
sika-guard-eval --condition C --offline review --limit 5 # pipeline check, no credentials
sika-guard-eval --condition C --limit 3                  # live; needs SIKA_GUARD_LLM_* variables
python -m sika_guard.benchmark pilot-run --out-dir /tmp/pilot   # benchmark tooling on the dev-only pilot; scripted evaluators, no model
sika-guard-eval --condition A+                           # the deterministic scanner baseline: no model, no network
python -m sika_guard.benchmark run --cases C --out-dir D --mode deterministic|fake|dev-live   # development runner (dev-live needs --max-calls and --confirm-spend; see DEMO.md)
python -m sika_guard.benchmark manifest --out manifests/m.json --note "..."            # a development baseline
python -m sika_guard.benchmark manifest --out manifests/m.json --freeze --approved-by NAME   # refused unless the tree is clean and committed
```

## Conventions

- Python 3.12+, Pydantic v2 models (frozen, `extra="forbid"`), standard library elsewhere.
- Prefer small readable functions over abstractions. Keep `policy.py` free of I/O; only `evaluators/llm.py` touches the network.
- When changing a rule, add or update a test and re-run the evaluator. If a fixture's result changes, say why in the commit.
- Use `tests/fakes.py` (an `httpx.MockTransport` provider) for any test involving the client; no test may need network access or credentials.
- Out of scope until the user says otherwise: adaptive routing, held-out evaluation, the 320-case benchmark, frontend, live payments, wallets, PSPs, cross-rail budgets, production auth, deployment.
