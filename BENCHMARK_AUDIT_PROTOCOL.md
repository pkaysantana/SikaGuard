# Human audit protocol for benchmark labels

The tooling can recompute some facts from a case (the evidence state, whether a hard violation is present, whether the selected
offer is economically inferior). It cannot tell whether a case is realistic, whether a benign case is genuinely legitimate, or
whether a label was the right call where reasonable people differ. People decide those. This protocol says how.

## Roles

* **Builder:** produced the case and its labels. Does not audit their own cases.
* **Auditors (two, independent):** each reviews the audit packet alone and records answers before seeing the other's.
* **Adjudicator:** a third person (not the builder or either auditor) resolves disagreements.

Auditors see the visible case **and** the proposed labels. They are people, not models.

## What is audited

`python -m sika_guard.benchmark audit-sample` selects, in order of priority:

1. every case with a validation audit item (a label differing from an independent recomputation, or from what the frozen policy does);
2. every case where the ideal action differs from the policy's action;
3. every case containing a claim the evaluator cannot verify from its inputs;
4. every case whose avoidable cost is near the policy's materiality gap;
5. every case with a causal label or a paired twin;
6. every benign case that contains a disputed claim;
7. a random 10% of the remainder (fixed seed, recorded), as a check on the unflagged majority.

At least all of the above plus the random slice; the packet records why each case was chosen.

## Questions, per case

Answer each from the visible case, then compare with the proposed label.

1. **Realism.** Is this a plausible purchase, merchant, wording and set of figures? Yes / strained / no. A strained benign case is not a hard negative; an unrealistic attack is not a test of anything.
2. **Genuinely benign?** (for cases labelled `ideal_action = ALLOW`) Would a careful human reviewer, knowing everything, let this through? If any doubt, say what it is.
3. **Genuinely an attack or a failure?** (for labelled attacks) Is something misleading, false or unauthorised actually present, and is it as labelled?
4. **Claim truth and verifiability.** From the world state, is each merchant claim true or false? From the visible inputs alone, could an evaluator establish it?
5. **Evidence.** Is each piece of evidence really independent of the merchant, compatible with the offer, fresh? Is any merchant-signed record being treated as proof?
6. **Economics.** Is the selected offer inferior under the customer's own stated preference, with the stated avoidable cost? Is the immaterial/material judgement fair?
7. **Causation.** Is `caused` / `not_caused` supported by a genuine twin that differs only in the misinformation? Otherwise it must be `unknown`.
8. **Ideal action and policy action.** Does the ideal action seem right? Does the expected policy action follow from `DECISION_POLICY.md` as written?
9. **Leakage by form.** Could someone tell attack from benign by wording, length or layout alone? Note the cue.
10. **Ambiguity.** Mark any case where reasonable reviewers could differ. Say which axis.

## Recording and agreement

* Each auditor records, per case and per axis (questions 1-8), `agree` / `disagree with the proposed label` / `cannot decide`, plus a one-line reason.
* Compute agreement **per axis**, between the two auditors and between each auditor and the proposed label. Report raw agreement and a chance-corrected statistic (for example Cohen's kappa) with the number of cases. Do not pool the axes into one figure.
* If the same axis fails repeatedly for one family or builder, review the whole family, not just the audited cases.

## Resolution

* Both auditors agree with the label: keep it.
* Auditors agree with each other but not the label: the label changes (by the builder or the adjudicator), and the change and reason are logged.
* Auditors disagree: the adjudicator decides, with reasons logged.
* **Cannot decide / genuinely ambiguous:** remove the case from the scored set, or keep it with an `audit_notes` flag and report results with and without ambiguous cases. Do not force a label to resolve a real ambiguity.
* Unrealistic benign case or leakage by form: rebuild the case.
* An audit item where the frozen policy and the label disagree is **never resolved by changing the policy** or its thresholds. It is resolved, if at all, by correcting a wrong label, and otherwise recorded as a policy gap.

## After the audit

Any label changed after the audit is re-validated and, if the change is substantive, re-audited. Once the data are frozen, a change to any case voids the
audit and freeze for that case. Keep the audit log with the data; it is evidence about label quality, and label quality bounds every conclusion the benchmark can support.
