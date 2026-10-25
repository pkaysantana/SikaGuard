"""Validation of case/label pairs against independent recomputation, provenance rules, pairs and quotas. Offline."""

import json
from datetime import timedelta

import pytest

from sika_guard.benchmark import independent
from sika_guard.benchmark.audit import audit_packet
from sika_guard.benchmark.hidden import BenchFamily, EvidenceState, HiddenLabels, ProvenanceEntry, SourceClass, Split
from sika_guard.benchmark.pilot import build_pilot
from sika_guard.benchmark.quotas import FAMILY_TARGETS, TARGET_TOTAL, band_of, check_quotas
from sika_guard.benchmark.validate import validate_dataset, validate_record
from sika_guard.fixtures.cases import build_cases
from sika_guard.models import Case, Decision
from sika_guard.policy import PolicyConfig


def pilot():
    cases, labels = build_pilot()
    return {c.case_id: c for c in cases}, {x.case_id: x for x in labels}


def pick(labels, family):
    return next(x for x in labels.values() if x.attack_family.value == family)


def relabel(x: HiddenLabels, **changes) -> HiddenLabels:
    return HiddenLabels.model_validate({**x.model_dump(), **changes})


def aref(case):
    """The reference to a case's availability assertion. Ids are neutral hashes now, so tests read them from the case."""
    return f"assertion:{case.assertions[0].assertion_id}"


def eref(case, n=0):
    return f"evidence:{case.evidence[n].evidence_id}"


def chose(case):
    return case.transaction.offer_id


def errors_for(case, label, **kw):
    return validate_record(case, label, **kw).errors


# --- the pilot ------------------------------------------------------------------------------------------------


def test_the_pilot_has_twelve_development_only_cases_and_validates():
    cases, labels = pilot()
    assert len(cases) == 12 and {x.split for x in labels.values()} == {Split.PILOT}
    report = validate_dataset(list(cases.values()), labels, development=build_cases())
    assert report.ok, report.all_errors()


def test_the_pilot_exercises_the_features_it_exists_to_exercise():
    _, labels = pilot()
    states = {x.evidence_state for x in labels.values()}
    assert {EvidenceState.NONE, EvidenceState.FRESH_CONTRADICTS, EvidenceState.FRESH_CONFLICT, EvidenceState.MERCHANT_CHANNEL_ONLY, EvidenceState.NOT_APPLICABLE} <= states
    assert any(x.counterfactual for x in labels.values())
    assert any(x.selected_inferior and x.ideal_action is Decision.ALLOW for x in labels.values())  # inferior but immaterial
    assert any(x.hard_violation_present for x in labels.values())
    assert any(x.ideal_action is not x.expected_action_under_policy for x in labels.values())  # a recorded policy gap
    assert any(any(not c.verifiable_from_inputs for c in x.claims) for x in labels.values())  # unverifiable by the evaluator
    assert len({c.mandate.max_amount.currency for c in pilot()[0].values()}) == 3


def test_labels_written_by_construction_agree_with_the_frozen_policy_without_being_derived_from_it():
    cases, labels = pilot()
    report = validate_dataset(list(cases.values()), labels)
    mismatches = [a for a in report.audit_items if "under the frozen policy" in a or "expects bases" in a]
    assert mismatches == []  # only the deliberate ideal-versus-policy gap is flagged, below
    flagged = [a for a in report.audit_items if "ideal action" in a]
    assert len(flagged) == 1 and "REVIEW" in flagged[0]


# --- independent recomputation catches wrong labels ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "family, change, fragment",
    [
        ("bm_false_availability_verifiable", {"evidence_state": EvidenceState.NONE, "contradictory_evidence_available": False}, "evidence_state"),
        ("bm_false_availability_verifiable", {"avoidable_cost_minor": 1}, "avoidable_cost_minor"),
        ("bm_false_availability_verifiable", {"selected_inferior": False, "avoidable_cost_minor": 0}, "selected_inferior"),
        ("bm_false_availability_verifiable", {"claim_material_to_selection": False}, "claim_material_to_selection"),
        ("bm_recipient_substitution", {"transaction_authorised": True}, "transaction_authorised"),
        ("bm_benign_fits_preference", {"transaction_authorised": False, "hard_violation_present": True,
                                       "ideal_action": Decision.BLOCK, "expected_action_under_policy": Decision.BLOCK,
                                       "expected_bases": ("block_hard_violation",)}, "hard_violation_present"),
    ],
)
def test_a_wrong_label_is_caught_by_recomputation_from_the_visible_case(family, change, fragment):
    cases, labels = pilot()
    label = pick(labels, family)
    errs = errors_for(cases[label.case_id], relabel(label, **change))
    assert any(fragment in e for e in errs), errs


def test_a_claim_label_that_disagrees_with_the_world_state_is_caught():
    cases, labels = pilot()
    label = pick(labels, "bm_false_availability_verifiable")
    wrong = relabel(label, claims=[{"claim_ref": aref(cases[label.case_id]), "truth": True, "verifiable_from_inputs": True}], causal_selection="not_applicable", counterfactual=None)
    assert any("disagrees with the world state" in e for e in errors_for(cases[label.case_id], wrong))


def test_verifiability_must_follow_the_evidence():
    cases, labels = pilot()
    label = pick(labels, "bm_false_availability_conflicting")
    wrong = relabel(label, claims=[{"claim_ref": aref(cases[label.case_id]), "truth": False, "verifiable_from_inputs": True}])
    assert any("verifiable_from_inputs" in e for e in errors_for(cases[label.case_id], wrong))


def test_a_label_that_misstates_what_the_frozen_policy_does_is_flagged_for_audit_not_corrected():
    cases, labels = pilot()
    label = pick(labels, "bm_inferior_plain")
    wrong = relabel(label, ideal_action=Decision.ALLOW, expected_action_under_policy=Decision.ALLOW, expected_bases=("allow",))
    rep = validate_record(cases[label.case_id], wrong)
    assert rep.errors == [] and any("under the frozen policy" in a for a in rep.audit)


def test_the_policy_check_can_be_skipped_for_label_construction_without_calling_the_policy():
    cases, labels = pilot()
    label = pick(labels, "bm_inferior_plain")
    wrong = relabel(label, ideal_action=Decision.ALLOW, expected_action_under_policy=Decision.ALLOW, expected_bases=("allow",))
    assert not any("frozen policy" in a for a in validate_record(cases[label.case_id], wrong, check_policy=False).audit)


# --- provenance -------------------------------------------------------------------------------------------------------------


def test_every_visible_item_needs_a_provenance_entry_and_the_right_class():
    cases, labels = pilot()
    label = pick(labels, "bm_false_availability_verifiable")
    case = cases[label.case_id]
    ref = aref(case)
    dropped = relabel(label, provenance=[p.model_dump() for p in label.provenance if p.ref != ref])
    assert any(f"no provenance for {ref}" in e for e in errors_for(case, dropped))
    swapped = relabel(label, provenance=[{**p.model_dump(), "source_class": SourceClass.AUTHENTICATED_ISSUER} if p.ref == ref else p.model_dump() for p in label.provenance])
    assert any(ref in e and "expected" in e for e in errors_for(case, swapped))
    extra = relabel(label, provenance=[*(p.model_dump() for p in label.provenance), ProvenanceEntry(ref="evidence:id-ffffffff", source_class=SourceClass.AUTHENTICATED_ISSUER).model_dump()])
    assert any("not in the visible case" in e for e in errors_for(case, extra))


def test_a_signed_merchant_record_stays_a_merchant_source_and_is_not_independent_evidence():
    cases, labels = pilot()
    label = pick(labels, "bm_false_availability_unverifiable")
    case = cases[label.case_id]
    evidence = next(e for e in case.evidence)
    assert not evidence.source.independent_of_merchant
    ev = eref(case)
    assert {p.ref: p for p in label.provenance}[ev].signed and {p.ref: p for p in label.provenance}[ev].source_class is SourceClass.MERCHANT_STRUCTURED
    promoted = relabel(label, provenance=[{**p.model_dump(), "source_class": SourceClass.AUTHENTICATED_ISSUER} if p.ref == ev else p.model_dump() for p in label.provenance])
    assert any(ev in e for e in errors_for(case, promoted))
    assert independent.evidence_state(case) is EvidenceState.MERCHANT_CHANNEL_ONLY


def test_independent_evidence_must_be_classed_as_an_issuer_or_a_reference():
    cases, labels = pilot()
    label = pick(labels, "bm_false_availability_conflicting")
    case = cases[label.case_id]
    classes = {p.ref: p.source_class for p in label.provenance}
    kinds = {eref(case, i): case.evidence[i].source.kind.value for i in range(len(case.evidence))}
    lender = next(r for r, k in kinds.items() if k == "lender")
    register = next(r for r, k in kinds.items() if k == "aggregator")
    assert classes[lender] is SourceClass.AUTHENTICATED_ISSUER and classes[register] is SourceClass.REGULATORY_REFERENCE


# --- evidence states, recomputed on constructed variants -------------------------------------------------------------------------


def variant(case: Case, **changes) -> Case:
    return case.model_copy(update=changes)


def test_the_evidence_state_responds_to_freshness_provenance_and_the_lenders_identity():
    cases, labels = pilot()
    arc = cases[pick(labels, "bm_false_availability_verifiable").case_id]
    assert independent.evidence_state(arc) is EvidenceState.FRESH_CONTRADICTS
    e = arc.evidence[0]
    stale = e.model_copy(update={"observed_at": arc.assertions[0].made_at - timedelta(hours=1)})
    assert independent.evidence_state(variant(arc, evidence=(stale,))) is EvidenceState.STALE_ONLY
    old = e.model_copy(update={"observed_at": arc.transaction.proposed_at - timedelta(hours=3)})
    assert independent.evidence_state(variant(arc, evidence=(old,))) is EvidenceState.STALE_ONLY
    wrong_lender = e.model_copy(update={"source": e.source.model_copy(update={"source_id": "LND-OTHER"})})
    assert independent.evidence_state(variant(arc, evidence=(wrong_lender,))) is EvidenceState.NONE  # another lender cannot speak for this offer
    agrees = e.model_copy(update={"offer_available": False})
    assert independent.evidence_state(variant(arc, evidence=(agrees,))) is EvidenceState.FRESH_AGREES
    assert independent.evidence_state(variant(arc, assertions=())) is EvidenceState.NOT_APPLICABLE


def test_schema_v1_allows_one_availability_assertion_per_case():
    cases, labels = pilot()
    case = cases[pick(labels, "bm_false_availability_verifiable").case_id]
    second = case.assertions[0].model_copy(update={"assertion_id": "CL-2"})
    with pytest.raises(ValueError, match="at most one"):
        independent.evidence_state(variant(case, assertions=(case.assertions[0], second)))


def test_the_benchmark_freshness_window_matches_the_frozen_policys():
    assert independent.FRESHNESS_WINDOW == PolicyConfig().max_evidence_age


# --- paired constructions ---------------------------------------------------------------------------------------------------------


def test_a_causal_label_stands_only_on_a_genuine_paired_twin():
    cases, labels = pilot()
    case_list = list(cases.values())
    caused = pick(labels, "bm_false_availability_verifiable")
    assert validate_dataset(case_list, labels).ok

    missing = {k: v for k, v in labels.items() if k != caused.counterfactual.paired_case_id}
    assert any("not in the set" in e for e in validate_dataset([c for c in case_list if c.case_id in missing], missing).all_errors())

    wrong_choice = relabel(caused, counterfactual={"paired_case_id": caused.counterfactual.paired_case_id, "choice_without_misinformation": chose(cases[caused.case_id])})
    errs = validate_dataset(case_list, {**labels, caused.case_id: wrong_choice}).all_errors()
    assert any("the twin selects" in e for e in errs)

    liar_twin = {**labels, caused.counterfactual.paired_case_id: relabel(labels[caused.counterfactual.paired_case_id], claims=[{"claim_ref": aref(cases[caused.case_id]), "truth": False, "verifiable_from_inputs": False}],
                                                                       causal_selection="unknown")}
    assert any("itself contains a false claim" in e for e in validate_dataset(case_list, liar_twin).all_errors())


def _twin_selecting(cases, twin_id, offer_id):
    twin = cases[twin_id]
    offer = next(o for o in twin.offers if o.offer_id == offer_id)
    txn = twin.transaction.model_copy(update={"offer_id": offer_id, "amount": offer.schedule.initial_charge()})
    return {**cases, twin_id: twin.model_copy(update={"transaction": txn})}


def test_caused_is_refused_when_the_twin_makes_the_same_choice_so_the_claim_changed_nothing():
    cases, labels = pilot()
    caused = pick(labels, "bm_false_availability_verifiable")
    twin_id = caused.counterfactual.paired_case_id
    pick_ = chose(cases[caused.case_id])
    same = _twin_selecting(cases, twin_id, pick_)  # the twin now also picks the costlier plan the false claim "caused"
    lab = relabel(caused, counterfactual={"paired_case_id": twin_id, "choice_without_misinformation": pick_})
    errs = validate_dataset(list(same.values()), {**labels, caused.case_id: lab}, check_policy=False).all_errors()
    assert any("labelled caused, but the twin makes the same choice" in e for e in errs)


def test_not_caused_is_refused_when_the_twin_makes_a_different_choice():
    cases, labels = pilot()
    caused = pick(labels, "bm_false_availability_verifiable")
    lab = relabel(caused, causal_selection="not_caused")  # the twin picks AR-3 while this case picks BY-8: the claim did change the choice
    errs = validate_dataset(list(cases.values()), {**labels, caused.case_id: lab}, check_policy=False).all_errors()
    assert any("labelled not_caused, but the twin makes a different choice" in e for e in errs)


def test_a_genuine_not_caused_pair_is_accepted():
    cases, labels = pilot()
    caused = pick(labels, "bm_false_availability_verifiable")
    twin_id = caused.counterfactual.paired_case_id
    pick_ = chose(cases[caused.case_id])
    same = _twin_selecting(cases, twin_id, pick_)
    lab = relabel(caused, causal_selection="not_caused", counterfactual={"paired_case_id": twin_id, "choice_without_misinformation": pick_})
    errs = validate_dataset(list(same.values()), {**labels, caused.case_id: lab}, check_policy=False).all_errors()
    assert not [e for e in errs if "twin" in e or "paired" in e or "labelled" in e]  # the pair is coherent (the twin's own economics label would need updating, and is reported)
    assert any("selected_inferior" in e for e in errs)


def test_a_twin_that_differs_in_the_world_is_not_a_paired_construction():
    cases, labels = pilot()
    caused = pick(labels, "bm_false_availability_verifiable")
    twin_id = caused.counterfactual.paired_case_id
    twin = cases[twin_id]
    cheaper = twin.offers[0].model_copy(update={"schedule": twin.offers[1].schedule})
    altered = {**cases, twin_id: twin.model_copy(update={"offers": (cheaper, twin.offers[1])})}
    errs = validate_dataset(list(altered.values()), labels, check_policy=False).all_errors()
    assert any("not a paired construction" in e for e in errs)


# --- dataset level --------------------------------------------------------------------------------------------------------------------


def test_duplicate_ids_and_mismatched_files_are_errors():
    cases, labels = pilot()
    case_list = list(cases.values())
    assert any("duplicate" in e for e in validate_dataset([*case_list, case_list[0]], labels).all_errors())
    dropped = dict(list(labels.items())[1:])
    assert any("disagree" in e for e in validate_dataset(case_list, dropped).all_errors())


def test_a_heldout_file_may_not_contain_pilot_cases():
    cases, labels = pilot()
    rep = validate_dataset(list(cases.values()), labels, development=build_cases(), heldout=True)
    assert any("may contain only held-out cases" in e for e in rep.all_errors())


def test_a_cluster_cannot_straddle_the_pilot_and_the_heldout_split():
    cases, labels = pilot()
    one = next(iter(labels.values()))
    other = next(x for x in labels.values() if x.case_id != one.case_id)
    mixed = {**labels, other.case_id: relabel(other, cluster_id=one.cluster_id, split=Split.HELDOUT)}
    assert any("spans several splits" in e for e in validate_dataset(list(cases.values()), mixed, check_policy=False).all_errors())


# --- quotas -----------------------------------------------------------------------------------------------------------------------------


def test_the_intended_composition_is_about_240_cases_with_every_family_present():
    assert TARGET_TOTAL == 240 and sum(FAMILY_TARGETS.values()) == 240
    assert set(FAMILY_TARGETS) == set(BenchFamily)  # every family has a stated target, none is forgotten


def test_a_set_that_is_just_the_pilot_fails_the_quotas_loudly():
    cases, labels = pilot()
    short = check_quotas(list(cases.values()), labels)
    assert any("12 cases, target 240" in s for s in short) and any("non-held-out" in s for s in short)
    assert len(short) > 10


def test_materiality_bands_are_relative_to_the_frozen_gap_so_they_straddle_it():
    gap = PolicyConfig().material_gap_minor
    assert band_of(gap // 4) == "below half the gap" and band_of(gap) == "around the gap" and band_of(gap * 5) == "2x to 10x" and band_of(gap * 20) == "above 10x"


# --- the human audit sample -----------------------------------------------------------------------------------------------------------------


def test_the_audit_packet_prioritises_the_contestable_cases_and_is_reproducible():
    cases, labels = pilot()
    case_list = list(cases.values())
    report = validate_dataset(case_list, labels)
    packet = audit_packet(case_list, labels, report, fraction=0.0)
    ids = {p["case_id"] for p in packet}
    contested = pick(labels, "bm_benign_truthful_unverifiable")
    assert any(x.ideal_action is not x.expected_action_under_policy and x.case_id in ids for x in labels.values())
    assert pick(labels, "bm_false_availability_verifiable").case_id in ids and pick(labels, "bm_false_availability_conflicting").case_id in ids
    assert contested.case_id in ids
    again = audit_packet(case_list, labels, report, fraction=0.5, seed=3)
    assert [p["case_id"] for p in again] == [p["case_id"] for p in audit_packet(case_list, labels, report, fraction=0.5, seed=3)]
    assert json.loads(json.dumps(packet))  # serialisable for a human reviewer
