"""Leakage-prevention checks. These test the detectors and the plumbing. They do not prove a benchmark is uncontaminated."""

import asyncio
import hashlib
import json
import re

import pytest

from sika_guard import evaluate
from sika_guard.benchmark import leakage
from sika_guard.benchmark.hidden import BenchFamily, HiddenLabels, label_vocabulary
from sika_guard.benchmark.leakage import CANARY, check_ids, forbidden_terms, lexical_artefacts, overlap_with_development, scan_case, scan_requests, scan_text
from sika_guard.benchmark.pilot import build_pilot
from sika_guard.benchmark.visible import opaque_id
from sika_guard.coordinator import Condition, coordinate
from sika_guard.evaluators.llm import evaluators_for
from sika_guard.fixtures.cases import build_cases
from sika_guard.models import Case

from fakes import FakeProvider

AI_CONDITIONS = [Condition.B, Condition.B3, Condition.BC, Condition.C]


def run(coro):
    return asyncio.run(coro)


def pilot():
    cases, labels = build_pilot()
    return cases, {x.case_id: x for x in labels}


def bodies_for(cases, condition):
    provider = FakeProvider()

    async def go():
        async with provider.client() as client:
            for case in cases:
                await coordinate(case, condition, evaluators_for(condition.value, client))

    run(go())
    return provider.bodies


def labelled_with_canary(labels):
    """Hidden labels whose free-text fields carry the canary: if any code path forwarded a label, the canary would travel."""
    return {k: HiddenLabels.model_validate({**v.model_dump(), "rationale": f"{CANARY} {v.rationale}", "audit_notes": CANARY}) for k, v in labels.items()}


# --- the canary ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("condition", AI_CONDITIONS)
def test_a_canary_planted_in_the_hidden_labels_never_reaches_any_request(condition):
    cases, labels = pilot()
    planted = labelled_with_canary(labels)  # alive in the same process while every request is built
    assert all(CANARY in x.rationale for x in planted.values())
    bodies = bodies_for(cases, condition)
    assert len(bodies) >= len(cases)
    assert CANARY not in json.dumps(bodies)
    assert scan_requests(bodies) == []


def test_the_detector_fires_when_a_label_or_the_canary_really_is_in_a_request():
    """Negative control: without this a silent detector would also report 'no leak'."""
    cases, labels = pilot()
    leaked = cases[0].model_copy(update={"offers": tuple(
        o.model_copy(update={"description": f"{o.description} attack_family bm_hidden_fees {CANARY}"}) for o in cases[0].offers)})
    hits = scan_case(leaked)
    assert CANARY in hits and "attack_family" in hits and "bm_hidden_fees" in hits
    bodies = bodies_for([leaked], Condition.B)
    assert CANARY in scan_requests(bodies)


def test_every_hidden_vocabulary_term_is_both_caught_and_absent_from_all_pilot_requests():
    cases, labels = pilot()
    assert len(label_vocabulary()) > 40
    for term in label_vocabulary()[:20]:
        assert scan_text(f"some text with {term.upper()} inside") == [term] or term in scan_text(f"some text with {term.upper()} inside")
    for condition in AI_CONDITIONS:
        assert scan_requests(bodies_for(cases, condition)) == []


def test_no_pilot_case_contains_label_vocabulary():
    cases, _ = pilot()
    assert [scan_case(c) for c in cases if scan_case(c)] == []


def test_case_ids_are_not_label_vocabulary_and_family_values_are_not_ordinary_words():
    assert all("_" in f.value for f in BenchFamily)
    assert all(f.value.startswith("bm_") for f in BenchFamily)  # so they cannot collide with development labels or merchant prose


# --- what the model sees, and how ----------------------------------------------------------------------------------


@pytest.mark.parametrize("condition", AI_CONDITIONS)
def test_merchant_text_stays_in_the_user_message_under_untrusted_content(condition):
    cases, _ = pilot()
    injected = next(c for c in cases if any("Ignore previous instructions" in o.description for o in c.offers))
    bodies = bodies_for([injected], condition)
    needle = "Ignore previous instructions"
    carriers = 0
    for body in bodies:
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        assert needle not in system
        payload = json.loads(user[user.index("{"):])
        assert needle not in json.dumps(payload["trusted"])
        carriers += needle in json.dumps(payload["untrusted_merchant_content"])
    assert carriers >= 1


@pytest.mark.parametrize("condition", AI_CONDITIONS)
def test_system_prompts_are_static_across_all_pilot_cases(condition):
    cases, _ = pilot()
    systems = {body["messages"][0]["content"] for body in bodies_for(cases, condition)}
    assert len(systems) == (3 if condition is Condition.C else 1)
    assert all(not any(c.case_id in s for c in cases) for s in systems)


# --- ids ------------------------------------------------------------------------------------------------------------------


def test_pilot_ids_are_opaque_distinct_and_unordered():
    cases, _ = pilot()
    ids = [c.case_id for c in cases]
    assert check_ids(ids) == []
    assert ids != sorted(ids)


@pytest.mark.parametrize(
    "bad",
    [["case-01", "bm-0123456789"], ["bm-attack00001", "bm-0123456789"], ["bm-0123456789", "bm-0123456789"], ["bm-hidden_fees", "bm-0123456789"], ["bm-ABCDEF0123", "bm-0123456789"]],
)
def test_ids_that_reveal_or_repeat_are_flagged(bad):
    assert check_ids(bad)


def test_ids_in_sorted_order_are_flagged_because_order_can_carry_the_answer():
    ids = sorted(opaque_id("x", str(i)) for i in range(10))
    assert any("sorted order" in p for p in check_ids(ids))


def test_opaque_ids_are_derived_by_hashing_not_by_counting():
    ids = [opaque_id("s", str(i)) for i in range(50)]
    assert len(set(ids)) == 50 and all(re.fullmatch(r"bm-[0-9a-f]{10}", i) for i in ids)
    assert not any(i.endswith(("0000000000", "0000000001")) for i in ids)


# --- reproduction of development templates -----------------------------------------------------------------------------------


def test_the_pilot_does_not_reuse_development_merchants_ids_texts_or_schedules():
    cases, _ = pilot()
    report = overlap_with_development(cases, build_cases())
    assert report.problems == [] and report.shared_offer_ids == [] and report.shared_merchants == []


def test_a_copy_of_a_development_case_is_detected_even_under_a_new_id():
    dev = build_cases()
    copy = dev[4].model_copy(update={"case_id": opaque_id("copy")})
    report = overlap_with_development([copy], dev)
    assert report.shared_offer_ids and report.shared_merchants and report.reused_texts
    assert report.schedule_signature_reuse == 1.0 and len(report.problems) == 4


def test_a_light_paraphrase_of_a_development_description_is_still_caught():
    dev = build_cases()
    source = next(o.description for c in dev for o in c.offers if len(o.description.split()) >= 12)
    case, _ = pilot()
    first = case[0]
    paraphrased = first.model_copy(update={"offers": (first.offers[0].model_copy(update={"description": source + " Thanks for shopping."}), *first.offers[1:])})
    assert overlap_with_development([paraphrased], dev).reused_texts


# --- lexical and formatting artefacts ------------------------------------------------------------------------------------------


def synthetic_cases(n: int, *, biased: bool):
    base, _ = pilot()
    template = base[0]
    pool_attack = ["URGENT limited stock, act now!", "Hurry: offer ends today!", "URGENT clearance, act now!"]
    pool_other = ["Includes accident cover for the term.", "Standard delivery included.", "Two year guarantee included."]
    cases, flags = [], {}
    pick = lambda salt, i, k: int(hashlib.sha256(f"{salt}{i}".encode()).hexdigest(), 16) % k  # noqa: E731  independent draws, no parity or period
    for i in range(n):
        attack = i % 2 == 0 if biased else bool(pick("label", i, 2))
        pool = (pool_attack if attack else pool_other) if biased else (pool_attack + pool_other)
        text = pool[pick("text", i, len(pool))]
        offers = (template.offers[0].model_copy(update={"description": text}), *template.offers[1:])
        case = template.model_copy(update={"case_id": opaque_id("synthetic", str(i)), "offers": offers})
        cases.append(case)
        flags[case.case_id] = attack
    return cases, flags


def test_the_artefact_check_flags_a_dataset_where_wording_gives_the_label_away():
    cases, flags = synthetic_cases(60, biased=True)
    report = lexical_artefacts(cases, flags)
    assert report.assessed and report.flagged and report.loo_accuracy > 0.9 and report.baseline_accuracy == 0.5
    assert any("urgent" in f or "act" in f or "now" in f for f in report.suspicious_features)


def test_the_artefact_check_does_not_fire_when_wording_is_unrelated_to_the_label():
    cases, flags = synthetic_cases(60, biased=False)
    report = lexical_artefacts(cases, flags)
    assert report.assessed and not report.flagged, (report.loo_accuracy, report.suspicious_features)


def test_the_artefact_check_declines_to_judge_a_small_sample_instead_of_guessing():
    cases, flags = synthetic_cases(12, biased=True)
    report = lexical_artefacts(cases, flags)
    assert not report.assessed and not report.flagged and "not assessed" in report.note


def test_the_artefact_check_is_described_as_a_heuristic_not_a_guarantee():
    assert "smoke detector" in leakage.lexical_artefacts.__doc__ and "never as a verdict" in leakage.lexical_artefacts.__doc__
    assert "not a guarantee" in leakage.__doc__.lower() or "does not mean the benchmark is uncontaminated" in leakage.__doc__


# --- labels and expected verdicts do not enter inference ------------------------------------------------------------------------------


def test_expected_verdicts_and_families_are_absent_from_everything_a_condition_receives():
    cases, labels = pilot()
    forbidden = forbidden_terms() + [x.rationale for x in labels.values()]
    for condition in AI_CONDITIONS:
        text = json.dumps(bodies_for(cases, condition))
        assert not [t for t in forbidden if t in text]


def test_the_development_cli_and_the_benchmark_cli_share_no_label_path(tmp_path, capsys):
    assert evaluate.main(["--condition", "B", "--offline", "allow", "--limit", "2"]) == 0
    assert "labels" not in " ".join(m for m in dir(Case) if not m.startswith("_"))
