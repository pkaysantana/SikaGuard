"""The specification documents and the tooling must not contradict each other."""

import inspect
import re
from pathlib import Path

from sika_guard.benchmark import __main__ as cli
from sika_guard.benchmark.audit import audit_packet
from sika_guard.benchmark.quotas import FAMILY_TARGETS, TARGET_TOTAL

ROOT = Path(__file__).resolve().parents[1]
SPEC = (ROOT / "BENCHMARK_SPEC.md").read_text(encoding="utf-8")
GUIDE = (ROOT / "BENCHMARK_BUILD_GUIDE.md").read_text(encoding="utf-8")
AUDIT = (ROOT / "BENCHMARK_AUDIT_PROTOCOL.md").read_text(encoding="utf-8")


def spec_family_counts() -> list[int]:
    section = SPEC.split("## 3. Content requirements")[1].split("## 4.")[0]
    return [int(m.group(1)) for m in re.finditer(r"^\| [A-Z][^|]+\| (\d+) \|", section, re.M)]


def test_the_spec_table_matches_the_quota_code_exactly():
    counts = spec_family_counts()
    assert sum(counts) == TARGET_TOTAL == 240 and sorted(counts) == sorted(FAMILY_TARGETS.values())


def test_the_target_size_is_240_and_the_old_320_survives_only_in_the_changelog():
    assert "**240**" in SPEC
    body = SPEC.split("## 1. What counts as development data")[1]
    assert "320" not in body


def test_the_spec_names_the_manifest_the_seal_and_the_frozen_policy_it_depends_on():
    for needle in ("manifest-0001-initial.json", "persisted seal that the inference run wrote over a complete, validated run log", "DECISION_POLICY.md", "not Git HEAD", "never edited"):
        assert needle in SPEC, needle


def test_every_cli_command_the_guide_tells_the_worker_to_run_exists():
    commands = set(re.findall(r"python -m sika_guard\.benchmark (\S+)", GUIDE))
    parser_source = inspect.getsource(cli.main)
    assert {"validate", "quotas", "leakage-scan", "audit-sample"} <= commands
    for command in commands:
        assert f'"{command}"' in parser_source, command


def test_the_audit_protocol_matches_the_sampler_default():
    assert "random 10%" in AUDIT and inspect.signature(audit_packet).parameters["fraction"].default == 0.10


def test_the_guide_defers_to_the_quota_file_instead_of_restating_counts():
    assert "quotas.py" in GUIDE and "must not contradict it" in GUIDE


def test_the_documents_do_not_claim_guarantees():
    assert "none of these gates is a guarantee" in SPEC
    assert "A pass does not show the dataset is unbiased or uncontaminated" in GUIDE
    for text in (SPEC, GUIDE):
        assert not re.search(r"\bguarantees? (that )?(the )?(dataset|benchmark) is (free|uncontaminated|unbiased)", text.lower())
    assert "synthetic distribution only" in SPEC
