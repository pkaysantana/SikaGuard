"""Reproducibility manifest: what exactly was evaluated.

The repository has uncommitted changes, so a Git commit does not identify the code that runs. This
records the working tree: hashes of the files themselves, of every prompt text, of the decision policy
and its materiality assumptions, plus the condition definitions, the aggregation rule, the model
configuration requirements and the ground-truth isolation rules.

A manifest is a statement made at one moment. `verify_manifest` reports every later difference and never
rewrites the manifest; an intended change is recorded as a new, separately dated manifest with a reason.
Nothing here is a security control: it makes drift visible, it does not prevent it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "sika-guard-manifest/1"

# Files whose behaviour defines the evaluation. A change to any of them after a manifest is recorded
# means results are no longer comparable with that manifest.
FROZEN_CORE = (
    "src/sika_guard/money.py",
    "src/sika_guard/models.py",
    "src/sika_guard/policy.py",
    "src/sika_guard/coordinator.py",
    "src/sika_guard/experiment.py",
    "src/sika_guard/evaluate.py",
    "src/sika_guard/evaluators/adjudicate.py",
    "src/sika_guard/evaluators/amounts.py",
    "src/sika_guard/evaluators/contract.py",
    "src/sika_guard/evaluators/crosscheck.py",
    "src/sika_guard/evaluators/grounding.py",
    "src/sika_guard/evaluators/llm.py",
    "src/sika_guard/evaluators/mocks.py",
    "src/sika_guard/evaluators/payloads.py",
    "src/sika_guard/evaluators/prompts.py",
    "src/sika_guard/evaluators/scanner.py",
)
DEVELOPMENT_DATA = (
    "src/sika_guard/labels.py",
    "src/sika_guard/fixtures/builders.py",
    "src/sika_guard/fixtures/cases.py",
    "src/sika_guard/fixtures/labels.py",
    "tests/data/recorded_replies.json",
)
DOCUMENTS = ("DECISION_POLICY.md", "BENCHMARK_SPEC.md", "README.md", "CLAUDE.md")

# Hidden-label modules and the modules that must never be able to reach them.
HIDDEN_MODULES = ("sika_guard.labels", "sika_guard.fixtures.labels", "sika_guard.benchmark.hidden")
INFERENCE_SIDE = (
    "sika_guard.policy",
    "sika_guard.coordinator",
    "sika_guard.evaluators.*",
    "sika_guard.benchmark.visible",
    "sika_guard.benchmark.decisions",
    "sika_guard.benchmark.seal",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def hash_variants(path: Path) -> set[str]:
    """The hashes this file has under each line-ending convention. A manifest records the bytes that were on disk when it was recorded, and
    the same source has different bytes on a machine that checks out CRLF than on one that checks out LF. Hashing the content under both
    conventions lets a recorded manifest still recognise unchanged text without rewriting it. A real edit matches neither."""
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return {sha256_bytes(raw), sha256_bytes(lf), sha256_bytes(lf.replace(b"\n", b"\r\n"))}


def matches_recorded(path: Path, recorded_sha256: str) -> bool:
    return recorded_sha256 in hash_variants(path)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def working_tree_state(root: Path) -> dict[str, Any]:
    """Git HEAD plus every modified or untracked path. When anything is dirty, HEAD is *not* the code that runs."""
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    if head is None or status is None:
        return {"git_available": False, "note": "git was not available; only file hashes identify the code"}
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    return {
        "git_available": True,
        "head": head.strip(),
        "head_matches_working_tree": not changed,
        "changed_or_untracked_paths": sorted(changed),
        "note": "when head_matches_working_tree is false, HEAD does not identify the evaluated code; use the file hashes",
    }


def _group_of(rel: str) -> str:
    if rel in FROZEN_CORE:
        return "frozen_core"
    if rel in DEVELOPMENT_DATA:
        return "development_data"
    if rel in DOCUMENTS:
        return "document"
    if rel.startswith("src/sika_guard/benchmark/"):
        return "benchmark_tooling"
    if rel.startswith("tests/"):
        return "tests"
    if rel == "pyproject.toml":
        return "project_config"
    return "source_other"


def tracked_files(root: Path) -> list[str]:
    paths = {p.relative_to(root).as_posix() for p in (root / "src" / "sika_guard").rglob("*.py")}
    paths |= {p.relative_to(root).as_posix() for p in (root / "tests").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".json"}}
    paths |= {rel for rel in (*DOCUMENTS, "pyproject.toml") if (root / rel).is_file()}
    return sorted(paths)


def file_hashes(root: Path) -> dict[str, dict[str, str]]:
    return {rel: {"sha256": sha256_file(root / rel), "group": _group_of(rel)} for rel in tracked_files(root)}


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("pydantic", "pydantic_core", "httpx", "httpcore", "anyio", "pytest"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _condition_definitions() -> dict[str, Any]:
    from sika_guard.evaluators.prompts import specs_for

    conditions: dict[str, Any] = {
        "A": {"description": "deterministic rule engine only; no model is called", "evaluators": []},
    }
    from sika_guard.evaluators.payloads import generalist_payload
    from sika_guard.evaluators.scanner import SCANNER_VERSION

    scanner_source = Path(__file__).resolve().parents[1] / "evaluators" / "scanner.py"
    conditions["A+"] = {
        "description": "deterministic rule engine plus a non-LLM scanner of the same source passages the generalist sees; no model is called",
        "evaluators": [
            {
                "evaluator_id": "deterministic-scanner",
                "role": "deterministic",
                "prompt_version": SCANNER_VERSION,
                "system_prompt_sha256": sha256_bytes(scanner_source.read_bytes()),  # a scanner has no prompt; its source is what defines it
                "payload_builder": generalist_payload.__name__,
            }
        ],
    }
    descriptions = {
        "B": "baseline plus one general-purpose analyst, one call per case",
        "B3": "baseline plus three independent calls of the same general-purpose analyst (the same number of calls as C; compute is not claimed to match)",
        "Bc": "baseline plus one analyst applying all three specialist rubrics in one call (given the same information as C)",
        "C": "baseline plus three concurrent specialist analysts (merchant claims, credit terms, authority)",
    }
    for name, description in descriptions.items():
        conditions[name] = {
            "description": description,
            "evaluators": [
                {
                    "evaluator_id": s.evaluator_id,
                    "role": s.role.value,
                    "prompt_version": s.prompt_version,
                    "system_prompt_sha256": sha256_bytes(s.system_prompt.encode("utf-8")),
                    "payload_builder": s.build_payload.__name__,
                }
                for s in specs_for(name)
            ],
        }
    return conditions


def _policy_section(root: Path) -> dict[str, Any]:
    from dataclasses import asdict

    from sika_guard.policy import DEFAULT_CONFIG

    config = {k: (v.total_seconds() if hasattr(v, "total_seconds") else v) for k, v in asdict(DEFAULT_CONFIG).items()}
    return {
        "document": "DECISION_POLICY.md",
        "document_sha256": sha256_file(root / "DECISION_POLICY.md") if (root / "DECISION_POLICY.md").is_file() else None,
        "rule_engine_sha256": sha256_file(root / "src/sika_guard/policy.py"),
        "deterministic_materiality_assumptions": {
            "status": "provisional policy assumptions; not tuned on any held-out data",
            "values": config,
            "units": {"max_evidence_age": "seconds", "*_gap_minor": "minor currency units (pence, cents)"},
        },
    }


def build_manifest(root: Path | None = None, *, note: str = "", status: str = "development_baseline", approved_by: str | None = None) -> dict[str, Any]:
    """Describe the working tree. `status` is "development_baseline" (a description, never an approval) or "frozen": a manifest someone
    named as approved, which is refused unless the Git tree is clean so that HEAD identifies the code. `approved_by` is a name a person
    typed; it is a record of who said so, not an authentication of them."""
    from sika_guard.coordinator import AGGREGATION_POLICY, CoordinatorConfig
    from sika_guard.evaluators import llm

    if status not in ("development_baseline", "frozen"):
        raise ValueError(f"unknown manifest status {status!r}")
    root = root or repo_root()
    tree = working_tree_state(root)
    if status == "frozen":
        if not (approved_by or "").strip():
            raise ValueError("a frozen manifest needs the name of whoever approved it")
        if not tree.get("git_available") or not tree.get("head_matches_working_tree"):
            raise ValueError("a frozen manifest needs a clean, committed Git tree, so that HEAD identifies the code; this one has uncommitted changes")
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "status": status,
        **({"approval": {"approved_by": approved_by.strip(), "statement": "approved as the definition of the evaluation; not an authentication"}} if status == "frozen" else {}),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "dependencies": _versions(),
        "working_tree": tree,
        "files": file_hashes(root),
        "decision_policy": _policy_section(root),
        "conditions": _condition_definitions(),
        "aggregation": {
            "policy": AGGREGATION_POLICY,
            "rule": "any single enforcing route counts: a verified and material finding, a verified extraction that trusted code finds adverse, "
            "material and about the selected offer (the same test for every condition), or (only if fail_closed) an evaluator failure. "
            "No vote, no risk score.",
            "fail_closed_default": CoordinatorConfig().fail_closed,
        },
        "model_configuration_requirements": {
            "environment_variables_required": list(llm.REQUIRED_ENV),
            "model_id": "one fixed model id for the entire run; declared in the run declaration; requested and returned ids recorded per call",
            "decoding": {"temperature": "unset (provider default) unless declared in the run declaration", "json_mode": True, "timeout_seconds": llm.DEFAULT_TIMEOUT_SECONDS},
            "retries": "none; failures are recorded as abstentions",
            "same_for_all_conditions": ["model id", "decoding settings", "authoritative case information", "aggregation rule", "adjudication and decision policy"],
            "uncontrolled": "the upstream provider behind the endpoint may vary between calls; it is recorded per call and reported as a reproducibility limitation, never forced",
            "secrets": "the API key is never recorded in a manifest, a decision log or a report",
        },
        "ground_truth_isolation": {
            "hidden_modules": list(HIDDEN_MODULES),
            "inference_side_modules_that_must_not_import_them": list(INFERENCE_SIDE),
            "requirements": [
                "labels live in a separate file and module and are joined only after the decision log is sealed",
                "case ids are opaque and carry no information about family, label or split",
                "no label field name, family name, policy vocabulary or canary string appears in any request",
                "merchant text stays under untrusted_merchant_content, never in a system prompt",
            ],
        },
    }
    manifest["digest"] = manifest_digest(manifest)
    return manifest


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Hash of the manifest's content, excluding its own digest and creation time."""
    body = {k: v for k, v in manifest.items() if k not in {"digest", "created_utc"}}
    return sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"))


@dataclass
class Drift:
    changed: dict[str, str] = field(default_factory=dict)  # path -> group
    removed: dict[str, str] = field(default_factory=dict)
    added: dict[str, str] = field(default_factory=dict)
    prompt_changes: list[str] = field(default_factory=list)
    policy_changes: list[str] = field(default_factory=list)
    other_changes: list[str] = field(default_factory=list)

    @property
    def frozen_core_drifted(self) -> bool:
        return any(g == "frozen_core" for g in (*self.changed.values(), *self.removed.values(), *self.added.values())) or bool(
            self.prompt_changes or self.policy_changes
        )

    @property
    def clean(self) -> bool:
        return not (self.changed or self.removed or self.added or self.prompt_changes or self.policy_changes or self.other_changes)

    def describe(self) -> str:
        lines = []
        for label, items in (("changed", self.changed), ("removed", self.removed), ("added", self.added)):
            lines += [f"  {label}: {path} [{group}]" for path, group in sorted(items.items())]
        lines += [f"  prompt: {c}" for c in self.prompt_changes] + [f"  policy: {c}" for c in self.policy_changes] + [f"  other: {c}" for c in self.other_changes]
        return "\n".join(lines) or "  no differences"


def verify_manifest(manifest: dict[str, Any], root: Path | None = None) -> Drift:
    """Compare a recorded manifest with the working tree as it is now. Reports; never rewrites."""
    root = root or repo_root()
    now = build_manifest(root)
    drift = Drift()
    old_files, new_files = manifest["files"], now["files"]
    for rel, entry in old_files.items():
        if rel not in new_files:
            drift.removed[rel] = entry["group"]
        elif new_files[rel]["sha256"] != entry["sha256"] and not matches_recorded(root / rel, entry["sha256"]):
            drift.changed[rel] = entry["group"]  # a difference in line endings alone is not a change
    for rel, entry in new_files.items():
        if rel not in old_files:
            drift.added[rel] = entry["group"]

    for name, cond in manifest["conditions"].items():
        for old, new in zip(cond["evaluators"], now["conditions"].get(name, {}).get("evaluators", []), strict=False):
            if old != new:
                drift.prompt_changes.append(f"{name}/{old['evaluator_id']}: {old['prompt_version']}@{old['system_prompt_sha256'][:10]} -> {new['prompt_version']}@{new['system_prompt_sha256'][:10]}")
        if len(cond["evaluators"]) != len(now["conditions"].get(name, {}).get("evaluators", [])):
            drift.prompt_changes.append(f"{name}: number of evaluators changed")
    if manifest["decision_policy"] != now["decision_policy"]:
        drift.policy_changes.append("decision policy document, rule engine hash or materiality assumptions differ")
    for key in ("aggregation", "model_configuration_requirements", "ground_truth_isolation", "dependencies"):
        if manifest[key] != now[key]:
            drift.other_changes.append(f"{key} differs")
    if manifest["python"]["version"] != now["python"]["version"]:
        drift.other_changes.append(f"python {manifest['python']['version']} -> {now['python']['version']}")
    return drift


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists; a recorded manifest is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("digest") != manifest_digest(manifest):
        raise ValueError(f"{path} has been edited: its digest does not match its content")
    return manifest
