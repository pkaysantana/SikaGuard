"""The gate between inference and hidden labels.

Labels may be opened only for a *finished, validated experiment log that the inference run itself sealed*:

  1. The runner, having finished, calls `write_seal_file(log, cases)`. That refuses anything that is not a complete v2 run
     log: arbitrary files, logs without a closing summary, logs with a failure line, logs missing a declared condition or
     any (condition, repeat, case) the declaration promises, logs with extra or duplicate entries, and logs whose declared
     case-file digest is not the real digest of the case file. The seal file records real digests.
  2. `verify_gate` is what the scorer must pass before `hidden.load_labels` will read anything. It needs the persisted seal
     file, re-computes the digests of the log and the case file as they are now, and re-runs the whole validation. A seal the
     scorer makes for itself does not exist: there is no function that seals without validating.

The run is also bound to a manifest (`binding`): the declared digest must be a recorded, untampered manifest that describes the frozen
core, the prompts and the condition definitions that ran, a live run needs a manifest approved as frozen, and the frozen core must not have
changed while the run was in progress. The seal records the manifest and those hashes, and the gate compares seal, declaration and
manifest with one another.

This is a procedural guard against sealing the wrong thing, against an unfinished or mislabelled experiment, and against editing a log
after the fact. It is the integrity of a COMPLETED run. It is not protection against a malicious operator who can rewrite the manifest
directory, the log and the seal together: that needs signatures made outside this repository, and nothing here is authentication.

This module must not import the hidden-label module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sika_guard.benchmark import binding
from sika_guard.benchmark.decisions import DETERMINISTIC_CONDITIONS, RUN_LOG_SCHEMA, RunLog, read_run_log
from sika_guard.benchmark.visible import load_visible, visible_file_digest

SEAL_SCHEMA = "sika-guard-seal/3"


class LabelsLockedError(RuntimeError):
    """The log cannot be sealed, or the labels cannot be opened, because the gate's conditions are not met."""


@dataclass(frozen=True)
class Seal:
    path: str
    sha256: str
    records: int
    cases_sha256: str


def seal_path_for(log: Path) -> Path:
    return log.with_name(log.name + ".seal.json")


def _log_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_manifest_binding(d, manifests_dir: Path | None, root: Path | None, *, still_running: bool) -> None:
    if not (d.manifest_status and d.frozen_core_sha256 and d.conditions_sha256):
        raise LabelsLockedError("the run declaration is not bound to a manifest (this log predates binding or was written without it)")
    try:
        binding.check_recorded(
            manifest_digest=d.manifest_digest, status=d.manifest_status, frozen_core_sha256=d.frozen_core_sha256, conditions_sha256=d.conditions_sha256,
            mode=d.mode, conditions=d.conditions, manifests_dir=manifests_dir, root=root, still_running=still_running,
        )
    except binding.ManifestBindingError as exc:
        raise LabelsLockedError(f"the run is not bound to a manifest that describes it: {exc}") from None


def validate_log(log_path: Path, cases_path: Path, *, manifests_dir: Path | None = None, root: Path | None = None, at_seal: bool = False) -> RunLog:
    """Everything that must be true of a log before its labels may be opened. Raises LabelsLockedError, naming the first failure."""
    if not log_path.is_file():
        raise LabelsLockedError(f"no decision log at {log_path.name}")
    try:
        log = read_run_log(log_path)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise LabelsLockedError(f"{log_path.name} is not a valid run log ({type(exc).__name__})") from None
    d = log.declaration
    if d.schema_id != RUN_LOG_SCHEMA:
        raise LabelsLockedError(f"only {RUN_LOG_SCHEMA} run logs can be sealed (this is {d.schema_id})")
    _check_manifest_binding(d, manifests_dir, root, still_running=at_seal)
    if log.summary is None:
        raise LabelsLockedError("the log has no closing summary: the run did not finish")
    if log.failures or not log.summary.complete:
        raise LabelsLockedError(f"the run is incomplete ({len(log.failures)} failure line(s); summary complete={log.summary.complete})")
    if not cases_path.is_file():
        raise LabelsLockedError(f"no case file at {cases_path.name}")
    if visible_file_digest(cases_path) != d.visible_cases_sha256:
        raise LabelsLockedError("the declared case-file digest is not the digest of the case file supplied")
    case_ids = {c.case_id for c in load_visible(cases_path)}
    if d.cases is not None and d.cases != len(case_ids):
        raise LabelsLockedError("the declaration's case count does not match the case file")

    declared = set(d.conditions)
    present = {r.condition.value for r in log.records}
    if not declared or present != declared:
        raise LabelsLockedError(f"declared conditions {sorted(declared)} but the log holds {sorted(present)}")
    expected = {
        (cond, repeat, case_id)
        for cond in declared
        for repeat in (range(1, d.repeats + 1) if (cond not in DETERMINISTIC_CONDITIONS or d.verify_a_repeat) else (1,))
        for case_id in case_ids
    }
    got = [(r.condition.value, r.repeat, r.record.case_id) for r in log.records]
    if len(got) != len(set(got)):
        raise LabelsLockedError("the log contains the same (condition, repeat, case) more than once")
    if missing := expected - set(got):
        raise LabelsLockedError(f"{len(missing)} expected (condition, repeat, case) record(s) are missing, e.g. {sorted(missing)[:2]}")
    if extra := set(got) - expected:
        raise LabelsLockedError(f"{len(extra)} unexpected record(s), e.g. {sorted(extra)[:2]}")
    if any(r.record.condition != r.condition for r in log.records):
        raise LabelsLockedError("a record's condition disagrees with its log entry")
    s = log.summary
    if s.recorded != len(got) or s.expected != len(expected) or s.failures != 0 or s.missing != 0:
        raise LabelsLockedError("the closing summary's counts do not match the log")
    return log


def write_seal_file(log_path: Path, cases_path: Path, *, manifests_dir: Path | None = None, root: Path | None = None) -> Seal:
    """Seal a finished log. Called by the runner. Refuses any log that is not complete, valid and bound to a manifest that describes the
    code that ran (and that is still the code in the tree); writes real digests."""
    log = validate_log(log_path, cases_path, manifests_dir=manifests_dir, root=root, at_seal=True)
    d = log.declaration
    target = seal_path_for(log_path)
    if target.exists():
        raise LabelsLockedError(f"{target.name} already exists; a seal is written once")
    seal = Seal(path=str(log_path), sha256=_log_digest(log_path), records=sum(1 for line in log_path.read_bytes().splitlines() if line.strip()), cases_sha256=visible_file_digest(cases_path))
    target.write_text(
        json.dumps(
            {
                "schema": SEAL_SCHEMA, "log": log_path.name, "log_sha256": seal.sha256, "lines": seal.records, "cases_sha256": seal.cases_sha256,
                "manifest_digest": d.manifest_digest, "manifest_status": d.manifest_status, "frozen_core_sha256": d.frozen_core_sha256,
                "conditions_sha256": d.conditions_sha256, "mode": d.mode,
            },
            indent=1,
        ) + "\n",
        encoding="utf-8",
    )
    return seal


def verify_gate(log_path: Path, cases_path: Path, *, manifests_dir: Path | None = None, root: Path | None = None) -> tuple[Seal, RunLog]:
    """What the scorer must pass. Requires the persisted seal, then re-validates the log against the real case file and the manifest."""
    target = seal_path_for(log_path)
    if not target.is_file():
        raise LabelsLockedError(f"no persisted seal ({target.name}): the inference run did not finish and seal, so labels stay closed")
    try:
        obj = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise LabelsLockedError("the seal file is not valid JSON") from None
    if obj.get("schema") != SEAL_SCHEMA or obj.get("log") != log_path.name:
        raise LabelsLockedError("the seal file does not belong to this log")
    if not log_path.is_file() or _log_digest(log_path) != obj.get("log_sha256"):
        raise LabelsLockedError("the decision log was modified after it was sealed")
    if not cases_path.is_file() or visible_file_digest(cases_path) != obj.get("cases_sha256"):
        raise LabelsLockedError("the case file is not the one the log was sealed against")
    log = validate_log(log_path, cases_path, manifests_dir=manifests_dir, root=root)
    d = log.declaration
    for key, declared in (("manifest_digest", d.manifest_digest), ("manifest_status", d.manifest_status), ("frozen_core_sha256", d.frozen_core_sha256),
                          ("conditions_sha256", d.conditions_sha256), ("mode", d.mode)):
        if obj.get(key) != declared:
            raise LabelsLockedError(f"the seal and the run declaration disagree about {key}")
    return Seal(path=str(log_path), sha256=obj["log_sha256"], records=obj["lines"], cases_sha256=obj["cases_sha256"]), log
