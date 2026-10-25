"""The decision log: what each condition decided for each case, written before any label is opened.

Two formats share one declaration model:

  v1  `write_decision_log` / `read_decision_log`: one record per condition and case (used by the mock pilot).
  v2  `RunLogWriter` / `read_run_log`: repeat-aware, append-only, with explicit failure lines and a closing
      summary. This is what the experiment runner writes.

A v2 log is JSON lines: a declaration, then `record` lines (a `CaseRecord` for one condition, repeat and case, with a flat
`RecordSummary`), `failure` lines (a case that could not be recorded, so nothing is silently dropped), and a final
`summary` line. This module must not import the hidden-label module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from sika_guard.benchmark.summary import RecordSummary, summarize
from sika_guard.coordinator import CaseRecord, Condition

DECISION_LOG_SCHEMA = "sika-guard-decisions/1"
RUN_LOG_SCHEMA = "sika-guard-decisions/2"
DETERMINISTIC_CONDITIONS = frozenset({"A", "A+"})  # no model: one execution per case stands for every repeat (unless the run verifies it)


class RunDeclaration(BaseModel):
    """Declared before a run. For a live run `model_id` is the single fixed model; for a mock or fake run it names the fake."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_id: str = DECISION_LOG_SCHEMA
    mode: str  # "mock" (scripted), "fake" (a fake provider through the real client), or "live"
    manifest_digest: str
    visible_cases_sha256: str
    model_id: str | None = None
    temperature: float | None = None
    json_mode: bool = True
    conditions: tuple[str, ...]
    fail_closed: bool = False
    created_utc: str = ""
    # v2 additions, all defaulted so v1 logs still parse
    repeats: int = 1
    cases: int | None = None
    planned_calls: int | None = None
    max_calls: int | None = None
    case_order_seed: int | None = None
    case_concurrency: int = 1
    experiment_name: str = ""
    verify_a_repeat: bool = False
    runtime: dict[str, Any] = {}  # non-secret runtime configuration: endpoint host, timeout, pricing configured
    # What the run is bound to (benchmark/binding.py). Empty on a log written before binding existed, which therefore cannot be sealed.
    manifest_status: str = ""
    frozen_core_sha256: str = ""  # hash of the frozen-core files as they were when the run started
    conditions_sha256: str = ""  # hash of the manifest's definitions of the declared conditions


def declare(*, mode: str, manifest_digest: str, visible_cases_sha256: str, conditions: Sequence[Condition], model_id: str | None = None,
            temperature: float | None = None, fail_closed: bool = False, **extra: Any) -> RunDeclaration:
    return RunDeclaration(
        mode=mode, manifest_digest=manifest_digest, visible_cases_sha256=visible_cases_sha256, model_id=model_id, temperature=temperature,
        conditions=tuple(c.value for c in conditions), fail_closed=fail_closed, created_utc=datetime.now(timezone.utc).isoformat(), **extra,
    )


# --- v1 -----------------------------------------------------------------------------------------------------------------


def write_decision_log(path: Path, declaration: RunDeclaration, records: Mapping[Condition, Sequence[CaseRecord]]) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; a decision log is written once")
    lines = [json.dumps({"kind": "declaration", **json.loads(declaration.model_dump_json())}, sort_keys=True)]
    for condition, recs in records.items():
        lines += [json.dumps({"kind": "record", "condition": condition.value, "record": json.loads(r.model_dump_json())}, sort_keys=True) for r in recs]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_decision_log(path: Path) -> tuple[RunDeclaration, dict[Condition, list[CaseRecord]]]:
    declaration: RunDeclaration | None = None
    records: dict[Condition, list[CaseRecord]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("kind") == "declaration":
            obj.pop("kind")
            declaration = RunDeclaration.model_validate(obj)
        elif obj.get("kind") == "record":
            records.setdefault(Condition(obj["condition"]), []).append(CaseRecord.model_validate(obj["record"]))
        else:
            raise ValueError(f"line {number}: unknown decision-log entry")
    if declaration is None:
        raise ValueError("the decision log has no run declaration")
    return declaration, records


# --- v2 -----------------------------------------------------------------------------------------------------------------


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Condition
    repeat: int
    started_utc: str
    finished_utc: str
    record: CaseRecord
    summary: RecordSummary


class RunFailure(BaseModel):
    """A (condition, repeat, case) that could not be recorded. Only the exception's class is kept: a message could carry anything."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Condition
    repeat: int
    case_id: str
    error_type: str
    started_utc: str
    finished_utc: str


class RunSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    expected: int
    recorded: int
    failures: int
    missing: int
    planned_calls: int | None
    calls_recorded: int
    complete: bool  # every expected (condition, repeat, case) has a record and no failure line
    finished_utc: str


class RunLog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    declaration: RunDeclaration
    records: tuple[RunRecord, ...]
    failures: tuple[RunFailure, ...]
    summary: RunSummary | None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_record(condition: Condition, repeat: int, started: str, finished: str, record: CaseRecord) -> RunRecord:
    return RunRecord(condition=condition, repeat=repeat, started_utc=started, finished_utc=finished, record=record, summary=summarize(record))


def _line(kind: str, payload: BaseModel) -> str:
    return json.dumps({"kind": kind, **json.loads(payload.model_dump_json())}, sort_keys=True)


class RunLogWriter:
    """Append-only writer. Each line is flushed as it is written, so a crash loses nothing already recorded."""

    def __init__(self, path: Path, declaration: RunDeclaration, *, resume: bool = False):
        self.path, self.declaration = path, declaration
        self.done: set[tuple[str, int, str]] = set()
        if resume:
            existing = read_run_log(path)
            if existing.summary is not None:
                raise ValueError("this log is already finished and sealed; it cannot be resumed")
            if existing.declaration.model_dump(exclude={"created_utc"}) != declaration.model_dump(exclude={"created_utc"}):
                raise ValueError("the run declaration differs from the one in the log; refusing to resume a different experiment")
            self.done = {(r.condition.value, r.repeat, r.record.case_id) for r in existing.records} | {(f.condition.value, f.repeat, f.case_id) for f in existing.failures}
            self._file = path.open("a", encoding="utf-8")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("x", encoding="utf-8")  # refuses to overwrite
            self._file.write(_line("declaration", declaration) + "\n")
            self._file.flush()

    def _write(self, line: str) -> None:
        self._file.write(line + "\n")
        self._file.flush()

    def record(self, rr: RunRecord) -> None:
        self._write(_line("record", rr))
        self.done.add((rr.condition.value, rr.repeat, rr.record.case_id))

    def failure(self, f: RunFailure) -> None:
        self._write(_line("failure", f))
        self.done.add((f.condition.value, f.repeat, f.case_id))

    def finish(self, summary: RunSummary) -> None:
        self._write(_line("summary", summary))
        self._file.close()


def read_run_log(path: Path) -> RunLog:
    declaration: RunDeclaration | None = None
    records: list[RunRecord] = []
    failures: list[RunFailure] = []
    summary: RunSummary | None = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        kind = obj.pop("kind", None)
        if kind == "declaration":
            declaration = RunDeclaration.model_validate(obj)
        elif kind == "record":
            records.append(RunRecord.model_validate(obj))
        elif kind == "failure":
            failures.append(RunFailure.model_validate(obj))
        elif kind == "summary":
            summary = RunSummary.model_validate(obj)
        else:
            raise ValueError(f"line {number}: unknown run-log entry {kind!r}")
    if declaration is None:
        raise ValueError("the run log has no declaration")
    return RunLog(declaration=declaration, records=tuple(records), failures=tuple(failures), summary=summary)
