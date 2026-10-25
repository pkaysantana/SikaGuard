"""The experiment runner: A, B, B3, Bc and C over an explicitly supplied model-visible case file, with repetitions.

It composes the frozen pieces (`evaluators_for`, `coordinate`, `LLMClient`) and adds only orchestration. It never
loads hidden labels, never adds a repeat id, run id or experiment name to anything a model is sent (a test compares
request bodies across repetitions), never retries a request, and records every (condition, repeat, case) as a record or
an explicit failure. A finished log is sealed, and the scorer refuses to open labels without that seal.

Cost control: `plan_calls` reports the exact call count before anything runs; live execution needs an explicit
confirmation, and a hard cap on requests is enforced at the transport. A condition's calls per case are the frozen
definitions (B 1, B3 3, Bc 1, C 3, A none), so the plan is exact when no condition is skipped.

This module must not import the hidden-label module.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from sika_guard.benchmark.decisions import (
    RUN_LOG_SCHEMA,
    RunDeclaration,
    RunFailure,
    RunLogWriter,
    RunSummary,
    declare,
    make_run_record,
    now_utc,
    read_run_log,
)
from sika_guard.benchmark import binding as manifest_binding
from sika_guard.benchmark.offline import CountingTransport, FakeProviderTransport
from sika_guard.benchmark.seal import Seal, write_seal_file
from sika_guard.benchmark.visible import visible_file_digest
from sika_guard.coordinator import Condition, CoordinatorConfig, coordinate
from sika_guard.evaluators.llm import LLMClient, LLMConfig, evaluators_for
from sika_guard.models import Case

CALLS_PER_CASE = {Condition.A: 0, Condition.A_PLUS: 0, Condition.B: 1, Condition.B3: 3, Condition.BC: 1, Condition.C: 3}
ALL_CONDITIONS = (Condition.A, Condition.B, Condition.B3, Condition.BC, Condition.C)  # the preregistered comparison; A+ is an optional reference
DETERMINISTIC = (Condition.A, Condition.A_PLUS)  # no model: one execution stands for every repeat
DEFAULT_REPEATS = 3
DEFAULT_ORDER_SEED = 20_240_617


@dataclass(frozen=True)
class ExperimentSpec:
    conditions: tuple[Condition, ...] = ALL_CONDITIONS
    repeats: int = DEFAULT_REPEATS
    fail_closed: bool = False
    order_seed: int = DEFAULT_ORDER_SEED
    case_concurrency: int = 1  # >1 runs cases at once; per-case latency is then measured under load and says so
    max_calls: int | None = None  # None means the planned total
    verify_a_repeat: bool = False  # run A in every repeat to check it is deterministic
    experiment_name: str = ""

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        if self.case_concurrency < 1:
            raise ValueError("case_concurrency must be at least 1")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("a condition is listed twice")


@dataclass(frozen=True)
class CallPlan:
    cases: int
    repeats: int
    per_condition: dict[Condition, int]  # model calls
    executions: dict[Condition, int]  # (repeat, case) runs
    total_calls: int


def plan_calls(n_cases: int, spec: ExperimentSpec = ExperimentSpec()) -> CallPlan:
    """Exact model calls and case executions for a spec, computed before anything runs."""
    executions = {c: n_cases * (spec.repeats if (c not in DETERMINISTIC or spec.verify_a_repeat) else 1) for c in spec.conditions}
    per = {c: executions[c] * CALLS_PER_CASE[c] for c in spec.conditions}
    return CallPlan(cases=n_cases, repeats=spec.repeats, per_condition=per, executions=executions, total_calls=sum(per.values()))


def format_plan(plan: CallPlan, spec: ExperimentSpec) -> str:
    lines = [
        f"Planned model calls: {plan.cases} cases x {plan.repeats} repeats (A is deterministic and runs once per case unless --verify-a-repeat).",
        "  condition  calls/case  case-executions  model calls",
    ]
    lines += [f"  {c.value:<9}  {CALLS_PER_CASE[c]:<10}  {plan.executions[c]:<15}  {plan.per_condition[c]}" for c in spec.conditions]
    lines.append(f"  TOTAL model calls: {plan.total_calls}" + (f"  (hard cap {spec.max_calls})" if spec.max_calls is not None else ""))
    lines.append("  Each call is one request; failures are recorded, never retried, so the plan is also the ceiling.")
    return "\n".join(lines)


def order_cases(cases: Sequence[Case], seed: int, repeat: int, condition: Condition) -> list[Case]:
    """A deterministic order that differs by repeat and condition, so position effects average out without randomness."""
    return sorted(cases, key=lambda c: hashlib.sha256(f"{seed}|{repeat}|{condition.value}|{c.case_id}".encode()).hexdigest())


@dataclass
class RunOutcome:
    log_path: Path
    expected: int
    recorded: int
    failures: int
    missing: int
    calls_made: int
    calls_refused: int
    complete: bool
    seal: Seal | None
    plan: CallPlan
    resumed_skipped: int = 0
    notes: list[str] = field(default_factory=list)


def runtime_description(config: LLMConfig | None, mode: str, spec: ExperimentSpec) -> dict:
    """Non-secret runtime configuration only. The key and the full URL are never recorded."""
    if config is None:
        return {"mode": mode, "case_concurrency": spec.case_concurrency}
    described = config.describe()
    return {"mode": mode, "case_concurrency": spec.case_concurrency, "endpoint": described["base_url"], "timeout_seconds": described["timeout_seconds"], "pricing_configured": described["pricing_configured"]}


def build_declaration(cases_path: Path, cases: Sequence[Case], spec: ExperimentSpec, plan: CallPlan, *, mode: str, config: LLMConfig | None,
                      manifest_digest: str, bound: manifest_binding.Binding) -> RunDeclaration:
    return declare(
        mode=mode, manifest_digest=manifest_digest, visible_cases_sha256=visible_file_digest(cases_path), conditions=spec.conditions,
        manifest_status=bound.status, frozen_core_sha256=bound.frozen_core_sha256, conditions_sha256=bound.conditions_sha256,
        model_id=config.model if config else None, temperature=config.temperature if config else None, json_mode=config.json_mode if config else True,
        fail_closed=spec.fail_closed, schema_id=RUN_LOG_SCHEMA, repeats=spec.repeats, cases=len(cases), planned_calls=plan.total_calls,
        max_calls=spec.max_calls if spec.max_calls is not None else plan.total_calls, case_order_seed=spec.order_seed,
        case_concurrency=spec.case_concurrency, experiment_name=spec.experiment_name, verify_a_repeat=spec.verify_a_repeat,
        runtime=runtime_description(config, mode, spec),
    )


async def run_experiment(
    cases: Sequence[Case], cases_path: Path, spec: ExperimentSpec, log_path: Path, *, mode: str, config: LLMConfig,
    inner_transport: httpx.AsyncBaseTransport, manifest_digest: str, resume: bool = False, manifests_dir: Path | None = None, root: Path | None = None,
) -> RunOutcome:
    """Run the experiment and write the log. `config` and the transport are supplied by the caller: a fake for offline
    runs, a real transport for live ones. Hidden labels are not an argument and are never reachable from here."""
    # Bound to a manifest that describes this code BEFORE anything is sent: an unbound experiment makes no call at all.
    bound = manifest_binding.bind(manifest_digest, mode=mode, conditions=[c.value for c in spec.conditions], manifests_dir=manifests_dir, root=root)
    plan = plan_calls(len(cases), spec)
    cap = spec.max_calls if spec.max_calls is not None else plan.total_calls
    declaration = build_declaration(cases_path, cases, spec, plan, mode=mode, config=config, manifest_digest=manifest_digest, bound=bound)
    writer = RunLogWriter(log_path, declaration, resume=resume)
    skipped = 0
    counting = CountingTransport(inner_transport, cap)
    coord = CoordinatorConfig(fail_closed=spec.fail_closed)
    expected: set[tuple[str, int, str]] = set()
    sem = asyncio.Semaphore(spec.case_concurrency)

    async def one(case: Case, condition: Condition, repeat: int, evaluators) -> None:
        nonlocal skipped
        key = (condition.value, repeat, case.case_id)
        if key in writer.done:
            skipped += 1
            return
        async with sem:
            started = now_utc()
            try:
                record = await coordinate(case, condition, evaluators, coord)
            except Exception as exc:  # explicit failure line; the exception class only, never its message
                writer.failure(RunFailure(condition=condition, repeat=repeat, case_id=case.case_id, error_type=type(exc).__name__, started_utc=started, finished_utc=now_utc()))
                return
            writer.record(make_run_record(condition, repeat, started, now_utc(), record))

    async with LLMClient(config, transport=counting) as client:
        for repeat in range(1, spec.repeats + 1):
            for condition in spec.conditions:
                if condition in DETERMINISTIC and repeat > 1 and not spec.verify_a_repeat:
                    continue  # deterministic: one run stands for every repeat
                evaluators = evaluators_for(condition.value, client) if condition is not Condition.A else []
                batch = order_cases(cases, spec.order_seed, repeat, condition)
                expected |= {(condition.value, repeat, c.case_id) for c in batch}
                await asyncio.gather(*(one(c, condition, repeat, evaluators) for c in batch))

    final = read_run_log_partial(log_path)
    recorded = len({(r.condition.value, r.repeat, r.record.case_id) for r in final.records})
    failed = len(final.failures)
    missing = len(expected - {(r.condition.value, r.repeat, r.record.case_id) for r in final.records} - {(f.condition.value, f.repeat, f.case_id) for f in final.failures})
    calls_recorded = sum(len(v.calls) for r in final.records for v in r.record.verdicts)
    complete = failed == 0 and missing == 0 and len(expected) == recorded
    writer.finish(RunSummary(expected=len(expected), recorded=recorded, failures=failed, missing=missing, planned_calls=plan.total_calls, calls_recorded=calls_recorded, complete=complete, finished_utc=now_utc()))
    notes = []
    seal = write_seal_file(log_path, cases_path, manifests_dir=manifests_dir, root=root) if complete else None  # an incomplete run is never sealed, so its labels stay closed
    if not complete:
        notes.append("the run is incomplete: it was NOT sealed and its labels stay closed")
    if counting.refused:
        notes.append(f"{counting.refused} request(s) were refused by the call cap and recorded as provider errors")
    if counting.calls > plan.total_calls:
        notes.append(f"{counting.calls} requests were sent against a plan of {plan.total_calls}")
    return RunOutcome(log_path, len(expected), recorded, failed, missing, counting.calls, counting.refused, complete, seal, plan, skipped, notes)


def read_run_log_partial(path: Path):
    return read_run_log(path)


def write_run_from_records(
    log_path: Path, cases_path: Path, cases: Sequence[Case], records: "dict[Condition, Sequence]", *, mode: str, manifest_digest: str, fail_closed: bool = False, seal: bool = True,
    manifests_dir: Path | None = None, root: Path | None = None,
) -> Seal | None:
    """Write already-computed records (the scripted mock evaluators) as a one-repeat v2 run, then seal it through the same validation
    a real run goes through. There is no separate path that seals without validating."""
    conditions = tuple(records)
    spec = ExperimentSpec(conditions=conditions, repeats=1, fail_closed=fail_closed)
    bound = manifest_binding.bind(manifest_digest, mode=mode, conditions=[c.value for c in conditions], manifests_dir=manifests_dir, root=root)
    plan = plan_calls(len(cases), spec)
    declaration = build_declaration(cases_path, cases, spec, plan, mode=mode, config=None, manifest_digest=manifest_digest, bound=bound)
    writer = RunLogWriter(log_path, declaration)
    n = 0
    for condition, recs in records.items():
        for record in recs:
            stamp = now_utc()
            writer.record(make_run_record(condition, 1, stamp, stamp, record))
            n += 1
    calls = sum(len(v.calls) for recs in records.values() for r in recs for v in r.verdicts)
    writer.finish(RunSummary(expected=len(cases) * len(conditions), recorded=n, failures=0, missing=len(cases) * len(conditions) - n, planned_calls=plan.total_calls, calls_recorded=calls,
                             complete=n == len(cases) * len(conditions), finished_utc=now_utc()))
    return write_seal_file(log_path, cases_path, manifests_dir=manifests_dir, root=root) if seal else None


def fake_config(mode: str) -> LLMConfig:
    """A config for offline runs. The key is a placeholder and the endpoint is never contacted."""
    return LLMConfig(base_url="https://fake.invalid/v1", api_key="fake-key-not-a-secret", model=f"fake-{mode}", timeout_seconds=30.0)


def offline_transport(mode: str) -> FakeProviderTransport:
    return FakeProviderTransport(mode=mode, model=f"fake-{mode}")
