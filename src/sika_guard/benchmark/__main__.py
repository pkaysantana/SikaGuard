"""Command line for the benchmark tooling: python -m sika_guard.benchmark <command>.

Construction-phase commands (validate, quotas, audit-sample, pilot-write) read labels openly. The scoring
commands (score, pilot-run) seal the decision log first and open labels only through that seal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sika_guard.benchmark import manifest as mf


def _read_cases(path: Path):
    from sika_guard.benchmark.visible import load_visible

    return load_visible(path)


def _dev_cases():
    from sika_guard.benchmark.pilot import build_pilot
    from sika_guard.fixtures.cases import build_cases

    return build_cases() + build_pilot()[0]


def cmd_manifest(a: argparse.Namespace) -> int:
    try:
        m = mf.build_manifest(note=a.note, status="frozen" if a.freeze else "development_baseline", approved_by=a.approved_by)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    mf.write_manifest(m, Path(a.out))
    print(f"wrote {a.out} (digest {m['digest'][:16]}, status {m['status']}); head_matches_working_tree={m['working_tree'].get('head_matches_working_tree')}")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    drift = mf.verify_manifest(mf.load_manifest(Path(a.manifest)))
    print("Differences from the recorded manifest:\n" + drift.describe())
    print(f"\nfrozen core drifted: {drift.frozen_core_drifted}")
    return 1 if drift.frozen_core_drifted else 0


def cmd_validate(a: argparse.Namespace) -> int:
    from sika_guard.benchmark.hidden import open_for_validation
    from sika_guard.benchmark.validate import validate_dataset

    cases, labels = _read_cases(Path(a.cases)), open_for_validation(Path(a.labels))
    report = validate_dataset(cases, labels, development=_dev_cases(), heldout=a.heldout)
    for e in report.all_errors():
        print("ERROR", e)
    for w in report.warnings:
        print("WARNING", w)
    for item in report.audit_items:
        print("AUDIT", item)
    print(f"\n{len(cases)} cases: {'VALID' if report.ok else 'INVALID'}; {len(report.audit_items)} audit item(s). A pass does not prove the dataset is uncontaminated.")
    return 0 if report.ok else 1


def cmd_quotas(a: argparse.Namespace) -> int:
    from sika_guard.benchmark.hidden import open_for_validation
    from sika_guard.benchmark.quotas import check_quotas

    short = check_quotas(_read_cases(Path(a.cases)), open_for_validation(Path(a.labels)))
    print("\n".join(short) or "every stated minimum is met")
    return 1 if short else 0


def cmd_audit(a: argparse.Namespace) -> int:
    from sika_guard.benchmark.audit import audit_packet
    from sika_guard.benchmark.hidden import open_for_validation
    from sika_guard.benchmark.validate import validate_dataset

    cases, labels = _read_cases(Path(a.cases)), open_for_validation(Path(a.labels))
    packet = audit_packet(cases, labels, validate_dataset(cases, labels), fraction=a.fraction, seed=a.seed)
    Path(a.out).write_text(json.dumps(packet, indent=1), encoding="utf-8")
    print(f"{len(packet)} of {len(cases)} cases selected for human audit -> {a.out}")
    return 0


def cmd_pilot_write(a: argparse.Namespace) -> int:
    from sika_guard.benchmark.hidden import dump_labels
    from sika_guard.benchmark.pilot import build_pilot
    from sika_guard.benchmark.visible import dump_visible

    cases, labels = build_pilot()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.jsonl").write_text(dump_visible(cases), encoding="utf-8")
    (out / "labels.jsonl").write_text(dump_labels(labels), encoding="utf-8")
    print(f"wrote {len(cases)} DEVELOPMENT-ONLY pilot cases and labels to {out}")
    return 0


def _score_and_print(cases_path: Path, decisions_path: Path, labels_path: Path, *, mock: bool, manifests_dir: Path | None = None) -> int:
    from sika_guard.benchmark.hidden import load_labels
    from sika_guard.benchmark.scoring import format_score, score_run
    from sika_guard.benchmark.seal import LabelsLockedError, verify_gate

    # The scorer cannot seal. The inference run must have sealed a complete log; the gate re-validates it before any label is read.
    try:
        _, log = verify_gate(decisions_path, cases_path, manifests_dir=manifests_dir)
    except LabelsLockedError as exc:
        print(f"labels stay closed: {exc}", file=sys.stderr)
        return 3
    if log.declaration.repeats > 1:
        print("labels stay closed: scoring of repeated runs is not implemented yet (see AUDIT_TRIAGE.md, open blockers)", file=sys.stderr)
        return 4
    records: dict = {}
    for r in log.records:
        records.setdefault(r.condition, []).append(r.record)
    labels = load_labels(labels_path, decisions_path, cases_path, manifests_dir=manifests_dir)
    currencies = {c.case_id: c.mandate.max_amount.currency for c in _read_cases(cases_path)}
    print(format_score(score_run(records, labels, currencies=currencies), mock=mock or log.declaration.mode == "mock"))
    return 0


def cmd_pilot_run(a: argparse.Namespace) -> int:
    from sika_guard.benchmark.mockrun import mock_records
    from sika_guard.benchmark.runner import write_run_from_records

    out = Path(a.out_dir)
    if not (out / "cases.jsonl").exists():
        cmd_pilot_write(argparse.Namespace(out_dir=a.out_dir))
    cases = _read_cases(out / "cases.jsonl")
    log = out / "decisions.jsonl"
    for stale in (log, log.with_name(log.name + ".seal.json")):
        if stale.exists():
            stale.unlink()
    # A mock run is plumbing, so a development baseline of the tree as it is now may bind it. The manifest is kept beside the log.
    manifests = out / "manifests"
    manifest_path = manifests / "manifest-pilot.json"
    if manifest_path.exists():
        manifest_path.unlink()
    manifest = mf.build_manifest(note="pilot mock run (development baseline; plumbing only)")
    mf.write_manifest(manifest, manifest_path)
    write_run_from_records(log, out / "cases.jsonl", cases, mock_records(cases), mode="mock", manifest_digest=manifest["digest"], fail_closed=True, manifests_dir=manifests)
    return _score_and_print(out / "cases.jsonl", log, out / "labels.jsonl", mock=True, manifests_dir=manifests)


def cmd_challenge_write(a: argparse.Namespace) -> int:
    """Write the development challenge cases (model-visible) and, SEPARATELY, their expectations. `--only` writes a subset for a smoke test."""
    from sika_guard.benchmark import devchallenge
    from sika_guard.benchmark.visible import dump_visible

    out = Path(a.out_dir)
    if a.set == "v5-validation":
        cases_path, exp_path = devchallenge.write_v5(out)
        print(f"wrote DEVELOPMENT-ONLY v5-validation cases to {cases_path} and, separately, their expectations to {exp_path}")
        return 0
    cases_path, exp_path = devchallenge.write(out)
    if a.only:
        wanted = {s.strip() for s in a.only.split(",") if s.strip()}
        built = [(slug, case, exp) for slug, case, exp in devchallenge.build() if slug in wanted]
        unknown = wanted - {slug for slug, _, _ in built}
        if unknown:
            print(f"unknown slug(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        cases_path = out / "smoke_cases.jsonl"
        cases_path.write_text(dump_visible([c for _, c, _ in built]), encoding="utf-8")
        import json as _json
        from dataclasses import asdict

        exp_path = out / "smoke_expectations.json"
        exp_path.write_text(_json.dumps([asdict(e) for _, _, e in built], indent=1), encoding="utf-8")
    print(f"wrote DEVELOPMENT-ONLY challenge cases to {cases_path} and, separately, their expectations to {exp_path}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    from sika_guard.benchmark import devrun

    try:
        conditions = devrun.parse_conditions(a.conditions)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    out = Path(a.out_dir)
    result = devrun.run_dev(
        Path(a.cases), out, conditions, repeats=a.repeats, mode=a.mode, fake_mode=a.fake_mode, max_calls=a.max_calls, confirm_spend=a.confirm_spend,
        fail_closed=a.fail_closed,
    )
    o = result.outcome
    print(f"\nrun {'COMPLETE and sealed' if o.complete else 'INCOMPLETE (not sealed; labels and expectations stay closed)'}: {o.recorded}/{o.expected} records, {o.failures} failure(s); "
          f"{o.calls_made} request(s) sent (plan {o.plan.total_calls}); log {result.log_path}")
    for note in o.notes:
        print("  note:", note)
    if not o.complete:
        return 1
    return _report(Path(a.cases), result.log_path, Path(a.expectations) if a.expectations else None, result.manifests_dir, out)


def _report(cases_path: Path, log_path: Path, expectations: Path | None, manifests_dir: Path | None, out: Path | None) -> int:
    """Print the report. It is written to `out` only when one is given: opening a saved run must never rewrite the saved run."""
    from sika_guard.benchmark import devrun
    from sika_guard.benchmark.seal import LabelsLockedError

    try:
        rep = devrun.report_from_files(cases_path, log_path, expectations, manifests_dir)
    except LabelsLockedError as exc:
        print(f"expectations stay closed: {exc}", file=sys.stderr)
        return 3
    if out is None:
        print("\n" + devrun.format_report(rep) + "\n\n(read only: nothing was written; use --save-to DIR to keep a copy)")
        return 0
    json_path, text_path = devrun.write_outputs(rep, out)
    print("\n" + devrun.format_report(rep) + f"\n\nsaved: {json_path} and {text_path}")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    log = Path(a.decisions)
    return _report(Path(a.cases), log, Path(a.expectations) if a.expectations else None, Path(a.manifests_dir) if a.manifests_dir else log.parent / "manifests",
                   Path(a.save_to) if a.save_to else None)


def cmd_score(a: argparse.Namespace) -> int:
    return _score_and_print(Path(a.cases), Path(a.decisions), Path(a.labels), mock=False)


def cmd_leakage(a: argparse.Namespace) -> int:
    from sika_guard.benchmark.leakage import check_ids, scan_case

    cases = _read_cases(Path(a.cases))
    problems = check_ids([c.case_id for c in cases]) + [f"{c.case_id}: {h}" for c in cases if (h := scan_case(c))]
    print("\n".join(problems) or "no label vocabulary, canary or id problems found by these checks (not a guarantee)")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m sika_guard.benchmark", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("manifest", help="record a reproducibility manifest (never overwrites)")
    s.add_argument("--out", required=True)
    s.add_argument("--note", default="")
    s.add_argument("--freeze", action="store_true", help="record an APPROVED manifest; refused unless the Git tree is clean and --approved-by names whoever approves")
    s.add_argument("--approved-by", default=None)
    s.set_defaults(fn=cmd_manifest)
    s = sub.add_parser("verify-manifest", help="report differences between a recorded manifest and the working tree")
    s.add_argument("manifest")
    s.set_defaults(fn=cmd_verify)
    for name, fn, help_ in (
        ("validate", cmd_validate, "validate a case file against its label file"),
        ("quotas", cmd_quotas, "check the intended composition"),
        ("audit-sample", cmd_audit, "select cases for human label audit"),
    ):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--cases", required=True)
        s.add_argument("--labels", required=True)
        if name == "validate":
            s.add_argument("--heldout", action="store_true", help="apply the held-out independence and artefact checks")
        if name == "audit-sample":
            s.add_argument("--out", required=True)
            s.add_argument("--fraction", type=float, default=0.10)
            s.add_argument("--seed", type=int, default=7)
        s.set_defaults(fn=fn)
    s = sub.add_parser("leakage-scan", help="scan a visible case file for label vocabulary and id problems")
    s.add_argument("--cases", required=True)
    s.set_defaults(fn=cmd_leakage)
    s = sub.add_parser("pilot-write", help="write the development-only pilot files")
    s.add_argument("--out-dir", required=True)
    s.set_defaults(fn=cmd_pilot_write)
    s = sub.add_parser("pilot-run", help="run the scripted mock evaluators over the pilot and score them (no model is called)")
    s.add_argument("--out-dir", required=True)
    s.set_defaults(fn=cmd_pilot_run)
    s = sub.add_parser("challenge-write", help="write the development-only challenge cases and, separately, their expectations")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--only", default="", help="comma-separated slugs: write a subset as smoke_cases.jsonl for a small live check")
    s.add_argument("--set", choices=("challenge", "v5-validation"), default="challenge", help="v5-validation: the two fresh cases for the first check of prompts v5")
    s.set_defaults(fn=cmd_challenge_write)
    s = sub.add_parser("run", help="run development cases through chosen conditions; a fake provider by default, a live one only with explicit spend limits")
    s.add_argument("--cases", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--conditions", default="A,A+,B,B3,Bc,C")
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--mode", choices=("deterministic", "fake", "dev-live"), default="fake")
    s.add_argument("--fake-mode", choices=("quiet", "alarmist", "extractor", "flaky"), default="quiet")
    s.add_argument("--max-calls", type=int, default=None, help="dev-live only: hard ceiling on requests, enforced where they leave the process")
    s.add_argument("--confirm-spend", type=int, default=None, help="dev-live only: must equal the exact planned number of calls")
    s.add_argument("--fail-closed", action="store_true")
    s.add_argument("--expectations", default=None, help="a separate expectations file, read only after the run is sealed complete")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("report", help="rebuild the report for a finished, sealed run")
    s.add_argument("--cases", required=True)
    s.add_argument("--decisions", required=True)
    s.add_argument("--expectations", default=None)
    s.add_argument("--manifests-dir", default=None)
    s.add_argument("--save-to", default=None, help="write results.json and results.txt here; by default a report only prints and never touches the saved run")
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("score", help="seal a decision log, open the labels through the seal, and score")
    s.add_argument("--cases", required=True)
    s.add_argument("--decisions", required=True)
    s.add_argument("--labels", required=True)
    s.set_defaults(fn=cmd_score)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
