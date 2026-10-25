"""Binding an experiment run to the manifest that says what was run.

A run log that merely quotes a manifest digest proves nothing: the digest may name no manifest, or a manifest of different code. So a
run is bound in three ways, checked BEFORE any inference and again when the run is sealed:

  * the declared digest must be that of a recorded, untampered manifest file (`locate_manifest`);
  * that manifest must describe the code that is running: the frozen core, every prompt and the decision policy are compared with the
    working tree now (`verify_manifest`), and the hash of the frozen core is recorded in the declaration so a file edited MID-RUN is caught
    at sealing;
  * the conditions the run declares must be defined, byte for byte, as the manifest defines them (`conditions_digest`).

A live run additionally needs a manifest with `status == "frozen"`: a manifest that someone approved after the tree was clean. A
development baseline may bind only a mock or fake run, which is plumbing and says nothing about a model.

What this is and is not. It makes a finished run's provenance checkable and stops an unfinished, mislabelled or edited-mid-run experiment
from opening labels. It is not authentication: someone who can rewrite the manifest directory, the log and the seal consistently can still
forge all three. That needs signatures made outside this repository, and nothing here pretends otherwise.

Must not import the hidden-label module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sika_guard.benchmark import manifest as mf

FROZEN = "frozen"
DEVELOPMENT_BASELINE = "development_baseline"


class ManifestBindingError(RuntimeError):
    """The run is not bound to a manifest that describes it."""


@dataclass(frozen=True)
class Binding:
    manifest_file: str
    manifest_digest: str
    status: str
    frozen_core_sha256: str
    conditions_sha256: str


def default_manifest_dir() -> Path:
    return mf.repo_root() / "manifests"


def status_of(manifest: dict[str, Any]) -> str:
    """Manifests recorded before the status field existed are development baselines: none of them was ever approved as a freeze."""
    return manifest.get("status", DEVELOPMENT_BASELINE)


def locate_manifest(digest: str, manifests_dir: Path | None = None) -> tuple[Path, dict[str, Any]]:
    directory = manifests_dir or default_manifest_dir()
    if not directory.is_dir():
        raise ManifestBindingError(f"there is no manifest directory at {directory.name}")
    for path in sorted(directory.glob("*.json")):
        try:
            manifest = mf.load_manifest(path)  # refuses a manifest whose digest does not match its content
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        if manifest.get("digest") == digest:
            return path, manifest
    raise ManifestBindingError("the declared manifest digest names no recorded, untampered manifest")


def _digest(items: Any) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def frozen_core_digest(manifest: dict[str, Any]) -> str:
    """A single hash over the (path, sha256) of every frozen-core file the manifest records."""
    return _digest(sorted((rel, e["sha256"]) for rel, e in manifest["files"].items() if e["group"] == "frozen_core"))


def current_frozen_core_digest(root: Path | None = None) -> str:
    return frozen_core_digest({"files": mf.file_hashes(root or mf.repo_root())})


def conditions_digest(manifest: dict[str, Any], conditions: Sequence[str]) -> str:
    missing = [c for c in conditions if c not in manifest["conditions"]]
    if missing:
        raise ManifestBindingError(f"the manifest does not define condition(s) {missing}")
    return _digest({c: manifest["conditions"][c] for c in sorted(conditions)})


def bind(manifest_digest: str, *, mode: str, conditions: Sequence[str], manifests_dir: Path | None = None, root: Path | None = None) -> Binding:
    """Check a run's manifest before it starts. Raises `ManifestBindingError`; nothing may be sent to a provider if it does."""
    path, manifest = locate_manifest(manifest_digest, manifests_dir)
    status = status_of(manifest)
    if mode == "live" and status != FROZEN:
        raise ManifestBindingError(f"a live run needs a frozen, approved manifest; {path.name} is a {status}")
    drift = mf.verify_manifest(manifest, root)
    if drift.frozen_core_drifted:
        raise ManifestBindingError(f"{path.name} does not describe the code that is running (the frozen core, a prompt or the decision policy changed):\n{drift.describe()}")
    return Binding(path.name, manifest_digest, status, frozen_core_digest(manifest), conditions_digest(manifest, conditions))


def check_recorded(*, manifest_digest: str, status: str, frozen_core_sha256: str, conditions_sha256: str, mode: str,
                   conditions: Sequence[str], manifests_dir: Path | None = None, root: Path | None = None, still_running: bool = False) -> None:
    """Everything a declaration (or a seal) claims about its manifest must be what the manifest says. With `still_running` (at sealing)
    the working tree must also still be the code the run started with; later, at scoring, the tree may legitimately have moved on."""
    path, manifest = locate_manifest(manifest_digest, manifests_dir)
    if status_of(manifest) != status:
        raise ManifestBindingError(f"the declared manifest status {status!r} is not {path.name}'s ({status_of(manifest)!r})")
    if mode == "live" and status != FROZEN:
        raise ManifestBindingError("a live run is bound to a frozen manifest only")
    if frozen_core_digest(manifest) != frozen_core_sha256:
        raise ManifestBindingError("the declared frozen-core hash is not the manifest's")
    if conditions_digest(manifest, conditions) != conditions_sha256:
        raise ManifestBindingError("the declared condition definitions are not the manifest's")
    if still_running and current_frozen_core_digest(root) != frozen_core_sha256:
        raise ManifestBindingError("the frozen core changed while the run was in progress")
