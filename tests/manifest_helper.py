"""Record a development-baseline manifest of the tree as it is now, into a test's own directory, and return what a run binds to."""

from __future__ import annotations

from pathlib import Path

from sika_guard.benchmark import manifest as mf


def record_manifest(tmp_path: Path, name: str = "manifest-test.json", **kwargs) -> tuple[str, Path]:
    """(digest, directory). The directory is `tmp_path / "manifests"`, which is where the F11/R10 tests look for it."""
    directory = tmp_path / "manifests"
    manifest = mf.build_manifest(note="recorded by a test", **kwargs)
    mf.write_manifest(manifest, directory / name)
    return manifest["digest"], directory
