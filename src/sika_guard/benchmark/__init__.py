"""Benchmark tooling: schema, validation, leakage checks, sealed decision logs, scoring and a dev-only pilot.

Nothing in this package is part of inference. Modules that build or read model-visible data
(`visible`, `decisions`, `seal`) never import the hidden-label module (`hidden`); a test enforces it.
"""
