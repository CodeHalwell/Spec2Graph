"""Spec2Graph extensions: MassSpecGym data pipeline and evaluation harness.

The core model stays in :mod:`spectral_diffusion`. Everything in this package
is additive — data loading, caching, evaluation and CLI scripts — and must
not mutate the core module.
"""
