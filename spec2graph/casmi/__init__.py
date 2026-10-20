"""Enveda CASMI 2026 competition layer for Spec2Graph.

Everything in this subpackage is additive. It imports the core model from
:mod:`spectral_diffusion` and the data/eval utilities from the rest of
:mod:`spec2graph`, and never mutates either.

What the competition asks for, and how this layer answers it:

* **Input**: LC-MS/MS spectra with precursor m/z, adduct and collision
  energy, grouped 1-16 spectra per ``molecule_id``. No molecular formula.
* **Output**: up to 25 ranked SMILES per molecule.
* **Metric**: MRR@25 on tautomer-canonical InChIKey14 (:mod:`.metric`).

The layer turns the spectral-subspace model into a *scoring function* over
mass-matched candidate structures (:mod:`.pool`, :mod:`.scoring`), ranks
them (:mod:`.ranker`), and only falls back to de novo generation for
formulas no pool contains (:mod:`.denovo`). Conditioning on adduct,
collision energy, neutral mass and polarity is added by wrapping the core
model (:mod:`.conditioning`) rather than editing it.

Competition data is CC BY-NC 4.0 and must not be redistributed: keep the
parquet files and every derived cache out of version control.
"""
