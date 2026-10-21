# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch>=2.6.0",
#     "rdkit==2026.3.3",
#     "numpy",
#     "pandas>=2.0.0",
#     "scipy>=1.10.0",
#     "pyarrow>=14.0.0",
#     "joblib>=1.3.0",
#     "tqdm>=4.65.0",
#     "matplotlib",
#     "seaborn",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Spec2Graph × CASMI 2026 — code explorer

    Walks real competition spectra through `spec2graph/casmi/` one stage at a time, with the real classes and functions:

    1. **Data** — `data/train.parquet` → `data.prepare` (column aliases, list-valued collision energies, neutral mass, molecule ids)
    2. **Adducts** — precursor m/z → neutral mass (`adducts.parse_adduct`)
    3. **Spectra** — peak cleaning (`data.Peaks`)
    4. **Spectral embedding** — graph Laplacian → eigenvectors → the invariant target $P_k = V_k V_k^\top$
    5. **Metric** — stereo-free, tautomer-canonical InChIKey14 (`chem.inchikey14`)
    6. **Candidate pool** — the library's structures plus same-formula isomers from the whole training set (`pool.CandidatePool`)
    7. **Model** — `CasmiModel` (core diffusion model + condition tokens) trained with `CasmiTrainer`
    8. **Heads, scoring, ranking, validation** — `Scorer`, `LinearRanker`, `Predictor`, `validation`
    9. **`data/test.parquet` → submission** — the Kaggle notebook's path, in miniature

    Default library: `enveda-np-examples`, 250 natural products on the test set's timsTOF, processed like the test spectra.
    """)
    return


@app.cell
def imports(mo):
    import os
    import sys
    from pathlib import Path

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    ROOT = mo.notebook_dir().parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.dataset as pds
    import pyarrow.parquet as pq
    import seaborn as sns
    import torch
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from torch.utils.data import DataLoader

    from spec2graph.casmi import chem, data, spectral, validation
    from spec2graph.casmi.adducts import AdductVocabulary, parse_adduct
    from spec2graph.casmi.conditioning import CasmiModel, CasmiModelConfig, CasmiTrainer, CasmiTrainerConfig
    from spec2graph.casmi.metric import Submission
    from spec2graph.casmi.pool import CandidatePool
    from spec2graph.casmi.predict import PredictConfig, Predictor
    from spec2graph.casmi.ranker import LinearRanker
    from spec2graph.casmi.scoring import Scorer, ScoringConfig
    from spectral_diffusion import Spec2GraphDiffusion, Spec2GraphDiffusionConfig, SpectralDataProcessor

    torch.backends.mha.set_fastpath_enabled(False)  # the encoder's eval fast path needs an op MPS lacks
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sns.set_theme(style="whitegrid", context="notebook")
    return (
        AdductVocabulary,
        CandidatePool,
        CasmiModel,
        CasmiModelConfig,
        CasmiTrainer,
        CasmiTrainerConfig,
        Chem,
        DEVICE,
        DataLoader,
        LinearRanker,
        Path,
        PredictConfig,
        Predictor,
        ROOT,
        Scorer,
        ScoringConfig,
        Spec2GraphDiffusion,
        Spec2GraphDiffusionConfig,
        SpectralDataProcessor,
        Submission,
        chem,
        data,
        np,
        os,
        parse_adduct,
        pc,
        pd,
        pds,
        plt,
        pq,
        rdMolDraw2D,
        sns,
        spectral,
        torch,
        validation,
    )


@app.cell
def constants(DEVICE, Path, ROOT, mo, os, torch):
    K = 8
    FP_BITS = 1024
    N_QUANTILES = 32
    MAX_ATOMS = 96
    MAX_PEAKS = 128
    N_JOBS = min(8, os.cpu_count() or 1)
    DATA_DIR = ROOT / "data"
    TRAIN_PARQUET, TEST_PARQUET = DATA_DIR / "train.parquet", DATA_DIR / "test.parquet"
    CACHE_DIR = Path.home() / ".cache" / "spec2graph-casmi-explorer"
    TEST_ADDUCTS = ("[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                    "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-")
    mo.md(f"Running on **{DEVICE}** with torch {torch.__version__}. Eigenvectors per molecule `K={K}`, fingerprint bits `{FP_BITS}`, spectral quantiles `{N_QUANTILES}`.")
    return (
        CACHE_DIR,
        DATA_DIR,
        FP_BITS,
        K,
        MAX_ATOMS,
        MAX_PEAKS,
        N_JOBS,
        N_QUANTILES,
        TEST_ADDUCTS,
        TEST_PARQUET,
        TRAIN_PARQUET,
    )


@app.cell
def mol_svg(Chem, mo, rdMolDraw2D):
    def mol_svg(smiles: str, size: tuple[int, int] = (300, 220)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return mo.md(f"`{smiles}` does not parse")
        drawer = rdMolDraw2D.MolDraw2DSVG(*size)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        return mo.Html(drawer.GetDrawingText())

    return (mol_svg,)


@app.cell
def train_index(TRAIN_PARQUET, mo, pq):
    mo.stop(not TRAIN_PARQUET.is_file(), mo.md(f"`{TRAIN_PARQUET}` not found: `uvx kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data`, then unzip it there."))
    _index = pq.read_table(TRAIN_PARQUET, columns=["ingest_lib", "inchikey14", "normalized_smiles", "molecular_formula"]).to_pandas()
    library_summary = (_index.groupby("ingest_lib").agg(spectra=("inchikey14", "size"), structures=("inchikey14", "nunique"))
                       .sort_values("spectra", ascending=False))
    train_structures = _index.drop_duplicates("inchikey14")[["inchikey14", "normalized_smiles", "molecular_formula"]].reset_index(drop=True)
    return library_summary, train_structures


@app.cell
def data_controls(library_summary, mo, train_structures):
    library_picker = mo.ui.dropdown(options=list(library_summary.index), value="enveda-np-examples", label="library", searchable=True)
    max_rows = mo.ui.number(start=100, stop=200_000, step=100, value=5_000, label="max spectra to load")
    mo.vstack([mo.md(f"""## 1 · Data

    `train.parquet` holds **{library_summary['spectra'].sum():,} spectra** of **{len(train_structures):,} structures** from {len(library_summary)} libraries.
    Only `enveda-180` and `enveda-np-examples` share the test set's timsTOF; `enveda-np-examples` is also natural products, like the test set."""),
        mo.ui.table(library_summary.reset_index(), selection=None, page_size=11), mo.hstack([library_picker, max_rows], justify="start")])
    return library_picker, max_rows


@app.cell
def load(TRAIN_PARQUET, data, library_picker, max_rows, pc, pds):
    raw_frame = pds.dataset(TRAIN_PARQUET).head(int(max_rows.value), filter=pc.field("ingest_lib") == library_picker.value).to_pandas()
    spectra_df = data.prepare(raw_frame, require_smiles=True)
    return raw_frame, spectra_df


@app.cell
def data_view(library_picker, mo, spectra_df):
    _cols = ["molecule_id", "adduct", "precursor_mz", "neutral_mass", "polarity", "collision_energy", "smiles"]
    mo.vstack([
        mo.md(f"""Loaded **{len(spectra_df):,} spectra** of **{spectra_df['molecule_id'].nunique():,} molecules** from `{library_picker.value}`
    ({spectra_df['adduct'].value_counts().to_dict()}).

    `data.prepare` maps the Kaggle names onto canonical ones (`ms2_mzs` → `mzs`, `normalized_smiles` → `smiles`, …),
    averages the list-valued `collision_energy_ev` (a merged acquisition like `[20, 50]` becomes 35), and derives `neutral_mass` and `polarity`
    from the adduct (NaN mass = unsupported adduct, dropped later). `train.parquet` has no `molecule_id`, so `prepare` groups each structure's
    spectra per library and splits them into molecules of at most 16 (`library:inchikey14:chunk`), the test set's largest."""),
        mo.ui.table(spectra_df[_cols].assign(n_peaks=spectra_df["mzs"].map(len)).round(4), page_size=8, selection=None),
    ])
    return


@app.cell
def adduct_controls(mo):
    precursor_mz_input = mo.ui.number(start=50.0, stop=2000.0, step=0.0001, value=300.0, label="precursor m/z")
    mo.vstack([mo.md("""## 2 · Adducts → neutral mass

    Every adduct is `[nM ± components]z±`: ion mass = n·M + shift − z·mₑ. The ten adducts in the hidden test set:"""), precursor_mz_input])
    return (precursor_mz_input,)


@app.cell
def adduct_table(TEST_ADDUCTS, mo, parse_adduct, pd, precursor_mz_input):
    _rows = []
    for _name in TEST_ADDUCTS:
        _adduct = parse_adduct(_name)
        _rows.append({"adduct": _name, "canonical": _adduct.name, "charge": _adduct.charge, "shift (Da)": _adduct.shift,
                      "neutral mass (Da)": _adduct.neutral_mass(precursor_mz_input.value)})
    mo.ui.table(pd.DataFrame(_rows).round(5), selection=None)
    return


@app.cell
def molecule_controls(mo, spectra_df):
    _ids = list(dict.fromkeys(spectra_df["molecule_id"]))
    molecule_picker = mo.ui.dropdown(options=_ids, value=_ids[0], label="molecule", searchable=True)
    mo.vstack([mo.md("## 3 · One molecule and its spectra\n\nEverything below follows this molecule. Predictions are per molecule, aggregated over all of its spectra."), molecule_picker])
    return (molecule_picker,)


@app.cell
def molecule(chem, mo, mol_svg, molecule_picker, spectra_df):
    mol_rows = spectra_df[spectra_df["molecule_id"] == molecule_picker.value].reset_index(drop=True)
    mol_smiles = mol_rows["smiles"].iloc[0]
    mol_record = chem.record(mol_smiles)
    mo.hstack([mol_svg(mol_smiles), mo.md(f"""
    **{molecule_picker.value}** · `{mol_smiles}`

    | | |
    |---|---|
    | formula | {mol_record['formula']} |
    | exact mass | {mol_record['exact_mass']:.5f} |
    | heavy atoms | {mol_record['n_heavy_atoms']} |
    | InChIKey14 | `{mol_record['inchikey14']}` |
    | spectra | {len(mol_rows)} ({', '.join(mol_rows['adduct'])}) |
    """)], justify="start", gap=2)
    return mol_record, mol_rows, mol_smiles


@app.cell
def peak_controls(MAX_PEAKS, mo):
    top_k_slider = mo.ui.slider(4, MAX_PEAKS, value=MAX_PEAKS, step=4, label="keep top-k peaks", show_value=True)
    sqrt_switch = mo.ui.switch(label="sqrt intensity")
    mo.vstack([mo.md("""`data.Peaks.clean` drops non-finite and zero peaks and the precursor echo (anything within 0.5 Da of the precursor or above it),
    keeps the top-k by intensity, sorts by m/z and rescales to a base peak of 1. Grey = raw, blue = what the model sees.
    These settings are the `peaks` used by training and prediction below, so changing them means retraining."""),
               mo.hstack([top_k_slider, sqrt_switch], justify="start")])
    return sqrt_switch, top_k_slider


@app.cell
def peaks_plot(data, mol_rows, np, plt, sqrt_switch, top_k_slider):
    peaks = data.Peaks(top_k=top_k_slider.value, sqrt_intensity=sqrt_switch.value)
    _records = mol_rows.to_dict("records")[:4]
    _fig, _axes = plt.subplots(len(_records), 1, figsize=(9, 2.1 * len(_records)), squeeze=False, sharex=True)
    for _ax, _row in zip(_axes[:, 0], _records):
        _raw_mz, _raw_it = np.asarray(_row["mzs"], float), np.asarray(_row["intensities"], float)
        _mz, _it = peaks.from_row(_row)
        _ax.vlines(_raw_mz, 0, _raw_it / _raw_it.max(), color="0.8", lw=2.5, label="raw")
        _ax.vlines(_mz, 0, _it, color="C0", lw=1.2, label="kept")
        _ax.axvline(_row["precursor_mz"], color="C3", ls="--", lw=1, label="precursor")
        _ce = "n/a" if np.isnan(_row["collision_energy"]) else f"{_row['collision_energy']:.0f} eV"
        _ax.set_title(f"{_row['adduct']} · CE {_ce} · {len(_raw_mz)} → {len(_mz)} peaks", fontsize=9, loc="left")
    _axes[0, 0].legend(fontsize=8, loc="upper left")
    _axes[-1, 0].set_xlabel("m/z")
    _fig.tight_layout()
    _fig
    return (peaks,)


@app.cell
def spectral_controls(K, mo):
    k_slider = mo.ui.slider(1, 16, value=K, label="k eigenvectors", show_value=True)
    mo.vstack([mo.md(r"""## 4 · Structure → spectral embedding

    `SpectralDataProcessor` turns the heavy-atom graph into the normalised Laplacian $L = I - D^{-1/2} A D^{-1/2}$ and keeps $k$ eigenvectors $V_k$.
    The diffusion model generates $V_k$, but it is trained against $P_k = V_k V_k^\top$. The candidate pool stores the eigenvalue spectrum as quantiles."""), k_slider])
    return (k_slider,)


@app.cell
def spectral_embedding(
    N_QUANTILES,
    SpectralDataProcessor,
    k_slider,
    mol_smiles,
    np,
    plt,
    sns,
    spectral,
):
    processor = SpectralDataProcessor(k=k_slider.value)
    adjacency = processor.smiles_to_adjacency(mol_smiles)
    eigvecs = processor.extract_eigenvectors(processor.compute_laplacian(adjacency))
    projection = processor.projection_matrix(eigvecs)
    eigvals = spectral.eigenvalues(adjacency, processor)
    _quantiles = spectral.quantiles(eigvals, N_QUANTILES)
    _fig, _axes = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1, 0.55, 1, 1.2]})
    _heat = dict(cbar=False, xticklabels=False, yticklabels=False)
    sns.heatmap(adjacency, ax=_axes[0], cmap="Greys", square=True, **_heat)
    sns.heatmap(eigvecs, ax=_axes[1], cmap="vlag", center=0, **_heat)
    sns.heatmap(projection, ax=_axes[2], cmap="vlag", center=0, square=True, **_heat)
    _axes[3].plot(eigvals, "o", ms=4, label="eigenvalues")
    _axes[3].plot(np.linspace(0, len(eigvals) - 1, N_QUANTILES), _quantiles, lw=1.2, label=f"{N_QUANTILES} quantiles")
    _axes[3].legend(fontsize=8)
    for _ax, _title in zip(_axes, ["adjacency A", f"V_k  ({eigvecs.shape[0]}×{eigvecs.shape[1]})", "P_k = V_k V_kᵀ", "Laplacian spectrum in [0, 2]"]):
        _ax.set_title(_title, fontsize=10)
    _fig.tight_layout()
    _fig
    return adjacency, eigvals, eigvecs, processor, projection


@app.cell
def invariance(
    adjacency,
    eigvals,
    eigvecs,
    mo,
    np,
    processor,
    projection,
    spectral,
):
    _rng = np.random.default_rng(0)
    _k = eigvecs.shape[1]
    _rotation, _ = np.linalg.qr(_rng.normal(size=(_k, _k)))
    _signs = _rng.choice([-1.0, 1.0], _k)
    _perm = _rng.permutation(len(adjacency))
    _change = lambda v: (np.abs(v - eigvecs).max(), np.abs(processor.projection_matrix(v) - projection).max())
    _flip, _rotate = _change(eigvecs * _signs), _change(eigvecs @ _rotation)
    _relabelled = np.abs(spectral.eigenvalues(adjacency[np.ix_(_perm, _perm)], processor) - eigvals).max()
    mo.md(f"""
    ### Why train on $P_k$ and not on $V_k$

    | transform | max change in $V_k$ | max change in $P_k$ |
    |---|---|---|
    | flip signs $V_k\\,\\mathrm{{diag}}(\\pm1)$ | {_flip[0]:.3f} | {_flip[1]:.1e} |
    | rotate $V_k Q$ (degenerate eigenvalues allow this) | {_rotate[0]:.3f} | {_rotate[1]:.1e} |

    Eigenvectors are only defined up to sign, and up to rotation inside a repeated eigenvalue, so a loss on $V_k$ would punish correct answers.
    $P_k$ is the same for all of them. The spectrum is also unchanged by relabelling atoms (max change {_relabelled:.1e}),
    which is why the pool can compare it with any candidate without an atom correspondence.
    """)
    return


@app.cell
def metric_controls(mo):
    answer_input = mo.ui.text(value="C=C(C)C1CCC2CCC(=O)C=C2C1", label="answer", full_width=True)
    guess_input = mo.ui.text(value="C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1", label="guess", full_width=True)
    mo.vstack([mo.md("""## 5 · How a guess is scored

    MRR@25: a molecule scores 1/rank of the first guess whose InChIKey14 matches the answer. `chem.inchikey14` ports the metric notebook:
    remove stereochemistry, canonicalise the tautomer, keep the first InChIKey block. The defaults are the metric notebook's regression case:
    a stereocentre pins the enone double bond, so without stripping stereo first the two would canonicalise to different tautomers."""), answer_input, guess_input])
    return answer_input, guess_input


@app.cell
def metric_view(answer_input, chem, guess_input, mo, mol_svg):
    _keys = {name: chem.inchikey14(s) for name, s in (("answer", answer_input.value), ("guess", guess_input.value))}
    _match = bool(_keys["answer"]) and _keys["answer"] == _keys["guess"]
    mo.vstack([
        mo.hstack([mol_svg(answer_input.value, (260, 180)), mol_svg(guess_input.value, (260, 180))], justify="start"),
        mo.md(f"""| | InChIKey14 |
    |---|---|
    | answer | `{_keys['answer']}` |
    | guess | `{_keys['guess']}` |

    **{'Match — this guess at rank r scores 1/r.' if _match else 'No match — scores 0.'}**"""),
    ])
    return


@app.cell
def pool_controls(mo):
    decoys_slider = mo.ui.slider(0, 200, value=30, step=10, label="isomer decoys per formula", show_value=True)
    ppm_slider = mo.ui.slider(1, 50, value=10, label="mass window (ppm)", show_value=True)
    mo.vstack([mo.md("""## 6 · Candidate pool

    In the competition the pool is every training structure plus COCONUT/PubChem, and retrieval is a `searchsorted` mass window. Here it is the
    loaded structures plus same-formula isomers drawn from the whole training set, the hardest decoys there are."""), mo.hstack([decoys_slider, ppm_slider], justify="start", gap=1.5)])
    return decoys_slider, ppm_slider


@app.cell
def pool(
    CandidatePool,
    FP_BITS,
    K,
    N_JOBS,
    N_QUANTILES,
    data,
    decoys_slider,
    mo,
    raw_frame,
    spectra_df,
    train_structures,
):
    _own = set(raw_frame["inchikey14"])
    _isomers = train_structures[train_structures["molecular_formula"].isin(set(raw_frame["molecular_formula"])) & ~train_structures["inchikey14"].isin(_own)]
    _decoys = _isomers.groupby("molecular_formula").head(decoys_slider.value)["normalized_smiles"]
    with mo.status.spinner(title=f"Featurising {len(_own) + len(_decoys):,} structures…"):
        structure_table = data.structures(list(spectra_df["smiles"].unique()) + list(_decoys), n_jobs=N_JOBS)
        pool = CandidatePool.build(structure_table, K, FP_BITS, N_QUANTILES, n_jobs=N_JOBS)
    mo.md(f"""The pool has **{len(pool):,} structures**: the {len(_own)} loaded ones plus {len(_decoys):,} isomers (same formula, so the same exact mass)
    from elsewhere in `train.parquet`. `data.structures` collapses SMILES to one row per InChIKey14, keeping every spelling as `aliases`;
    `CandidatePool.build` sorts by exact mass and precomputes eigenvectors, elements, spectral quantiles and packed fingerprints.""")
    return pool, structure_table


@app.cell
def window(mo, mol_record, mol_rows, molecule_picker, pd, pool, ppm_slider):
    _mass = float(mol_rows["neutral_mass"].median())
    _idx = pool.query_mass(_mass, ppm_slider.value, min_abs_da=0.002)
    _window = pd.DataFrame({"smiles": pool.smiles[_idx], "formula": pool.formula[_idx], "exact_mass": pool.exact_mass[_idx],
                            "ppm": pool.ppm_error(_mass, _idx), "answer": pool.inchikey14[_idx] == mol_record["inchikey14"]})
    mo.vstack([mo.md(f"Median neutral mass of {molecule_picker.value}'s spectra: **{_mass:.5f} Da** → **{len(_idx)} candidates**. "
                     "Isomers share an exact mass, so the window alone cannot separate them; that is the model's job."),
               mo.ui.table(_window.round(5), selection=None)])
    return


@app.cell
def train_controls(DEVICE, mo):
    d_model_pick = mo.ui.dropdown(options={"64": 64, "128": 128, "256": 256}, value="128", label="d_model")
    epochs_slider = mo.ui.slider(1, 200, value=30, label="epochs", show_value=True)
    heads_only_switch = mo.ui.switch(label="heads-only proxy (no denoiser)")
    train_button = mo.ui.run_button(label=f"Train on {DEVICE}")
    mo.vstack([mo.md("""## 7 · Model and training

    `CasmiModel` wraps the core `Spec2GraphDiffusion`: three condition tokens (adduct + polarity, collision energy, neutral mass) are prepended
    to the encoded peaks and handed to the core decoder as `memory`. The heads predict a fingerprint, the heavy-atom count, atom types and the
    Laplacian spectral quantiles; the denoiser learns to generate $V_k$, scored through $P_k$. Heads-only trains everything except the denoiser."""),
        mo.hstack([d_model_pick, epochs_slider, heads_only_switch, train_button], justify="start", gap=1.5)])
    return d_model_pick, epochs_slider, heads_only_switch, train_button


@app.cell
def dataset(
    AdductVocabulary,
    CACHE_DIR,
    FP_BITS,
    K,
    MAX_ATOMS,
    N_QUANTILES,
    data,
    mo,
    peaks,
    spectra_df,
    spectral,
    structure_table,
):
    vocab = AdductVocabulary.fit(spectra_df["adduct"])
    dataset = data.Dataset(spectra_df, spectral.SpectralCache(CACHE_DIR, k=K, fingerprint_bits=FP_BITS), vocab,
                           max_atoms=MAX_ATOMS, n_quantiles=N_QUANTILES, peaks=peaks, table=structure_table)
    mo.md(f"`data.Dataset` kept **{len(dataset):,}** of {len(spectra_df):,} spectra (dropped: {dataset.drops or 'none'}); "
          f"the adduct vocabulary has {len(vocab)} entries (index 0 = unknown).")
    return dataset, vocab


@app.cell
def train(
    CasmiModel,
    CasmiModelConfig,
    CasmiTrainer,
    CasmiTrainerConfig,
    DEVICE,
    DataLoader,
    FP_BITS,
    K,
    MAX_ATOMS,
    MAX_PEAKS,
    N_QUANTILES,
    Spec2GraphDiffusion,
    Spec2GraphDiffusionConfig,
    d_model_pick,
    data,
    dataset,
    epochs_slider,
    heads_only_switch,
    mo,
    pd,
    torch,
    train_button,
    vocab,
):
    mo.stop(not train_button.value, mo.md("Press **Train** to fit a model. Sections 8–10 need it."))
    torch.manual_seed(0)
    _d = d_model_pick.value
    _core = Spec2GraphDiffusion(Spec2GraphDiffusionConfig(
        d_model=_d, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=4 * _d, k=K, max_atoms=MAX_ATOMS,
        max_peaks=MAX_PEAKS, fingerprint_dim=FP_BITS, enable_atom_count_head=True, enable_atom_type_head=True))
    trainer = CasmiTrainer(
        CasmiModel(_core, CasmiModelConfig(n_adducts=len(vocab), n_quantiles=N_QUANTILES)),
        CasmiTrainerConfig(fingerprint_loss_weight=0.5, atom_count_loss_weight=0.05, atom_type_loss_weight=0.5, heads_only=heads_only_switch.value),
        DEVICE)
    _optimiser = torch.optim.AdamW(trainer.wrapper.parameters(), lr=3e-4)
    _loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=data.Collator(MAX_ATOMS, MAX_PEAKS, K, FP_BITS))
    _log = []
    for _epoch in mo.status.progress_bar(range(epochs_slider.value), title=f"training on {DEVICE}", show_eta=True):
        for _batch in _loader:
            _loss, _parts = trainer.train_step(_optimiser, _batch.to(DEVICE), return_components=True, grad_clip=1.0)
            _log.append({"step": len(_log), "total": _loss, **_parts})
    loss_history = pd.DataFrame(_log)
    return loss_history, trainer


@app.cell
def loss_plot(loss_history, plt, sns):
    _long = loss_history.melt(id_vars="step", var_name="loss", value_name="value")
    _long["smoothed"] = _long.groupby("loss")["value"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    sns.lineplot(data=_long, x="step", y="smoothed", hue="loss", ax=_ax)
    _ax.set(yscale="log", ylabel="loss (rolling mean of 10)", title=f"{len(loss_history)} steps")
    _fig.tight_layout()
    _fig
    return


@app.cell
def heads(
    DEVICE,
    MAX_PEAKS,
    N_QUANTILES,
    data,
    eigvals,
    mo,
    mol_record,
    mol_rows,
    molecule_picker,
    np,
    peaks,
    plt,
    pool,
    sns,
    spectral,
    trainer,
    vocab,
):
    mol_spectra = data.Spectra.from_rows(mol_rows, vocab, MAX_PEAKS, peaks, device=DEVICE)
    mo.stop(mol_spectra is None, mo.md("This molecule has no usable spectra (unsupported adduct or too few peaks)."))
    heads = trainer.heads(mol_spectra)
    _prob = heads["fingerprint_prob"].mean(0).cpu().numpy()
    _answer = pool.index_of(mol_record["inchikey14"])
    _bits = pool.fingerprints([_answer])[0].astype(bool) if _answer is not None else np.zeros(_prob.shape, bool)
    _predicted_q = heads["quantiles"].cpu().numpy()
    _counts = heads["atom_count"].cpu().numpy()
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for _q in _predicted_q:
        _axes[0].plot(_q, color="C0", alpha=0.35, lw=1)
    _axes[0].plot(_predicted_q.mean(0), color="C0", lw=2, label="predicted (mean over spectra)")
    _axes[0].plot(spectral.quantiles(eigvals, N_QUANTILES), color="k", lw=2, ls="--", label="true")
    _axes[0].set(title="spectral quantiles", xlabel="quantile index")
    _axes[0].legend(fontsize=8)
    sns.stripplot(x=np.where(_bits, "bit set", "bit unset"), y=_prob, ax=_axes[1], size=2.5, alpha=0.6, order=["bit set", "bit unset"])
    _axes[1].set(title="fingerprint probabilities", ylabel="predicted P(bit)")
    _axes[2].bar([f"spectrum {i + 1}" for i in range(len(_counts))], _counts, color="C0")
    _axes[2].axhline(mol_record["n_heavy_atoms"], color="k", ls="--", label="true")
    _axes[2].set(title="heavy-atom count")
    _axes[2].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(f"""## 8 · What the heads predict for {molecule_picker.value}

    `trainer.heads` encodes each spectrum once (the conditioned `memory`) and reads every head from it. At prediction time the per-spectrum outputs
    are averaged. A good fingerprint head separates the two strips; a good quantile head tracks the dashed line."""), _fig])
    return (mol_spectra,)


@app.cell
def scoring_controls(mo):
    n_samples_slider = mo.ui.slider(1, 16, value=4, label="eigenvector draws per spectrum", show_value=True)
    n_steps_slider = mo.ui.slider(5, 100, value=20, step=5, label="DDIM steps", show_value=True)
    projection_switch = mo.ui.switch(value=True, label="projection overlap (samples the denoiser)")
    mo.vstack([mo.md("""## 9 · Scoring and ranking candidates

    `Scorer.score` turns the heads into features for every candidate in the mass window: fingerprint log-likelihood and soft Tanimoto,
    quantile distance, atom-count error and, with sampling on, the projection overlap $\\max_\\Pi \\langle \\Pi^\\top \\hat P \\Pi, P_c \\rangle$
    (Sinkhorn alignment seeded by the atom-type head) plus an element log-likelihood. `LinearRanker` combines them; before it is fitted on
    held-out molecules it uses hand-set weights."""), mo.hstack([n_samples_slider, n_steps_slider, projection_switch], justify="start", gap=1.5)])
    return n_samples_slider, n_steps_slider, projection_switch


@app.cell
def scoring_config(
    ScoringConfig,
    heads_only_switch,
    n_samples_slider,
    n_steps_slider,
    ppm_slider,
    projection_switch,
):
    scoring_config = ScoringConfig(ppm=ppm_slider.value, n_samples=n_samples_slider.value, n_steps=n_steps_slider.value,
                                   use_projection=projection_switch.value and not heads_only_switch.value)
    return (scoring_config,)


@app.cell
def predictor(
    MAX_PEAKS,
    PredictConfig,
    Predictor,
    peaks,
    pool,
    scoring_config,
    trainer,
    vocab,
):
    predictor = Predictor(trainer, pool, vocab, config=PredictConfig(scoring=scoring_config, peaks=peaks, max_peaks=MAX_PEAKS))
    return (predictor,)


@app.cell
def scores(
    LinearRanker,
    Scorer,
    mo,
    mol_record,
    mol_spectra,
    molecule_picker,
    pool,
    scoring_config,
    trainer,
):
    _scores = Scorer(trainer, pool, scoring_config).score(mol_spectra, molecule_picker.value)
    _ranked = (_scores.frame(pool)
               .assign(ranker_score=LinearRanker.heuristic().score(_scores.features),
                       answer=lambda f: f["inchikey14"] == mol_record["inchikey14"])
               .sort_values("ranker_score", ascending=False).reset_index(drop=True))
    _hit = _ranked.index[_ranked["answer"]]
    _verdict = f"The answer ranks **#{_hit[0] + 1}** of {len(_ranked)} (reciprocal rank {1 / (_hit[0] + 1):.2f})." if len(_hit) else "The answer is not in the window."
    mo.vstack([mo.md(_verdict + (" Projection features are off (heads-only model or switch)." if not scoring_config.use_projection else "")),
               mo.ui.table(_ranked.drop(columns="inchikey14").round(4), page_size=10, selection=None)])
    return


@app.cell
def predict_controls(mo, spectra_df):
    _n = spectra_df["molecule_id"].nunique()
    n_molecules_slider = mo.ui.slider(1, _n, value=min(_n, 30), label="molecules", show_value=True)
    predict_button = mo.ui.run_button(label="Predict")
    mo.vstack([mo.md("""## 10 · Predict and validate

    `Predictor.predict` scores and ranks each molecule, tops up to 25 guesses with the nearest pool masses and falls back per molecule so one
    failure never zeroes the file. `validation.evaluate` reports MRR@25 with a bootstrap interval and window recall. With the answers removed from the
    pool (competition class 3) only de novo generation could score, and this explorer runs without it, so that case is left out."""),
        mo.hstack([n_molecules_slider, predict_button], justify="start", gap=1.5)])
    return n_molecules_slider, predict_button


@app.cell
def predict(
    mo,
    n_molecules_slider,
    pd,
    pool,
    predict_button,
    predictor,
    spectra_df,
    validation,
):
    mo.stop(not predict_button.value, mo.md("Press **Predict** to run the full pipeline on the first N molecules."))
    _ids = list(dict.fromkeys(spectra_df["molecule_id"]))[: n_molecules_slider.value]
    _subset = spectra_df[spectra_df["molecule_id"].isin(_ids)].reset_index(drop=True)
    with mo.status.spinner(title=f"Predicting {len(_ids)} molecules…"):
        _predictions, _scores = predictor.predict_frame(_subset)
        reports = [validation.evaluate("known_structure", _predictions, validation.truth_map(_subset), _scores, pool)]
    mo.vstack([
        mo.md("These molecules were in the training data, so the known-structure score is in-sample and optimistic."),
        mo.ui.table(pd.DataFrame([vars(r) for r in reports]).round(4), selection=None),
    ])
    return


@app.cell
def test_controls(TEST_PARQUET, data, mo):
    test_df = data.load(TEST_PARQUET)
    _n = test_df["molecule_id"].nunique()
    test_limit = mo.ui.slider(1, _n, value=min(_n, 40), label="test molecules", show_value=True)
    test_button = mo.ui.run_button(label="Predict test.parquet")
    mo.vstack([mo.md(f"""## 11 · `test.parquet` → submission

    **{len(test_df):,} spectra** of **{_n} molecules**, all timsTOF ({test_df['adduct'].value_counts().to_dict()}). Kaggle swaps in the hidden test
    set (about the same size) when the notebook is re-run. This is `casmi_predict`'s path; the guesses come from this explorer's small pool,
    so expect them to be wrong: the real pool comes from `casmi_prepare`."""), mo.hstack([test_limit, test_button], justify="start", gap=1.5)])
    return test_button, test_df, test_limit


@app.cell
def test_predict(
    DATA_DIR,
    Submission,
    mo,
    pd,
    predictor,
    test_button,
    test_df,
    test_limit,
):
    mo.stop(not test_button.value, mo.md("Press **Predict test.parquet** to build submission rows."))
    _ids = list(dict.fromkeys(test_df["molecule_id"]))[: test_limit.value]
    with mo.status.spinner(title=f"Predicting {len(_ids)} test molecules…"):
        _predictions, _ = predictor.predict_frame(test_df[test_df["molecule_id"].isin(_ids)].reset_index(drop=True))
    test_submission = Submission(_predictions, _ids).frame()
    _all_ids = list(dict.fromkeys(test_df["molecule_id"]))
    _problems = Submission.problems(test_submission, _all_ids)
    _sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    mo.vstack([
        mo.md(f"Kaggle's rejection rules against all {len(_all_ids)} test molecules: **{'; '.join(_problems) if _problems else 'passes'}**"
              + (f" (a preview of {len(_ids)}; a submission must cover every molecule)" if len(_ids) < len(_all_ids) else "")
              + f". `sample_submission.csv` has {len(_sample)} rows "
              f"and columns {list(_sample.columns)}. `Submission(...).write('submission.csv')` writes the file."),
        mo.ui.table(test_submission, page_size=8, selection=None),
    ])
    return


@app.cell(hide_code=True)
def next(mo):
    mo.md(r"""
    ## Running it for real

    The same classes drive the CLI (`docs/17_casmi_competition.md`):

    ```bash
    python -m spec2graph.scripts.casmi_prepare  --train-parquet data/train.parquet --cache-dir ~/.cache/spec2graph-casmi --output-dir assets/casmi
    python -m spec2graph.scripts.casmi_train    --train-parquet data/train.parquet --assets-dir assets/casmi --cache-dir ~/.cache/spec2graph-casmi --output-dir runs/casmi --device mps
    python -m spec2graph.scripts.casmi_validate --checkpoint runs/casmi/epoch_010.pt --pool assets/casmi/pool.npz --train-parquet data/train.parquet --holdout-keys runs/casmi/holdout_keys.json --fit-ranker runs/casmi/ranker.json --output runs/casmi/validation.json
    python -m spec2graph.scripts.casmi_predict  --checkpoint runs/casmi/epoch_010.pt --pool assets/casmi/pool.npz --ranker runs/casmi/ranker.json --test-parquet data/test.parquet --output submission.csv
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
