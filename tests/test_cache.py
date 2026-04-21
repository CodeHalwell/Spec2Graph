"""Tests for :mod:`spec2graph.data.cache`.

These tests compute real eigenvectors (via RDKit) so they require the
full environment. None of them hit the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")

from spec2graph.data.cache import EigenvectorCache


# Small molecules whose eigenvectors compute reliably and quickly.
ETHANOL_SMILES = "CCO"
ETHANOL_INCHIKEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
BENZENE_SMILES = "c1ccccc1"
BENZENE_INCHIKEY = "UHOVQNZJYSORNB-UHFFFAOYSA-N"


@pytest.fixture
def cache(tmp_path: Path) -> EigenvectorCache:
    return EigenvectorCache(cache_dir=tmp_path, k=4, bond_weighting="order")


class TestEigvecCache:
    def test_miss_then_hit_returns_identical_array(self, cache):
        first = cache.get_or_compute(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        assert first is not None
        assert first.shape == (3, 4)  # 3 heavy atoms, k=4 eigenvectors
        assert first.dtype == np.float32

        # Second call reads from disk and must be bit-for-bit identical.
        second = cache.get_or_compute(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        assert second is not None
        np.testing.assert_array_equal(first, second)

    def test_cached_file_is_a_valid_numpy_array(self, cache):
        """Regression: _atomic_save must promote the real .npy file.

        A previous bug moved the empty mkstemp placeholder into place
        instead of the array written by np.save, leaving an empty cache
        file that the corruption-recovery branch silently re-filled on
        every read. Loading the file directly catches that class of bug.
        """
        expected = cache.get_or_compute(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        path = cache.eigvec_path(ETHANOL_INCHIKEY)
        assert path.stat().st_size > 0, "cache file is empty"
        loaded = np.load(path, allow_pickle=False)
        np.testing.assert_array_equal(loaded, expected)

    def test_cache_hit_does_not_recompute(self, cache, monkeypatch):
        """A cache hit must read from disk, not invoke the processor.

        If the cache silently recomputes on every call (as it did when
        _atomic_save wrote empty files), this test catches it by making
        any recomputation attempt raise.
        """
        cache.get_or_compute(ETHANOL_SMILES, ETHANOL_INCHIKEY)

        def _refuse(*args, **kwargs):
            raise RuntimeError("processor must not be called on cache hit")

        monkeypatch.setattr(cache._processor, "process_smiles", _refuse)
        # This must succeed by reading from disk; the monkeypatched
        # processor would raise if we tried to recompute.
        result = cache.get_or_compute(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        assert result is not None

    def test_cache_miss_creates_sharded_file(self, cache):
        cache.get_or_compute(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        expected_path = (
            cache.eigvec_root / ETHANOL_INCHIKEY[:2].upper() / f"{ETHANOL_INCHIKEY}.npy"
        )
        assert expected_path.exists()

    def test_invalid_smiles_returns_none(self, cache, caplog):
        caplog.set_level("WARNING")
        result = cache.get_or_compute("not-a-smiles", "INVALIDKEY0001-UHFFFAOYSA-N")
        assert result is None
        # Make sure no cache file was created — we do not want to cache failures.
        expected = cache.eigvec_path("INVALIDKEY0001-UHFFFAOYSA-N")
        assert not expected.exists()

    def test_corrupted_file_is_detected_and_recomputed(self, cache, caplog):
        first = cache.get_or_compute(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        assert first is not None
        path = cache.eigvec_path(ETHANOL_INCHIKEY)

        # Overwrite the cached file with garbage.
        path.write_bytes(b"this is not a numpy file at all")

        caplog.set_level("WARNING")
        recomputed = cache.get_or_compute(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        assert recomputed is not None
        np.testing.assert_array_equal(first, recomputed)
        assert any("corrupted" in message.lower() for message in caplog.messages)

    def test_different_molecules_get_different_shards(self, cache):
        cache.get_or_compute(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        cache.get_or_compute(BENZENE_SMILES, BENZENE_INCHIKEY)

        # InChIKeys starting with different prefixes go in different shards.
        ethanol_shard = cache.eigvec_root / ETHANOL_INCHIKEY[:2].upper()
        benzene_shard = cache.eigvec_root / BENZENE_INCHIKEY[:2].upper()
        assert ethanol_shard != benzene_shard
        assert (ethanol_shard / f"{ETHANOL_INCHIKEY}.npy").exists()
        assert (benzene_shard / f"{BENZENE_INCHIKEY}.npy").exists()

    def test_short_inchikey_raises(self, cache):
        with pytest.raises(ValueError):
            cache.eigvec_path("A")

    def test_has_eigvec_reflects_state(self, cache):
        assert not cache.has_eigvec(ETHANOL_INCHIKEY)
        cache.get_or_compute(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        assert cache.has_eigvec(ETHANOL_INCHIKEY)


class TestFingerprintCache:
    def test_fingerprint_miss_then_hit(self, cache):
        first = cache.get_or_compute_fingerprint(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        assert first is not None
        assert first.shape == (cache.fingerprint_bits,)
        assert first.dtype == np.float32
        assert set(np.unique(first).tolist()).issubset({0.0, 1.0})

        second = cache.get_or_compute_fingerprint(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        np.testing.assert_array_equal(first, second)

    def test_fingerprint_sharded_layout(self, cache):
        cache.get_or_compute_fingerprint(ETHANOL_SMILES, ETHANOL_INCHIKEY)
        expected = (
            cache.fingerprint_root
            / ETHANOL_INCHIKEY[:2].upper()
            / f"{ETHANOL_INCHIKEY}.npy"
        )
        assert expected.exists()


class TestPrecomputeAll:
    def _tiny_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"smiles": ETHANOL_SMILES, "inchikey": ETHANOL_INCHIKEY},
                {"smiles": BENZENE_SMILES, "inchikey": BENZENE_INCHIKEY},
            ]
        )

    def test_precompute_populates_cache(self, cache):
        counts = cache.precompute_all(
            self._tiny_df(), n_jobs=1, show_progress=False
        )
        assert counts["requested"] == 2
        assert counts["computed"] == 2
        assert counts["failed"] == 0
        assert cache.has_eigvec(ETHANOL_INCHIKEY)
        assert cache.has_fingerprint(BENZENE_INCHIKEY)

    def test_precompute_is_idempotent(self, cache):
        cache.precompute_all(self._tiny_df(), n_jobs=1, show_progress=False)
        counts = cache.precompute_all(
            self._tiny_df(), n_jobs=1, show_progress=False
        )
        # Second run should skip both entries.
        assert counts["skipped"] == 2
        assert counts["computed"] == 0
        assert counts["failed"] == 0

    def test_precompute_deduplicates_by_inchikey(self, cache):
        df = pd.DataFrame(
            [
                {"smiles": ETHANOL_SMILES, "inchikey": ETHANOL_INCHIKEY},
                {"smiles": ETHANOL_SMILES, "inchikey": ETHANOL_INCHIKEY},
                {"smiles": BENZENE_SMILES, "inchikey": BENZENE_INCHIKEY},
            ]
        )
        counts = cache.precompute_all(df, n_jobs=1, show_progress=False)
        assert counts["requested"] == 2  # duplicate row collapsed

    def test_precompute_requires_expected_columns(self, cache):
        df = pd.DataFrame([{"foo": "bar"}])
        with pytest.raises(ValueError, match="must contain"):
            cache.precompute_all(df, n_jobs=1, show_progress=False)

    def test_precompute_counts_failures(self, cache):
        df = pd.DataFrame(
            [
                {"smiles": ETHANOL_SMILES, "inchikey": ETHANOL_INCHIKEY},
                {"smiles": "not-a-smiles", "inchikey": "BADKEY0000001-UHFFFAOYSA-N"},
            ]
        )
        counts = cache.precompute_all(df, n_jobs=1, show_progress=False)
        assert counts["computed"] == 1
        assert counts["failed"] == 1
