"""Tests for density_v2.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import generic_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import density_v2 as dn  # noqa: E402


def _slow_nanmean(values):        # the v1 callback (hicberg.utils.nanmean_filter)
    vals = values[~np.isnan(values)]
    return np.mean(vals) if len(vals) > 0 else np.nan


@pytest.mark.parametrize("shape, size", [((30, 30), 5), ((12, 40), 7), ((4, 4), 15), ((3, 50), 45)])
def test_fast_nanmean_filter_equals_generic_filter(shape, size):
    rng = np.random.default_rng(0)
    z = rng.random(shape)
    z[rng.random(shape) < 0.3] = np.nan
    slow = generic_filter(z, _slow_nanmean, size=size, mode="mirror")
    np.testing.assert_allclose(dn.nanmean_filter(z, size), slow, rtol=1e-10, equal_nan=True)


def test_iterative_fill_only_fills_holes():
    z = np.ones((20, 20)) * 2.0
    z[5:8, 5:8] = np.nan
    z[0, 0] = 7.0
    out = dn.iterative_fill(z, 3)
    assert not np.isnan(out).any()
    assert out[0, 0] == 7.0 and out[10, 10] == 2.0


def test_detrend_intra_mean_of_positive_values_per_diagonal():
    mat = np.array([[4., 2., 0.],
                    [2., 6., 1.],
                    [0., 1., np.nan]])
    out = dn.detrend_intra(mat)
    # diagonal 0: positives 4, 6 → mean 5 ; diagonal 1: 2, 1, 2, 1 → 1.5 ; diagonal 2: none
    np.testing.assert_allclose(out[0, 0], 4 / 5)
    np.testing.assert_allclose(out[1, 2], 1 / 1.5)
    assert out[0, 2] == 0 and np.isnan(out[2, 2])


def test_detrend_intra_caps_large_values_to_one():
    mat = np.full((5, 5), 1.0)
    mat[0, 4] = mat[4, 0] = 1000.0     # diagonal 4: only these two → ratio 1 (not capped)
    mat[1, 2] = 100.0                  # diagonal 1: mean of (1,1,1,1,100,1,1,1) → ratio ≈ 7.8
    out = dn.detrend_intra(mat, max_val=5)
    assert out[1, 2] == 1.0            # >= max_val → 1
    assert out[0, 4] == pytest.approx(1.0)


def test_contact_matrix_counts_and_symmetry(tmp_path):
    pairs = tmp_path / "x.pairs"
    pairs.write_text(
        "## pairs format v1.0\n"
        "r1\tchrA\t1\tchrA\t150\t+\t-\n"       # bins 0 and 1 (bin_size 100)
        "r2\tchrA\t150\tchrA\t1\t-\t+\n"       # same contact, other order
        "r3\tchrA\t100\tchrB\t50\t+\t+\n"      # chrA bin 0 (v2 binning) – chrB bin 0
        "r4\tchrA\t250\tchrA\t250\t+\t-\n"     # diagonal, last (partial) bin 2
    )
    sizes = {"chrA": 250, "chrB": 100}
    m = dn.contact_matrix(pairs, sizes, 100).toarray()
    assert m.shape == (4, 4)                    # 3 bins (ceil) + 1
    assert m[0, 1] == m[1, 0] == 2
    assert m[0, 3] == m[3, 0] == 1
    assert m[2, 2] == 1                         # diagonal counted once
    assert np.array_equal(m, m.T)


def test_unbiased_keeps_zeros_and_averages_one():
    rng = np.random.default_rng(0)
    mat = rng.poisson(0.15, (60, 60)).astype(float)       # sparse, ~86% zeros
    mat = np.triu(mat) + np.triu(mat, 1).T
    oe = dn.obs_exp_unbiased(mat, intra=True)
    dist = np.abs(np.subtract.outer(np.arange(60), np.arange(60)))
    empty_diag = np.array([np.diagonal(mat, k).sum() == 0 for k in range(60)])[dist]
    assert np.all(oe[(mat == 0) & ~empty_diag] == 0)       # zeros stay zeros
    assert np.all(np.isnan(oe[empty_diag]))                 # no expected → NaN (filled later)
    for k in range(0, 50):
        diag = np.diagonal(oe, k)
        if np.nansum(np.diagonal(mat, k)) > 0:
            assert np.nanmean(diag) == pytest.approx(1.0)


def test_sparse_trans_block_not_inflated():
    rng = np.random.default_rng(1)
    mat = rng.poisson(0.13, (200, 300)).astype(float)      # like yeast trans at 2 kb
    v1 = dn.block_density(mat, intra=False, method="v1")
    ub = dn.block_density(mat, intra=False, method="unbiased")
    assert np.median(v1) > 1.6                              # v1: ≈ 1 + fraction of zeros
    assert np.median(ub) == pytest.approx(1.0, abs=0.1)


def test_unbiased_bad_bins_filled():
    mat = np.full((40, 40), 3.0)
    mat[10:13, :] = np.nan
    mat[:, 10:13] = np.nan
    d = dn.block_density(mat, intra=True, patch_small=5, patch_large=9, method="unbiased")
    assert np.isfinite(d).all()
    np.testing.assert_allclose(d, 1.0)


def test_block_density_neutral_on_uniform_matrix():
    mat = np.full((30, 30), 5.0)
    d = dn.block_density(mat, intra=True, patch_small=3, patch_large=5)
    np.testing.assert_allclose(d, 1.0)
    d = dn.block_density(np.full((10, 20), 5.0), intra=False, patch_small=3, patch_large=5)
    np.testing.assert_allclose(d, 1.0)


def test_adaptive_filter_small_window_where_dense():
    counts = np.full((100, 100), 0.1)          # sparse background
    counts[45:55, 45:55] = 20.0                # dense peak (like CEN-CEN)
    z = counts / counts.mean()
    out, chosen = dn.adaptive_filter(z, counts, windows=(5, 11, 21, 45), n_min=100)
    assert chosen[50, 50] == 5                 # 25 px × 20 ≥ 100
    assert chosen[5, 5] == 45                  # background never reaches 100 → largest
    fixed = dn.nanmean_filter(z, 45)
    assert out[50, 50] > 3 * fixed[50, 50]     # the peak is not diluted


def test_adaptive_density_unbiased_peak():
    rng = np.random.default_rng(0)
    lam = np.full((150, 200), 0.13)
    lam[70:80, 95:105] = 3.0                   # centromere-like enrichment
    mat = rng.poisson(lam).astype(float)
    fixed = dn.block_density(mat, intra=False, method="unbiased", smoothing="fixed")
    adapt = dn.block_density(mat, intra=False, method="unbiased", smoothing="adaptive")
    assert np.median(adapt) == pytest.approx(np.median(fixed), abs=0.15)   # neutral background kept
    assert adapt[75, 100] > 2 * fixed[75, 100]                              # sharper peak
