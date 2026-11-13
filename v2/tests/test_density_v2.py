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


def test_block_density_neutral_on_uniform_matrix():
    mat = np.full((30, 30), 5.0)
    d = dn.block_density(mat, intra=True, patch_small=3, patch_large=5)
    np.testing.assert_allclose(d, 1.0)
    d = dn.block_density(np.full((10, 20), 5.0), intra=False, patch_small=3, patch_large=5)
    np.testing.assert_allclose(d, 1.0)
