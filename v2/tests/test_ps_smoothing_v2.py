"""Tests for the P(s) smoothing by pooling of distance bins (statistics_v2)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import statistics_v2 as st  # noqa: E402


def test_rich_bins_keep_their_exact_raw_value():
    xs = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    counts = np.array([500, 400, 300, 200, 150, 120], dtype=float)
    areas = np.array([10, 10, 20, 20, 30, 30], dtype=float)
    law = st.pooled_law(xs, counts, areas, n_min=100, d_max=6)
    np.testing.assert_allclose(law, counts / areas)


def test_poor_bins_are_pooled():
    # flat law, 25 counts per bin: groups of 4 bins (100 counts); the last 2
    # bins (50 counts < 100) are merged back into the previous group
    xs = np.arange(0, 100, 10, dtype=float)
    counts = np.array([25, 25, 25, 25, 25, 0, 50, 25, 25, 25], dtype=float)
    law = st.pooled_law(xs, counts, np.full(10, 100.0), n_min=100, d_max=100)
    np.testing.assert_allclose(law, 0.25)               # empty bin 5 is not 0 any more


def test_bins_without_area_stay_zero():
    xs = np.array([0, 10, 20, 30], dtype=float)
    law = st.pooled_law(xs, np.array([200., 200., 5., 0.]), np.array([10., 10., 0., 0.]), 100)
    assert law[2] == 0 and law[3] == 0


def test_recovers_power_law_from_poisson_counts():
    rng = np.random.default_rng(0)
    L = 1_000_000
    xs = st.generate_xs(L)
    areas = st.trapezoid_areas(xs, L)
    hi = np.append(xs[1:], L)
    centres = np.sqrt(np.maximum(xs, 0.5) * hi)
    truth = 1e-3 * centres ** -1.2
    counts = rng.poisson(truth * areas).astype(float)

    law = st.pooled_law(xs, counts, areas, n_min=200, d_max=L)
    far = (xs > 1e4) & (xs < 0.8 * L)
    raw_err = np.abs(counts[far] / areas[far] / truth[far] - 1)
    err = np.abs(law[far] / truth[far] - 1)
    # 5-20 pairs per bin here: the pooled law halves the error of the raw law
    assert np.median(err) < 0.15
    assert np.median(err) < 0.5 * np.median(raw_err)
    # no systematic bias (the group value is not the law at the group centre),
    # averaged over draws (one draw fluctuates by ~5% at these counts)
    bias = [np.median(st.pooled_law(xs, np.random.default_rng(s).poisson(truth * areas)
                                    .astype(float), areas, n_min=200, d_max=L)[far]
                      / truth[far]) for s in range(20)]
    assert abs(np.mean(bias) - 1) < 0.03


def test_pooled_law_conserves_counts_per_group():
    rng = np.random.default_rng(1)
    xs = st.generate_xs(100_000)
    areas = st.trapezoid_areas(xs, 100_000)
    counts = rng.poisson(5.0, len(xs)).astype(float)
    law = st.pooled_law(xs, counts, areas, n_min=50, d_max=100_000)
    assert np.sum(law * areas) == pytest.approx(counts.sum(), rel=1e-3)


def test_orientations_merged_beyond_d_merge():
    xs = np.array([0, 1000, 3000, 6000, 9000], dtype=float)
    areas = np.full(5, 1000.0)
    counts = np.array([          # rows: +-, -+, ++, --
        [900, 600, 300, 100, 100],
        [100, 200, 300, 100, 100],
        [100, 200, 300, 100, 100],
        [100, 200, 300, 100, 100],
    ], dtype=float)
    laws = st.smooth_ps_laws(xs, counts, areas, n_min=1, d_merge=5000, d_max=12000)
    # short distances: own counts (weird = ++ and -- over 2 × area)
    assert laws["uncut"][0] == pytest.approx(0.9)
    assert laws["weird"][0] == pytest.approx(200 / 2000)
    # beyond 5 kb: one law for all = 400 / (4 × 1000)
    for key in ("uncut", "circle", "weird"):
        np.testing.assert_allclose(laws[key][3:], 0.1)


def test_short_distances_keep_raw_values_and_zeros():
    # like chrI weird: no ++/-- pair at all below ~19 bp, then sparse counts
    xs = st.generate_xs(230_000)
    areas = st.trapezoid_areas(xs, 230_000)
    counts = np.zeros((4, len(xs)))
    rich = (xs >= 19) & (xs < 1000)
    counts[2, rich] = np.resize([4, 1, 7, 10, 0, 14], rich.sum())
    counts[:, xs >= 1000] = 500
    laws = st.smooth_ps_laws(xs, counts, areas, n_min=100, d_merge=5000,
                             d_max=230_000, d_pool=1000)
    weird_raw = counts[2] / (2 * areas)
    short = xs < 1000
    np.testing.assert_allclose(laws["weird"][short], weird_raw[short])
    assert np.all(laws["weird"][xs < 19] == 0)          # never observed → stays 0


def test_compute_intra_ps_rejects_unknown_smoothing():
    with pytest.raises(ValueError):
        st.compute_intra_ps(None, smoothing="spline")
