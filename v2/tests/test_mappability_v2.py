"""Tests for mappability_v2.py: binned mask and mappable areas."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mappability_v2 as mp   # noqa: E402
import statistics_v2 as st    # noqa: E402


# --- Mappable areas --------------------------------------------------------

@pytest.mark.parametrize("L, r", [(10_000, 100), (12_345, 100), (230_490, 1000), (1_000, 1)])
def test_full_mask_gives_trapezoids(L, r):
    xs = st.generate_xs(L)
    f = np.ones(-(-L // r))
    np.testing.assert_allclose(mp.mappable_areas(xs, L, f, r),
                               st.trapezoid_areas(xs, L), rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("L, r", [(6_300, 100), (84_800, 100), (1_000, 1)])
def test_full_mask_gives_circular_rectangles(L, r):
    xs = st.generate_xs(L)
    f = np.ones(L // r)
    np.testing.assert_allclose(mp.mappable_areas(xs, L, f, r, circular=True),
                               st.trapezoid_areas(xs, L, circular=True), rtol=1e-9, atol=1e-6)


def test_half_mappable_chromosome():
    # first half mappable, second half not: pairs at distance d = max(L/2 - d, 0)
    L, r = 10_000, 10
    f = np.r_[np.ones(500), np.zeros(500)]
    xs = st.generate_xs(L)
    areas = mp.mappable_areas(xs, L, f, r)
    assert areas.sum() == pytest.approx((L / 2) ** 2 / 2, rel=1e-6)
    assert np.all(areas[xs >= L / 2] == pytest.approx(0, abs=1e-6))
    i = st.attribute_xs(xs, 1000)
    lo, hi = xs[i], xs[i + 1]
    assert areas[i] == pytest.approx((L / 2 - lo + L / 2 - hi) * (hi - lo) / 2, rel=1e-6)


def test_unmappable_hole_reduces_areas_at_all_distances():
    L, r = 100_000, 100
    f = np.ones(1000)
    f[400:450] = 0                       # 5 kb unmappable
    xs = st.generate_xs(L)
    full = st.trapezoid_areas(xs, L)
    holed = mp.mappable_areas(xs, L, f, r)
    assert np.all(holed <= full + 1e-6)
    assert holed.sum() == pytest.approx((L - 5000) ** 2 / 2, rel=1e-3)


def test_mappable_length():
    assert mp.mappable_length(np.array([1, 0.5, 1]), 250, 100) == pytest.approx(100 + 50 + 50)


# --- Binned mask from a bedgraph -------------------------------------------

def test_binned_unique_fraction(tmp_path):
    bg = tmp_path / "x.bedgraph"
    bg.write_text(
        "chrA\t0\t150\t1\n"        # unique
        "chrA\t150\t250\t0.5\n"    # 2 copies → not unique
        "chrA\t250\t400\t1\n"
        # 400-500 not in the bedgraph → 0
        "chrB\t0\t100\t0.333333\n"
    )
    out = mp.binned_unique_fraction(bg, {"chrA": 500, "chrB": 100, "chrC": 50}, bin_size=100)
    np.testing.assert_allclose(out["chrA"], [1.0, 0.5, 0.5, 1.0, 0.0])
    np.testing.assert_allclose(out["chrB"], [0.0])
    np.testing.assert_allclose(out["chrC"], [0.0])   # absent from the bedgraph


def test_binned_unique_fraction_last_bin_shorter(tmp_path):
    bg = tmp_path / "x.bedgraph"
    bg.write_text("chrA\t0\t230\t1\n")
    out = mp.binned_unique_fraction(bg, {"chrA": 230}, bin_size=100)
    np.testing.assert_allclose(out["chrA"], [1.0, 1.0, 1.0])
