"""Tests for mappability_v2.py: binned mask and mappable areas."""

import shutil
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


@pytest.mark.skipif(shutil.which("bowtie2") is None or shutil.which("bowtie2-build") is None,
                    reason="bowtie2 not installed")
def test_bowtie2_mappability_unique_vs_duplicated(tmp_path):
    import subprocess
    rng = np.random.default_rng(3)
    rand = lambda n: "".join(rng.choice(list("ACGT"), n))
    rep = rand(400)                                         # 400 bp present twice
    chrA = rand(1000) + rep + rand(1000)                    # 2400 bp
    chrB = rand(600) + rep + rand(600) + "N" * 100          # 1700 bp
    fa = tmp_path / "g.fa"
    fa.write_text(f">chrA\n{chrA}\n>chrB\n{chrB}\n")
    subprocess.run(["bowtie2-build", "-q", str(fa), str(tmp_path / "idx")], check=True)
    bg = mp.bowtie2_mappability(fa, tmp_path / "idx", tmp_path / "out", kmer=35, step=10, threads=2)
    frac = mp.binned_unique_fraction(bg, {"chrA": 2400, "chrB": 1700}, 100)
    # unique flanks mappable, the repeat (and the k-mers overlapping it) not
    assert frac["chrA"][:9].min() == 1.0 and frac["chrA"][15:].mean() > 0.95
    assert frac["chrA"][10:13].max() == 0.0 and 0 < frac["chrA"][13] < 0.5
    assert frac["chrB"][6:9].max() == 0.0
    # k-mers starting near the end of the repeat overlap it by < ~24 bp: the
    # second copy scores below the local min score (20 + 8 ln 35 ≈ 48), no XS
    assert 0 < frac["chrB"][9] < 0.5
    assert frac["chrB"][-1] == 0.0                          # N
    rows = [l.split("\t") for l in bg.read_text().splitlines()]
    assert all(int(e) <= {"chrA": 2400, "chrB": 1700}[c] for c, s, e, v in rows)


@pytest.mark.skipif(shutil.which("bowtie2") is None or shutil.which("bowtie2-build") is None,
                    reason="bowtie2 not installed")
def test_repeat_links_direct_and_inverted(tmp_path):
    import subprocess
    rng = np.random.default_rng(5)
    rand = lambda n: "".join(rng.choice(list("ACGT"), n))
    rc = lambda x: x.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    rep, inv = rand(300), rand(300)
    # chrA: rep at 1000 and 3000 (direct); inv at 5000 and its reverse complement at 7000
    chrA = rand(1000) + rep + rand(1700) + rep + rand(1700) + inv + rand(1700) + rc(inv) + rand(1000)
    fa = tmp_path / "g.fa"
    fa.write_text(f">chrA\n{chrA}\n>chrB\n{rand(3000)}\n")
    subprocess.run(["bowtie2-build", "-q", str(fa), str(tmp_path / "idx")], check=True)
    path, summary = mp.repeat_links(fa, tmp_path / "idx", tmp_path, kmer=35, step=10,
                                    resolution=1000, threads=2)
    links = mp.read_repeat_links(path)
    agg = links.groupby(["start1", "start2", "orientation"])["n_kmers"].sum()
    assert agg[(1000, 3000, "+")] > 15 and agg[(3000, 1000, "+")] > 15     # direct, both ways
    assert agg[(5000, 7000, "-")] > 15 and agg[(7000, 5000, "-")] > 15     # inverted
    assert set(links["chrom1"]) == {"chrA"} and summary["linked"] > 0


def test_repeat_families():
    import pandas as pd
    rows = [("chrA", 1000, "chrA", 3000, "+", 10), ("chrA", 3000, "chrA", 1000, "+", 10),   # tandem
            ("chrA", 3000, "chrA", 5000, "+", 10),
            ("chrA", 0, "chrB", 9000, "-", 5), ("chrB", 0, "chrA", 19000, "-", 5),         # ends
            ("chrA", 0, "chrB", 0, "+", 5),
            ("chrA", 10000, "chrB", 4000, "+", 1)]                                          # < min
    links = pd.DataFrame(rows, columns=mp.LINK_COLUMNS)
    bins, fams = mp.repeat_families(links, 1000, min_kmers=3,
                                    chrom_sizes={"chrA": 20000, "chrB": 10000}, end_kb=2)
    f = {(c, s): fam for c, s, fam in zip(bins.chrom, bins.start, bins.family)}
    assert f[("chrA", 1000)] == f[("chrA", 3000)] == f[("chrA", 5000)]
    assert f[("chrA", 0)] == f[("chrB", 9000)] == f[("chrB", 0)] == f[("chrA", 19000)]
    assert ("chrA", 10000) not in f
    kinds = dict(zip(fams.family, fams.kind))
    assert kinds[f[("chrA", 1000)]] == "tandem" and kinds[f[("chrA", 0)]] == "subtelomeric"
    assert sorted(fams.family) == [1, 2] and fams.n_bins.tolist() == [4, 3]
