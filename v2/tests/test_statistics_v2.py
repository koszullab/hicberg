"""Unit tests for statistics_v2.py on small synthetic pairs files."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import statistics_v2 as st  # noqa: E402


CHROM_SIZES = {"chrA": 10_000, "chrB": 5_000, "circ": 1_000}

# readID chrom1 pos1 chrom2 pos2 strand1 strand2
PAIRS = [
    ("r1", "chrA", 100,  "chrA", 600,  "+", "-"),   # uncut,  d=500
    ("r2", "chrA", 600,  "chrA", 100,  "-", "+"),   # uncut (reads swapped), d=500
    ("r3", "chrA", 100,  "chrA", 600,  "-", "+"),   # circle, d=500
    ("r4", "chrA", 100,  "chrA", 2100, "+", "+"),   # ++,     d=2000
    ("r5", "chrA", 2100, "chrA", 100,  "-", "-"),   # --,     d=2000
    ("r6", "circ", 50,   "circ", 950,  "+", "-"),   # circular: d=min(900, 100)=100
    ("r7", "chrA", 1,    "chrB", 5000, "+", "-"),   # trans
    ("r8", "chrB", 2000, "chrA", 2001, "-", "-"),   # trans (other order)
    ("r9", "chrZ", 10,   "chrA", 10,   "+", "+"),   # unknown chromosome
]


@pytest.fixture
def pairs_file(tmp_path):
    path = tmp_path / "unrescued.pairs"
    with open(path, "w") as fh:
        fh.write("## pairs format v1.0\n")
        fh.write("#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2\n")
        for p in PAIRS:
            fh.write("\t".join(map(str, p)) + "\n")
    return path


@pytest.fixture
def acc(pairs_file):
    a = st.Accumulator(CHROM_SIZES, bin_size=1000, base=1.1, circular={"circ"})
    for chunk in st.iter_pairs(pairs_file, a.names, chunksize=3):   # several chunks
        a.add(*chunk)
    return a


def test_generate_xs_is_prefix_of_larger_chromosome():
    small, large = st.generate_xs(5_000), st.generate_xs(250_000_000)
    assert np.array_equal(large[:len(small)], small)


def test_stats(acc):
    s = acc.stats
    assert s["pairs"] == 9
    assert s["skipped_unknown_chrom"] == 1
    assert s["cis"] == 6 and s["trans"] == 2
    assert (s["+-"], s["-+"], s["++"], s["--"]) == (3, 1, 1, 1)


def test_ps_counts_orientation_and_distance(acc):
    counts = acc.ps_counts_by_chrom()
    xs_a = acc.xs["chrA"]
    i500, i2000 = st.attribute_xs(xs_a, 500), st.attribute_xs(xs_a, 2000)

    assert counts["chrA"][st.UNCUT,       i500]  == 2
    assert counts["chrA"][st.CIRCLE,      i500]  == 1
    assert counts["chrA"][st.PLUS_PLUS,   i2000] == 1
    assert counts["chrA"][st.MINUS_MINUS, i2000] == 1
    assert counts["chrA"].sum() == 5


def test_circular_distance(acc):
    counts = acc.ps_counts_by_chrom()["circ"]
    assert counts[st.UNCUT, st.attribute_xs(acc.xs["circ"], 100)] == 1
    assert counts.sum() == 1


def test_coverage_bins(acc):
    cov = acc.coverage_by_chrom()
    assert len(cov["chrA"]) == 11              # 10_000 // 1000 + 1
    # chrA 5' positions: 100,600,600,100,100,600,100,2100,2100,100 (cis) + 1, 2001 (trans)
    assert cov["chrA"][0] == 9                 # positions 1..1000
    assert cov["chrA"][2] == 3                 # 2001, 2100, 2100
    assert cov["chrB"][4] == 1                 # pos 5000 → (5000-1)//1000 = 4
    assert cov["chrB"][1] == 1                 # pos 2000 → bin 1
    assert sum(v.sum() for v in cov.values()) == 2 * 8


def test_trans_ps_symmetric(acc):
    tps = st.compute_trans_ps(acc)
    expected = 2 / (10_000 * 5_000 * 4)
    assert tps[("chrA", "chrB")] == pytest.approx(expected)
    assert tps[("chrB", "chrA")] == pytest.approx(expected)
    assert tps[("chrA", "circ")] == 0


def test_intra_ps_normalisation(acc):
    uncuts, circles, weirds = st.compute_intra_ps(acc, smoothing="none")
    areas = st.trapezoid_areas(acc.xs["chrA"], 10_000)
    i500, i2000 = st.attribute_xs(acc.xs["chrA"], 500), st.attribute_xs(acc.xs["chrA"], 2000)
    assert uncuts["chrA"][i500]   == pytest.approx(2 / areas[i500])
    assert circles["chrA"][i500]  == pytest.approx(1 / areas[i500])
    assert weirds["chrA"][i2000]  == pytest.approx(2 / (2 * areas[i2000]))


def test_trapezoid_areas_cover_all_position_pairs():
    L = 10_000
    xs = st.generate_xs(L)
    # sum over bins = number of ordered pairs (i <= j) ~ L^2 / 2
    assert st.trapezoid_areas(xs, L).sum() == pytest.approx(L * L / 2, rel=1e-3)


def test_circular_areas_cover_all_position_pairs():
    L = 6_318
    xs = st.generate_xs(L)
    areas = st.trapezoid_areas(xs, L, circular=True)
    # L position pairs at each distance d <= L/2 → L * L/2 in total
    assert areas.sum() == pytest.approx(L * L / 2)
    # flat: area proportional to bin width below L/2, nothing beyond
    i = st.attribute_xs(xs, 1000)
    assert areas[i] == pytest.approx(L * (xs[i + 1] - xs[i]))
    assert np.all(areas[xs >= L / 2] == 0)


def test_intra_ps_circular_normalisation(acc):
    uncuts, _, _ = st.compute_intra_ps(acc, smoothing="none")
    areas = st.trapezoid_areas(acc.xs["circ"], 1_000, circular=True)
    i100 = st.attribute_xs(acc.xs["circ"], 100)
    assert uncuts["circ"][i100] == pytest.approx(1 / areas[i100])
    assert np.all(uncuts["circ"][acc.xs["circ"] >= 500] == 0)


def test_correct_coverage_divides_and_interpolates():
    raw = np.array([100, 50, 0, 0, 80, 100], dtype=float)
    f   = np.array([1.0, 0.5, 0.1, 0.0, 0.8, 1.0])
    cov, valid = st.correct_coverage(raw, f, 600, 100, min_fraction=0.5,
                                     min_mappable_bp=0, min_ratio=0)
    np.testing.assert_array_equal(valid, [True, True, False, False, True, True])
    # 50/0.5 = 100, 80/0.8 = 100, holes interpolated between 100 and 100
    np.testing.assert_allclose(cov, [100, 100, 100, 100, 100, 100])


def test_correct_coverage_ends_and_circular():
    raw = np.array([0, 10, 20, 30, 0], dtype=float)
    f   = np.array([0, 1, 1, 1, 0], dtype=float)
    lin, _ = st.correct_coverage(raw, f, 500, 100, min_ratio=0)
    np.testing.assert_allclose(lin, [10, 10, 20, 30, 30])        # constant beyond ends
    circ, _ = st.correct_coverage(raw, f, 500, 100, min_ratio=0, circular=True)
    # wraps around: index 3 (30) → index 6 ≡ 1 (10) in 3 steps
    np.testing.assert_allclose(circ, [30 - 2 * 20 / 3, 10, 20, 30, 30 - 20 / 3])


def test_correct_coverage_short_last_bin_is_scaled_not_excluded():
    # like the end of chrI: bins of 2 kb, last bin 490 bp wide
    cov, valid = st.correct_coverage(np.array([2000., 2000., 490.]), np.ones(3), 4490, 2000,
                                     min_ratio=0)
    assert valid.all()
    np.testing.assert_allclose(cov, [2000, 2000, 2000])          # 490 reads in 490 bp → 2000 / 2 kb


def test_correct_coverage_tiny_remnant_excluded():
    # last bin only 10 bp wide: below min_mappable_bp → interpolated, not × 10
    cov, valid = st.correct_coverage(np.array([100., 100., 5.]), np.ones(3), 210, 100,
                                     min_mappable_bp=50, min_ratio=0)
    assert valid.tolist() == [True, True, False]
    np.testing.assert_allclose(cov, [100, 100, 100])


def test_correct_coverage_low_outlier_excluded():
    raw = np.full(15, 100.)
    raw[7] = 30                                                  # mappable but under-covered
    cov, valid = st.correct_coverage(raw, np.ones(15), 1500, 100, min_ratio=0.5)
    assert not valid[7] and valid.sum() == 14
    np.testing.assert_allclose(cov, 100)
    cov, valid = st.correct_coverage(raw, np.ones(15), 1500, 100, min_ratio=0)
    assert valid.all() and cov[7] == 30


def test_correct_coverage_isolated_low_bin_after_gap():
    # like the end of chrI: 10 good bins, unmappable stretch, one low last bin
    raw = np.r_[np.full(10, 2000.), np.zeros(5), 800.]
    f = np.r_[np.ones(10), np.zeros(5), 1.]
    cov, valid = st.correct_coverage(raw, f, 16 * 2000, 2000, min_ratio=0.5)
    assert not valid[15]                                          # compared to real neighbours
    np.testing.assert_allclose(cov[10:], 2000)                    # constant beyond the last kept bin


def test_correct_coverage_short_end_bin_is_not_an_anchor():
    # end of chrI: 10 good bins, 5 unmappable bins, last bin 490 bp wide and low
    raw = np.r_[np.full(10, 2000.), np.zeros(5), 200.]
    f = np.r_[np.ones(10), np.zeros(5), 1.]
    cov, valid = st.correct_coverage(raw, f, 15 * 2000 + 490, 2000)
    assert valid[15]                                              # its own value is kept...
    assert cov[15] == pytest.approx(200 * 2000 / 490)
    np.testing.assert_allclose(cov[10:15], 2000)                  # ...but it does not pull the gap down


def test_correct_coverage_keeps_low_coverage_mappable_bins_by_default():
    # a mappable bin with a low coverage (e.g. few restriction sites) is real data
    raw = np.r_[np.full(10, 2000.), 300., np.full(10, 2000.)]
    cov, valid = st.correct_coverage(raw, np.ones(21), 21 * 2000, 2000)
    assert valid.all() and cov[10] == 300


def test_correct_coverage_longer_than_mask_and_empty():
    cov, valid = st.correct_coverage(np.array([10., 10., 0.]), np.array([1., 1.]), 200, 100,
                                     min_ratio=0)
    np.testing.assert_allclose(cov, [10, 10, 10])                # extra bin = 0 bp wide
    assert valid.tolist() == [True, True, False]
    cov, valid = st.correct_coverage(np.array([5., 5.]), np.array([0., 0.]), 200, 100)
    assert np.isnan(cov).all() and not valid.any()


def test_load_chrom_sizes_from_sam_header(tmp_path):
    sam = tmp_path / "x.sam"
    sam.write_text("@HD\tVN:1.0\n@SQ\tSN:chrI\tLN:230490\n@SQ\tSN:chrII\tLN:822297\n"
                   "@PG\tID:bowtie2\nread\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\tIIII\n")
    assert st.load_chrom_sizes(sam) == {"chrI": 230490, "chrII": 822297}


def test_rebin_counts():
    np.testing.assert_array_equal(st.rebin_counts(np.arange(7), 100, 200, 4), [1, 5, 9, 6])
    with pytest.raises(ValueError):
        st.rebin_counts(np.arange(7), 100, 250, 3)


def test_fill_from_all_reads(tmp_path):
    from types import SimpleNamespace
    # chrA: 6 bins of 200 bp, bins 2-3 unmappable (excluded), last bin 100 bp wide
    acc = SimpleNamespace(names=["chrA"], sizes=np.array([1100]), bin_size=200)
    final = {"chrA": np.array([100., 100., 999., 999., 100., 100.])}
    kept = {"chrA": np.array([True, True, False, False, True, True])}
    mapp = {"chrA": np.array([1., 1., 0., 0., 1., 1.])}
    # all aligned reads at 100 bp: 2× the group1 level, more reads in the repeat
    z = tmp_path / "all.npz"
    np.savez(z, resolution=100, chrA=np.array([100, 100, 100, 100, 150, 150, 0, 0, 100, 100, 100]))
    out = st.fill_from_all_reads(acc, final, kept, mapp, z)["chrA"]
    # scale = median(100 / 200) = 0.5 → bin 2: 0.5 × 300, bin 3: 0 (no read at all)
    np.testing.assert_allclose(out, [100, 100, 150, 0, 100, 100])
