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
    uncuts, circles, weirds = st.compute_intra_ps(acc, smooth=False)
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
    uncuts, _, _ = st.compute_intra_ps(acc, smooth=False)
    areas = st.trapezoid_areas(acc.xs["circ"], 1_000, circular=True)
    i100 = st.attribute_xs(acc.xs["circ"], 100)
    assert uncuts["circ"][i100] == pytest.approx(1 / areas[i100])
    assert np.all(uncuts["circ"][acc.xs["circ"] >= 500] == 0)


def test_load_chrom_sizes_from_sam_header(tmp_path):
    sam = tmp_path / "x.sam"
    sam.write_text("@HD\tVN:1.0\n@SQ\tSN:chrI\tLN:230490\n@SQ\tSN:chrII\tLN:822297\n"
                   "@PG\tID:bowtie2\nread\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\tIIII\n")
    assert st.load_chrom_sizes(sam) == {"chrI": 230490, "chrII": 822297}
