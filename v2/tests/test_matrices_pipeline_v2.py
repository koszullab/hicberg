"""Tests for matrices_v2.py and the helpers of hicberg_v2.py."""

import gzip
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matrices_v2 as mx     # noqa: E402
import hicberg_v2 as hb      # noqa: E402


def test_write_bins(tmp_path):
    p = mx.write_bins({"chrA": 450, "chrB": 200}, 200, tmp_path / "bins.bed")
    assert p.read_text().splitlines() == ["chrA\t0\t200", "chrA\t200\t400", "chrA\t400\t450",
                                          "chrB\t0\t200"]


def test_gc_content(tmp_path):
    fa = tmp_path / "g.fa"
    fa.write_text(">chrA\nGGGGCCCCAA\nAATTNNNN\n>chrB\nATAT\n")
    gc = mx.gc_content(fa, {"chrA": 18, "chrB": 4}, 10)
    np.testing.assert_allclose(gc["chrA"], [0.8, 0.0])          # 2nd bin: AATT + 4 N → 0 / 4
    np.testing.assert_allclose(gc["chrB"], [0.0])


def _hicstuff_ok():
    try:
        import hicstuff.pipeline  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hicstuff_ok(), reason="hicstuff not installed")
def test_hicstuff_filter_given_thresholds_and_microc(tmp_path):
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 20000))
    fa = tmp_path / "g.fa"
    fa.write_text(">chrA\n" + seq + "\n")
    rows = [("chrA", 1000, "chrA", 15000, "+", "-"),      # far apart: kept
            ("chrA", 1000, "chrA", 15000, "+", "-"),      # PCR duplicate
            ("chrA", 2000, "chrA", 2050, "+", "-"),       # uncut (same fragment, +-)
            ("chrA", 5000, "chrA", 12000, "-", "+")]      # far: kept
    pairs = tmp_path / "x.pairs"
    pairs.write_text("## pairs format v1.0\n#chromsize: chrA 20000\n"
                     "#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n" +
                     "".join(f"r{k}\t{a}\t{p}\t{b}\t{q}\t{s1}\t{s2}\n"
                             for k, (a, p, b, q, s1, s2) in enumerate(rows)))
    out, thr = mx.hicstuff_filter(pairs, fa, "DpnII", tmp_path / "o", 1, "x", thresholds=(3, 3))
    kept = [l.split("\t")[2] for l in out.read_text().splitlines() if not l.startswith("#")]
    assert thr == (3, 3) and sorted(kept) == ["1000", "5000"]
    # Micro-C: virtual 100 bp fragments; the 50 bp +- pair is an uncut there too
    out, thr = mx.hicstuff_filter(pairs, fa, "mnase", tmp_path / "m", 1, "x", thresholds=(3, 3))
    kept = [l.split("\t")[2] for l in out.read_text().splitlines() if not l.startswith("#")]
    assert sorted(kept) == ["1000", "5000"]
    frags = {int(l.split("\t")[7]) for l in out.read_text().splitlines() if not l.startswith("#")}
    assert frags == {9, 49}           # 1-based pos 1000 = last base of the 10th 100 bp segment


def test_restriction_sites(tmp_path):
    fa = tmp_path / "g.fa"
    # DpnII GATC (palindromic) and HinfI GANTC; bins of 10 bp
    fa.write_text(">chrA\nGATCAGAATC\nAAAAAAAAAA\nGACTCGATCG\n>chrB\nACGT\n")
    s = mx.restriction_sites(fa, {"chrA": 30, "chrB": 4}, 10, "DpnII,HinfI")
    np.testing.assert_array_equal(s["chrA"], [2, 0, 2])
    np.testing.assert_array_equal(s["chrB"], [0])
    assert mx.enzyme_sites("DpnII,HinfI") == ["GATC", "GANTC"]
    assert mx.enzyme_sites("mnase") == [] and mx.restriction_sites(fa, {"chrA": 30}, 10, "150") is None
    # non-palindromic site found on both strands: GAAGAC (BbsI) and its reverse complement GTCTTC
    fa.write_text(">c\nGAAGACGTCTTC\n")
    assert mx.restriction_sites(fa, {"c": 12}, 100, "GAAGAC")["c"].tolist() == [2]


@pytest.mark.skipif(shutil.which("cooler") is None, reason="cooler not installed")
def test_build_cool_counts(tmp_path):
    import cooler
    pairs = tmp_path / "x.pairs.gz"
    with gzip.open(pairs, "wt") as fh:
        fh.write("## pairs format v1.0\n#chromsize: chrA 1000\n#chromsize: chrB 500\n"
                 "#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n")
        rows = [("chrA", 10, "chrA", 150), ("chrA", 150, "chrA", 10), ("chrA", 910, "chrB", 20),
                ("chrB", 499, "chrB", 499)]
        for k, (a, p, b, q) in enumerate(rows):
            fh.write(f"r{k}\t{a}\t{p}\t{b}\t{q}\t+\t-\n")
    c = cooler.Cooler(str(mx.build_cool(pairs, {"chrA": 1000, "chrB": 500}, 100,
                                        tmp_path / "x.cool", balance=False)))
    m = c.matrix(balance=False)[:]
    assert m.sum() - np.trace(m) == 6        # 3 off-diagonal contacts, counted twice (symmetric)
    assert m[0, 1] == 2 and m[9, 10] == 1 and m[14, 14] == 1


def test_fasta_sizes_and_read_length(tmp_path):
    fa = tmp_path / "g.fa.gz"
    with gzip.open(fa, "wt") as fh:
        fh.write(">chrA desc\nACGT\nAC\n>chrB\nA\n")
    assert hb.fasta_sizes(fa) == {"chrA": 6, "chrB": 1}
    fq = tmp_path / "r.fq"
    fq.write_text("@r1\nACGTA\n+\nIIIII\n@r2\nACG\n+\nIII\n")
    assert hb.read_length(fq) == 5


def test_run_stage():
    class A:
        start_stage, exit_stage = "classify", "stats"
    assert hb.run_stage("classify", A) and hb.run_stage("stats", A)
    assert not hb.run_stage("align", A) and not hb.run_stage("rescue", A)


def test_auto_plot_and_density_bin_sizes():
    import density_v2 as dn
    yeast = {"chrIV": 1_531_933, "chrXII": 1_078_177}
    assert mx.auto_plot_bin_size(yeast, list(yeast), 2000) == 2000
    assert dn.auto_density_bin_size(yeast, 2000, memory=8 * 1024 ** 3) == 2000
    mosq = {"c1": 132_876_167, "c2": 225_161_840, "w": 1_369_678}
    assert mx.auto_plot_bin_size(mosq, ["c1", "c2"], 2000) == 100_000
    assert dn.auto_density_bin_size(mosq, 2000, memory=20 * 1024 ** 3) == 25_000
    many = {f"s{k}": 10_000 for k in range(40)} | mosq
    assert mx.auto_plot_chroms(many) == ["c1", "c2", "w"]
    np.testing.assert_allclose(mx.rebin_mean([1, 3, 5, np.nan, 7], 2), [2, 5, 7])
