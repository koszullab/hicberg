"""Tests for omics_v2.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import omics_v2 as om  # noqa: E402


def write_pairs(path, rows):
    path.write_text("## pairs format v1.0\n#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n"
                    + "".join(f"r{k}\t{a}\t{p}\t{b}\t{q}\t{s}\t{t}\n" for k, (a, p, b, q, s, t) in enumerate(rows)))


def test_fragments_orientation_and_length(tmp_path):
    p = tmp_path / "x.pairs"
    write_pairs(p, [("c", 101, "c", 300, "+", "-"),      # fragment 101-300 (200 bp)
                    ("c", 500, "c", 351, "-", "+"),      # reverse read listed first: 351-500
                    ("c", 100, "c", 300, "+", "+"),      # same strand
                    ("c", 300, "c", 100, "+", "-"),      # outward
                    ("c", 100, "c", 140, "+", "-"),      # too short
                    ("c", 100, "d", 300, "+", "-")])     # trans
    out = list(om.fragments(p, 100, 1000))
    frags = [(c, s.tolist(), e.tolist()) for c, s, e in out if c is not None]
    counters = out[-1][1]
    assert frags == [("c", [100, 350], [300, 500])]
    assert (counters["fragments"], counters["same_strand"], counters["outward"],
            counters["short"], counters["trans"]) == (2, 1, 1, 1, 1)


@pytest.mark.parametrize("res", [1, 10])
def test_fragment_coverage_and_bigwig(tmp_path, res):
    p = tmp_path / "x.pairs"
    write_pairs(p, [("c", 101, "c", 300, "+", "-"), ("c", 151, "c", 300, "+", "-")])
    cov, lengths, n = om.fragment_coverage(p, {"c": 1000}, res, 100, 1000)
    exact = np.zeros(1000); exact[100:300] += 1; exact[150:300] += 1
    np.testing.assert_allclose(cov["c"], exact.reshape(-1, res).mean(axis=1))
    assert lengths[200] == 1 and lengths[150] == 1
    bw_path = tmp_path / "s.bw"
    om.write_bigwig(bw_path, cov, {"c": 1000}, res)
    import pyBigWig
    bw = pyBigWig.open(str(bw_path))
    assert bw.stats("c", 150, 300, type="mean")[0] == pytest.approx(2.0)
    assert bw.stats("c", 100, 150, type="mean")[0] == pytest.approx(1.0)


def test_omics_tracks(tmp_path):
    u, r = tmp_path / "u.pairs", tmp_path / "r.pairs"
    write_pairs(u, [("c", 101, "c", 300, "+", "-")])
    write_pairs(r, [("c", 101, "c", 300, "+", "-"), ("c", 601, "c", 800, "+", "-")])
    (tmp_path / "sizes").write_text("c\t1000\n")
    om.omics_tracks(u, r, tmp_path / "sizes", tmp_path / "out")
    for f in ("signal_unrescued.bw", "signal_rescued.bw", "rescued_fraction.bw", "omics_stats.txt",
              "omics_plots.pdf"):
        assert (tmp_path / "out" / f).is_file()
    assert "signal rescued / unrescued : 2.000" in (tmp_path / "out" / "omics_stats.txt").read_text()


def test_ratio_tracks(tmp_path):
    sizes = {"c": 1000}
    for name, ip_extra in (("ip", True), ("inp", False)):
        d = tmp_path / name; d.mkdir()
        for kind in ("unrescued", "rescued"):
            cov = {"c": np.ones(1000)}
            if ip_extra:
                cov["c"][:100] += 3                      # IP: 4x enriched in the first 100 bp
            om.write_bigwig(d / f"signal_{kind}.bw", cov, sizes, 1)
    om.ratio_tracks(tmp_path / "ip", tmp_path / "inp", tmp_path / "out", bin_size=50, pseudocount=0)
    import pyBigWig
    bw = pyBigWig.open(str(tmp_path / "out" / "log2ratio_rescued.bw"))
    scale = 1000 / 1300                                  # IP scaled to the input total
    assert bw.stats("c", 0, 100)[0] == pytest.approx(np.log2(4 * scale), abs=1e-3)
    assert bw.stats("c", 500, 600)[0] == pytest.approx(np.log2(scale), abs=1e-3)
