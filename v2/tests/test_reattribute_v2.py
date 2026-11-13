"""Tests for reattribute_v2.py."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reattribute_v2 as ra    # noqa: E402
import statistics_v2 as st     # noqa: E402


def sam(name, flag, chrom, pos, score, cigar="50M"):
    return (f"{name}\t{flag}\t{chrom}\t{pos}\t255\t{cigar}\t*\t0\t0\t{'A' * 50}\t{'I' * 50}"
            f"\tAS:i:{score}\tYT:Z:UU\n")


def test_combination_indices():
    i1, i2, starts, k = ra.combination_indices([2, 1], [3, 2])
    assert k.tolist() == [6, 2] and starts.tolist() == [0, 6]
    # pair 0: R1 0-1 × R2 0-2 ; pair 1: R1 2 × R2 3-4
    assert i1.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    assert i2.tolist() == [0, 1, 2, 0, 1, 2, 3, 4]


def test_weighted_draw_frequencies_and_zero_segments():
    rng = np.random.default_rng(0)
    w = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 5.0])
    starts, sizes = np.array([0, 2, 5]), np.array([2, 3, 1])
    draws = np.array([ra.weighted_draw(w, starts, sizes, rng) for _ in range(4000)])
    assert np.mean(draws[:, 0] == 1) == pytest.approx(0.75, abs=0.03)    # 3 / (1 + 3)
    assert set(np.unique(draws[:, 1])) == {2, 3, 4}                      # all 0 → uniform
    assert np.all(draws[:, 2] == 5)
    w_bad = np.array([np.nan, 1.0, -2.0, np.inf])
    d = ra.weighted_draw(w_bad, np.array([0, 2]), np.array([2, 2]), rng)
    assert d[0] == 1 and d[1] in (2, 3)


def test_best_candidates():
    block = [("chrA", 10, False, 60), ("chrA", 500, True, 60), ("chrB", 7, False, 52)]
    assert len(ra.best_candidates(block)) == 2
    assert len(ra.best_candidates(block, as_delta=10)) == 3
    assert ra.best_candidates([]) == []


def test_read_blocks_groups_by_read_and_drops_unmapped():
    lines = ["@HD\tVN:1.0\n",
             sam("r1", 0, "chrA", 100, 60), sam("r1", 256, "chrB", 200, 60),
             sam("r2", 16, "chrA", 1000, 50),
             "r3\t4\t*\t0\t0\t*\t*\t0\t0\tAAAA\tIIII\tYT:Z:UU\n"]
    blocks = list(ra.read_blocks(lines))
    assert [b[0] for b in blocks] == ["r1", "r2", "r3"]
    assert blocks[0][1] == [("chrA", 100, False, 60), ("chrB", 200, False, 60)]
    assert blocks[1][1] == [("chrA", 1049, True, 50)]                    # reverse: 5' = end
    assert blocks[2][1] == []


@pytest.fixture
def stats_dir(tmp_path):
    """Tiny statistics folder: two chromosomes, contacts mostly at short distance."""
    rng = np.random.default_rng(1)
    lines = ["## pairs format v1.0\n"]
    for k in range(20000):
        c = "chrA" if rng.random() < 0.5 else "chrB"
        p = int(rng.integers(1, 90_000))
        if rng.random() < 0.97:
            q = min(p + int(rng.exponential(3000)) + 1, 100_000)
            lines.append(f"r{k}\t{c}\t{p}\t{c}\t{q}\t+\t-\n")
        else:
            lines.append(f"r{k}\tchrA\t{p}\tchrB\t{int(rng.integers(1, 100_000))}\t+\t+\n")
    pairs = tmp_path / "unrescued.pairs"
    pairs.write_text("".join(lines))
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chrA\t100000\nchrB\t100000\n")
    out = tmp_path / "stats"
    st.generate_statistics(pairs, sizes, out, bin_size=2000, smooth=False, plots=False)
    return out


@pytest.mark.parametrize("mode", ra.MODES)
def test_reattribute_stream_end_to_end(stats_dir, mode):
    laws = ra.Laws(stats_dir, mode)
    # R1 has 2 copies (chrA 10000 and chrB 50000); R2 is unique near the first copy
    r1 = [sam("p1", 0, "chrA", 10_000, 60), sam("p1", 256, "chrB", 50_000, 60),
          sam("p1", 256, "chrB", 80_000, 40)]            # worse score: dropped
    r2 = [sam("p1", 16, "chrA", 11_951, 60)]              # 5' = 12000
    out = io.StringIO()
    stats = ra.reattribute_stream(r1 * 1, r2 * 1, laws, out, np.random.default_rng(0))
    assert stats["placed"] == 1 and stats["cand1"] == 2 and stats["cand2"] == 1
    fields = out.getvalue().split("\t")
    assert len(fields) == 9 and fields[-2:] == ["2", "1\n"]
    if mode != "random":
        # cis at 2 kb is far more likely than trans: draw many times
        wins = 0
        for s in range(200):
            o = io.StringIO()
            ra.reattribute_stream(r1, r2, laws, o, np.random.default_rng(s))
            wins += o.getvalue().split("\t")[1] == "chrA"
        assert wins > 190


def test_reattribute_stream_detects_desync(stats_dir):
    laws = ra.Laws(stats_dir, "ps")
    with pytest.raises(ValueError):
        ra.reattribute_stream([sam("a", 0, "chrA", 10, 60)], [sam("b", 0, "chrA", 10, 60)],
                              laws, io.StringIO(), np.random.default_rng(0))
