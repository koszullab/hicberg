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
          sam("p1", 256, "chrB", 80_000, 40)]            # worse score: weight ~1e-6
    r2 = [sam("p1", 16, "chrA", 11_951, 60)]              # 5' = 12000
    out = io.StringIO()
    stats = ra.reattribute_stream(r1 * 1, r2 * 1, laws, out, np.random.default_rng(0))
    assert stats["placed"] == 1 and stats["cand1"] == 3 and stats["cand2"] == 1   # AS 40 kept (weight e^-14)
    fields = out.getvalue().split("\t")
    assert len(fields) == 9 and fields[-2:] == ["3", "1\n"]
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


def test_score_lambda_and_weights():
    assert ra.score_lambda(0.01) == pytest.approx(np.log(297) / 8)
    assert np.isinf(ra.score_lambda(0))
    w = ra.alignment_weights([70, 62, 70], ra.score_lambda(0.01))
    np.testing.assert_allclose(w, [1, 1 / 297, 1], rtol=1e-6)        # 1 mismatch = ×1/297
    np.testing.assert_allclose(ra.alignment_weights([70, 62, 70], np.inf), [1, 0, 1])


def test_many_near_copies_share_the_reads(stats_dir):
    # poly-T-like read: 1 perfect locus, 300 loci at 1 mismatch (AS -8), mode random:
    # the perfect copy gets ~1 / (1 + 300/297) ≈ 50% of the draws, not 100%
    laws = ra.Laws(stats_dir, "random")
    r1 = [sam("p", 0, "chrA", 1000, 70)] + [sam("p", 256, "chrB", 100 + 200 * k, 62) for k in range(300)]
    r2 = [sam("p", 0, "chrA", 50_000, 70)]
    hits = 0
    for seed in range(400):
        out = io.StringIO()
        ra.reattribute_stream(r1, r2, laws, out, np.random.default_rng(seed))
        hits += out.getvalue().split("\t")[2] == "1000"
    assert 0.40 < hits / 400 < 0.60
    out = io.StringIO()                                                  # error_rate 0: best only
    ra.reattribute_stream(r1, r2, laws, out, np.random.default_rng(0), error_rate=0)
    assert out.getvalue().split("\t")[2] == "1000"


@pytest.mark.parametrize("mode", ["ps", "full"])
def test_hierarchical_draw_matches_exact(stats_dir, mode):
    # 40 × 40 candidates spread on two chromosomes, some R1/R2 copies close together:
    # the hierarchical draw (near exact + importance-sampled far) must reproduce the exact
    # distribution of the chosen combination
    laws = ra.Laws(stats_dir, mode)
    rng = np.random.default_rng(3)
    n = 40
    c1 = rng.integers(0, 2, n); p1 = rng.integers(1, 99_000, n)
    c2 = c1.copy(); p2 = np.clip(p1 + rng.integers(-3000, 3000, n), 1, 99_999)
    c2[::2] = rng.integers(0, 2, n // 2); p2[::2] = rng.integers(1, 99_000, n // 2)
    r1, r2 = rng.random(n) < .5, rng.random(n) < .5
    a1, a2 = rng.random(n) + .1, rng.random(n) + .1
    i1, i2, _, _ = ra.combination_indices([n], [n])
    w = laws.propensity(c1[i1], p1[i1], r1[i1], c2[i2], p2[i2], r2[i2]) * a1[i1] * a2[i2]
    exact = w / w.sum()
    counts = np.zeros(n * n)
    draws = 6000
    for s in range(draws):
        i, j, n_eval = ra.hierarchical_draw(laws, (c1, p1, r1, a1), (c2, p2, r2, a2),
                                            np.random.default_rng(s), near=10_000, n_samples=64)
        counts[i * n + j] += 1
    # total variation between the empirical and exact distributions, vs the sampling noise of
    # an exact sampler with the same number of draws
    tv = 0.5 * np.abs(counts / draws - exact).sum()
    ref = np.bincount(np.random.default_rng(0).choice(n * n, draws, p=exact), minlength=n * n)
    tv_ref = 0.5 * np.abs(ref / draws - exact).sum()
    assert tv < tv_ref + 0.03
    assert n_eval < n * n


def test_reattribute_stream_hierarchical_switch(stats_dir):
    laws = ra.Laws(stats_dir, "full")
    r1 = [sam("p", 0, "chrA", 10_000, 60)] + [sam("p", 256, "chrB", 1000 + 3000 * k, 60) for k in range(30)]
    r2 = [sam("p", 16, "chrA", 11_951, 60)] + [sam("p", 256, "chrB", 60_000 + 1000 * k, 60) for k in range(30)]
    out = io.StringIO()
    stats = ra.reattribute_stream(r1, r2, laws, out, np.random.default_rng(0), max_combinations=100,
                                  near=5000, n_samples=64)
    assert stats["sampled"] == 1 and stats["placed"] == 1 and stats["evaluated"] < 31 * 31
    assert len(out.getvalue().split("\t")) == 9


def test_density_of_uses_the_refined_map(stats_dir):
    laws = ra.Laws(stats_dir, "full")
    # fake refined map for chrA only, at 1 kb, constant 5
    laws.fine = np.array([c == "chrA" for c in laws.names])
    laws.dens_bin_fine = 1000
    laws.dens_n_fine = np.array([100 if c == "chrA" else 1 for c in laws.names])
    laws.density_fine = {("chrA", "chrA"): np.full((100, 100), 5.0)}
    a, b = laws.code["chrA"], laws.code["chrB"]
    out = laws.density_of(np.array([a, a]), np.array([1000, 1000]), np.array([a, b]), np.array([9000, 9000]))
    assert out[0] == 5.0 and out[1] != 5.0          # chrA x chrA refined; chrA x chrB global


def test_flat_density_is_memory_mapped_and_identical(stats_dir):
    path = stats_dir / st.DENSITY_MAP
    ref = ra.Laws(stats_dir, "full")
    assert isinstance(ref.density, dict)
    ra.flatten_density(path)
    flat, index = ra.flat_density_paths(path)
    assert flat.is_file() and index.is_file()
    laws = ra.Laws(stats_dir, "full")
    assert isinstance(laws.density, ra.FlatBlocks)
    assert set(laws.density) == set(ref.density)
    for k, v in ref.density.items():
        assert isinstance(laws.density[k], np.memmap)
        np.testing.assert_array_equal(laws.density[k], v)
    rng = np.random.default_rng(0)
    c1, c2 = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    p1, p2 = rng.integers(1, 100_000, 500), rng.integers(1, 100_000, 500)
    np.testing.assert_array_equal(laws.density_of(c1, p1, c2, p2), ref.density_of(c1, p1, c2, p2))
    np.testing.assert_array_equal(laws.coarse(100_000), ref.coarse(100_000))
    # a newer density file (statistics run again) makes the flat copy stale
    import os
    os.utime(path, (flat.stat().st_mtime + 10,) * 2)
    assert isinstance(ra.Laws(stats_dir, "full").density, dict)


def test_chunk_resume(tmp_path, monkeypatch):
    out = tmp_path / "predicted_chunk0.pairs"
    calls = []

    def fake(job):
        calls.append(job["out"])
        Path(job["out"]).write_text("x\n")
        return dict(ra._new_stats(), pairs=3, seconds=1.5)

    monkeypatch.setattr(ra, "_reattribute_chunk", fake)
    stats = ra.reattribute_chunk({"out": out})
    assert ra.chunk_done(out) and stats["pairs"] == 3
    out.write_text("x\n")                    # pairs file rewritten after the stats → not done
    import os
    os.utime(out, (ra.chunk_stats_path(out).stat().st_mtime + 10,) * 2)
    assert not ra.chunk_done(out)


def test_excluded_chromosome_never_receives_reads(stats_dir):
    laws = ra.Laws(stats_dir, "full", exclude=["chrB"])
    assert "chrB" not in laws.code and laws.excluded == {"chrB"} and laws.names == ["chrA", "chrB"]
    # R1 identical on chrA and chrB (PAR-like), R2 on chrA far away → always chrA
    r1 = [sam("p1", 0, "chrA", 10_000, 60), sam("p1", 256, "chrB", 10_000, 60)]
    r2 = [sam("p1", 16, "chrA", 60_000, 60), sam("p1", 272, "chrB", 60_000, 60)]
    for s in range(50):
        out = io.StringIO()
        ra.reattribute_stream(r1, r2, laws, out, np.random.default_rng(s))
        f = out.getvalue().split("\t")
        assert f[1] == "chrA" and f[3] == "chrA"
    # a mate only on the excluded chromosome → unplaced
    out = io.StringIO()
    stats = ra.reattribute_stream(r1, [sam("p1", 16, "chrB", 60_000, 60)], laws, out, np.random.default_rng(0))
    assert stats["unplaced"] == 1 and out.getvalue() == ""
    # index-only path: a unique first alignment on the excluded chromosome gives no candidate
    c, bins = ra.index_read_candidates(None, laws.code, b"A", b"I", ("chrB", 100, False, 60, True), 30, 1.0)
    assert c == [] and bins is False


def test_density_diagonal_below_min_distance(stats_dir):
    laws = ra.Laws(stats_dir, "full")
    a, b = laws.code["chrA"], laws.code["chrB"]
    c = np.array([a, a, a]); p1 = np.array([10_000, 10_000, 10_000])
    c2 = np.array([a, a, b]); p2 = np.array([12_000, 60_000, 12_000])
    laws.density_fine = {}
    laws.dens_min = 0
    raw = laws.density_of(c, p1, c2, p2)
    laws.dens_min = 5000
    out = laws.density_of(c, p1, c2, p2)
    diag = laws.density_any(c[:1], p1[:1], c[:1], p1[:1])[0]
    assert out[0] == diag and out[1] == raw[1] and out[2] == raw[2]   # < 5 kb: diagonal value


def test_read_hits_with_and_without_nm(tmp_path):
    old = tmp_path / "old.tsv"
    old.write_text("r1\tchrA\t10\t+\t60\t0\tchrB\t20\t-\t50\t1\n")
    new = tmp_path / "new.tsv"
    new.write_text("r1\tchrA\t10\t+\t60\t0\t5\tchrB\t20\t-\t50\t1\t0\n")
    (_, a1, a2), = ra.read_hits(old)
    (_, b1, b2), = ra.read_hits(new)
    assert a1 == ("chrA", 10, False, 60, False, -1) and a2 == ("chrB", 20, True, 50, True, -1)
    assert b1 == ("chrA", 10, False, 60, False, 5) and b2 == ("chrB", 20, True, 50, True, 0)


class _NoCandidates:
    def candidates(self, seq, min_fraction=0.8):
        return [], np.zeros(0, np.int64), np.zeros(0)


def test_degraded_read_without_index_candidate_not_placed():
    codes = {"chrA": 0}
    idx = _NoCandidates()
    st = ra._new_stats()
    hit = ("chrA", 100, False, 150, False, 5)               # ambiguous, NM 5, no index candidate
    assert ra.index_read_candidates(idx, codes, b"A", b"I", hit, 30, 1.0, stats=st)[0]   # rule off
    c, _ = ra.index_read_candidates(idx, codes, b"A", b"I", hit, 30, 1.0, stats=st, max_fallback_nm=4)
    assert c == [] and st["unplaceable_fallback"] == 1
    ok = ("chrA", 100, False, 150, False, 1)                 # NM 1: kept at its first alignment
    assert ra.index_read_candidates(idx, codes, b"A", b"I", ok, 30, 1.0, stats=st, max_fallback_nm=4)[0]
    uq = ("chrA", 100, False, 150, True, 6)                  # unique but degraded, no copy
    c, _ = ra.index_read_candidates(idx, codes, b"A", b"I", uq, 30, 1.0, stats=st, max_fallback_nm=4)
    assert c == [] and st["unplaceable_unique"] == 1
    old = ("chrA", 100, False, 150, False, -1)               # NM unknown (old hits): kept
    assert ra.index_read_candidates(idx, codes, b"A", b"I", old, 30, 1.0, stats=st, max_fallback_nm=4)[0]


def test_fallback_read_found_through_its_reference(tmp_path):
    """A short read with one error in the middle has no intact k-mer (no repeated minimizer): its
    copies are found through the reference at its first alignment, then the read is scored in each."""
    import repeat_index_v2 as rix
    rng = np.random.default_rng(0)
    g = rng.choice(list("ACGT"), 30000)
    g[20000:20600] = g[5000:5600]                             # two identical copies
    genome = tmp_path / "g.fa"
    genome.write_text(">chrA\n" + "".join(g) + "\n")
    rix.build_repeat_index(genome, tmp_path / "rix", log=lambda m: None)
    idx = rix.RepeatIndex(tmp_path / "rix")
    read = list(g[5100:5140])
    read[20] = {"A": "C", "C": "G", "G": "T", "T": "A"}[read[20]]
    read = "".join(read).encode()
    assert len(idx.candidates(read, 0.3)[0]) == 0             # no repeated minimizer in the read
    assert idx.reference_segment("chrA", 5101, False, 40) == "".join(g[5100:5140]).encode()
    st = ra._new_stats()
    c, bins = ra.index_read_candidates(idx, {"chrA": 0}, read, b"I" * 40, ("chrA", 20101, False, 70, False),
                                       30, 1.0, stats=st)
    assert st["index_ref_lookup"] == 1 and st["index_fallback"] == 0
    assert {p // 1000 for _, p, *_ in c} >= {5, 20}           # both copies are candidates
    # same read sequenced from the reverse strand (5' = 0-based exclusive end of the alignment)
    rc = read.translate(bytes.maketrans(b"ACGT", b"TGCA"))[::-1]
    st = ra._new_stats()
    c, _ = ra.index_read_candidates(idx, {"chrA": 0}, rc, b"I" * 40, ("chrA", 20140, True, 70, False),
                                    30, 1.0, stats=st)
    assert st["index_ref_lookup"] == 1 and st["index_fallback"] == 0
    assert {p // 1000 for _, p, *_ in c} >= {5, 20} and {r for _, _, r, *_ in c} == {1}


def test_degraded_unique_read_requalified(tmp_path):
    """A unique read (MAPQ) with NM >= requalify_nm is re-examined: its copies become candidates."""
    import repeat_index_v2 as rix
    rng = np.random.default_rng(1)
    g = rng.choice(list("ACGT"), 30000)
    g[20000:20600] = g[5000:5600]
    genome = tmp_path / "g.fa"
    genome.write_text(">chrA\n" + "".join(g) + "\n")
    rix.build_repeat_index(genome, tmp_path / "rix", log=lambda m: None)
    idx = rix.RepeatIndex(tmp_path / "rix")
    read = "".join(g[5100:5250]).encode()
    st = ra._new_stats()
    hit = ("chrA", 5101, False, 280, True, 5)                 # unique by MAPQ, NM 5
    c, _ = ra.index_read_candidates(idx, {"chrA": 0}, read, b"I" * 150, hit, 30, 1.0, stats=st)
    assert len(c) == 1                                          # off: kept at its first alignment
    c, _ = ra.index_read_candidates(idx, {"chrA": 0}, read, b"I" * 150, hit, 30, 1.0, stats=st, requalify_nm=4)
    assert st["unique_requalified"] == 1 and {p // 1000 for _, p, *_ in c} >= {5, 20}
    ok = ("chrA", 5101, False, 300, True, 1)                    # NM 1: unique as before
    assert len(ra.index_read_candidates(idx, {"chrA": 0}, read, b"I" * 150, ok, 30, 1.0, stats=st, requalify_nm=4)[0]) == 1
