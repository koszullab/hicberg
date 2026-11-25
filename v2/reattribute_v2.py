#!/usr/bin/env python3
"""
reattribute_v2.py

Re-attribution of the ambiguous pairs (group2) to one of their possible
positions, drawn at random with a probability proportional to a propensity
computed from the statistical laws of the unambiguous pairs (statistics_v2.py).

For each group2 chunk (group2_1_chunk{i}.fastq / group2_2_chunk{i}.fastq,
streaming_classifier_v2.py):
  1. bowtie2 --very-sensitive-local --reorder -k K (or -a) on R1 and R2,
     read as a stream (no SAM written on disk), alignments grouped by read
  2. candidates of a read = its alignments within --as_delta of the best
     score (AS), each weighted by its likelihood exp(-lambda × (best AS - AS)),
     lambda from --error_rate (1 mismatch = ×1/300 at 1%). The previous
     version kept all the -k alignments with the same weight (~26% of them
     have at least one more mismatch than the best one on yeast); a first v2
     kept the best-score ones only, which sent every low-complexity read
     (poly-T: 1 perfect locus, hundreds at 1 mismatch) to the same few loci
     (artefactual contact spots); the likelihood weighting spreads them
  3. propensity of every R1 × R2 combination, vectorised over batches of
     pairs, with the same conventions as the statistics (5' positions,
     orientation after ordering by position, circular distances):
        random    1
        ps        P(s) (orientation, distance) or trans P(s)
        standard  P(s) × coverage(R1) × coverage(R2)
        full      P(s) × density(R1, R2)
  4. one combination drawn per pair (weighted, seeded → reproducible)
  5. predicted.pairs: same format as unrescued.pairs + the numbers of
     candidates of R1 and R2

CLI:
  python3 reattribute_v2.py classifier_dir stats_dir output_dir \
      --index genome_index --mode full --max_alignments all --n_workers 4
"""

import argparse
import gc
import json
import multiprocessing as mp
import re
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

import statistics_v2 as st
from streaming_classifier_v2 import split_sam, fast_outer_5prime

MODES = ("random", "ps", "standard", "full")
_AS = re.compile(r"(?:^|\t)AS:i:(-?\d+)")


# ---------------------------------------------------------------------------
# Alignments
# ---------------------------------------------------------------------------

def bowtie2_command(index, fastq, max_alignments="100", threads=1):
    """bowtie2 command writing SAM to stdout (-k K, or -a for 'all')."""
    k = ["-a"] if str(max_alignments) == "all" else ["-k", str(max_alignments)]
    return ["bowtie2", "--very-sensitive-local", "--reorder", *k, "-p", str(threads),
            "-x", str(index), "-U", str(fastq)]


_COMP = bytes.maketrans(b"ACGTNacgtn", b"TGCANtgcan")


def read_blocks(lines, with_seq=False):
    """
    Group the alignment lines of a SAM stream by read (consecutive with
    --reorder). Yields (read name, [(chrom, 5' position, is_reverse, AS), ...]),
    unmapped alignments left out (empty list if the read is unmapped).
    with_seq: yields (name, block, read sequence in its sequencing orientation, bytes).
    """
    name, block, seq = None, [], b""
    for line in lines:
        f = split_sam(line)
        if f is None:
            continue
        if f[0] != name:
            if name is not None:
                yield (name, block, seq) if with_seq else (name, block)
            name, block = f[0], []
            if with_seq:
                seq = f[9].encode()
                if int(f[1]) & 16:
                    seq = seq.translate(_COMP)[::-1]
        flag = int(f[1])
        if flag & 4:
            continue
        m = _AS.search(f[11])
        score = int(m.group(1)) if m else 0
        rev = bool(flag & 16)
        block.append((f[2], fast_outer_5prime(f, rev), rev, score))
    if name is not None:
        yield (name, block, seq) if with_seq else (name, block)


def best_candidates(block, as_delta=0):
    """Alignments whose score is >= best score - as_delta."""
    if not block:
        return block
    best = max(a[3] for a in block)
    return [a for a in block if a[3] >= best - as_delta]


def score_lambda(error_rate, mismatch_cost=8):
    """
    Log-likelihood cost of one alignment-score point. In bowtie2 local mode a
    high-quality mismatch replaces the +2 match bonus by a -6 penalty (8
    points), and is e / (3 (1 - e)) times less likely than a correct base:
    lambda = ln(3 (1 - e) / e) / 8 (0.71 for e = 1%). error_rate <= 0 →
    infinite (only the best-score alignments count).
    """
    if error_rate <= 0:
        return np.inf
    return np.log(3 * (1 - error_rate) / error_rate) / mismatch_cost


def alignment_weights(scores, lam):
    """Relative likelihood of each alignment of a read: exp(-lambda × (best AS - AS))."""
    scores = np.asarray(scores, dtype=np.float64)
    gap = scores.max() - scores
    if np.isinf(lam):
        return (gap == 0).astype(np.float64)
    return np.exp(-lam * gap)


# ---------------------------------------------------------------------------
# Statistical laws (outputs of statistics_v2.py)
# ---------------------------------------------------------------------------

def flat_density_paths(path):
    """Memory-mappable copy of a density file: density_map.flat.npy + density_map.flat_index.npy."""
    path = Path(path)
    return path.with_suffix(".flat.npy"), path.with_suffix(".flat_index.npy")


def flatten_density(path):
    """
    Write the dict of density blocks of `path` (pickled, loaded whole by np.load) as one float32
    array, memory-mappable, + an index {(chrom_a, chrom_b): (offset, shape)}. The workers then
    share a single copy of the density through the page cache instead of loading one each
    (human, 50 kb bins: 8 GB per worker). Skipped when the flat copy is newer than `path`.
    """
    path = Path(path)
    flat, index_path = flat_density_paths(path)
    if flat.is_file() and index_path.is_file() \
            and min(flat.stat().st_mtime, index_path.stat().st_mtime) >= path.stat().st_mtime:
        return flat, index_path
    maps = np.load(path, allow_pickle=True).item()
    tmp = flat.with_name(flat.stem + ".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                    shape=(sum(v.size for v in maps.values()),))
    index, off = {}, 0
    for k in list(maps):
        v = maps.pop(k)                                   # freed block by block
        arr[off:off + v.size] = np.asarray(v, dtype=np.float32).ravel()
        index[k] = (off, tuple(v.shape))
        off += v.size
    arr.flush()
    del arr, maps
    np.save(index_path, index)
    tmp.replace(flat)                                     # complete copy only
    return flat, index_path


class FlatBlocks(Mapping):
    """Read-only dict of density blocks backed by the memory-mapped flat copy (flatten_density)."""

    def __init__(self, path):
        flat, index_path = flat_density_paths(path)
        self._arr = np.load(flat, mmap_mode="r")
        self._index = np.load(index_path, allow_pickle=True).item()

    def __getitem__(self, key):
        off, shape = self._index[key]
        return self._arr[off:off + int(np.prod(shape))].reshape(shape)

    def __iter__(self):
        return iter(self._index)

    def __len__(self):
        return len(self._index)


def load_density(path):
    """Density blocks: memory-mapped flat copy if present (and up to date), else the dict."""
    flat, index_path = flat_density_paths(path)
    if flat.is_file() and index_path.is_file() and flat.stat().st_mtime >= Path(path).stat().st_mtime:
        return FlatBlocks(path)
    return np.load(path, allow_pickle=True).item()


class Laws:
    """Laws of a statistics folder, stored as flat arrays for vectorised lookups."""

    def __init__(self, stats_dir, mode, circular=None, bin_size=2000, exclude=()):
        d = Path(stats_dir)
        load = lambda name: np.load(d / name, allow_pickle=True).item()
        sizes = load(st.CHROM_SIZES)
        self.names = list(sizes)
        # excluded chromosomes (e.g. chrY of a female sample): absent from `code`, so that no
        # candidate is placed on them (the candidate filters test `chrom in laws.code`); their
        # rows stay in the arrays (indices unchanged)
        self.excluded = set(exclude) & set(self.names)
        self.code = {c: i for i, c in enumerate(self.names) if c not in self.excluded}
        self.sizes = np.array([sizes[c] for c in self.names], dtype=np.int64)
        self.bin_size = bin_size
        self.mode = mode
        if circular is None:
            circular = read_circular(d)
        self.circular = np.array([c in circular for c in self.names])

        # P(s): law[code, type, xs index], types 0 uncut, 1 circle, 2 weird, 3 orientation
        # unknown (repeat-index candidates, strand set after the draw): mean of the 4 orientations
        xs = load(st.XS)
        self.xs_glob = max(xs.values(), key=len)
        self.n_xs = np.array([len(xs[c]) for c in self.names])
        self.law = np.zeros((len(self.names), 4, len(self.xs_glob)))
        for t, name in enumerate((st.UNCUTS, st.CIRCLES, st.WEIRDS)):
            law = load(name)
            for k, c in enumerate(self.names):
                self.law[k, t, :len(law[c])] = law[c]
        self.law[:, 3] = (self.law[:, 0] + self.law[:, 1] + 2 * self.law[:, 2]) / 4
        self.index_bin = 0                                  # set by set_index_bin()
        trans = load(st.TRANS_PS)
        self.trans = np.array([[trans.get((a, b), 0.0) for b in self.names] for a in self.names])

        if mode == "standard":
            cov = load(st.COVERAGE)
            self.cov_n = np.array([len(cov[c]) for c in self.names])
            self.cov_off = np.concatenate([[0], np.cumsum(self.cov_n)[:-1]])
            self.cov = np.concatenate([np.asarray(cov[c], dtype=np.float64) for c in self.names])
        if mode == "full":
            self.density = load_density(d / st.DENSITY_MAP)
            self.dens_n = np.array([self.density[(c, c)].shape[0] for c in self.names])
            # the density may be binned more coarsely than the coverage (large genomes)
            self.dens_bin = int(np.load(d / st.DENSITY_BIN)) if (d / st.DENSITY_BIN).is_file() \
                else bin_size
            # cis pairs closer than this were left out of the density: density 1 (P(s) only)
            self.dens_min = int(np.load(d / st.DENSITY_MIN_DIST)) \
                if (d / st.DENSITY_MIN_DIST).is_file() else 0
            # refined density of the small, densely covered molecules (statistics_v2)
            self.fine = np.zeros(len(self.names), dtype=bool)
            if (d / st.DENSITY_FINE_MAP).is_file():
                self.density_fine = load_density(d / st.DENSITY_FINE_MAP)
                self.dens_bin_fine = int(np.load(d / st.DENSITY_FINE_BIN))
                fine_names = {a for a, _ in self.density_fine}
                self.fine = np.array([c in fine_names for c in self.names])
                self.dens_n_fine = np.array([self.density_fine[(c, c)].shape[0] if c in fine_names else 1
                                             for c in self.names])

    def set_index_bin(self, B):
        """Repeat-index candidates sit at bin centres: two of them in the same bin are at an
        unknown distance in [0, B) (triangular law of |x - y| for x, y uniform in the bin), so
        their P(s) is the orientation-averaged law averaged over that distance law."""
        self.index_bin = B
        x = self.xs_glob.astype(np.float64)
        width = np.diff(np.append(x, x[-1] * 1.1))
        w = width * np.clip(B - x, 0, None)
        self.law_same_bin = np.array([(self.law[c, 3, :self.n_xs[c]] * w[:self.n_xs[c]]).sum()
                                      / max(w[:self.n_xs[c]].sum(), 1e-300)
                                      for c in range(len(self.names))])

    def ps(self, c1, p1, r1, c2, p2, r2):
        """P(s) of each combination (intra: by orientation and distance; trans: trans P(s)).
        Strands: False / 0 forward, True / 1 reverse, 2 unknown (orientation-averaged law)."""
        out = self.trans[c1, c2].astype(np.float64)
        cis = c1 == c2
        if cis.any():
            c, a, b = c1[cis], p1[cis], p2[cis]
            s1, s2 = np.asarray(r1)[cis], np.asarray(r2)[cis]
            orient = st.pair_orientation(a, b, s1 == 1, s2 == 1)
            kind = np.minimum(orient, 2)                       # ++ and -- → weird
            unknown = (s1 == 2) | (s2 == 2)
            kind[unknown] = 3
            d = st.cis_distance(a, b, self.sizes[c], self.circular[c])
            idx = np.minimum(st.attribute_xs(self.xs_glob, d), self.n_xs[c] - 1)
            out[cis] = self.law[c, kind, idx]
            if self.index_bin:
                same = (s1 == 2) & (s2 == 2) & (d < self.index_bin)
                if same.any():
                    sub = out[cis]
                    sub[same] = self.law_same_bin[c[same]]
                    out[cis] = sub
        return out

    def coverage(self, c, p):
        b = np.minimum((p - 1) // self.bin_size, self.cov_n[c] - 1)
        return self.cov[self.cov_off[c] + np.maximum(b, 0)]

    def density_of(self, c1, p1, c2, p2):
        """Density of each combination (density_any); below the density's minimum distance
        (statistics_v2 density_min_distance: those pairs were left out of the density, P(s) by
        orientation describes them) the density is held constant at its diagonal value (the
        density of the first position's own bin): no step at that distance. A density of 1
        there made a step against the interpolated value of the satellite arrays (0.5-0.9) and
        pulled their contacts below 5 kb (known-truth test)."""
        out = self.density_any(c1, p1, c2, p2)
        if getattr(self, "dens_min", 0) > 0:
            cis = np.flatnonzero(c1 == c2)
            if len(cis):
                c = c1[cis]
                near = cis[st.cis_distance(p1[cis], p2[cis], self.sizes[c], self.circular[c]) < self.dens_min]
                if len(near):
                    out[near] = self.density_any(c1[near], p1[near], c1[near], p1[near])
        return out

    def density_any(self, c1, p1, c2, p2):
        """Density of each combination; blocks are stored for chromosome a <= b. Pairs of
        molecules that have a refined density map (both in it) use it."""
        both = self.fine[c1] & self.fine[c2]
        if both.any():
            out = np.empty(len(c1))
            out[both] = self._density(c1[both], p1[both], c2[both], p2[both], self.density_fine,
                                      self.dens_bin_fine, self.dens_n_fine)
            if (~both).any():
                out[~both] = self._density(c1[~both], p1[~both], c2[~both], p2[~both], self.density,
                                           self.dens_bin, self.dens_n)
            return out
        return self._density(c1, p1, c2, p2, self.density, self.dens_bin, self.dens_n)

    def _density(self, c1, p1, c2, p2, density, dens_bin, dens_n):
        b1 = np.clip((p1 - 1) // dens_bin, 0, dens_n[c1] - 1)
        b2 = np.clip((p2 - 1) // dens_bin, 0, dens_n[c2] - 1)
        swap = c1 > c2
        ca, cb = np.where(swap, c2, c1), np.where(swap, c1, c2)
        ba, bb = np.where(swap, b2, b1), np.where(swap, b1, b2)
        out = np.empty(len(c1))
        key = ca * len(self.names) + cb
        for k in np.unique(key):
            sel = key == k
            blk = density[(self.names[k // len(self.names)], self.names[k % len(self.names)])]
            out[sel] = blk[ba[sel], bb[sel]]
        return out

    def coarse(self, near):
        """
        Chromosome × chromosome proposal law of the hierarchical draw (far combinations):
        trans P(s) × mean trans density, intra: mean P(s) beyond `near` (pairs of positions
        uniform on the chromosome, orientations averaged) × mean intra density. Only the
        efficiency of the draw depends on it (the exact weights correct the proposal).
        """
        if getattr(self, "_coarse_near", None) == near:
            return self._coarse
        n = len(self.names)
        g = np.ones((n, n)) if self.mode == "random" else self.trans.astype(np.float64).copy()
        if self.mode != "random":
            x = self.xs_glob.astype(np.float64)
            width = np.diff(np.append(x, x[-1] * 1.1))
            for c in range(n):
                m = self.n_xs[c]
                law = (self.law[c, 0, :m] + self.law[c, 1, :m] + 2 * self.law[c, 2, :m]) / 4
                L = self.sizes[c] / (2 if self.circular[c] else 1)
                w = width[:m] * np.clip(L - x[:m], 0, None) * (x[:m] >= near)
                g[c, c] = (law * w).sum() / w.sum() if w.sum() > 0 else law[-1]
        if self.mode == "full":
            for a in range(n):
                for b in range(a, n):
                    blk = self.density[(self.names[a], self.names[b])]
                    v = float(np.nanmean(blk)) if np.isfinite(blk).any() else 1.0
                    g[a, b] *= v
                    if a != b:
                        g[b, a] *= v
        self._coarse_near, self._coarse = near, np.where(np.isfinite(g) & (g > 0), g, 0.0)
        return self._coarse

    def propensity(self, c1, p1, r1, c2, p2, r2):
        if self.mode == "random":
            return np.ones(len(c1))
        ps = self.ps(c1, p1, r1, c2, p2, r2)
        if self.mode == "ps":
            return ps
        if self.mode == "standard":
            return ps * self.coverage(c1, p1) * self.coverage(c2, p2)
        return ps * self.density_of(c1, p1, c2, p2)


def read_circular(stats_dir):
    """Circular chromosomes recorded in statistics_stats.txt (statistics_v2.py)."""
    path = Path(stats_dir) / "statistics_stats.txt"
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("circular"):
                value = line.split(":", 1)[1].strip()
                return set() if value == "-" else {c.strip() for c in value.split(",")}
    return set()


# ---------------------------------------------------------------------------
# Vectorised draw over a batch of pairs
# ---------------------------------------------------------------------------

def combination_indices(n1, n2):
    """
    For pairs with n1[p] R1 candidates and n2[p] R2 candidates, the index (in
    the concatenated candidate arrays) of R1 and R2 of every combination, and
    the start of each pair in the combination array.
    """
    n1, n2 = np.asarray(n1, dtype=np.int64), np.asarray(n2, dtype=np.int64)
    k = n1 * n2
    starts = np.concatenate([[0], np.cumsum(k)[:-1]])
    pair = np.repeat(np.arange(len(k)), k)
    q = np.arange(k.sum()) - starts[pair]
    i = q // n2[pair]
    j = q % n2[pair]
    start1 = np.concatenate([[0], np.cumsum(n1)[:-1]])
    start2 = np.concatenate([[0], np.cumsum(n2)[:-1]])
    return start1[pair] + i, start2[pair] + j, starts, k


def weighted_draw(weights, starts, sizes, rng):
    """
    One index per segment [starts[p], starts[p] + sizes[p]) of `weights`,
    drawn with probability proportional to the weights (uniform if they are
    all 0 in the segment). Non-finite or negative weights count as 0.
    """
    w = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    cs = np.cumsum(w)
    before = np.concatenate([[0.0], cs])[starts]
    total = np.concatenate([[0.0], cs])[starts + sizes] - before
    u = rng.random(len(starts))
    idx = np.searchsorted(cs, before + u * total, side="right")
    uniform = starts + np.minimum((u * sizes).astype(np.int64), sizes - 1)
    idx = np.where(total > 0, idx, uniform)
    return np.clip(idx, starts, starts + sizes - 1)


# ---------------------------------------------------------------------------
# Hierarchical draw of the pairs with too many combinations
# ---------------------------------------------------------------------------

def near_combinations(laws, c1, p1, c2, p2, near):
    """
    Indices (i, j) of the R1 × R2 candidate combinations on the same chromosome at a
    distance < near (circular distance on circular chromosomes), found by sorting the R2
    positions per chromosome: cost ~ (n1 + n2) log n + number of near combinations.
    """
    out_i, out_j = [], []
    for c in np.intersect1d(c1, c2):
        i = np.flatnonzero(c1 == c)
        j = np.flatnonzero(c2 == c)
        if laws.circular[c]:                      # small molecules: all pairs, then filter
            ii, jj = np.repeat(i, len(j)), np.tile(j, len(i))
            d = st.cis_distance(p1[ii], p2[jj], laws.sizes[c], laws.circular[c])
            keep = d < near
            out_i.append(ii[keep]); out_j.append(jj[keep])
            continue
        order = j[np.argsort(p2[j], kind="stable")]
        ps2 = p2[order]
        lo = np.searchsorted(ps2, p1[i] - near, side="right")
        hi = np.searchsorted(ps2, p1[i] + near, side="left")
        cnt = hi - lo
        if cnt.sum() == 0:
            continue
        ii = np.repeat(i, cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        out_i.append(ii); out_j.append(order[np.repeat(lo, cnt) + offs])
    if not out_i:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(out_i), np.concatenate(out_j)


def hierarchical_draw(laws, cand1, cand2, rng, near=100_000, n_samples=256):
    """
    One combination of a pair with many candidates, without evaluating all n1 × n2
    combinations (coarse-to-fine):
      - near combinations (same chromosome, distance < near), where P(s) varies steeply:
        all evaluated exactly (exact sum W_near);
      - far combinations: n_samples drawn from a hierarchical proposal — chromosome pair
        (A, B) ∝ coarse law(A, B) × Σ alignment weights of the R1 candidates on A × the same
        for R2 on B, then one candidate on A and one on B ∝ their alignment weights — and
        re-weighted by exact weight / proposal probability (importance sampling): the mean
        ratio estimates W_far without bias, and resampling ∝ ratio gives the far draw;
      - near vs far chosen ∝ W_near vs estimated W_far.
    cand = (codes, positions, is_reverse, alignment weights). Returns (i, j, n evaluated).
    """
    c1, p1, r1, a1 = cand1
    c2, p2, r2, a2 = cand2
    ni, nj = near_combinations(laws, c1, p1, c2, p2, near)
    w_near = laws.propensity(c1[ni], p1[ni], r1[ni], c2[nj], p2[nj], r2[nj]) * a1[ni] * a2[nj]
    w_near = np.where(np.isfinite(w_near) & (w_near > 0), w_near, 0.0)
    W_near = w_near.sum()

    # proposal: chromosome pair, then candidates ∝ alignment weights
    g = laws.coarse(near)
    u1, inv1 = np.unique(c1, return_inverse=True)
    u2, inv2 = np.unique(c2, return_inverse=True)
    S1 = np.bincount(inv1, weights=a1)
    S2 = np.bincount(inv2, weights=a2)
    G = g[np.ix_(u1, u2)] * S1[:, None] * S2[None, :]
    Z = G.sum()
    W_far, si, sj, ratio = 0.0, None, None, None
    if Z > 0:
        k = rng.choice(G.size, size=n_samples, p=(G / Z).ravel())
        A, B = k // len(u2), k % len(u2)

        def pick(inv, a, groups, S):
            order = np.argsort(inv, kind="stable")
            cs = np.cumsum(a[order])
            base = np.concatenate([[0.0], np.cumsum(S)])[groups]
            idx = np.searchsorted(cs, base + rng.random(len(groups)) * S[groups], side="right")
            return order[np.minimum(idx, len(order) - 1)]

        si, sj = pick(inv1, a1, A, S1), pick(inv2, a2, B, S2)
        q = g[c1[si], c2[sj]] * a1[si] * a2[sj] / Z
        w = laws.propensity(c1[si], p1[si], r1[si], c2[sj], p2[sj], r2[sj]) * a1[si] * a2[sj]
        same = c1[si] == c2[sj]
        if same.any():                              # near combinations are counted above
            d = st.cis_distance(p1[si][same], p2[sj][same], laws.sizes[c1[si][same]],
                                laws.circular[c1[si][same]])
            w[np.flatnonzero(same)[d < near]] = 0.0
        ratio = np.where(np.isfinite(w) & (w > 0) & (q > 0), w / np.where(q > 0, q, 1), 0.0)
        W_far = ratio.mean()
    total = W_near + W_far
    n_eval = len(ni) + (n_samples if Z > 0 else 0)
    if total <= 0:                                  # nothing informative: uniform
        return int(rng.integers(len(c1))), int(rng.integers(len(c2))), n_eval
    if rng.random() * total < W_near:
        t = np.searchsorted(np.cumsum(w_near), rng.random() * W_near, side="right")
        t = min(t, len(w_near) - 1)
        return int(ni[t]), int(nj[t]), n_eval
    t = np.searchsorted(np.cumsum(ratio), rng.random() * ratio.sum(), side="right")
    t = min(t, len(ratio) - 1)
    return int(si[t]), int(sj[t]), n_eval


# ---------------------------------------------------------------------------
# One chunk
# ---------------------------------------------------------------------------

def index_candidates(repeat_index, seq):
    """
    Candidates of a read whose bowtie2 list is saturated, from the repeat index: every bin
    sharing the read's repeated minimizers, strand unknown (2: orientation-averaged P(s); exact
    position and strand set after the draw by RepeatIndex.place). Returns [(chrom, bin centre,
    2, weight, bin start)] (weight = copies of the read sequence in the bin), or [].
    """
    chroms, starts, w = repeat_index.candidates(seq)
    half = repeat_index.bin_size // 2
    return [(c, b + half, 2, x, b) for c, b, x in zip(chroms, starts.tolist(), w.tolist())]


def _draw_pairs(pairs, laws, out_fh, rng, stats, repeat_index=None, batch_combinations=2_000_000,
                max_combinations=0, near=100_000, n_samples=256, index_limit=200_000,
                near_index=20_000):
    """
    Draw one combination per pair and write it (shared by the bowtie2 and the index-only
    re-attributions). pairs yields (name, cands1, bins1, cands2, bins2, seq1, seq2, quals1,
    quals2): cands = [(chrom, pos, strand, weight, bin start)], strand 0 / 1 / 2 (unknown: a bin
    of the repeat index, bin start >= 0, resolved to an exact position after the draw); bins =
    True if the candidates are index bins. Pairs with more than max_combinations (> 0), or an
    index-bin side and more than index_limit, combinations use the hierarchical draw (exact
    combinations below near, or near_index for index bins).
    """

    def as_arrays(c):
        return (np.array([laws.code[a[0]] for a in c], dtype=np.int64),
                np.array([a[1] for a in c], dtype=np.int64),
                np.array([a[2] for a in c], dtype=np.int8),
                np.array([a[3] for a in c], dtype=np.float64))

    def choices(cand, seq, quals):
        """Exact placements [(pos, rev)] of a chosen candidate (several ties for an index bin)."""
        chrom, pos, rev, _, b = cand
        if b < 0:
            return [(pos, bool(rev))]
        ch = repeat_index.place_choices(seq, chrom, b, quals)
        if not ch:
            stats["index_unplaced"] += 1
            return [(pos, False)]
        return ch

    def pick(c_fixed, fixed, chrom, ch):
        """One placement among ch ∝ P(s) relative to the fixed mate (uniform if trans or 0)."""
        if len(ch) == 1:
            return ch[0]
        if c_fixed == chrom:
            n = len(ch)
            code = np.full(n, laws.code[chrom])
            w = laws.ps(code, np.full(n, fixed[0]), np.full(n, int(fixed[1])),
                        code, np.array([x[0] for x in ch]), np.array([int(x[1]) for x in ch]))
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            if w.sum() > 0:
                return ch[int(np.searchsorted(np.cumsum(w), rng.random() * w.sum(), side="right").clip(0, n - 1))]
        return ch[int(rng.integers(len(ch)))]

    def write(name, a, b, n1, n2, seq1, seq2, qual1, qual2):
        """Exact placement of the pair: a mate with tied placements (tandem copies in the drawn
        bin) is placed ∝ P(s) relative to the other one, so that short-range contacts keep their
        distance and orientation."""
        ch_a, ch_b = choices(a, seq1, qual1), choices(b, seq2, qual2)
        if len(ch_a) > 1 and len(ch_b) > 1:
            pa, ra = ch_a[int(rng.integers(len(ch_a)))]
            pb, rb = pick(a[0], (pa, ra), b[0], ch_b)
        elif len(ch_a) > 1:
            pb, rb = ch_b[0]
            pa, ra = pick(b[0], (pb, rb), a[0], ch_a)
        else:
            pa, ra = ch_a[0]
            pb, rb = pick(a[0], (pa, ra), b[0], ch_b)
        out_fh.write(f"{name}\t{a[0]}\t{pa}\t{b[0]}\t{pb}\t{'-' if ra else '+'}\t{'-' if rb else '+'}\t"
                     f"{n1}\t{n2}\n")
        stats["intra"] += int(a[0] == b[0])

    batch = []                     # (name, cands1, cands2, seq1, seq2, quals1, quals2)
    n_comb = 0

    def flush():
        nonlocal batch, n_comb
        if not batch:
            return
        cands = [b[1] for b in batch], [b[2] for b in batch]
        n1 = np.array([len(c) for c in cands[0]])
        n2 = np.array([len(c) for c in cands[1]])
        arrays = [as_arrays([a for c in side for a in c]) for side in cands]
        i1, i2, starts, k = combination_indices(n1, n2)
        (c1, p1, r1, a1), (c2, p2, r2, a2) = arrays
        # statistical propensity × likelihood (or copies) of each of the two candidates
        w = laws.propensity(c1[i1], p1[i1], r1[i1], c2[i2], p2[i2], r2[i2]) * a1[i1] * a2[i2]
        chosen = weighted_draw(w, starts, k, rng)
        s1, s2 = i1[chosen], i2[chosen]
        o1 = np.concatenate([[0], np.cumsum(n1)[:-1]])
        o2 = np.concatenate([[0], np.cumsum(n2)[:-1]])
        for p, (name, x1, x2, q1, q2, u1, u2) in enumerate(batch):
            write(name, x1[s1[p] - o1[p]], x2[s2[p] - o2[p]], n1[p], n2[p], q1, q2, u1, u2)
        stats["placed"] += len(batch)
        stats["combinations"] += int(k.sum())
        stats["evaluated"] += int(k.sum())
        stats["cand1"] += int(n1.sum())
        stats["cand2"] += int(n2.sum())
        batch, n_comb = [], 0

    for name, c1, x1, c2, x2, q1, q2, u1, u2 in pairs:
        n = len(c1) * len(c2)
        if (max_combinations and n > max_combinations) or ((x1 or x2) and n > index_limit):
            y1, y2 = as_arrays(c1), as_arrays(c2)
            i, j, n_eval = hierarchical_draw(laws, y1, y2, rng, near_index if (x1 or x2) else near,
                                             n_samples)
            write(name, c1[i], c2[j], len(c1), len(c2), q1, q2, u1, u2)
            stats["placed"] += 1
            stats["sampled"] += 1
            stats["combinations"] += n
            stats["evaluated"] += n_eval
            stats["cand1"] += len(c1)
            stats["cand2"] += len(c2)
            continue
        batch.append((name, c1, c2, q1, q2, u1, u2))
        n_comb += n
        if n_comb >= batch_combinations:
            flush()
    flush()
    return stats


def _new_stats():
    return dict(pairs=0, placed=0, unplaced=0, combinations=0, cand1=0, cand2=0, intra=0,
                sampled=0, evaluated=0, index_reads=0, index_bins=0, index_unplaced=0,
                index_scored=0, index_fallback=0, unplaceable_fallback=0, unplaceable_unique=0,
                index_ref_lookup=0, unique_requalified=0)


def reattribute_stream(lines1, lines2, laws, out_fh, rng, as_delta=30, error_rate=0.01,
                       batch_combinations=2_000_000, max_combinations=0, near=100_000,
                       n_samples=256, repeat_index=None, saturation=None, index_limit=200_000,
                       near_index=20_000):
    """
    Re-attribute the pairs of two SAM streams (R1, R2) and write the chosen
    combination of each pair to out_fh. Candidates: alignments within as_delta
    of the best score, each weighted by its likelihood (alignment_weights,
    error_rate; 0 → best-score ties only). Pairs with more than max_combinations
    R1 × R2 combinations (> 0) are drawn by hierarchical_draw (near combinations
    exact, far ones importance-sampled); 0 → all combinations always evaluated.

    repeat_index (RepeatIndex) and saturation (bowtie2 -k): a read with `saturation`
    alignments (the list was cut by -k, so it is an arbitrary subset of the copies) and at
    least 2 candidates takes its candidates from the repeat index instead (all the bins
    carrying its sequence); the drawn bin is then resolved to an exact position and strand.
    Pairs with an index side and more than index_limit combinations use the hierarchical draw
    with exact combinations below near_index (bins of 2 kb: fewer exact combinations).
    Returns counters.
    """
    lam = score_lambda(error_rate)
    if np.isinf(lam):
        as_delta = 0
    stats = _new_stats()
    use_index = repeat_index is not None and saturation
    if use_index:
        laws.set_index_bin(repeat_index.bin_size)

    def weighted(block):
        """[(chrom, pos, rev, weight, bin start = -1)] of the best candidates of a read."""
        c = [a for a in best_candidates(block, as_delta) if a[0] in laws.code]
        if not c:
            return c
        w = alignment_weights([a[3] for a in c], lam)
        return [(a[0], a[1], int(a[2]), x, -1) for a, x in zip(c, w)]

    def candidates(block, seq):
        c = weighted(block)
        if use_index and len(block) >= saturation and len(c) >= 2:
            x = [a for a in index_candidates(repeat_index, seq) if a[0] in laws.code]
            if x:
                stats["index_reads"] += 1
                stats["index_bins"] += len(x)
                return x, True
        return c, False

    def pairs():
        for (name1, b1, q1), (name2, b2, q2) in zip(read_blocks(lines1, True), read_blocks(lines2, True)):
            if name1 != name2:
                raise ValueError(f"R1 / R2 out of sync: {name1} vs {name2}")
            stats["pairs"] += 1
            (c1, x1), (c2, x2) = candidates(b1, q1), candidates(b2, q2)
            if not c1 or not c2:
                stats["unplaced"] += 1
                continue
            yield name1, c1, x1, c2, x2, q1, q2, None, None

    return _draw_pairs(pairs(), laws, out_fh, rng, stats, repeat_index, batch_combinations,
                       max_combinations, near, n_samples, index_limit, near_index)


def read_fastq(path):
    """Yield (name, sequence bytes, qualities bytes) of a FASTQ file."""
    with open(path, "rb") as fh:
        while True:
            h = fh.readline()
            if not h:
                return
            seq = fh.readline().rstrip()
            fh.readline()
            qual = fh.readline().rstrip()
            yield h[1:].split()[0].decode(), seq, qual


def read_hits(path):
    """Yield (name, hit R1, hit R2) of a group2_hits_chunk file: hit = (chrom, 5' position,
    is_reverse, AS, unique, NM). Files of 5 fields per mate (before 2026-11-25, no NM): NM = -1."""
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            k = (len(f) - 1) // 2
            h1, h2 = ((f[o], int(f[o + 1]), f[o + 2] == "-", int(f[o + 3]), f[o + 4] == "1",
                       int(f[o + 5]) if k >= 6 else -1) for o in (1, 1 + k))
            yield f[0], h1, h2


_COMP = bytes.maketrans(b"ACGTN", b"TGCAN")


def index_read_candidates(repeat_index, codes, seq, qual, hit, as_delta, lam, max_scored=50,
                          min_fraction=0.3, stats=None, max_fallback_nm=None, requalify_nm=None):
    """
    Candidates of one group2 read for the index-only re-attribution (see
    reattribute_index_stream). hit = first alignment (chrom, 5' position, is_reverse, AS,
    unique[, NM]); codes = chromosomes known to the laws. Returns (candidates, are_bins).
    max_fallback_nm: a read without any index candidate (ambiguous, or unique) whose first
    alignment has NM >= max_fallback_nm is not placeable (no candidate: the pair is unplaced):
    degraded reads of the satellites (errors in low-complexity arrays, chimeras) match nothing
    exactly and their first alignments converged on a few loci (piles chr16:52.18 Mb,
    chr14:8.27 Mb, 2026-11-25). None = off; needs the NM of the hits (classification of
    2026-11-25 or later; -1 = unknown, kept).
    """
    stats = stats if stats is not None else _new_stats()
    anchor = None
    chrom, pos, rev, _, unique = hit[:5]
    nm = hit[5] if len(hit) > 5 else -1
    degraded = max_fallback_nm is not None and nm >= max_fallback_nm
    if unique and requalify_nm is not None and nm >= requalify_nm:
        # unique by MAPQ but a poor first alignment (NM >= requalify_nm): degraded satellite reads matched
        # a singular array edge best by a small margin and seeded coverage piles (2026-11-25); treated
        # as an ambiguous read: its copies (index, through the read or the reference at its first
        # alignment) compete with its first alignment through the alignment weights
        unique = False
        stats["unique_requalified"] += 1
    if unique:
        if degraded and len(repeat_index.candidates(seq, min_fraction)[0]) == 0:
            stats["unplaceable_unique"] += 1
            return [], False
        return ([(chrom, pos, int(rev), 1.0, -1)] if chrom in codes else []), False
    chroms, starts, w = repeat_index.candidates(seq, min_fraction)
    if len(chroms) == 0 and hasattr(repeat_index, "reference_segment"):
        # no repeated minimizer in the read itself (often one error in a short read): the reference
        # at its first alignment carries the copies; the read itself is then scored in each bin
        # (mosquito: mito reads with an error had been left on the chr3 NUMT, 2026-11-25)
        ref = repeat_index.reference_segment(chrom, pos, rev, len(seq))
        if ref:
            chroms, starts, w = repeat_index.candidates(ref, min_fraction)
            if len(chroms):
                stats["index_ref_lookup"] += 1
                # the read in its sequenced orientation, the reference on the forward strand: the
                # anchor must have the read's orientation (diagonals of the read)
                anchor = ref if not rev else ref.translate(_COMP)[::-1]
    if len(chroms) == 0:
        if degraded:
            stats["unplaceable_fallback"] += 1
            return [], False
        stats["index_fallback"] += 1
        return ([(chrom, pos, int(rev), 1.0, -1)] if chrom in codes else []), False
    stats["index_reads"] += 1
    B = repeat_index.bin_size
    if len(chroms) + 1 > max_scored:
        stats["index_bins"] += len(chroms)
        return [(c, b + B // 2, 2, x, b) for c, b, x in zip(chroms, starts.tolist(), w.tolist())
                if c in codes], True
    stats["index_scored"] += 1
    bins = set(zip(chroms, starts.tolist())) | {(chrom, (pos - 1) // B * B)}
    found = {}
    for c, b in bins:
        if c not in codes:
            continue
        sc, ch = (repeat_index.score_bin(seq, c, b, qual) if anchor is None
                  else repeat_index.score_bin(seq, c, b, qual, anchor=anchor))
        if sc is None:
            continue
        for p, r in ch:
            if found.get((c, p, r), -1) < sc:
                found[(c, p, r)] = sc
    if not found:
        stats["index_fallback"] += 1
        return ([(chrom, pos, int(rev), 1.0, -1)] if chrom in codes else []), False
    best = max(found.values())
    keep = [(k, v) for k, v in found.items() if v >= best - as_delta]
    wts = alignment_weights([v for _, v in keep], lam)
    return [(c, p, int(r), x, -1) for ((c, p, r), _), x in zip(keep, wts)], False


def reattribute_index_stream(fastq1, fastq2, hits, laws, out_fh, rng, repeat_index, as_delta=30,
                             error_rate=0.01, max_scored=50, min_fraction=0.3,
                             batch_combinations=2_000_000, max_combinations=0, near=100_000,
                             n_samples=256, index_limit=200_000, near_index=20_000,
                             max_fallback_nm=None, requalify_nm=None):
    """
    Index-only re-attribution (no second bowtie2 alignment): the candidates of each group2
    read come from the first alignment and the repeat index (repeat_index_v2.RepeatIndex):
      - unique read (first alignment): its position;
      - ambiguous read: the bins of the index sharing >= min_fraction of the best bin's read
        minimizers, plus the bin of its first alignment;
          <= max_scored bins (low-copy repeats): the read is aligned in each bin (score_bin:
          ungapped local alignment, bowtie2 local scores and qualities) → exact candidates
          within as_delta of the best score, weighted by their likelihood as the bowtie2
          candidates (alignment_weights);
          more bins (satellites, young repeats): the bins themselves, weighted by the copies of
          the read sequence they carry, strand unknown, resolved to an exact position after
          the draw;
      - ambiguous read without any repeated minimizer: the reference sequence at its first
        alignment is looked up instead (its copies, the read scored in each); still nothing: its
        first alignment (counted), or no candidate if its NM >= max_fallback_nm.
    hits: group2_hits_chunk file (streaming_classifier_v2), same order as the FASTQ files.
    """
    lam = score_lambda(error_rate)
    if np.isinf(lam):
        as_delta = 0
    stats = _new_stats()
    laws.set_index_bin(repeat_index.bin_size)

    def candidates(seq, qual, hit):
        return index_read_candidates(repeat_index, laws.code, seq, qual, hit, as_delta, lam,
                                     max_scored, min_fraction, stats, max_fallback_nm, requalify_nm)

    def pairs():
        for (n1, s1, u1), (n2, s2, u2), (nh, h1, h2) in zip(read_fastq(fastq1), read_fastq(fastq2),
                                                            read_hits(hits)):
            if not (n1 == n2 == nh):
                raise ValueError(f"R1 / R2 / hits out of sync: {n1} {n2} {nh}")
            stats["pairs"] += 1
            if not ({h1[0], h2[0]} <= set(laws.code) | laws.excluded):     # unknown chromosome
                stats["unplaced"] += 1
                continue
            (c1, x1), (c2, x2) = candidates(s1, u1, h1), candidates(s2, u2, h2)
            if not c1 or not c2:
                stats["unplaced"] += 1
                continue
            yield n1, c1, x1, c2, x2, s1, s2, u1, u2

    return _draw_pairs(pairs(), laws, out_fh, rng, stats, repeat_index, batch_combinations,
                       max_combinations, near, n_samples, index_limit, near_index)


def pairs_header(chrom_sizes, extra_columns=()):
    """4DN pairs header: #chromsize lines and chr1 / chr2 column names (pairtools, hicstuff)."""
    lines = ["## pairs format v1.0"] + [f"#chromsize: {c} {L}" for c, L in chrom_sizes.items()]
    lines.append("#columns: readID chr1 pos1 chr2 pos2 strand1 strand2"
                 + "".join(f" {c}" for c in extra_columns))
    return "\n".join(lines) + "\n"


def merge_pairs(unrescued, predicted, out_path, chrom_sizes):
    """rescued.pairs = unrescued.pairs + predicted.pairs (7 standard columns). Returns the number of pairs."""
    n = 0
    with open(out_path, "w") as out:
        out.write(pairs_header(chrom_sizes))
        for path in (unrescued, predicted):
            with open(path) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    f = line.rstrip("\n").split("\t")
                    out.write("\t".join(f[:7]) + "\n")
                    n += 1
    return n


def chunk_stats_path(out):
    """Statistics of a finished chunk, written once its pairs file is complete (resume)."""
    return Path(str(out) + ".stats.json")


def chunk_done(out):
    """A chunk is finished when its stats file exists and is not older than its pairs file."""
    s = chunk_stats_path(out)
    return Path(out).is_file() and s.is_file() and s.stat().st_mtime >= Path(out).stat().st_mtime


def reattribute_chunk(job):
    """Worker: one group2 chunk (_reattribute_chunk), then its stats file (resume)."""
    stats = _reattribute_chunk(job)
    chunk_stats_path(job["out"]).write_text(json.dumps({k: int(v) if k != "seconds" else float(v)
                                                        for k, v in stats.items()}))
    return stats


def _reattribute_chunk(job):
    """
    One group2 chunk. job = dict(fastq1, fastq2 | sam1, sam2, out, stats_dir,
    mode, circular, bin_size, index, max_alignments, threads, as_delta, seed).
    Runs bowtie2 on both FASTQ files (stdout read as a stream), or reads SAM files.
    """
    t0 = time.time()
    laws = Laws(job["stats_dir"], job["mode"], job.get("circular"), job["bin_size"],
                job.get("exclude", ()))
    rng = np.random.default_rng(job["seed"])
    if job.get("index_only"):
        from repeat_index_v2 import RepeatIndex
        with open(job["out"], "w") as out:
            stats = reattribute_index_stream(job["fastq1"], job["fastq2"], job["hits"], laws, out, rng,
                                             RepeatIndex(job["repeat_index"]), job["as_delta"],
                                             job["error_rate"],
                                             max_combinations=job.get("max_combinations", 0),
                                             near=job.get("near", 100_000),
                                             n_samples=job.get("n_samples", 256),
                                             max_fallback_nm=job.get("max_fallback_nm"),
                                             requalify_nm=job.get("requalify_nm"))
        stats["seconds"] = time.time() - t0
        return stats
    repeat_index = saturation = None
    if job.get("repeat_index") and str(job["max_alignments"]) != "all":
        from repeat_index_v2 import RepeatIndex
        repeat_index = RepeatIndex(job["repeat_index"])
        saturation = int(job["max_alignments"])
    procs, handles = [], []
    try:
        if job.get("sam1"):
            handles = [open(job["sam1"]), open(job["sam2"])]
            streams = handles
        else:
            for fq in (job["fastq1"], job["fastq2"]):
                log = open(str(job["out"]) + f".{Path(fq).stem}.bowtie2.log", "w")
                handles.append(log)
                procs.append(subprocess.Popen(
                    bowtie2_command(job["index"], fq, job["max_alignments"], job["threads"]),
                    stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1 << 20))
            streams = [p.stdout for p in procs]
        with open(job["out"], "w") as out:
            stats = reattribute_stream(streams[0], streams[1], laws, out, rng, job["as_delta"],
                                       job["error_rate"],
                                       max_combinations=job.get("max_combinations", 0),
                                       near=job.get("near", 100_000),
                                       n_samples=job.get("n_samples", 256),
                                       repeat_index=repeat_index, saturation=saturation)
        for p in procs:
            if p.wait() != 0:
                raise RuntimeError(f"bowtie2 failed (see {job['out']}.*.bowtie2.log)")
    finally:
        for h in handles:
            h.close()
    stats["seconds"] = time.time() - t0
    return stats


def reattribute(classifier_dir, stats_dir, output_dir, index=None, mode="full",
                max_alignments="100", as_delta=30, n_workers=1, threads=1, circular=None,
                bin_size=2000, seed=0, sam_pairs=None, error_rate=0.01, max_combinations=0,
                near=100_000, n_samples=256, repeat_index=None, index_only=False,
                exclude_chroms=(), max_fallback_nm=None, requalify_nm=None):
    """
    Re-attribute all the group2 chunks of classifier_dir (in parallel, one
    process per chunk) and write output_dir/predicted.pairs.
    sam_pairs: optional list of (sam1, sam2) already aligned, instead of running bowtie2.
    repeat_index: prefix of a repeat index (repeat_index_v2.py) for the reads whose bowtie2
    list is saturated (-k reached), None = bowtie2 candidates only.
    index_only: no second bowtie2 alignment, the candidates come from the first alignment and
    the repeat index (reattribute_index_stream; needs repeat_index and the group2_hits_chunk
    files of the classification).
    """
    if index_only and not repeat_index:
        raise ValueError("index_only needs a repeat index")
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode}, choose among {MODES}")
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = dict(stats_dir=str(stats_dir), mode=mode, circular=circular, bin_size=bin_size,
                index=index, max_alignments=max_alignments, threads=threads, as_delta=as_delta,
                error_rate=error_rate, max_combinations=max_combinations, near=near,
                n_samples=n_samples, repeat_index=str(repeat_index) if repeat_index else None,
                index_only=index_only, exclude=list(exclude_chroms), max_fallback_nm=max_fallback_nm,
                requalify_nm=requalify_nm)
    jobs = []
    if sam_pairs:
        for i, (s1, s2) in enumerate(sam_pairs):
            jobs.append(dict(base, sam1=s1, sam2=s2, seed=seed + i,
                             out=output_dir / f"predicted_chunk{i}.pairs"))
    else:
        r1 = sorted(Path(classifier_dir).glob("group2_1_chunk*.fastq"),
                    key=lambda p: int(re.search(r"chunk(\d+)", p.name).group(1)))
        if not r1:
            raise FileNotFoundError(f"No group2_1_chunk*.fastq in {classifier_dir}")
        if index is None and not index_only:
            raise ValueError("--index (bowtie2 index) is required to align the group2 chunks")
        for i, f1 in enumerate(r1):
            f2 = f1.with_name(f1.name.replace("group2_1_", "group2_2_"))
            hits = f1.with_name(f1.name.replace("group2_1_", "group2_hits_").replace(".fastq", ".tsv"))
            if index_only and not hits.is_file():
                raise FileNotFoundError(f"{hits} missing: run the classification again (index-only "
                                        f"re-attribution needs the first alignments of the group2 reads)")
            jobs.append(dict(base, fastq1=str(f1), fastq2=str(f2), hits=str(hits), seed=seed + i,
                             out=output_dir / f"predicted_chunk{i}.pairs"))

    # one memory-mapped copy of the density, shared by all the workers (page cache), instead of
    # one unpickled copy per worker (human, 50 kb: 8 GB each → OOM with 18 workers)
    if mode == "full":
        for name in (st.DENSITY_MAP, st.DENSITY_FINE_MAP):
            if (Path(stats_dir) / name).is_file():
                flatten_density(Path(stats_dir) / name)
        gc.collect()

    # resume: chunks finished by a previous run (stats file written) are not drawn again
    done = {i for i, j in enumerate(jobs) if chunk_done(j["out"])}
    if done:
        print(f"Resume: {len(done)} chunk(s) already re-attributed, {len(jobs) - len(done)} to do")
    todo = [j for i, j in enumerate(jobs) if i not in done]
    if n_workers > 1 and len(todo) > 1:
        # forkserver, not fork: the workers start from a fresh process, not from a copy of the
        # caller (hicberg_v2 running stats + rescue in one process: ~18 GB of laws / matrices
        # inherited by each forked worker, copied page by page as Python touches the objects:
        # human full run stalled in memory thrashing, 2026-11-24)
        with mp.get_context("forkserver").Pool(min(n_workers, len(todo))) as pool:
            new = pool.map(reattribute_chunk, todo)
    else:
        new = [reattribute_chunk(j) for j in todo]
    new = iter(new)
    results = [json.loads(chunk_stats_path(j["out"]).read_text()) if i in done else next(new)
               for i, j in enumerate(jobs)]

    sizes = np.load(Path(stats_dir) / st.CHROM_SIZES, allow_pickle=True).item()
    with open(output_dir / "predicted.pairs", "w") as out:
        out.write(pairs_header(sizes, ("n_candidates1", "n_candidates2")))
        for j in jobs:
            with open(j["out"]) as fh:
                for line in fh:
                    out.write(line)
            Path(j["out"]).unlink()
            chunk_stats_path(j["out"]).unlink(missing_ok=True)

    unrescued = Path(classifier_dir) / "unrescued.pairs" if classifier_dir else None
    if unrescued is not None and unrescued.is_file():
        n_res = merge_pairs(unrescued, output_dir / "predicted.pairs", output_dir / "rescued.pairs", sizes)
        print(f"rescued.pairs: {n_res:,} pairs (unrescued + predicted)")

    total = {k: sum(r[k] for r in results) for k in results[0] if k != "seconds"}
    p = max(total["placed"], 1)
    if index_only:
        cand_line = (f"candidates            : index only (no second alignment): unique reads at their "
                     f"first alignment; ambiguous reads from the repeat index: {total['index_scored']} "
                     f"aligned in each of their <= 50 bins (exact scores), "
                     f"{total['index_reads'] - total['index_scored']} as bins (many copies, "
                     f"{total['index_unplaced']} left at the bin centre), {total['index_fallback']} "
                     f"without repeated minimizers kept at their first alignment "
                     f"({total.get('index_ref_lookup', 0)} other reads without repeated minimizer: copies found "
                     f"through the reference at their first alignment)"
                     + (f"; unique reads with NM >= {requalify_nm} re-examined as ambiguous: "
                        f"{total.get('unique_requalified', 0)}" if requalify_nm is not None else "")
                     + (f"; not placeable (no index candidate, NM >= {max_fallback_nm}): "
                        f"{total['unplaceable_fallback']} ambiguous, {total['unplaceable_unique']} unique reads"
                        if max_fallback_nm is not None else ""))
    else:
        cand_line = (f"alignments            : "
                     f"{'-a (all)' if str(max_alignments) == 'all' else f'-k {max_alignments}'}, "
                     + (f"candidates within {as_delta} of the best score, weighted by their likelihood "
                        f"(error rate {error_rate:g}, {score_lambda(error_rate):.2f} per AS point)"
                        if error_rate > 0 else "best-score candidates only"))
    lines = [
        f"mode                  : {mode}",
        cand_line,
        f"pairs                 : {total['pairs']}",
        f"placed                : {total['placed']}",
        f"unplaced              : {total['unplaced']} (a mate without candidate"
        + (f" outside the excluded {', '.join(sorted(exclude_chroms))}" if exclude_chroms else "") + ")",
        f"candidates per read   : R1 {total['cand1'] / p:.2f}, R2 {total['cand2'] / p:.2f}",
        f"combinations per pair : {total['combinations'] / p:.2f} "
        f"(evaluated {total['evaluated'] / p:.2f})",
        f"hierarchical draw     : "
        + (f"{total['sampled']} pairs ({100 * total['sampled'] / p:.1f}%) with > {max_combinations} "
           f"combinations: near (< {near / 1000:g} kb) exact + {n_samples} importance samples"
           if max_combinations else "off (all combinations evaluated)"),
        f"repeat index          : "
        + (f"{total['index_reads']} reads ({100 * total['index_reads'] / max(2 * total['pairs'], 1):.1f}%) "
           f"with a saturated list (-k {max_alignments}) took their candidates from the index: "
           f"{total['index_bins'] / max(total['index_reads'], 1):.0f} bins per read, "
           f"{total['index_unplaced']} left at the bin centre (no anchor)"
           if repeat_index and not index_only else ("see candidates" if index_only else
                                                     "off (bowtie2 candidates only)")),
        f"placed intra          : {total['intra']} ({100 * total['intra'] / p:.1f}%)",
        f"time                  : {time.time() - t0:.0f} s ({n_workers} workers)",
    ]
    (output_dir / "reattribution_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n=== Re-attribution ===\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Re-attribute the ambiguous pairs (group2).")
    parser.add_argument("classifier_dir", help="Output folder of streaming_classifier_v2.py (group2 chunks)")
    parser.add_argument("stats_dir", help="Output folder of statistics_v2.py")
    parser.add_argument("output_dir", help="Output folder (predicted.pairs)")
    parser.add_argument("--index", help="bowtie2 index of the genome")
    parser.add_argument("--mode", choices=MODES, default="full",
                        help="standard = P(s) × coverage, full = P(s) × density (default=full)")
    parser.add_argument("--max_alignments", default="100",
                        help="bowtie2 -k value, or 'all' for -a (default=100)")
    parser.add_argument("--error_rate", type=float, default=0.01,
                        help="Sequencing error rate of the likelihood weight of each alignment, "
                             "exp(-lambda x (best AS - AS)) (default=0.01: 1 mismatch = x1/300). "
                             "0 = best-score alignments only")
    parser.add_argument("--as_delta", type=int, default=30,
                        help="Ignore the alignments more than this AS below the best one "
                             "(default=30: weight < 1e-9 with error_rate 0.01)")
    parser.add_argument("--n_workers", type=int, default=1, help="Chunks processed in parallel")
    parser.add_argument("--threads", type=int, default=1, help="bowtie2 threads per worker")
    parser.add_argument("--circular", default=None,
                        help="Circular chromosomes (default: those of statistics_stats.txt)")
    parser.add_argument("--bin_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_combinations", type=int, default=0,
                        help="Pairs with more R1 x R2 combinations are drawn hierarchically (near "
                             "combinations exact, far ones importance-sampled); 0 = always exact")
    parser.add_argument("--near", type=int, default=100_000,
                        help="Hierarchical draw: cis combinations closer than this are exact")
    parser.add_argument("--n_samples", type=int, default=256,
                        help="Hierarchical draw: importance samples of the far combinations")
    parser.add_argument("--repeat_index", default=None,
                        help="Prefix of a repeat index (repeat_index_v2.py): reads whose bowtie2 list "
                             "is saturated take all the copies of their sequence as candidates")
    parser.add_argument("--index_only", action="store_true",
                        help="No second bowtie2 alignment: candidates from the first alignment and "
                             "the repeat index (needs --repeat_index)")
    args = parser.parse_args()
    circular = None if args.circular is None else {c for c in args.circular.split(",") if c}
    reattribute(args.classifier_dir, args.stats_dir, args.output_dir, args.index, args.mode,
                args.max_alignments, args.as_delta, args.n_workers, args.threads, circular,
                args.bin_size, args.seed, error_rate=args.error_rate,
                max_combinations=args.max_combinations, near=args.near, n_samples=args.n_samples,
                repeat_index=args.repeat_index, index_only=args.index_only)


if __name__ == "__main__":
    main()
