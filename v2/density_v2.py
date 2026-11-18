#!/usr/bin/env python3
"""
density_v2.py

2D contact density maps (local observed / expected enrichment), used by the
"full" re-attribution mode: propensity = P(s) × density(bin_i, bin_j).

Same algorithm as the previous version (hicberg.utils.get_local_density),
computed directly from a pairs file (no cooler) and with fast filters:
  1. contact matrix of each chromosome pair (bins of bin_size)
  2. bad bins (rows / columns) → NaN
  3. intra: observed / expected, expected = mean of the positive values of
     each diagonal (values >= max_val → 1); trans: observed / mean
  4. zeros → 1 (neutral)
  5. NaN filled by iterated nan-mean over patch_small × patch_small windows
  6. nan-mean over patch_large × patch_large windows
  7. zeros → min non-zero value; intra: diagonal → 1

The nan-mean windows (scipy generic_filter with a Python callback in the
previous version, very slow) are computed as uniform_filter(values) /
uniform_filter(defined), which gives the same result in O(1) per pixel.

Dense blocks: fine for small genomes. Human scale: TODO (banded storage).
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import uniform_filter

import statistics_v2 as st

DENSITY_MAP = "density_map.npy"


# ---------------------------------------------------------------------------
# Contact matrix
# ---------------------------------------------------------------------------

def genome_bins(chrom_sizes, bin_size):
    """Number of bins per chromosome (ceil, as cooler) and offsets in the genome."""
    n_bins = np.array([-(-L // bin_size) for L in chrom_sizes.values()])
    offsets = np.concatenate([[0], np.cumsum(n_bins)[:-1]])
    return n_bins, offsets


# adaptive window sizes (bins). 61-181 added for sparse far fields (mosquito
# 50 kb: ~200 contacts need ~3 × 3 Mb at the telomere-telomere corners)
DENSITY_WINDOWS = (1, 3, 5, 11, 21, 31, 45, 61, 91, 121, 181)


def available_memory():
    """Available RAM in bytes (Linux /proc/meminfo MemAvailable; None elsewhere)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


# Peak memory of block_density (unbiased + adaptive window) per cell of the
# largest block: measured ~52 bytes (n = 4000: +0.84 GB over the input); x2 for
# the split halves of --density_n_min auto.
DENSITY_BYTES_PER_CELL = 100
NICE_BIN_SIZES = (2_000, 5_000, 10_000, 20_000, 25_000, 50_000, 100_000, 200_000, 250_000,
                  500_000, 1_000_000)


def auto_density_bin_size(chrom_sizes, bin_size, memory=None, fraction=0.5):
    """
    Smallest "nice" bin size (>= bin_size) such that the density of the
    largest chromosome fits in `fraction` of the available memory:
    DENSITY_BYTES_PER_CELL × (L_max / b)^2 <= fraction × memory.
    Yeast (L_max 1.5 Mb): 2 kb; mosquito (225 Mb, 16 GB free): 25 kb.
    """
    memory = memory or available_memory() or 8 * 1024 ** 3
    l_max = max(chrom_sizes.values())
    b_min = l_max * np.sqrt(DENSITY_BYTES_PER_CELL / (fraction * memory))
    for b in NICE_BIN_SIZES:
        if b >= max(b_min, bin_size):
            return int(b)
    return int(max(b_min, bin_size))


def data_density_bin_size(ps_counts, xs, chrom_sizes, bin_size, n_min=200, window=45,
                           max_fallback=0.05, candidates=NICE_BIN_SIZES):
    """
    Smallest bin size (>= bin_size, in `candidates`) at which the adaptive
    window can reach n_min contacts for most contacts: at distance d the
    contact density per position pair is rho(d) = N(d) / ((L - d) Δd) (from
    the intra P(s) counts), a window of `window` × `window` bins of size b
    holds ~ window² rho(d) b² contacts; the fraction of the intra contacts
    whose window holds < n_min ("fallback") must be <= max_fallback.
    Yeast AC1 (36 M pairs, 12 Mb): 0 at 2 kb → 2 kb. Mosquito (3.3 M unique
    pairs, 575 Mb): 21 % at 2 kb, 7 % at 25 kb, 4.8 % at 50 kb → 50 kb.
    ps_counts: {chrom: (orientations, n_xs)} raw counts; xs: {chrom: left
    edges of the log distance bins}.
    """
    def fallback(b):
        num = den = 0.0
        for c, L in chrom_sizes.items():
            if c not in ps_counts:
                continue
            n = np.asarray(ps_counts[c], dtype=np.float64).sum(axis=0)
            x = np.asarray(xs[c], dtype=np.float64)[:len(n)]
            edges = np.append(x, max(L, x[-1] + 1))
            area = np.clip(L - (edges[:-1] + edges[1:]) / 2, 0, None) * np.diff(edges)
            ok = area > 0
            rho = np.zeros_like(area)
            rho[ok] = n[ok] / area[ok]
            fb = ok & (window * window * rho * b * b < n_min)
            num += n[fb].sum()
            den += n[ok].sum()
        return num / den if den else 0.0

    for b in candidates:
        if b >= bin_size and fallback(b) <= max_fallback:
            return int(b), fallback(b)
    return int(candidates[-1]), fallback(candidates[-1])


def contact_matrix(pairs_path, chrom_sizes, bin_size, chunksize=5_000_000,
                   split_halves=False, seed=0):
    """
    Genome-wide symmetric contact matrix (scipy CSR, int64) from a pairs file.
    bin = (pos - 1) // bin_size (1-based positions, same as the coverage and
    as cooler cload pairs: identical to the v1 cooler, checked pixel by pixel).
    Each pair is counted once in the upper triangle, then symmetrised.
    split_halves: also return the matrices of two random halves of the pairs
    (cross-validation of the density threshold): (full, [half_0, half_1]).
    """
    names = list(chrom_sizes)
    n_bins, offsets = genome_bins(chrom_sizes, bin_size)
    n = int(n_bins.sum())
    rng = np.random.default_rng(seed)

    uppers = [sparse.csr_matrix((n, n), dtype=np.int64) for _ in range(2 if split_halves else 1)]
    for c1, p1, c2, p2, _, _ in st.iter_pairs(pairs_path, names, chunksize):
        ok = (c1 >= 0) & (c2 >= 0)
        c1, p1, c2, p2 = c1[ok], p1[ok], c2[ok], p2[ok]
        b1 = offsets[c1] + np.minimum((p1 - 1) // bin_size, n_bins[c1] - 1)
        b2 = offsets[c2] + np.minimum((p2 - 1) // bin_size, n_bins[c2] - 1)
        i, j = np.minimum(b1, b2), np.maximum(b1, b2)
        if split_halves:
            h = rng.random(len(i)) < 0.5
            parts = [(i[h], j[h]), (i[~h], j[~h])]
        else:
            parts = [(i, j)]
        for k, (ii, jj) in enumerate(parts):
            uppers[k] = uppers[k] + sparse.csr_matrix(
                (np.ones(len(ii), dtype=np.int64), (ii, jj)), shape=(n, n))
    sym = [(u + u.T - sparse.diags(u.diagonal())).tocsr() for u in uppers]
    if split_halves:
        return (sym[0] + sym[1]).tocsr(), sym
    return sym[0]


# ---------------------------------------------------------------------------
# Density of one block
# ---------------------------------------------------------------------------

def nanmean_filter(z, size):
    """Mean of the non-NaN values in each size × size window ('mirror' edges)."""
    defined = np.isfinite(z)
    total = uniform_filter(np.where(defined, z, 0.0), size=size, mode="mirror")
    count = uniform_filter(defined.astype(np.float64), size=size, mode="mirror")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count < 1e-12] = np.nan
    return out


def iterative_fill(z, patch_size, max_iter=5):
    """Fill NaN with the nan-mean of their window, iterated (fills large holes)."""
    z = z.copy()
    for _ in range(max_iter):
        holes = np.isnan(z)
        if not holes.any():
            break
        z[holes] = nanmean_filter(z, patch_size)[holes]
    return z


def detrend_intra(mat, max_val=10):
    """
    Observed / expected for an intra-chromosomal block: expected = mean of the
    positive values of each diagonal (NaN and zeros excluded, as v1). Values
    >= max_val (or divided by an empty diagonal) → 1. Zeros and NaN unchanged.
    """
    n = mat.shape[0]
    rows, cols = np.indices(mat.shape)
    dist = np.abs(rows - cols)
    positive = mat > 0                      # False for NaN and zeros
    sums = np.bincount(dist[positive], weights=mat[positive], minlength=n)
    counts = np.bincount(dist[positive], minlength=n)
    expected = np.divide(sums, counts, out=np.zeros(n), where=counts > 0)

    out = mat.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out[positive] = mat[positive] / expected[dist[positive]]
    out[positive & ~(out < max_val)] = 1.0  # >= max_val or inf
    return out


def expected_unbiased(mat, intra, w_row=None, w_col=None, log_pool=1.1):
    """
    Expected contacts of each pixel, zeros included:
      intra: expected(i, j) = E_d × w_i × w_j, E_d = sum of the diagonals /
             sum of w_i w_j over their defined (non-NaN) pixels, the diagonals
             pooled in log-distance groups (log_pool; None = per diagonal);
      trans: expected(i, j) = E × w_i × w_j, E = block sum / sum of w_i w_j.
    w = mappable length of each bin / bin_size (default 1).
    Returns (expected, defined); expected is 0 where nothing is expected.
    """
    w_row = np.ones(mat.shape[0]) if w_row is None else np.asarray(w_row, dtype=np.float64)
    w_col = np.ones(mat.shape[1]) if w_col is None else np.asarray(w_col, dtype=np.float64)
    W = np.outer(w_row, w_col)
    defined = np.isfinite(mat) & (W > 0)
    if not intra:
        total_w = W[defined].sum()
        e = mat[defined].sum() / total_w if total_w > 0 else 0.0
        return np.where(defined, e * W, 0.0), defined
    n = mat.shape[0]
    rows, cols = np.indices(mat.shape)
    dist = np.abs(rows - cols)
    sums = np.bincount(dist[defined], weights=mat[defined], minlength=n)
    wsum = np.bincount(dist[defined], weights=W[defined], minlength=n)
    if log_pool:
        # diagonals pooled in log-distance groups (base log_pool, as the P(s)):
        # the last diagonals hold a few pixels only, all near the corner, so a
        # per-diagonal expected absorbs the telomere-telomere contacts
        # (mosquito chr1 corner obs / exp 1.08 per diagonal, 2.10 pooled)
        g = np.zeros(n, dtype=np.int64)
        d = np.arange(1, n)
        g[1:] = 1 + np.floor(np.log(d) / np.log(log_pool)).astype(np.int64)
        _, g = np.unique(g, return_inverse=True)
        gs, gw = np.bincount(g, weights=sums), np.bincount(g, weights=wsum)
        e_d = np.divide(gs, gw, out=np.zeros(len(gs)), where=gw > 0)[g]
    else:
        e_d = np.divide(sums, wsum, out=np.zeros(n), where=wsum > 0)
    return np.where(defined, e_d[dist] * W, 0.0), defined


def obs_exp_unbiased(mat, intra, w_row=None, w_col=None):
    """
    Observed / expected with the zeros kept as information (expected_unbiased).
    w = mappable length of each bin / bin_size (default 1): a short last bin
    or a partially mappable bin is not seen as depleted (end of chrI: one
    490-bp bin had obs / exp 0.14 only because of its size).
    Pixels whose expected is 0 (no contact at all) → NaN (filled later).
    With w = 1 the mean of the result is 1 on each diagonal / block.
    """
    expected, defined = expected_unbiased(mat, intra, w_row, w_col)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(expected > 0, mat / expected, np.nan)
    out[~defined] = np.nan
    return out


def block_sum(z, k):
    """Sum over k × k blocks (edges padded with 0)."""
    if k <= 1:
        return z
    n0, n1 = -(-z.shape[0] // k) * k, -(-z.shape[1] // k) * k
    p = np.zeros((n0, n1))
    p[:z.shape[0], :z.shape[1]] = z
    return p.reshape(n0 // k, k, n1 // k, k).sum(axis=(1, 3))


def obs_exp_binned(mat, intra, w_row=None, w_col=None, k=1):
    """
    Raw (not smoothed) obs / exp aggregated over k × k bins: sum of the
    observed / sum of the expected of the defined pixels. NaN where nothing
    is defined. k = 1 → obs_exp_unbiased.
    """
    if k <= 1:
        return obs_exp_unbiased(mat, intra, w_row, w_col)
    expected, defined = expected_unbiased(mat, intra, w_row, w_col)
    obs = block_sum(np.where(defined, mat, 0.0), k)
    exp = block_sum(expected, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(exp > 0, obs / exp, np.nan)


def adaptive_filter(z, counts, windows=DENSITY_WINDOWS, n_min=200, expected=None):
    """
    For each pixel, nan-mean of z over the smallest window (in `windows`) that
    holds >= n_min raw contacts (Poisson error <= 1/sqrt(n_min)). Sharp where
    contacts are dense (centromeres, near the diagonal), wide on sparse
    regions (arms in trans).
    Where even the largest window holds fewer than n_min contacts:
      expected None : nan-mean over the largest window (previous behaviour)
      expected given: shrinkage towards 1 (no information beyond the P(s)),
                      (O + δ) / (E + δ) with O, E the observed / expected
                      contacts of the largest window and δ = n_min - O: the
                      missing contacts are taken at density 1. Continuous at
                      O = n_min; avoids the 0 / huge values of sparse data
                      (mosquito 25 kb: 0.006 contact per far-field pixel,
                      densities up to 230 before).
    Returns (smoothed, chosen window size per pixel; 0 = shrunk fallback).
    """
    c = np.where(np.isfinite(counts), counts, 0.0)
    out = np.full(z.shape, np.nan)
    chosen = np.zeros(z.shape, dtype=np.int32)
    todo = np.ones(z.shape, dtype=bool)
    last = len(windows) - 1
    for k, w in enumerate(windows):
        o_w = uniform_filter(c, size=w, mode="mirror") * w * w
        enough = o_w >= n_min
        if k == last and expected is not None:
            sel = todo & enough
            fb = todo & ~enough
            if fb.any():
                e = np.where(np.isfinite(expected), expected, 0.0)
                e_w = uniform_filter(e, size=w, mode="mirror") * w * w
                delta = np.clip(n_min - o_w, 0, None)
                out[fb] = ((o_w + delta) / (e_w + delta))[fb]
                todo &= ~fb
        else:
            sel = todo & (enough | (k == last))
        if sel.any():
            out[sel] = nanmean_filter(z, w)[sel]
            chosen[sel] = w
            todo &= ~sel
    return out, chosen


GAUSSIAN_SIGMAS = tuple(float(x) for x in 0.5 * np.sqrt(2.0) ** np.arange(16))  # 0.5 … 90 bins


def _box3(x, sigma):
    """
    Gaussian-like smoothing of standard deviation sigma (bins) by 3 passes of
    a box filter (cost independent of sigma); zero padding outside the block.
    Returns (smoothed, sum of the squared normalised 2D weights).
    """
    w = max(1, int(round(np.sqrt(4 * sigma * sigma + 1))))
    if w % 2 == 0:
        w += 1
    y = x
    for _ in range(3):
        y = uniform_filter(y, size=w, mode="constant")
    k1 = np.ones(w) / w
    k = np.convolve(np.convolve(k1, k1), k1)
    return y, float((k ** 2).sum()) ** 2


def cap_outlier_pixels(counts, expected, cap=20, ratio=100):
    """
    Pixels with more than `cap` contacts AND more than `ratio` × their expected are
    capped at max(cap, ratio × expected) before smoothing: a single pixel then cannot
    bring the kernel to n_min contacts on its own (mosquito: one PCR jackpot of 104
    copies in a chr1 x chr3 pixel whose expected was 0.012 gave a density of 62 over
    a ~5 Mb disc). Near-diagonal pixels (large expected) and moderate far-cis / trans
    enrichments (telomere-telomere, centromere clustering) are untouched.
    """
    counts = np.asarray(counts, dtype=np.float64)
    e = np.where(np.isfinite(expected), expected, 0.0)
    limit = np.maximum(cap, ratio * e)
    out = np.where(np.isfinite(counts) & (counts > limit), limit, counts)
    return out


def adaptive_gaussian(obs, expected, n_min=200, sigmas=GAUSSIAN_SIGMAS):
    """
    Density = smoothed observed / smoothed expected, with a Gaussian-like
    kernel whose width adapts continuously to the local number of contacts.
    For each sigma of the ladder: N_eff = smoothed counts / sum(w²) (equal to
    the number of contacts for a box window, the Poisson effective sample
    size of a weighted kernel), D_sigma = smooth(obs) / smooth(expected)
    (ratio of sums: zeros and edges handled, no mean of noisy ratios). Each
    pixel takes the smallest sigma with N_eff >= n_min, interpolated in log
    N_eff with the previous sigma (geometric blend of the two densities): no
    jump between window sizes, hence no square blocks. Where even the
    largest sigma has N_eff < n_min: shrinkage towards 1, (O + δ) / (E + δ)
    in contact units, δ = n_min - O (as adaptive_filter).
    obs, expected: counts and expected counts (0 on excluded pixels).
    Returns (density, sigma used per pixel; 0 = shrunk fallback).
    """
    obs = np.where(np.isfinite(obs), obs, 0.0)
    expected = np.where(np.isfinite(expected), expected, 0.0)
    out = np.full(obs.shape, np.nan)
    used = np.zeros(obs.shape)
    todo = np.ones(obs.shape, dtype=bool)
    prev_d = prev_logn = None
    for k, s in enumerate(sigmas):
        so, sw2 = _box3(obs, s)
        se, _ = _box3(expected, s)
        n_eff = so / sw2
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(se > 0, so / se, np.nan)
        logn = np.log(np.maximum(n_eff, 1e-12))
        reached = todo & (n_eff >= n_min)
        if reached.any():
            if prev_d is None:
                out[reached] = d[reached]
            else:
                # blend with the previous sigma, in log N_eff
                t = (np.log(n_min) - prev_logn) / np.maximum(logn - prev_logn, 1e-12)
                t = np.clip(t, 0, 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    blend = np.exp((1 - t) * np.log(np.maximum(prev_d, 1e-12))
                                   + t * np.log(np.maximum(d, 1e-12)))
                ok_prev = np.isfinite(prev_d) & (prev_d > 0)
                out[reached] = np.where(ok_prev, blend, d)[reached]
            used[reached] = s
            todo &= ~reached
        if not todo.any():
            break
        prev_d, prev_logn = d, logn
    if todo.any():                              # shrinkage at the largest sigma
        o_cnt = so / sw2
        e_cnt = se / sw2
        delta = np.clip(n_min - o_cnt, 0, None)
        out[todo] = ((o_cnt + delta) / (e_cnt + delta))[todo]
    return out, used


def block_density(mat, intra, patch_small=15, patch_large=45, max_val=10, method="v1",
                  smoothing="fixed", windows=DENSITY_WINDOWS, n_min=200,
                  w_row=None, w_col=None):
    """
    Density map of one chromosome pair (mat: counts with NaN on bad bins).
    method "v1": previous version (expected from the positive values only,
    values >= max_val → 1, zeros → 1: biased upwards on sparse blocks,
    trans median 1.84 on yeast). "unbiased": obs_exp_unbiased(), zeros kept.
    smoothing "fixed": nan-mean over patch_large × patch_large (v1);
    "adaptive": adaptive_filter(windows, n_min) (box windows);
    "gaussian": adaptive_gaussian(n_min) (default of the pipeline: continuous
    kernel width, no square blocks; better held-out likelihood, see
    DEVELOPMENT.md).
    """
    mat = mat.astype(np.float64)
    counts = mat.copy()
    expected = None
    if method == "unbiased":
        expected, defined = expected_unbiased(mat, intra, w_row, w_col)
        expected = np.where(defined, expected, 0.0)
        mat = obs_exp_unbiased(mat, intra, w_row, w_col)
    elif method == "v1":
        if intra:
            mat = detrend_intra(mat, max_val)
        else:
            mean = np.nanmean(mat)
            mat = mat / mean if mean > 0 else mat
        mat[mat == 0] = 1.0
    else:
        raise ValueError(f"Unknown density method: {method}")

    filled = iterative_fill(mat, patch_small, max_iter=5)
    if smoothing == "fixed":
        smooth = nanmean_filter(filled, patch_large)
    elif smoothing == "adaptive":
        smooth, _ = adaptive_filter(filled, counts, windows, n_min, expected)
        smooth = np.clip(smooth, 0, None)
    elif smoothing == "gaussian":
        if expected is None:
            raise ValueError("smoothing 'gaussian' needs method 'unbiased'")
        smooth, _ = adaptive_gaussian(cap_outlier_pixels(counts, expected), expected, n_min)
        smooth = np.clip(np.where(np.isfinite(smooth), smooth, 1.0), 0, None)
    else:
        raise ValueError(f"Unknown density smoothing: {smoothing}")

    nonzero = smooth[(smooth != 0) & np.isfinite(smooth)]
    if len(nonzero):
        smooth[smooth == 0] = nonzero.min()
    if intra:
        np.fill_diagonal(smooth, 1.0)
    return smooth


# ---------------------------------------------------------------------------
# All blocks
# ---------------------------------------------------------------------------

def density_maps(matrix, chrom_sizes, bin_size, bad_bins=None,
                 patch_small=15, patch_large=45, max_val=10, method="v1",
                 smoothing="fixed", windows=DENSITY_WINDOWS, n_min=200, weights=None,
                 n_min_trans=None):
    """
    {(chrom_a, chrom_b): density} for every pair of chromosomes (a <= b in
    the chromosome order, as v1). bad_bins: boolean array over the genome bins
    (True = excluded row / column), e.g. from the cooler balancing (v1) or
    from the mappability. weights: mappable length / bin_size of each genome
    bin (bin_weights), used by the unbiased obs / exp.
    """
    names = list(chrom_sizes)
    n_bins, offsets = genome_bins(chrom_sizes, bin_size)
    out = {}
    for (i, a), (j, b) in itertools.combinations_with_replacement(enumerate(names), 2):
        ra = slice(offsets[i], offsets[i] + n_bins[i])
        rb = slice(offsets[j], offsets[j] + n_bins[j])
        mat = matrix[ra, rb].toarray().astype(np.float64)
        if bad_bins is not None:
            # as v1: a row is masked if it is entirely NaN in the balanced
            # block, i.e. its bin is bad or all the column bins are bad
            bad_a, bad_b = bad_bins[ra], bad_bins[rb]
            mat[bad_a | bad_b.all(), :] = np.nan
            mat[:, bad_b | bad_a.all()] = np.nan
        out[(a, b)] = block_density(mat, intra=(a == b), patch_small=patch_small,
                                    patch_large=patch_large, max_val=max_val, method=method,
                                    smoothing=smoothing, windows=windows,
                                    n_min=n_min if (a == b or n_min_trans is None) else n_min_trans,
                                    w_row=None if weights is None else weights[ra],
                                    w_col=None if weights is None else weights[rb])
    return out


def reproducible_n_min(halves, chrom_sizes, bin_size, bad_bins=None, weights=None,
                       grid=(25, 50, 100, 150, 200, 300, 400), target=0.8, min_dist=3,
                       windows=DENSITY_WINDOWS, smoothing="gaussian"):
    """
    Choose the adaptive-window threshold n_min, separately for intra and trans,
    as the smallest one whose fine fluctuations are reproducible: on each of
    two random halves of the pairs, fluctuations = log2(adaptive density) −
    log2(density with the largest window); r = correlation of the
    fluctuations between the halves; reproducible part of the full map =
    2r / (1 + r) (Spearman-Brown, the full map has twice the contacts). The
    smallest n_min with a reproducible part >= target is kept (the largest of
    the grid if none). The halves use n_min / 2 (same windows as the full map).
    A prediction (cross-validation) criterion was tried first: it kept more
    noise than wanted (trans n_min 50: only ~50% of the extra fluctuations real).
    Intra pixels closer than min_dist bins to the diagonal are left out.
    Returns ({"intra": n_min, "trans": n_min}, {group: {n_min: reproducible part}}).
    """
    names = list(chrom_sizes)
    n_bins, offsets = genome_bins(chrom_sizes, bin_size)
    sl = [slice(o, o + n) for o, n in zip(offsets, n_bins)]
    fluct = {g: {nm: ([], []) for nm in grid} for g in ("intra", "trans")}
    for i, j in itertools.combinations_with_replacement(range(len(names)), 2):
        intra = i == j
        wr = None if weights is None else weights[sl[i]]
        wc = None if weights is None else weights[sl[j]]
        keep = None
        if intra:
            k = np.arange(n_bins[i])
            keep = np.abs(np.subtract.outer(k, k)) >= min_dist
        for h, half in enumerate(halves):
            x = half[sl[i], sl[j]].toarray().astype(np.float64)
            if bad_bins is not None:
                x[bad_bins[sl[i]], :] = np.nan
                x[:, bad_bins[sl[j]]] = np.nan
            smooth = block_density(x, intra, method="unbiased", smoothing="fixed",
                                   patch_large=windows[-1], w_row=wr, w_col=wc)
            for nm in grid:
                d = block_density(x, intra, method="unbiased", smoothing=smoothing,
                                  windows=windows, n_min=nm / 2, w_row=wr, w_col=wc)
                with np.errstate(divide="ignore", invalid="ignore"):
                    f = np.log2(d) - np.log2(smooth)
                if keep is not None:
                    f = np.where(keep, f, np.nan)
                fluct["intra" if intra else "trans"][nm][h].append(f.ravel())
    part = {}
    for g, by_nm in fluct.items():
        part[g] = {}
        for nm, (a, b) in by_nm.items():
            a, b = np.concatenate(a), np.concatenate(b)
            ok = np.isfinite(a) & np.isfinite(b)
            r = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 and np.std(a[ok]) > 0 else 0.0
            part[g][nm] = 2 * r / (1 + r) if r > -1 else 0.0
    best = {g: next((nm for nm in grid if p[nm] >= target), grid[-1]) for g, p in part.items()}
    return best, part


def bin_weights(mappability_path, chrom_sizes, bin_size):
    """
    Mappable length / bin_size of each genome bin, from statistics_v2's
    mappability.npy (fraction of mappable positions per bin) and the
    effective width of each bin (the last bin of a chromosome is shorter).
    """
    frac = np.load(mappability_path, allow_pickle=True).item()
    out = []
    for c, L in chrom_sizes.items():
        n = -(-L // bin_size)
        f = np.zeros(n)
        f[:min(n, len(frac[c]))] = frac[c][:n]
        widths = np.clip(L - np.arange(n) * bin_size, 0, bin_size)
        out.append(f * widths / bin_size)
    return np.concatenate(out)


def kept_bad_bins(kept_path, chrom_sizes, bin_size):
    """Bad bins = coverage bins excluded by statistics_v2 (coverage_kept.npy, mappability)."""
    kept = np.load(kept_path, allow_pickle=True).item()
    n_bins, _ = genome_bins(chrom_sizes, bin_size)
    return np.concatenate([~kept[c][:n] for c, n in zip(chrom_sizes, n_bins)])


def cooler_bad_bins(cool_path):
    """Bins without balancing weight in a cooler file (v1 mask)."""
    import cooler
    return cooler.Cooler(str(cool_path)).bins()[:]["weight"].isna().to_numpy()


def main():
    parser = argparse.ArgumentParser(description="Contact density maps from a pairs file.")
    parser.add_argument("pairs", help="unrescued.pairs")
    parser.add_argument("output_dir", help="Output directory (density_map.npy)")
    parser.add_argument("--chrom_sizes", required=True,
                        help="SAM file (@SQ header), chrom.sizes, .fai or chromosome_sizes.npy")
    parser.add_argument("--bin_size", type=int, default=2000)
    parser.add_argument("--patch_small", type=int, default=15)
    parser.add_argument("--patch_large", type=int, default=45)
    parser.add_argument("--method", choices=["unbiased", "v1"], default="unbiased",
                        help="unbiased: zeros kept, expected including zeros (default); "
                             "v1: previous version (zeros → 1, trans biased upwards)")
    parser.add_argument("--smoothing", choices=["gaussian", "adaptive", "fixed"], default="gaussian",
                        help="gaussian (default): adaptive Gaussian kernel with >= n_min "
                             "effective contacts; adaptive: smallest box window with >= n_min "
                             "contacts; fixed: patch_large × patch_large (v1)")
    parser.add_argument("--n_min", default="200",
                        help="adaptive: min contacts per window in intra, or 'auto': smallest "
                             "threshold (intra and trans) whose fine fluctuations are >= --reproducibility "
                             "reproducible between two halves of the pairs (default=200)")
    parser.add_argument("--reproducibility", type=float, default=0.8,
                        help="--n_min auto: target reproducible part of the fine fluctuations "
                             "(default=0.8)")
    parser.add_argument("--n_min_trans", type=int, default=200,
                        help="adaptive: min contacts per window in trans (default=200; "
                             "ignored with --n_min auto)")
    parser.add_argument("--mappability", default=None,
                        help="statistics_v2 mappability.npy: the expected of each pixel uses the "
                             "mappable length of its two bins (short / partially mappable bins "
                             "are not seen as depleted)")
    mask = parser.add_mutually_exclusive_group()
    mask.add_argument("--coverage_kept", default=None,
                      help="Bad bins = bins excluded from the coverage (statistics_v2.py "
                           "coverage_kept.npy, mappability)")
    mask.add_argument("--bad_bins_cooler", default=None,
                      help="Bad bins from the balancing of this cooler (v1)")
    args = parser.parse_args()

    t0 = time.time()
    sizes = st.load_chrom_sizes(args.chrom_sizes)
    auto = args.n_min == "auto"
    result = contact_matrix(args.pairs, sizes, args.bin_size, split_halves=auto)
    matrix, halves = result if auto else (result, None)
    print(f"Contact matrix: {matrix.shape[0]} bins ({time.time() - t0:.0f} s)")
    if args.coverage_kept:
        bad = kept_bad_bins(args.coverage_kept, sizes, args.bin_size)
    elif args.bad_bins_cooler:
        bad = cooler_bad_bins(args.bad_bins_cooler)
    else:
        bad = None
    weights = bin_weights(args.mappability, sizes, args.bin_size) if args.mappability else None
    if auto:
        best, part = reproducible_n_min(halves, sizes, args.bin_size, bad, weights,
                                        target=args.reproducibility)
        for g, d in part.items():
            print(f"  reproducible fluctuations {g}: "
                  + "  ".join(f"{k}:{v:.0%}" for k, v in d.items()) + f" → n_min = {best[g]}")
        n_min, n_min_trans = best["intra"], best["trans"]
    else:
        n_min, n_min_trans = int(args.n_min), args.n_min_trans
    maps = density_maps(matrix, sizes, args.bin_size, bad, args.patch_small, args.patch_large,
                        method=args.method, smoothing=args.smoothing, n_min=n_min,
                        n_min_trans=n_min_trans, weights=weights)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / DENSITY_MAP, maps)
    print(f"{len(maps)} density blocks saved in {out / DENSITY_MAP} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
