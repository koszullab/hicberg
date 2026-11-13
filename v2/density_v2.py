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


def expected_unbiased(mat, intra, w_row=None, w_col=None):
    """
    Expected contacts of each pixel, zeros included:
      intra: expected(i, j) = E_d × w_i × w_j, E_d = sum of each diagonal /
             sum of w_i w_j over its defined (non-NaN) pixels;
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


def adaptive_filter(z, counts, windows=(1, 3, 5, 11, 21, 31, 45), n_min=200):
    """
    For each pixel, nan-mean of z over the smallest window (in `windows`) that
    holds >= n_min raw contacts (Poisson error <= 1/sqrt(n_min)); the largest
    window where none does. Sharp where contacts are dense (centromeres,
    near the diagonal), wide on sparse regions (arms in trans).
    Returns (smoothed, chosen window size per pixel).
    """
    c = np.where(np.isfinite(counts), counts, 0.0)
    out = np.full(z.shape, np.nan)
    chosen = np.zeros(z.shape, dtype=np.int32)
    todo = np.ones(z.shape, dtype=bool)
    for k, w in enumerate(windows):
        enough = uniform_filter(c, size=w, mode="mirror") * w * w >= n_min
        sel = todo & (enough | (k == len(windows) - 1))
        if sel.any():
            out[sel] = nanmean_filter(z, w)[sel]
            chosen[sel] = w
            todo &= ~sel
    return out, chosen


def block_density(mat, intra, patch_small=15, patch_large=45, max_val=10, method="v1",
                  smoothing="fixed", windows=(1, 3, 5, 11, 21, 31, 45), n_min=200,
                  w_row=None, w_col=None):
    """
    Density map of one chromosome pair (mat: counts with NaN on bad bins).
    method "v1": previous version (expected from the positive values only,
    values >= max_val → 1, zeros → 1: biased upwards on sparse blocks,
    trans median 1.84 on yeast). "unbiased": obs_exp_unbiased(), zeros kept.
    smoothing "fixed": nan-mean over patch_large × patch_large (v1);
    "adaptive": adaptive_filter(windows, n_min).
    """
    mat = mat.astype(np.float64)
    counts = mat.copy()
    if method == "unbiased":
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
        smooth, _ = adaptive_filter(filled, counts, windows, n_min)
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
                 smoothing="fixed", windows=(1, 3, 5, 11, 21, 31, 45), n_min=200, weights=None,
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
                       windows=(1, 3, 5, 11, 21, 31, 45)):
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
                d = block_density(x, intra, method="unbiased", smoothing="adaptive",
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
    parser.add_argument("--smoothing", choices=["adaptive", "fixed"], default="adaptive",
                        help="adaptive: smallest window with >= n_min contacts (default); "
                             "fixed: patch_large × patch_large (v1)")
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
