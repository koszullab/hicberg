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


def contact_matrix(pairs_path, chrom_sizes, bin_size, chunksize=5_000_000):
    """
    Genome-wide symmetric contact matrix (scipy CSR, int64) from a pairs file.
    bin = (pos - 1) // bin_size (1-based positions, same as the coverage and
    as cooler cload pairs: identical to the v1 cooler, checked pixel by pixel).
    Each pair is counted once in the upper triangle, then symmetrised.
    """
    names = list(chrom_sizes)
    sizes = np.array(list(chrom_sizes.values()))
    n_bins, offsets = genome_bins(chrom_sizes, bin_size)
    n = int(n_bins.sum())

    upper = sparse.csr_matrix((n, n), dtype=np.int64)
    for c1, p1, c2, p2, _, _ in st.iter_pairs(pairs_path, names, chunksize):
        ok = (c1 >= 0) & (c2 >= 0)
        c1, p1, c2, p2 = c1[ok], p1[ok], c2[ok], p2[ok]
        b1 = offsets[c1] + np.minimum((p1 - 1) // bin_size, n_bins[c1] - 1)
        b2 = offsets[c2] + np.minimum((p2 - 1) // bin_size, n_bins[c2] - 1)
        i, j = np.minimum(b1, b2), np.maximum(b1, b2)
        upper = upper + sparse.csr_matrix((np.ones(len(i), dtype=np.int64), (i, j)),
                                          shape=(n, n))
    return (upper + upper.T - sparse.diags(upper.diagonal())).tocsr()


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


def block_density(mat, intra, patch_small=15, patch_large=45, max_val=10):
    """Density map of one chromosome pair (mat: counts with NaN on bad bins)."""
    mat = mat.astype(np.float64)
    if intra:
        mat = detrend_intra(mat, max_val)
    else:
        mean = np.nanmean(mat)
        mat = mat / mean if mean > 0 else mat
    mat[mat == 0] = 1.0

    filled = iterative_fill(mat, patch_small, max_iter=5)
    smooth = nanmean_filter(filled, patch_large)

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
                 patch_small=15, patch_large=45, max_val=10):
    """
    {(chrom_a, chrom_b): density} for every pair of chromosomes (a <= b in
    the chromosome order, as v1). bad_bins: boolean array over the genome bins
    (True = excluded row / column), e.g. from the cooler balancing (v1) or
    from the mappability.
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
                                    patch_large=patch_large, max_val=max_val)
    return out


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
    parser.add_argument("--bad_bins_cooler", default=None,
                        help="Take the bad bins from the balancing of this cooler (v1)")
    args = parser.parse_args()

    t0 = time.time()
    sizes = st.load_chrom_sizes(args.chrom_sizes)
    matrix = contact_matrix(args.pairs, sizes, args.bin_size)
    print(f"Contact matrix: {matrix.shape[0]} bins, {matrix.sum() // 2:,} contacts "
          f"(off-diagonal counted twice) ({time.time() - t0:.0f} s)")
    bad = cooler_bad_bins(args.bad_bins_cooler) if args.bad_bins_cooler else None
    maps = density_maps(matrix, sizes, args.bin_size, bad, args.patch_small, args.patch_large)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / DENSITY_MAP, maps)
    print(f"{len(maps)} density blocks saved in {out / DENSITY_MAP} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
