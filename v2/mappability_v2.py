#!/usr/bin/env python3
"""
mappability_v2.py

Theoretical mappability (GenMap) and its use to normalise the P(s).

1. run_genmap()          : GenMap index + map → per-position bedgraph
                           (score = 1 / number of occurrences of the k-mer
                           starting at this position, with up to E errors)
2. binned_unique_fraction(): per bin, fraction of positions whose k-mer is
                           unique (score >= min_score). A read from a
                           multi-copy position carries an XS tag and goes to
                           group2, so this fraction is what group1 can observe.
3. mappable_areas()      : number of mappable position pairs (i, j) per log
                           distance bin = autocorrelation of the mask (FFT).
                           Replaces the trapezoid area: with a mask equal to 1
                           everywhere it gives exactly the trapezoid (linear)
                           or rectangle (circular) areas.

CLI (step 1, run once per genome and read length):
    python3 mappability_v2.py genome.fa out_dir -K 35 -E 2 -T 8
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. GenMap
# ---------------------------------------------------------------------------

def run_genmap(genome, output_dir, kmer, errors=2, threads=8):
    """
    Run GenMap on a genome. The index (genome-specific, independent of K/E)
    is reused if it already exists. Returns the path of the bedgraph.
    """
    if shutil.which("genmap") is None:
        raise RuntimeError("genmap not found in PATH (conda install -c bioconda genmap)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir = output_dir / "genmap_index"
    prefix = output_dir / f"genmap_K{kmer}_E{errors}"

    if not index_dir.is_dir():   # genmap refuses an existing folder
        subprocess.run(["genmap", "index", "-F", str(genome), "-I", str(index_dir)],
                       check=True)

    subprocess.run(["genmap", "map", "-K", str(kmer), "-E", str(errors),
                    "-I", str(index_dir), "-O", str(prefix), "-bg", "-T", str(threads)],
                   check=True)

    bedgraph = prefix.with_name(prefix.name + ".bedgraph")
    if not bedgraph.is_file():
        raise FileNotFoundError(f"GenMap output not found: {bedgraph}")
    return bedgraph


# ---------------------------------------------------------------------------
# 2. Binned mask
# ---------------------------------------------------------------------------

def read_bedgraph(path):
    """{chrom: (starts, ends, scores)} from a bedgraph (0-based, half-open)."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     names=["chrom", "start", "end", "score"],
                     dtype={"chrom": str, "start": np.int64, "end": np.int64,
                            "score": np.float64})
    return {c: (g["start"].to_numpy(), g["end"].to_numpy(), g["score"].to_numpy())
            for c, g in df.groupby("chrom", sort=False)}


def _binned_mean(starts, ends, values, chrom_size, bin_size):
    """
    Mean of a piecewise-constant signal over consecutive bins, exact and
    vectorised: integral of the signal at the bin edges by interpolation of
    its cumulative sum (positions not covered by the bedgraph count as 0).
    """
    order = np.argsort(starts)
    starts, ends, values = starts[order], ends[order], values[order]
    cum = np.cumsum(values * (ends - starts))
    cum_before = np.concatenate([[0.0], cum[:-1]])

    # integral F(x): flat over gaps, linear inside each interval
    xp = np.ravel(np.column_stack([starts, ends]))
    fp = np.ravel(np.column_stack([cum_before, cum]))

    edges = np.minimum(np.arange(0, chrom_size + bin_size, bin_size), chrom_size)
    integral = np.interp(edges, xp, fp, left=0.0, right=cum[-1])
    widths = np.diff(edges).astype(np.float64)
    return np.divide(np.diff(integral), widths, out=np.zeros_like(widths), where=widths > 0)


def binned_unique_fraction(bedgraph, chrom_sizes, bin_size, min_score=1.0):
    """
    {chrom: fraction of positions with score >= min_score, per bin}.
    The last bin may be shorter than bin_size. Chromosomes absent from the
    bedgraph get 0 everywhere.
    """
    data = read_bedgraph(bedgraph) if not isinstance(bedgraph, dict) else bedgraph
    out = {}
    for chrom, size in chrom_sizes.items():
        n_bins = -(-size // bin_size)
        if chrom not in data:
            out[chrom] = np.zeros(n_bins)
            continue
        s, e, score = data[chrom]
        out[chrom] = _binned_mean(s, e, (score >= min_score - 1e-9).astype(np.float64),
                                  size, bin_size)[:n_bins]
    return out


# ---------------------------------------------------------------------------
# 3. Mappable areas
# ---------------------------------------------------------------------------

def mask_autocorrelation(f, circular=False):
    """
    C[Δ] = sum_b f[b] f[b+Δ] (linear) or f[(b+Δ) mod n] (circular), Δ = 0..n-1,
    by FFT. C[0] is replaced by sum_b f[b]: at distance 0 a position is paired
    with itself (sum M(i)^2 = sum M(i)), which f^2 underestimates in partially
    mappable bins.
    """
    n = len(f)
    size = n if circular else int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(f, size)
    c = np.fft.irfft(spec * np.conj(spec), size)[:n]
    c = np.clip(c, 0, None)
    c[0] = f.sum()
    return c


def mappable_areas(xs, chrom_size, fraction, resolution, circular=False):
    """
    Number of mappable position pairs per log distance bin [xs[j], xs[j+1])
    (last bin up to L, or L/2 if circular).

    Pairs at distance d: a(d) = resolution × C(d / resolution), linearly
    interpolated between bin offsets; the area is the integral of a(d) over
    each distance bin. With fraction = 1 everywhere, a(d) = L - d (linear) or
    L (circular): identical to statistics_v2.trapezoid_areas().
    """
    L = float(chrom_size)
    r = float(resolution)
    f = np.asarray(fraction, dtype=np.float64).copy()
    n = len(f)
    f[-1] *= (L - r * (n - 1)) / r          # last bin may be shorter than r
    c = mask_autocorrelation(f, circular)

    # a(d) known at the nodes d = k·r, piecewise linear in between
    if circular:
        # (exact for L multiple of r, otherwise approximation of the period)
        d_max = L / 2
        nodes, a_nodes = np.arange(n) * r, r * c
    else:
        d_max = L
        nodes = np.append(np.arange(n) * r, L)   # a(L) = 0: no pair beyond L
        a_nodes = np.append(r * c, 0.0)

    seg = np.diff(nodes)
    slopes = np.diff(a_nodes) / seg
    I_nodes = np.concatenate([[0.0], np.cumsum(seg * (a_nodes[:-1] + a_nodes[1:]) / 2)])

    def integral(x):
        x = np.clip(np.asarray(x, dtype=np.float64), 0, d_max)
        k = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(seg) - 1)
        t = x - nodes[k]
        return I_nodes[k] + a_nodes[k] * t + slopes[k] * t * t / 2

    edges = np.append(np.asarray(xs, dtype=np.float64), d_max)
    return np.clip(np.diff(integral(edges)), 0, None)


def mappable_length(fraction, chrom_size, resolution):
    """Number of mappable positions of a chromosome."""
    widths = np.full(len(fraction), float(resolution))
    widths[-1] = chrom_size - resolution * (len(fraction) - 1)
    return float(np.sum(fraction * widths))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute theoretical mappability with GenMap.")
    parser.add_argument("genome", help="Genome FASTA")
    parser.add_argument("output_dir", help="Output directory (GenMap index + bedgraph)")
    parser.add_argument("-K", "--kmer", type=int, required=True, help="k-mer length = read length")
    parser.add_argument("-E", "--errors", type=int, default=2, help="Mismatches allowed (default=2)")
    parser.add_argument("-T", "--threads", type=int, default=8, help="Threads (default=8)")
    args = parser.parse_args()

    bedgraph = run_genmap(args.genome, args.output_dir, args.kmer, args.errors, args.threads)
    print(f"Mappability bedgraph: {bedgraph}")


if __name__ == "__main__":
    main()
