#!/usr/bin/env python3
"""
statistics_v2.py

Computes the statistical laws used to re-attribute ambiguous reads, directly
from `unrescued.pairs` (group1, output of streaming_classifier_v2.py), in a
single streaming pass (constant memory, vectorised with numpy):

  - intra-chromosomal P(s), split by pair orientation
        uncut  (+/-, inward)   → uncuts.npy
        circle (-/+, outward)  → circles.npy
        weird  (++ and --)     → weirds.npy
  - trans-chromosomal pseudo P(s)   → trans_ps.npy
  - 1D coverage (raw + smoothed)    → coverage1.npy / coverage.npy

Output file names and normalisations are the same as in the previous
hicberg version (hicberg/statistics.py) so that both pipelines can be
compared directly. Raw counts for the 4 orientations (+-, -+, ++, --) are
additionally saved in ps_counts.npz for QC.

Position conventions (shared with the re-attribution step)
----------------------------------------------------------
`pos` in unrescued.pairs is the 1-based outer-most 5' position (pairtools):
  - distance  : d = |pos2 - pos1|  (min(d, L - d) on circular chromosomes)
  - coverage  : bin = (pos - 1) // bin_size   (0-based coordinate of the 5' base)
Orientation is defined after ordering the two reads by position
(leftmost read first), as in hicberg.utils.get_ordered_reads().
"""

import argparse
import contextlib
import io
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Same file names as hicberg/statistics.py
XS             = "xs.npy"
CHROM_SIZES    = "chromosome_sizes.npy"
UNCUTS         = "uncuts.npy"
CIRCLES        = "circles.npy"
WEIRDS         = "weirds.npy"
TRANS_PS       = "trans_ps.npy"
COVERAGE_RAW   = "coverage1.npy"   # without smoothing
COVERAGE       = "coverage.npy"    # smoothed
PS_COUNTS      = "ps_counts.npz"   # raw counts, 4 orientations (QC)
MAPPABILITY    = "mappability.npy" # fraction of mappable positions per coverage bin

# Orientation codes (after ordering reads by position)
ORIENTATIONS = ("+-", "-+", "++", "--")   # uncut, circle, weird, weird
UNCUT, CIRCLE, PLUS_PLUS, MINUS_MINUS = range(4)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_chrom_sizes(path):
    """
    Load chromosome sizes, keeping the file order. Accepted formats:
      - SAM file          : @SQ header lines (SN:, LN:)
      - .npy              : dict saved by the previous hicberg version
      - any other text    : 2 first columns = name, size (chrom.sizes, .fai)
    """
    path = Path(path)

    if path.suffix == ".npy":
        return dict(np.load(path, allow_pickle=True).item())

    sizes = {}
    with open(path) as fh:
        if path.suffix == ".sam":
            for line in fh:
                if not line.startswith("@"):
                    break
                if line.startswith("@SQ"):
                    tags = dict(t.split(":", 1) for t in line.rstrip("\n").split("\t")[1:])
                    sizes[tags["SN"]] = int(tags["LN"])
        else:
            for line in fh:
                if line.strip() and not line.startswith("#"):
                    fields = line.split()
                    sizes[fields[0]] = int(fields[1])

    if not sizes:
        raise ValueError(f"No chromosome sizes found in {path}")
    return sizes


def count_header_lines(pairs_path):
    """Number of leading '#' lines in a .pairs file."""
    n = 0
    with open(pairs_path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            n += 1
    return n


def iter_pairs(pairs_path, chrom_names, chunksize):
    """
    Stream unrescued.pairs by chunks.
    Yields numpy arrays: c1, p1, c2, p2 (int64), rev1, rev2 (bool).
    Chromosome codes follow chrom_names order; unknown names → -1.
    """
    categories = pd.Index(chrom_names)
    reader = pd.read_csv(
        pairs_path,
        sep="\t",
        header=None,
        skiprows=count_header_lines(pairs_path),
        usecols=[1, 2, 3, 4, 5, 6],
        names=["chrom1", "pos1", "chrom2", "pos2", "strand1", "strand2"],
        dtype={"chrom1": "category", "pos1": np.int64,
               "chrom2": "category", "pos2": np.int64,
               "strand1": "category", "strand2": "category"},
        chunksize=chunksize,
    )
    for chunk in reader:
        codes = []
        for col in ("chrom1", "chrom2"):
            cat = chunk[col].cat
            lookup = categories.get_indexer(cat.categories)   # -1 if unknown
            codes.append(lookup[cat.codes.to_numpy()].astype(np.int64))
        yield (
            codes[0], chunk["pos1"].to_numpy(),
            codes[1], chunk["pos2"].to_numpy(),
            (chunk["strand1"] == "-").to_numpy(),
            (chunk["strand2"] == "-").to_numpy(),
        )


# ---------------------------------------------------------------------------
# Log binning (identical to hicberg.statistics.generate_xs)
# ---------------------------------------------------------------------------

def generate_xs(chromosome_size, base=1.1):
    """Log-spaced distance bins, starting with 0."""
    n_bins = np.divide(np.log1p(chromosome_size), np.log(base)).astype(int)
    xs = np.unique(np.logspace(0, n_bins, num=n_bins + 1, base=base, dtype=int))
    return np.insert(xs, 0, 0)


def attribute_xs(xs, distances):
    """Index of the log bin of each distance (vectorised)."""
    return np.searchsorted(xs, distances, side="right") - 1


def trapezoid_areas(xs, chrom_size, circular=False):
    """
    Number of (i, j) positions of a chromosome whose distance falls in each
    log bin: normalises counts into a contact probability.

    Linear chromosome  : L - d position pairs at distance d
                         → trapezoid area (same as the previous version).
    Circular chromosome: L position pairs at every distance d ≤ L/2, none
                         beyond → rectangle L × (bin width ∩ [0, L/2]).
    """
    xs = xs.astype(np.float64)
    L = float(chrom_size)

    if circular:
        edges = np.minimum(np.append(xs, L / 2), L / 2)
        return L * np.diff(edges)

    areas = (2 * L - xs[1:] - xs[:-1]) * (xs[1:] - xs[:-1]) * 0.5
    return np.append(areas, (L - xs[-1]) ** 2 / 2)


# ---------------------------------------------------------------------------
# Per-chunk accumulation
# ---------------------------------------------------------------------------

def cis_distance(pos1, pos2, chrom_sizes, is_circular):
    """d = |pos2 - pos1|, shortest way round on circular chromosomes."""
    d = np.abs(pos2 - pos1)
    if is_circular is not None and is_circular.any():
        d = np.where(is_circular, np.minimum(d, chrom_sizes - d), d)
    return d


def pair_orientation(pos1, pos2, rev1, rev2):
    """Orientation code after ordering the two reads by position."""
    swap = pos1 > pos2
    rev_left  = np.where(swap, rev2, rev1)
    rev_right = np.where(swap, rev1, rev2)
    orient = np.full(pos1.shape, PLUS_PLUS, dtype=np.int64)
    orient[~rev_left &  rev_right] = UNCUT
    orient[ rev_left & ~rev_right] = CIRCLE
    orient[ rev_left &  rev_right] = MINUS_MINUS
    return orient


class Accumulator:
    """Counts accumulated over all chunks of the pairs file."""

    def __init__(self, chrom_sizes, bin_size, base, circular):
        self.names    = list(chrom_sizes)
        self.sizes    = np.array([chrom_sizes[c] for c in self.names], dtype=np.int64)
        self.n_chrom  = len(self.names)
        self.bin_size = bin_size
        self.circular = np.array([c in circular for c in self.names])

        # Global log bins: xs of each chromosome is a prefix of the xs of the
        # largest one (same logspace values), so one searchsorted is enough.
        self.xs       = {c: generate_xs(L, base) for c, L in chrom_sizes.items()}
        self.xs_glob  = generate_xs(self.sizes.max(), base)
        self.n_xs     = np.array([len(self.xs[c]) for c in self.names])

        # Coverage: all chromosomes concatenated
        self.n_cov_bins = self.sizes // bin_size + 1
        self.cov_offset = np.concatenate([[0], np.cumsum(self.n_cov_bins)[:-1]])

        self.ps_counts  = np.zeros(self.n_chrom * 4 * len(self.xs_glob), dtype=np.int64)
        self.trans      = np.zeros(self.n_chrom * self.n_chrom, dtype=np.int64)
        self.coverage   = np.zeros(self.n_cov_bins.sum(), dtype=np.int64)
        self.stats      = dict(pairs=0, skipped_unknown_chrom=0, cis=0, trans=0,
                               **{o: 0 for o in ORIENTATIONS})

    def add(self, c1, p1, c2, p2, rev1, rev2):
        self.stats["pairs"] += len(c1)

        known = (c1 >= 0) & (c2 >= 0)
        if not known.all():
            self.stats["skipped_unknown_chrom"] += int((~known).sum())
            c1, p1, c2, p2, rev1, rev2 = (a[known] for a in (c1, p1, c2, p2, rev1, rev2))

        # --- Coverage (both reads, 5' position) ---
        for c, p in ((c1, p1), (c2, p2)):
            b = (np.clip(p, 1, self.sizes[c]) - 1) // self.bin_size
            self.coverage += np.bincount(self.cov_offset[c] + b,
                                         minlength=len(self.coverage))

        # --- Chromosome pair counts (cis + trans) ---
        self.trans += np.bincount(c1 * self.n_chrom + c2, minlength=len(self.trans))

        # --- Intra-chromosomal P(s) by orientation ---
        cis = c1 == c2
        self.stats["cis"]   += int(cis.sum())
        self.stats["trans"] += int((~cis).sum())
        c, p1, p2, rev1, rev2 = c1[cis], p1[cis], p2[cis], rev1[cis], rev2[cis]

        orient = pair_orientation(p1, p2, rev1, rev2)
        d      = cis_distance(p1, p2, self.sizes[c], self.circular[c])
        idx    = np.minimum(attribute_xs(self.xs_glob, d), self.n_xs[c] - 1)

        flat = (c * 4 + orient) * len(self.xs_glob) + idx
        self.ps_counts += np.bincount(flat, minlength=len(self.ps_counts))

        for k, o in enumerate(ORIENTATIONS):
            self.stats[o] += int((orient == k).sum())

    # -----------------------------------------------------------------------

    def ps_counts_by_chrom(self):
        """{chrom: array (4, n_xs)} of raw counts."""
        counts = self.ps_counts.reshape(self.n_chrom, 4, len(self.xs_glob))
        return {c: counts[i, :, :self.n_xs[i]] for i, c in enumerate(self.names)}

    def coverage_by_chrom(self):
        return {c: self.coverage[o:o + n].astype(np.float64)
                for c, o, n in zip(self.names, self.cov_offset, self.n_cov_bins)}


# ---------------------------------------------------------------------------
# Final laws
# ---------------------------------------------------------------------------

def _quiet(func, *args, **kwargs):
    """Call a (verbose) hicberg function, silencing its prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def compute_intra_ps(acc, smooth=True, mappability=None, map_resolution=None):
    """
    Normalise raw counts by the number of position pairs per distance bin:
    uncut and circle by the area, weird (++ and --) by 2 × area.
      - default: trapezoid areas (same as the previous version); circular
        chromosomes use the circular areas (laws are 0 beyond L/2)
      - mappability = {chrom: fraction of mappable positions per bin of
        map_resolution}: only pairs of mappable positions are counted
        (mappability_v2.mappable_areas)
    """
    if smooth:
        from hicberg.utils import fitting_ps
    if mappability is not None:
        from mappability_v2 import mappable_areas

    uncuts, circles, weirds = {}, {}, {}
    for i, (chrom, counts) in enumerate(acc.ps_counts_by_chrom().items()):
        xs    = acc.xs[chrom]
        if mappability is None:
            areas = trapezoid_areas(xs, acc.sizes[i], circular=acc.circular[i])
        else:
            areas = mappable_areas(xs, acc.sizes[i], mappability[chrom], map_resolution,
                                   circular=acc.circular[i])

        def norm(n, factor=1.0):
            return np.divide(n, areas * factor, out=np.zeros_like(areas), where=areas != 0)

        laws = {
            "uncut":  norm(counts[UNCUT]),
            "circle": norm(counts[CIRCLE]),
            "weird":  norm(counts[PLUS_PLUS] + counts[MINUS_MINUS], factor=2.0),
        }
        if smooth:
            for key, y in laws.items():
                try:
                    laws[key] = _quiet(fitting_ps, xs, y)
                except Exception as e:   # very small contigs: keep raw law
                    print(f"  [warning] P(s) fit failed on {chrom}/{key}: {e} → raw values kept")

        uncuts[chrom], circles[chrom], weirds[chrom] = laws["uncut"], laws["circle"], laws["weird"]

    return uncuts, circles, weirds


def compute_trans_ps(acc, lengths=None):
    """
    Pseudo P(s) between two chromosomes: number of contacts / (L1 × L2 × 4),
    × 4 for the 4 read orientations. Symmetric: both (a, b) and (b, a) keys.
    lengths: {chrom: length} to use instead of the chromosome sizes
    (e.g. number of mappable positions).
    """
    sizes = acc.sizes if lengths is None else np.array([lengths[c] for c in acc.names])
    counts = acc.trans.reshape(acc.n_chrom, acc.n_chrom)
    counts = counts + counts.T - np.diag(np.diag(counts))
    trans_ps = {}
    for (i, a), (j, b) in itertools.product(enumerate(acc.names), repeat=2):
        area = float(sizes[i]) * float(sizes[j]) * 4
        trans_ps[(a, b)] = counts[i, j] / area if area > 0 else 0.0
    return trans_ps


def compute_coverages(acc, smooth=True):
    raw = acc.coverage_by_chrom()
    if not smooth:
        return raw, raw

    from hicberg.utils import mad_smoothing2

    smoothed = {}
    for chrom, y in raw.items():
        try:
            smoothed[chrom] = _quiet(mad_smoothing2, y)
        except Exception as e:
            print(f"  [warning] coverage smoothing failed on {chrom}: {e} → raw values kept")
            smoothed[chrom] = y
    return raw, smoothed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_statistics(pairs_path, chrom_sizes_path, output_dir, bin_size=2000,
                        base=1.1, circular="", chunksize=5_000_000, smooth=True,
                        plots=True, mappability=None, map_resolution=100,
                        min_mappability=1.0):
    """
    Compute and save the statistical laws, then (plots=True) their PDF plots
    in output_dir/plots, to be checked before running the re-attribution.

    mappability: GenMap bedgraph (mappability_v2.py). If given, the P(s)
    areas and trans lengths only count mappable positions (fraction of
    positions with score >= min_score, binned at map_resolution).
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chrom_sizes = load_chrom_sizes(chrom_sizes_path)
    circular    = {c.strip() for c in circular.split(",") if c.strip()}
    unknown_circ = circular - set(chrom_sizes)
    if unknown_circ:
        raise ValueError(f"Circular chromosome(s) not in chromosome sizes: {sorted(unknown_circ)}")

    acc = Accumulator(chrom_sizes, bin_size, base, circular)

    print(f"Reading {pairs_path} ...")
    for chunk in iter_pairs(pairs_path, acc.names, chunksize):
        acc.add(*chunk)
        print(f"  {acc.stats['pairs']:>13,} pairs  ({time.time() - t0:.0f} s)")

    map_fine = map_lengths = map_coarse = None
    if mappability is not None:
        from mappability_v2 import read_bedgraph, binned_unique_fraction, mappable_length
        print(f"Reading mappability {mappability} ...")
        bg = read_bedgraph(mappability)
        missing = [c for c in chrom_sizes if c not in bg]
        if missing:
            print(f"  [warning] chromosomes absent from the mappability file "
                  f"(considered unmappable): {missing}")
        map_fine   = binned_unique_fraction(bg, chrom_sizes, map_resolution, min_mappability)
        map_coarse = binned_unique_fraction(bg, chrom_sizes, bin_size, min_mappability)
        map_lengths = {c: mappable_length(map_fine[c], L, map_resolution)
                       for c, L in chrom_sizes.items()}

    print("Computing intra P(s) ...")
    uncuts, circles, weirds = compute_intra_ps(acc, smooth, map_fine, map_resolution)
    print("Computing trans P(s) ...")
    trans_ps = compute_trans_ps(acc, map_lengths)
    print("Computing coverages ...")
    cov_raw, cov_smooth = compute_coverages(acc, smooth)

    np.save(output_dir / XS,           acc.xs)
    np.save(output_dir / CHROM_SIZES,  chrom_sizes)
    np.save(output_dir / UNCUTS,       uncuts)
    np.save(output_dir / CIRCLES,      circles)
    np.save(output_dir / WEIRDS,       weirds)
    np.save(output_dir / TRANS_PS,     trans_ps)
    np.save(output_dir / COVERAGE_RAW, cov_raw)
    np.save(output_dir / COVERAGE,     cov_smooth)
    np.savez_compressed(output_dir / PS_COUNTS,
                        orientations=np.array(ORIENTATIONS),
                        **{c: v for c, v in acc.ps_counts_by_chrom().items()})
    if map_coarse is not None:
        np.save(output_dir / MAPPABILITY, map_coarse)

    # --- Stats ---
    s = acc.stats
    n = max(s["pairs"], 1)
    lines = [
        f"pairs                 : {s['pairs']}",
        f"skipped_unknown_chrom : {s['skipped_unknown_chrom']}",
        f"cis                   : {s['cis']}  ({100 * s['cis'] / n:.1f}%)",
        f"trans                 : {s['trans']}  ({100 * s['trans'] / n:.1f}%)",
    ] + [
        f"cis {o} {name:<17}: {s[o]}  ({100 * s[o] / max(s['cis'], 1):.1f}% of cis)"
        for o, name in zip(ORIENTATIONS, ("(uncut)", "(circle)", "(weird)", "(weird)"))
    ] + [
        f"bin_size              : {bin_size}",
        f"log base              : {base}",
        f"circular              : {','.join(sorted(circular)) or '-'}",
        f"smoothing             : {smooth}",
    ]
    if map_lengths is not None:
        total, mappable = sum(chrom_sizes.values()), sum(map_lengths.values())
        lines += [
            f"mappability           : {mappability}",
            f"  resolution / min    : {map_resolution} bp / score >= {min_mappability}",
            f"  mappable genome     : {mappable:,.0f} / {total:,} bp ({100 * mappable / total:.1f}%)",
        ]
    else:
        lines.append("mappability           : - (trapezoid areas)")
    lines.append(f"time                  : {time.time() - t0:.0f} s")
    (output_dir / "statistics_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n=== Statistics ===\n" + "\n".join(lines))

    # --- Plots: check the laws before re-attribution ---
    if plots:
        from plots_v2 import plot_all
        print()
        plot_all(output_dir, output_dir / "plots", bin_size=bin_size)
        print("→ Check these plots before running the re-attribution step.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute P(s), trans P(s) and coverage from unrescued.pairs."
    )
    parser.add_argument("pairs",      help="unrescued.pairs (output of streaming_classifier_v2.py)")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--chrom_sizes", required=True,
                        help="Chromosome sizes: SAM file (@SQ header), chrom.sizes, .fai "
                             "or chromosome_sizes.npy")
    parser.add_argument("--bin_size",  type=int,   default=2000,
                        help="Coverage bin size in bp (default=2000)")
    parser.add_argument("--base",      type=float, default=1.1,
                        help="Log base of P(s) distance bins (default=1.1)")
    parser.add_argument("--circular",  default="",
                        help="Comma-separated circular chromosomes, e.g. plasmid2micron,chrMT")
    parser.add_argument("--chunksize", type=int,   default=5_000_000,
                        help="Pairs read per chunk (default=5,000,000)")
    parser.add_argument("--no_smoothing", action="store_true",
                        help="Save raw laws (no P(s) fit, no coverage smoothing)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Do not generate the PDF plots (output_dir/plots)")
    parser.add_argument("--mappability", default=None,
                        help="GenMap bedgraph (mappability_v2.py): P(s) areas and trans "
                             "lengths only count mappable positions")
    parser.add_argument("--map_resolution", type=int, default=100,
                        help="Resolution of the mappability mask for the P(s) areas (default=100)")
    parser.add_argument("--min_mappability", type=float, default=1.0,
                        help="Min GenMap score of a mappable position (default=1.0 = unique k-mer)")
    args = parser.parse_args()

    generate_statistics(
        pairs_path       = args.pairs,
        chrom_sizes_path = args.chrom_sizes,
        output_dir       = args.output_dir,
        bin_size         = args.bin_size,
        base             = args.base,
        circular         = args.circular,
        chunksize        = args.chunksize,
        smooth           = not args.no_smoothing,
        plots            = not args.no_plots,
        mappability      = args.mappability,
        map_resolution   = args.map_resolution,
        min_mappability  = args.min_mappability,
    )


if __name__ == "__main__":
    main()
