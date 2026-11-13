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
  - 1D coverage (raw + final)       → coverage1.npy / coverage.npy
        final = raw / mappable fraction (bins too unmappable interpolated)
        with --mappability, otherwise mad_smoothing2 (previous version)

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
COVERAGE_KEPT  = "coverage_kept.npy"  # coverage bins used (others interpolated)

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


def pooled_law(xs, counts, areas, n_min=100, d_max=None):
    """
    Law = counts / areas, with consecutive distance bins pooled (from short to
    long distances) until each group holds >= n_min counts: group value =
    sum(counts) / sum(areas), Poisson error <= 1/sqrt(n_min). The group values
    are then interpolated in log-log at the area-weighted (log) centre of
    each group → continuous curve (flat before the first centre, extended
    with the log-log slope of the last segment beyond the last centre); a bin alone in its group
    keeps its exact raw value. Bins with area 0 (circular, beyond L/2) stay 0.
    No point is ever replaced by a fit.
    """
    xs = np.asarray(xs, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    law = np.zeros(len(xs))
    idx = np.flatnonzero(areas > 0)
    if len(idx) == 0:
        return law

    # bin centres (geometric) for the log-log interpolation
    hi = np.append(xs[1:], d_max if d_max is not None else xs[-1] * 1.1)
    centres = np.sqrt(np.maximum(xs, 0.5) * np.maximum(hi, 1.0))

    # groups of consecutive bins with >= n_min counts (last one merged back)
    groups, current, total = [], [], 0.0
    for j in idx:
        current.append(j)
        total += counts[j]
        if total >= n_min:
            groups.append(current)
            current, total = [], 0.0
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)

    values = np.array([counts[g].sum() / areas[g].sum() for g in groups])
    g_centres = np.array([np.exp(np.average(np.log(centres[g]), weights=areas[g]))
                          for g in groups])

    if np.all(values > 0) and len(groups) > 1:
        lx, gx, gy = np.log(centres[idx]), np.log(g_centres), np.log(values)
        ly = np.interp(lx, gx, gy)
        # beyond the last group centre: extend the log-log slope of the last
        # segment (the law decays smoothly at long distances). Before the first
        # centre: flat (short distances are not a power law: religations,
        # overlapping reads)
        sel = lx > gx[-1]
        if sel.any() and gx[-1] > gx[-2]:
            ly[sel] = gy[-1] + (gy[-1] - gy[-2]) / (gx[-1] - gx[-2]) * (lx[sel] - gx[-1])
        law[idx] = np.exp(ly)

        # The group value is an area-weighted mean of a law that varies inside
        # the group, so it is not the law at the group centre (biased upwards
        # for a decaying law). Re-scale the curve so that each group gets
        # exactly its observed counts (sum of areas × law = sum of counts),
        # with the correction factor interpolated (log-log) between groups.
        for _ in range(3):
            ratio = np.array([counts[g].sum() / np.sum(areas[g] * law[g]) for g in groups])
            law[idx] *= np.exp(np.interp(lx, gx, np.log(ratio)))
    else:   # empty groups: piecewise-constant values
        for g, v in zip(groups, values):
            law[g] = v
    return law


def smooth_ps_laws(xs, counts, areas, n_min=100, d_merge=5000, d_max=None, d_pool=1000):
    """
    The 3 P(s) laws (uncut, circle, weird) from the raw counts of the 4
    orientations:
      - d < d_pool: raw values, zeros and fluctuations included. At short
        distances the law changes from one bp to the next (overlapping reads,
        religations): pooling would spread a group mean over configurations
        that are never observed (e.g. no ++/-- pair below 19 bp on chrI)
      - d_pool <= d < d_merge: each law from its own counts (weird: ++ and --
        over 2 × area), smoothed by pooled_law()
      - d >= d_merge: the orientations are equivalent (25% each), so the 3 laws
        share one estimate from all the counts over 4 × area (4× more counts)
    """
    xs = np.asarray(xs)
    raw_zone, short = xs < d_pool, xs < d_merge
    mid = short & ~raw_zone
    out = {}
    for key, n, factor in (("uncut", counts[UNCUT], 1.0),
                           ("circle", counts[CIRCLE], 1.0),
                           ("weird", counts[PLUS_PLUS] + counts[MINUS_MINUS], 2.0)):
        a = areas * factor
        raw = np.divide(n, a, out=np.zeros(len(xs)), where=a > 0)
        pooled = pooled_law(xs, np.where(mid, n, 0), np.where(mid, a, 0), n_min, d_max)
        out[key] = np.where(raw_zone, raw, pooled)
    merged = pooled_law(xs, np.where(short, 0, counts.sum(axis=0)),
                        np.where(short, 0, areas * 4.0), n_min, d_max)
    for key in out:
        out[key] = np.where(short, out[key], merged)
    return out


def compute_intra_ps(acc, smoothing="pool", mappability=None, map_resolution=None,
                     n_min=100, d_merge=5000, d_pool=1000):
    """
    Normalise raw counts by the number of position pairs per distance bin:
    uncut and circle by the area, weird (++ and --) by 2 × area.
      - default: trapezoid areas (same as the previous version); circular
        chromosomes use the circular areas (laws are 0 beyond L/2)
      - mappability = {chrom: fraction of mappable positions per bin of
        map_resolution}: only pairs of mappable positions are counted
        (mappability_v2.mappable_areas)
    smoothing: "pool" (smooth_ps_laws), "fitting_ps" (previous version) or "none".
    """
    if smoothing not in ("pool", "fitting_ps", "none"):
        raise ValueError(f"Unknown P(s) smoothing: {smoothing}")
    if smoothing == "fitting_ps":
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
        if smoothing == "pool":
            d_max = acc.sizes[i] / 2 if acc.circular[i] else acc.sizes[i]
            laws = smooth_ps_laws(xs, counts, areas, n_min, d_merge, d_max, d_pool)
        elif smoothing == "fitting_ps":
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


def correct_coverage(raw, fraction, chrom_size, bin_size, min_fraction=0.8,
                     min_mappable_bp=100, min_ratio=0.5, window=10, circular=False):
    """
    Coverage corrected by mappability, expressed per bin_size bp:
      1. mappable length of each bin = fraction × effective width (the last
         bin of a chromosome is shorter than bin_size)
      2. valid bins: fraction (of the bin's own width) >= min_fraction, and
         at least min_mappable_bp mappable positions (guard against tiny
         remnants, e.g. 20 bp with 1 read × 100);
         corrected = raw × bin_size / mappable length
      3. low outliers: valid bins below min_ratio × the median of the
         corrected coverage of the `window` closest valid bins on each side are excluded too (truly
         under-covered bins, repeats seen by bowtie2 but not by GenMap)
      4. excluded bins: linear interpolation between the closest valid bins
         (periodic on circular chromosomes, constant beyond the ends)
    Returns (corrected, valid) with valid = bins kept (not interpolated). If no
    bin is valid, corrected is NaN everywhere (to be filled by the caller).
    """
    raw = np.asarray(raw, dtype=np.float64)
    f = np.zeros(len(raw))
    n = min(len(raw), len(fraction))
    f[:n] = np.asarray(fraction)[:n]
    widths = np.clip(chrom_size - np.arange(len(raw)) * bin_size, 0, bin_size)
    mappable = f * widths

    valid = (f >= min_fraction) & (mappable >= max(min_mappable_bp, 1e-9))
    corrected = np.full(len(raw), np.nan)
    corrected[valid] = raw[valid] * bin_size / mappable[valid]

    if min_ratio > 0 and valid.sum() > 2:
        # median of the `window` closest valid bins on each side (whatever the
        # gap): an isolated bin after an unmappable stretch is compared to
        # real neighbours, not to itself
        idx = np.flatnonzero(valid)
        vals = corrected[idx]
        local = pd.Series(vals).rolling(2 * window + 1, center=True,
                                        min_periods=1).median().to_numpy()
        low = idx[vals < min_ratio * local]
        valid[low] = False
        corrected[low] = np.nan

    if valid.any() and not valid.all():
        idx = np.arange(len(raw))
        corrected[~valid] = np.interp(idx[~valid], idx[valid], corrected[valid],
                                      period=len(raw) if circular else None)
    return corrected, valid


def compute_coverages(acc, smooth=True, mappability=None, **correction):
    """
    Returns (raw, final, kept) coverages per chromosome.
      - mappability = {chrom: fraction of mappable positions per coverage
        bin}: final = correct_coverage(**correction) (no other smoothing),
        kept = bins used (not interpolated). Chromosomes without any kept bin
        get the genome-wide median.
      - otherwise: final = mad_smoothing2 (previous version), or raw if not
        smooth; kept = None.
    """
    raw = acc.coverage_by_chrom()

    if mappability is not None:
        final, kept, empty = {}, {}, []
        for i, (chrom, y) in enumerate(raw.items()):
            final[chrom], kept[chrom] = correct_coverage(
                y, mappability[chrom], acc.sizes[i], acc.bin_size,
                circular=acc.circular[i], **correction)
            if not kept[chrom].any():
                empty.append(chrom)
        if empty:
            median = np.median(np.concatenate([v[np.isfinite(v)] for v in final.values()]))
            for chrom in empty:
                final[chrom][:] = median
            print(f"  [warning] no kept bin on {empty}: genome-wide median coverage used")
        total = sum(len(v) for v in raw.values())
        n_out = sum(int((~v).sum()) for v in kept.values())
        print(f"  {n_out} / {total} coverage bins excluded (unmappable or low outliers) → interpolated")
        return raw, final, kept

    if not smooth:
        return raw, raw, None

    from hicberg.utils import mad_smoothing2

    smoothed = {}
    for chrom, y in raw.items():
        try:
            smoothed[chrom] = _quiet(mad_smoothing2, y)
        except Exception as e:
            print(f"  [warning] coverage smoothing failed on {chrom}: {e} → raw values kept")
            smoothed[chrom] = y
    return raw, smoothed, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_statistics(pairs_path, chrom_sizes_path, output_dir, bin_size=2000,
                        base=1.1, circular="", chunksize=5_000_000, smooth=True,
                        ps_smoothing="pool", ps_min_pairs=100, ps_merge_distance=5000,
                        ps_pool_distance=1000,
                        plots=True, mappability=None, map_resolution=100,
                        min_mappability=1.0, coverage_min_fraction=0.8,
                        coverage_min_bp=100, coverage_min_ratio=0.5):
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
    ps_smoothing = ps_smoothing if smooth else "none"
    uncuts, circles, weirds = compute_intra_ps(acc, ps_smoothing, map_fine, map_resolution,
                                               ps_min_pairs, ps_merge_distance, ps_pool_distance)
    print("Computing trans P(s) ...")
    trans_ps = compute_trans_ps(acc, map_lengths)
    print("Computing coverages ...")
    cov_raw, cov_smooth, cov_kept = compute_coverages(
        acc, smooth, map_coarse, min_fraction=coverage_min_fraction,
        min_mappable_bp=coverage_min_bp, min_ratio=coverage_min_ratio)

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
    if cov_kept is not None:
        np.save(output_dir / COVERAGE_KEPT, cov_kept)

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
        f"P(s) smoothing        : {ps_smoothing}"
        + (f" (raw below {ps_pool_distance} bp, >= {ps_min_pairs} pairs per group beyond, "
           f"orientations merged beyond {ps_merge_distance} bp)"
           if ps_smoothing == "pool" else ""),
    ]
    if map_lengths is not None:
        total, mappable = sum(chrom_sizes.values()), sum(map_lengths.values())
        lines += [
            f"mappability           : {mappability}",
            f"  resolution / min    : {map_resolution} bp / score >= {min_mappability}",
            f"  coverage            : raw x {bin_size} / mappable bp; bins kept if fraction >= "
            f"{coverage_min_fraction}, >= {coverage_min_bp} mappable bp, >= {coverage_min_ratio} x local median",
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
    parser.add_argument("--ps_smoothing", choices=["pool", "fitting_ps"], default="pool",
                        help="P(s) smoothing: pool = distance bins pooled until enough pairs, "
                             "log-log interpolation; fitting_ps = previous version (default=pool)")
    parser.add_argument("--ps_min_pairs", type=int, default=100,
                        help="pool: min number of pairs per group of distance bins (default=100)")
    parser.add_argument("--ps_pool_distance", type=int, default=1000,
                        help="pool: raw P(s) values (no pooling) below this distance, where the "
                             "law changes from one bp to the next (default=1000)")
    parser.add_argument("--ps_merge_distance", type=int, default=5000,
                        help="pool: beyond this distance the 4 orientations share one law "
                             "(default=5000; check orientations.pdf)")
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
    parser.add_argument("--coverage_min_fraction", type=float, default=0.8,
                        help="Coverage bins kept if their mappable fraction (of their own width) "
                             "is >= this value, else interpolated (default=0.8)")
    parser.add_argument("--coverage_min_bp", type=int, default=100,
                        help="... and if they contain >= this number of mappable bp (default=100)")
    parser.add_argument("--coverage_min_ratio", type=float, default=0.5,
                        help="... and if their corrected coverage is >= this ratio x the local "
                             "median of the 10 closest kept bins on each side; 0 to disable (default=0.5)")
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
        ps_smoothing     = args.ps_smoothing,
        ps_min_pairs     = args.ps_min_pairs,
        ps_merge_distance = args.ps_merge_distance,
        ps_pool_distance = args.ps_pool_distance,
        plots            = not args.no_plots,
        mappability      = args.mappability,
        map_resolution   = args.map_resolution,
        min_mappability  = args.min_mappability,
        coverage_min_fraction = args.coverage_min_fraction,
        coverage_min_bp  = args.coverage_min_bp,
        coverage_min_ratio = args.coverage_min_ratio,
    )


if __name__ == "__main__":
    main()
