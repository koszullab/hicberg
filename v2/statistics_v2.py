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
  - 2D density (mode full)         → density_map.npy (+ contacts.npz), density_v2
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
MAPPABILITY_EFF = "mappability_effective.npy"  # effective mappability e(f) per coverage bin
EFF_CURVE      = "effective_mappability.tsv"  # calibrated curve f -> e(f)
DENSITY_MIN_DIST = "density_min_distance.npy"  # cis pairs closer than this: density 1 (P(s) only)
DENSITY_MAP    = "density_map.npy"    # 2D density (density_v2), float32
DENSITY_BIN    = "density_bin_size.npy"  # bin size of the density (may differ from bin_size)
DENSITY_MASK   = "density_mask.npy"   # {chrom: (bad bins, mappable weights)} at the density bin
# refined density of the small, densely covered molecules (mitochondria, endosymbionts, plasmids)
DENSITY_FINE_MAP, DENSITY_FINE_BIN, DENSITY_FINE_MASK = \
    "density_fine_map.npy", "density_fine_bin_size.npy", "density_fine_mask.npy"
CONTACTS       = "contacts.npz"       # contact matrix at bin_size (upper triangle, scipy sparse)

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

def generate_xs(chromosome_size, base=1.1, linear_until=0, linear_step=2):
    """
    Log-spaced distance bins, starting with 0. linear_until > 0: linear bins of
    linear_step bp up to linear_until, log bins beyond (omics data: the fragment-
    length distribution of e.g. MNase-seq is sharp, ~15 bp log bins at 150 bp blur it).
    """
    n_bins = np.divide(np.log1p(chromosome_size), np.log(base)).astype(int)
    xs = np.unique(np.logspace(0, n_bins, num=n_bins + 1, base=base, dtype=int))
    xs = np.insert(xs, 0, 0)
    if linear_until > 0:
        lin = np.arange(0, min(linear_until, chromosome_size) + 1, linear_step)
        xs = np.unique(np.concatenate([lin, xs[xs > lin[-1]]]))
    return xs


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

    def __init__(self, chrom_sizes, bin_size, base, circular, linear_until=0, linear_step=2):
        self.names    = list(chrom_sizes)
        self.sizes    = np.array([chrom_sizes[c] for c in self.names], dtype=np.int64)
        self.n_chrom  = len(self.names)
        self.bin_size = bin_size
        self.circular = np.array([c in circular for c in self.names])

        # Global log bins: xs of each chromosome is a prefix of the xs of the
        # largest one (same logspace values), so one searchsorted is enough.
        self.xs       = {c: generate_xs(L, base, linear_until, linear_step) for c, L in chrom_sizes.items()}
        self.xs_glob  = generate_xs(self.sizes.max(), base, linear_until, linear_step)
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


def local_coverage_ratios(raw, fraction, chrom_sizes, min_length=200_000, cap=5.0, window=25,
                          min_full=5):
    """
    Data of the effective-mappability calibration: for each coverage bin of a sequence
    >= min_length (last, shorter bin left out) with >= min_full fully mappable bins (f >= 0.99)
    within +-window bins, r = raw coverage / mean raw coverage of these bins (capped at `cap`).
    Returns (r, f) as two flat arrays (empty if no data).
    """
    from scipy.ndimage import uniform_filter1d
    R, F = [], []
    for c, L in chrom_sizes.items():
        if L < min_length or c not in raw or c not in fraction:
            continue
        r = np.asarray(raw[c], np.float64)[:-1]
        f = np.zeros(len(r))
        n = min(len(r), len(fraction[c]))
        f[:n] = np.asarray(fraction[c], np.float64)[:n]
        full = (f >= 0.99).astype(np.float64)
        size = 2 * window + 1
        tot = uniform_filter1d(r * full, size, mode="constant") * size
        cnt = uniform_filter1d(full, size, mode="constant") * size
        ok = (cnt >= min_full - 1e-6) & (tot > 0)
        if not ok.any():
            continue
        R.append(np.minimum(r[ok] / (tot[ok] / cnt[ok]), cap))
        F.append(f[ok])
    if not R:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(R), np.concatenate(F)


def effective_mappability_curve(raw, fraction, chrom_sizes, bin_size, min_length=200_000,
                                step=0.05, min_bins=20, cap=5.0, window=25, min_full=5):
    """
    Effective mappability e(f): observed coverage of a coverage bin relative to the fully
    mappable bins, as a function of its mappable fraction f (bedgraph), calibrated on the run.
    The bedgraph overestimates what our reads can map (human: f = 0.28 → e = 0.06; reads
    shorter than the k of the track after the chimera split, MAPQ >= 30, bins next to repeats),
    so a coverage divided by f is biased low in the partially mappable bins (and the expected
    of the density, weighted by f, biased high). Local reference, so that the regional biology
    (repeat-rich regions are often heterochromatin) is not mixed with the mappability: per
    sequence >= min_length (last, shorter bin left out), r = raw / mean raw of the fully
    mappable bins (f >= 0.99) within +-window bins (>= min_full of them; capped at `cap`).
    Human: local e(0.2) / 0.2 = 0.40, against 0.23 with a chromosome-wide reference. Mean r per
    class of f (width `step`, >= min_bins bins); made non-decreasing (pool-adjacent-violators,
    weighted by the bins), scaled to 1 in the top class; (0, 0) and (1, 1) added.
    Returns (f grid, e) for np.interp; the identity if too few data.
    """
    R, F = local_coverage_ratios(raw, fraction, chrom_sizes, min_length, cap, window, min_full)
    if len(R) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    edges = np.arange(0, 1 + step / 2, step)
    k = np.clip(np.digitize(F, edges[1:-1]), 0, len(edges) - 2)
    x, y, w = [], [], []
    for i in range(len(edges) - 1):
        s = k == i
        if s.sum() >= min_bins:
            x.append(F[s].mean())
            y.append(R[s].mean())
            w.append(float(s.sum()))
    if len(x) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    # pool-adjacent-violators: non-decreasing fit
    blocks = [[yy, ww, 1] for yy, ww in zip(y, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            tw = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / tw, tw, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = np.concatenate([[b[0]] * b[2] for b in blocks])
    fit = fit / fit[-1] if fit[-1] > 0 else fit
    x = np.concatenate([[0.0], x, [1.0]])
    e = np.clip(np.concatenate([[0.0], fit, [1.0]]), 0, 1)
    return x, np.maximum.accumulate(e)


def correct_coverage(raw, fraction, chrom_size, bin_size, min_fraction=0.3,
                     min_mappable_bp=100, min_ratio=0.0, window=10, circular=False,
                     min_anchor_bp=None, keep_fraction=None, context_bins=0, context_fraction=0.8):
    """
    Coverage corrected by mappability, expressed per bin_size bp:
      1. mappable length of each bin = fraction × effective width (the last
         bin of a chromosome is shorter than bin_size)
      2. valid bins: fraction (of the bin's own width) >= min_fraction, and
         at least min_mappable_bp mappable positions (guard against tiny
         remnants, e.g. 20 bp with 1 read × 100);
         corrected = raw × bin_size / mappable length
      3. optional (min_ratio > 0, off by default): low outliers, i.e. valid
         bins below min_ratio × the median of the corrected coverage of the
         `window` closest valid bins on each side, are excluded too. Off by
         default: a mappable bin with a low coverage can be real (e.g. few
         restriction sites) and must be reconstructed faithfully.
      4. excluded bins: linear interpolation between the closest valid bins
         (periodic on circular chromosomes, constant beyond the ends). Only
         bins with >= min_anchor_bp mappable bp (default bin_size / 2) are
         used as anchors: a short valid bin at a chromosome end keeps its own
         value but does not impose its slope on the bins upstream (end of
         chrI: one 490-bp bin was pulling 26 kb of interpolated coverage down)
    keep_fraction: if given, the bins are kept on keep_fraction >= min_fraction (the bedgraph
    mappable fraction) while `fraction` (the effective mappability) normalises them.
    context_bins > 0: a bin is kept only if >= context_fraction of the bins within +-context_bins
    pass the fraction threshold too: isolated mappable islands inside the satellite arrays have a
    low coverage and were the anchors of the interpolated arrays (human pilot, anchors of the gaps
    >= 500 kb: 0.41 x the median with f >= 0.9 alone, 0.82 with 80 % of +-100 kb; 2 % of the kept
    bins removed).
    Returns (corrected, valid) with valid = bins kept (not interpolated). If no
    bin is valid, corrected is NaN everywhere (to be filled by the caller).
    """
    raw = np.asarray(raw, dtype=np.float64)
    f = np.zeros(len(raw))
    n = min(len(raw), len(fraction))
    f[:n] = np.asarray(fraction)[:n]
    widths = np.clip(chrom_size - np.arange(len(raw)) * bin_size, 0, bin_size)
    mappable = f * widths

    fk = f
    if keep_fraction is not None:
        fk = np.zeros(len(raw))
        fk[:min(len(raw), len(keep_fraction))] = np.asarray(keep_fraction)[:len(raw)]
    passes = fk >= min_fraction
    valid = passes & (mappable >= max(min_mappable_bp, 1e-9))
    if context_bins > 0 and len(raw) > 1:
        from scipy.ndimage import uniform_filter1d
        ctx = uniform_filter1d(passes.astype(np.float64), 2 * int(context_bins) + 1,
                               mode="wrap" if circular else "nearest")
        valid &= ctx >= context_fraction
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
        anchors = valid & (mappable >= (bin_size / 2 if min_anchor_bp is None else min_anchor_bp))
        if not anchors.any():
            anchors = valid
        idx = np.arange(len(raw))
        corrected[~valid] = np.interp(idx[~valid], idx[anchors], corrected[anchors],
                                      period=len(raw) if circular else None)
    return corrected, valid


def rebin_counts(counts, resolution, bin_size, n_bins):
    """Sum counts binned at `resolution` into bins of bin_size (multiple of it)."""
    if bin_size % resolution:
        raise ValueError(f"bin_size {bin_size} is not a multiple of {resolution}")
    k = bin_size // resolution
    padded = np.zeros(n_bins * k)
    m = min(len(counts), len(padded))
    padded[:m] = counts[:m]
    return padded.reshape(n_bins, k).sum(axis=1)


def fill_from_all_reads(acc, final, kept, mappability, all_reads):
    """
    Option: excluded (unmappable) bins get the coverage of ALL aligned reads
    (ambiguous ones at bowtie2's random best position, streaming_classifier_v2
    --coverage_all_reads), scaled to the group1 level by the genome-wide median
    ratio group1 / all reads over fully mappable full-width bins, and expressed
    per bin_size (short last bins). Bins without any read stay 0.
    """
    z = np.load(all_reads)
    res = int(z["resolution"])
    totals, ratios = {}, []
    for i, chrom in enumerate(acc.names):
        n = len(final[chrom])
        t = rebin_counts(z[chrom], res, acc.bin_size, n) if chrom in z.files else np.zeros(n)
        widths = np.clip(acc.sizes[i] - np.arange(n) * acc.bin_size, 0, acc.bin_size)
        totals[chrom] = (t, widths)
        f = np.zeros(n)
        f[:min(n, len(mappability[chrom]))] = mappability[chrom][:n]
        ref = kept[chrom] & (f >= 0.99) & (widths == acc.bin_size) & (t > 0)
        ratios.append(final[chrom][ref] / t[ref])
    scale = float(np.median(np.concatenate(ratios)))
    for chrom in acc.names:
        t, widths = totals[chrom]
        out = ~kept[chrom] & (widths > 0)
        final[chrom][out] = scale * t[out] * acc.bin_size / widths[out]
    print(f"  excluded bins filled from all aligned reads (scale group1 / all = {scale:.3f})")
    return final


def compute_coverages(acc, smooth=True, mappability=None, all_reads=None, keep_by=None, **correction):
    """
    Returns (raw, final, kept) coverages per chromosome.
      - mappability = {chrom: fraction of mappable positions per coverage
        bin}: final = correct_coverage(**correction) (no other smoothing),
        kept = bins used (not interpolated). Excluded bins: interpolated
        (default) or, with all_reads (coverage_all_reads.npz), estimated from
        all aligned reads (fill_from_all_reads). Chromosomes without any kept
        bin get the genome-wide median.
      - otherwise: final = mad_smoothing2 (previous version), or raw if not
        smooth; kept = None.
    """
    raw = acc.coverage_by_chrom()

    if mappability is not None:
        final, kept, empty = {}, {}, []
        for i, (chrom, y) in enumerate(raw.items()):
            final[chrom], kept[chrom] = correct_coverage(
                y, mappability[chrom], acc.sizes[i], acc.bin_size,
                circular=acc.circular[i], keep_fraction=None if keep_by is None else keep_by[chrom],
                **correction)
            if not kept[chrom].any():
                empty.append(chrom)
        if all_reads is not None:
            final = fill_from_all_reads(acc, final, kept, mappability, all_reads)
            empty = []
        if empty:
            median = np.median(np.concatenate([v[np.isfinite(v)] for v in final.values()]))
            for chrom in empty:
                final[chrom][:] = median
            print(f"  [warning] no kept bin on {empty}: genome-wide median coverage used")
        total = sum(len(v) for v in raw.values())
        n_out = sum(int((~v).sum()) for v in kept.values())
        print(f"  {n_out} / {total} coverage bins excluded (unmappable"
              f"{' or low outliers' if correction.get('min_ratio', 0) > 0 else ''})"
              f" → {'all aligned reads' if all_reads is not None else 'interpolated'}")
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
                        min_mappability=1.0, coverage_min_fraction=None, effective_mappability=True,
                        coverage_min_bp=100, coverage_min_ratio=0.0, coverage_all_reads=None,
                        density=True, density_n_min=200, density_n_min_trans=200,
                        density_reproducibility=0.8, density_bin_size=None, plot_chroms=None,
                        density_refine=True, ps_linear_until=0, ps_linear_step=2, display_names=None,
                        density_min_distance=2000, density_masked="interpolate",
                        coverage_context_kb=100, coverage_context_fraction=0.8):
    """
    Compute and save the statistical laws, then (plots=True) their PDF plots
    in output_dir/plots, to be checked before running the re-attribution.

    mappability: GenMap bedgraph (mappability_v2.py). If given, the P(s)
    areas and trans lengths only count mappable positions (fraction of
    positions with score >= min_score, binned at map_resolution).
    density_bin_size: bin size of the 2D density (None = bin_size; "auto" =
    density_v2.auto_density_bin_size, from the available memory): the dense
    blocks of large genomes do not fit in memory at the coverage bin size
    (mosquito chromosome 2, 225 Mb at 2 kb: ~1.3 TB).
    plot_chroms: chromosomes shown in the density plots (None = all).
    """
    t0 = time.time()
    density_n_min_auto = density_n_min == "auto"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chrom_sizes = load_chrom_sizes(chrom_sizes_path)
    circular    = {c.strip() for c in circular.split(",") if c.strip()}
    unknown_circ = circular - set(chrom_sizes)
    if unknown_circ:
        raise ValueError(f"Circular chromosome(s) not in chromosome sizes: {sorted(unknown_circ)}")
    if coverage_all_reads is not None and mappability is None:
        raise ValueError("--coverage_all_reads needs --mappability (it fills the unmappable bins)")

    acc = Accumulator(chrom_sizes, bin_size, base, circular, ps_linear_until, ps_linear_step)

    print(f"Reading {pairs_path} ...")
    for chunk in iter_pairs(pairs_path, acc.names, chunksize):
        acc.add(*chunk)
        print(f"  {acc.stats['pairs']:>13,} pairs  ({time.time() - t0:.0f} s)")

    map_fine = map_lengths = map_coarse = bg = None
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
    map_eff = eff_curve = None
    if map_coarse is not None and effective_mappability:
        eff_curve = effective_mappability_curve(acc.coverage_by_chrom(), map_coarse, chrom_sizes,
                                                bin_size)
        map_eff = {c: np.interp(v, *eff_curve) for c, v in map_coarse.items()}
        gx, ge = eff_curve
        print("  effective mappability (coverage vs mappable fraction f, local reference, calibrated "
              "on the run): " + "  ".join(f"e({q})={np.interp(q, gx, ge):.2f}" for q in (0.2, 0.4, 0.6, 0.8, 0.9)))
        # P(s) areas and trans lengths: each fine position weighted by e(f) / f of its coverage
        # bin (human: P(s) +2-7 %, nearly uniform in s)
        if bin_size % map_resolution == 0:
            k = bin_size // map_resolution
            for c in map_fine:
                fc = np.asarray(map_coarse[c], np.float64)
                ratio = np.where(fc > 0, map_eff[c] / np.maximum(fc, 1e-12), 0.0)
                rf = np.repeat(ratio, k)
                m = np.asarray(map_fine[c], np.float64)
                rf = np.concatenate([rf, np.full(max(0, len(m) - len(rf)), rf[-1] if len(rf) else 1.0)])
                map_fine[c] = np.minimum(m * rf[:len(m)], 1.0)
    if map_fine is not None:
        map_lengths = {c: mappable_length(map_fine[c], L, map_resolution)
                       for c, L in chrom_sizes.items()}

    print("Computing intra P(s) ...")
    ps_smoothing = ps_smoothing if smooth else "none"
    uncuts, circles, weirds = compute_intra_ps(acc, ps_smoothing, map_fine, map_resolution,
                                               ps_min_pairs, ps_merge_distance, ps_pool_distance)
    print("Computing trans P(s) ...")
    trans_ps = compute_trans_ps(acc, map_lengths)
    if coverage_min_fraction is None:
        # bins kept on their bedgraph mappable fraction f; with the effective mappability the
        # corrected coverage is only unbiased for nearly fully mappable bins (human: median
        # corrected / median 0.61 at e 0.2-0.3, 0.77 at 0.4-0.5, 0.91 at 0.6-0.7 (f ~0.9), 1.00
        # above), and the poorly mappable bins next to the repeats were the anchors of the
        # interpolated arrays (chr1 HSat2 interpolated at 0.4 x the median); low-mappability
        # genomes (mosquito: 76 % of the bins partially mappable) may need e.g. 0.3
        coverage_min_fraction = 0.9 if map_eff is not None else 0.3
    print("Computing coverages ...")
    cov_raw, cov_smooth, cov_kept = compute_coverages(
        acc, smooth, map_eff if map_eff is not None else map_coarse, all_reads=coverage_all_reads,
        keep_by=map_coarse if map_eff is not None else None, min_fraction=coverage_min_fraction,
        context_bins=int(round(coverage_context_kb * 1000 / bin_size)),
        context_fraction=coverage_context_fraction,
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
    (output_dir / MAPPABILITY_EFF).unlink(missing_ok=True)
    (output_dir / EFF_CURVE).unlink(missing_ok=True)
    if map_eff is not None:
        np.save(output_dir / MAPPABILITY_EFF, map_eff)
        (output_dir / EFF_CURVE).write_text("f\teffective\n" + "".join(
            f"{a:.4f}\t{b:.4f}\n" for a, b in zip(*eff_curve)))
    if cov_kept is not None:
        np.save(output_dir / COVERAGE_KEPT, cov_kept)

    # --- 2D density (mode full = P(s) × density) ---
    if density:
        import density_v2 as dn
        from scipy import sparse
        if density_bin_size == "auto":
            # the coarser of: fits in memory, and enough contacts for the
            # adaptive window (<= 5 % of the intra contacts in fallback pixels)
            b_mem = dn.auto_density_bin_size(chrom_sizes, bin_size)
            b_data, fb = dn.data_density_bin_size(acc.ps_counts_by_chrom(), acc.xs, chrom_sizes,
                                                  bin_size, n_min=200 if density_n_min == "auto"
                                                  else int(density_n_min))
            density_bin_size = max(b_mem, b_data)
            print(f"  density bin: memory → {b_mem / 1000:g} kb, data → {b_data / 1000:g} kb "
                  f"({fb:.1%} of the intra contacts in fallback windows)")
        db = int(density_bin_size or bin_size)
        print(f"Computing density map ({db / 1000:g} kb bins) ...")
        auto = density_n_min == "auto"
        # the cis pairs closer than density_min_distance are left out of the density (described
        # by P(s) by orientation; the re-attribution uses density 1 below that distance), so its
        # diagonal is informative (not forced to 1); masked bins interpolated (density_masked)
        diag = "informative" if density_min_distance > 0 else "one"
        result = dn.contact_matrix(pairs_path, chrom_sizes, db, chunksize, split_halves=auto,
                                   min_cis_distance=density_min_distance)
        matrix, halves = result if auto else (result, None)
        def density_mask(sizes_, db_):
            n_bins_, _ = dn.genome_bins(sizes_, db_)
            mask_ = {}
            for (c, L), n in zip(sizes_.items(), n_bins_):
                # a density bin is bad if none of its coverage bins is kept
                if cov_kept is not None:
                    k = np.asarray(cov_kept[c], dtype=np.float64)
                    idx = np.minimum(np.arange(len(k)) * bin_size // db_, n - 1)
                    bad_c = np.bincount(idx, weights=k, minlength=n)[:n] == 0
                else:
                    bad_c = np.zeros(n, dtype=bool)
                if map_eff is not None:
                    # effective mappable length of the coverage bins inside each density bin
                    e2 = np.asarray(map_eff[c], np.float64)
                    wid2 = np.clip(L - np.arange(len(e2)) * bin_size, 0, bin_size)
                    i2 = np.minimum(np.arange(len(e2)) * bin_size // db_, n - 1)
                    w_c = np.bincount(i2, weights=e2 * wid2, minlength=n)[:n] / db_
                elif bg is not None:
                    frac_c = binned_unique_fraction({c: bg[c]} if c in bg else {}, {c: L}, db_,
                                                    min_mappability)[c][:n]
                    w_c = frac_c * np.clip(L - np.arange(n) * db_, 0, db_) / db_
                else:
                    w_c = None
                mask_[c] = (bad_c, w_c)
            return mask_

        bad = weights = None
        mask = density_mask(chrom_sizes, db)
        if cov_kept is not None:
            bad = np.concatenate([mask[c][0] for c in chrom_sizes])
        if bg is not None:
            weights = np.concatenate([mask[c][1] for c in chrom_sizes])
        if auto:
            best, part = dn.reproducible_n_min(halves, chrom_sizes, db, bad, weights,
                                               target=density_reproducibility)
            for g, d in part.items():
                print(f"  reproducible fluctuations {g}: "
                      + "  ".join(f"{k}:{v:.0%}" for k, v in d.items()) + f" → n_min = {best[g]}")
            density_n_min, density_n_min_trans = best["intra"], best["trans"]
        maps = dn.density_maps(matrix, chrom_sizes, db, bad, method="unbiased",
                               smoothing="gaussian", n_min=int(density_n_min),
                               n_min_trans=int(density_n_min_trans), weights=weights,
                               masked=density_masked, diag=diag)
        maps = {k: v.astype(np.float32) for k, v in maps.items()}
        np.save(output_dir / DENSITY_MIN_DIST, int(density_min_distance))
        np.save(output_dir / DENSITY_MAP, maps)
        np.save(output_dir / DENSITY_BIN, db)
        np.save(output_dir / DENSITY_MASK, mask)
        sparse.save_npz(output_dir / CONTACTS, sparse.triu(matrix).tocsr())

        # refined density: molecules whose own contacts support a bin >= 4x finer than the global
        # one (the global bin is set by the largest, sparsest chromosomes: mosquito 200 kb, so
        # Wolbachia (1.37 Mb) had 7 density bins); one finer map for these molecules together
        for f in (DENSITY_FINE_MAP, DENSITY_FINE_BIN, DENSITY_FINE_MASK):
            (output_dir / f).unlink(missing_ok=True)
        if density_refine:
            pc = acc.ps_counts_by_chrom()
            own = {c: dn.data_density_bin_size({c: pc[c]}, acc.xs, {c: L}, bin_size,
                                               n_min=int(density_n_min))[0]
                   for c, L in chrom_sizes.items()}
            # enough intra contacts to estimate a fine density (else tiny scaffolds qualify by default)
            n_intra = {c: float(np.asarray(pc[c]).sum()) for c in chrom_sizes}
            fine = {c: b for c, b in own.items() if 4 * b <= db and n_intra[c] >= 20 * int(density_n_min)}
            if fine:
                sizes_f = {c: chrom_sizes[c] for c in chrom_sizes if c in fine}
                db_f = max(max(fine.values()), dn.auto_density_bin_size(sizes_f, bin_size))
                if 4 * db_f <= db:
                    mask_f = density_mask(sizes_f, db_f)
                    bad_f = np.concatenate([mask_f[c][0] for c in sizes_f]) if cov_kept is not None else None
                    w_f = np.concatenate([mask_f[c][1] for c in sizes_f]) if bg is not None else None
                    mat_f = dn.contact_matrix(pairs_path, sizes_f, db_f, chunksize,
                                              min_cis_distance=density_min_distance)
                    maps_f = dn.density_maps(mat_f, sizes_f, db_f, bad_f, method="unbiased",
                                             smoothing="gaussian", n_min=int(density_n_min),
                                             n_min_trans=int(density_n_min_trans), weights=w_f,
                                             masked=density_masked, diag=diag)
                    np.save(output_dir / DENSITY_FINE_MAP, {k: v.astype(np.float32) for k, v in maps_f.items()})
                    np.save(output_dir / DENSITY_FINE_BIN, db_f)
                    np.save(output_dir / DENSITY_FINE_MASK, mask_f)
                    print(f"  refined density ({db_f / 1000:g} kb bins) for: {', '.join(sizes_f)}")

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
            f"  resolution / min    : {map_resolution} bp / score >= {min_mappability}"
            + ("; P(s) areas and trans lengths: positions weighted by e(f) / f of their coverage bin"
               if map_eff is not None else ""),
            f"  coverage            : raw x {bin_size} / "
            f"{'effective mappable bp (e(f) calibrated on the run, ' + EFF_CURVE + ')' if map_eff is not None else 'mappable bp'}"
            f"; bins kept if mappable fraction f >= "
            f"{coverage_min_fraction}, >= {coverage_min_bp} mappable bp"
            + (f", >= {coverage_context_fraction:.0%} of the bins within +-{coverage_context_kb:g} kb too"
               if coverage_context_kb > 0 else "")
            + (f", >= {coverage_min_ratio} x local median" if coverage_min_ratio > 0
               else " (low-coverage bins kept)"),
            "  excluded bins       : " + (f"from all aligned reads ({coverage_all_reads})"
                                         if coverage_all_reads else "linear interpolation"),
            f"  mappable genome     : {mappable:,.0f} / {total:,} bp ({100 * mappable / total:.1f}%)",
        ]
    else:
        lines.append("mappability           : - (trapezoid areas)")
    lines.append("density               : " + (
        f"{db / 1000:g} kb bins, unbiased, adaptive Gaussian (>= {density_n_min} contacts intra, "
        f">= {density_n_min_trans} trans{f', auto: >= {density_reproducibility:.0%} reproducible' if density_n_min_auto else ''})"
        + (", mappability mask and weights" if map_coarse is not None else "")
        + (f" (effective mappability)" if map_eff is not None else "")
        + f", masked bins {density_masked}"
        + (f", cis pairs < {density_min_distance} bp left out (density held at its diagonal value there), diagonal informative"
           if density_min_distance > 0 else ", diagonal = 1")
        if density else "-"))
    lines.append(f"time                  : {time.time() - t0:.0f} s")
    (output_dir / "statistics_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n=== Statistics ===\n" + "\n".join(lines))

    # --- Plots: check the laws before re-attribution ---
    if plots:
        from plots_v2 import plot_all, DISPLAY_NAMES
        if display_names:
            np.save(output_dir / DISPLAY_NAMES, dict(display_names))
        print()
        plot_all(output_dir, output_dir / "plots", bin_size=bin_size, plot_chroms=plot_chroms)
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
    parser.add_argument("--coverage_min_fraction", type=float, default=None,
                        help="Coverage bins kept (not interpolated) if their mappable fraction f "
                             "(bedgraph) is >= this value; they also define the masked density bins "
                             "(default 0.9 with the effective mappability, 0.3 without; "
                             "low-mappability genomes, e.g. mosquito: 0.3)")
    parser.add_argument("--coverage_context_kb", type=float, default=100,
                        help="A coverage bin is kept only if >= --coverage_context_fraction of the bins "
                             "within +- this distance (kb) pass the fraction threshold too: isolated "
                             "mappable islands inside satellite arrays are interpolated (default 100; 0 = off)")
    parser.add_argument("--coverage_context_fraction", type=float, default=0.8)
    parser.add_argument("--no_effective_mappability", action="store_true",
                        help="Coverage and density weights proportional to the bedgraph mappable "
                             "fraction (previous behaviour) instead of the effective mappability "
                             "calibrated on the run (the bedgraph overestimates it: human f 0.28 → 0.06)")
    parser.add_argument("--coverage_min_bp", type=int, default=100,
                        help="... and if they contain >= this number of mappable bp (default=100)")
    parser.add_argument("--coverage_min_ratio", type=float, default=0.0,
                        help="Optional: also exclude mappable bins whose corrected coverage is < this "
                             "ratio x the local median (10 closest kept bins on each side). Off by "
                             "default: low coverage can be real (default=0)")
    parser.add_argument("--no_density", action="store_true",
                        help="Do not compute the 2D density map (mode full)")
    parser.add_argument("--density_n_min", default="200",
                        help="Density: min contacts per adaptive window in intra, or 'auto': "
                             "smallest threshold (intra and trans) whose fine fluctuations are "
                             "reproducible (--density_reproducibility) between two halves of the "
                             "pairs (default=200)")
    parser.add_argument("--density_reproducibility", type=float, default=0.8,
                        help="--density_n_min auto: target reproducible part of the fine "
                             "fluctuations of the density (default=0.8)")
    parser.add_argument("--density_bin_size", default=None,
                        help="Bin size of the 2D density (default = --bin_size; 'auto' = from "
                             "the available memory, for large genomes)")
    parser.add_argument("--density_min_distance", type=int, default=2000,
                        help="Cis pairs closer than this (bp) are left out of the density (uncuts, "
                             "self-ligations: the orientations are random from ~1 kb on) and the "
                             "re-attribution holds the density at its diagonal value below it (P(s) by "
                             "orientation shapes the short range); the density diagonal is then "
                             "informative (default 2000; 0 = previous behaviour, diagonal forced to 1)")
    parser.add_argument("--density_masked", choices=("interpolate", "neutral", "diagonal"), default="interpolate",
                        help="Density of the masked bins (no unique data): interpolated from the "
                             "flanks (default) or neutral (1)")
    parser.add_argument("--density_n_min_trans", type=int, default=200,
                        help="Density: min contacts per adaptive window in trans (default=200)")
    parser.add_argument("--coverage_all_reads", default=None,
                        help="Optional: coverage_all_reads.npz (streaming_classifier_v2.py "
                             "--coverage_all_reads): excluded bins get the coverage of all aligned "
                             "reads instead of an interpolation")
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
        effective_mappability = not args.no_effective_mappability,
        coverage_context_kb = args.coverage_context_kb,
        coverage_context_fraction = args.coverage_context_fraction,
        coverage_min_bp  = args.coverage_min_bp,
        coverage_min_ratio = args.coverage_min_ratio,
        coverage_all_reads = args.coverage_all_reads,
        density          = not args.no_density,
        density_n_min    = args.density_n_min if args.density_n_min == "auto" else int(args.density_n_min),
        density_n_min_trans = args.density_n_min_trans,
        density_min_distance = args.density_min_distance,
        density_masked   = args.density_masked,
        density_reproducibility = args.density_reproducibility,
        density_bin_size = (args.density_bin_size if args.density_bin_size in (None, "auto")
                            else int(args.density_bin_size)),
    )


if __name__ == "__main__":
    main()
