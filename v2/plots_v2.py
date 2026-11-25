#!/usr/bin/env python3
"""
plots_v2.py

PDF plots of the statistical laws produced by statistics_v2.py.

Works on any statistics folder (v1 or v2: same file names), and can overlay
a second folder (e.g. the previous hicberg version) to compare both:

  ps.pdf            P(s) per chromosome (uncut +-, circle -+, weird ++/--)
  ps_ratio.pdf      ratio P(s) v2 / reference per chromosome   (with --compare_dir)
  coverage.pdf      raw + smoothed coverage per chromosome
  trans_ps.pdf      trans P(s) heatmap (+ scatter v2 vs reference)
  density.pdf       log2 density map, genome + zooms (if density_map.npy)
  density_intra.pdf raw obs / exp and density, one page per chromosome
  density_trans.pdf same for each pair of chromosomes (if contacts.npz)
  orientations.pdf  fraction of each orientation vs distance  (v2 only, ps_counts.npz)

One page per chromosome in each PDF.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import numpy as np                                    # noqa: E402


def opaque_imshow(ax, z, cmap, vmin, vmax, bad="white", **kwargs):
    """imshow of a colour-mapped matrix as an opaque RGB image.

    -inf (log of zero contacts) gets the lowest colour of the scale, NaN the colour `bad`.
    A masked / NaN array is otherwise embedded in the PDF with a transparency mask, which
    PDF viewers render as grey speckles when the page is not zoomed. Returns a
    ScalarMappable for the colorbar."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, to_rgb
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = Normalize(vmin=vmin, vmax=vmax)
    z = np.asarray(np.ma.filled(z, np.nan) if np.ma.isMaskedArray(z) else z, dtype=float)
    nan = np.isnan(z)
    rgb = cmap(norm(np.where(np.isneginf(z) | nan, vmin, z)))[..., :3]
    rgb[nan] = to_rgb(bad)
    kwargs.setdefault("interpolation", "none")
    ax.imshow(rgb, **kwargs)
    return ScalarMappable(norm=norm, cmap=cmap)


XS, UNCUTS, CIRCLES, WEIRDS = "xs.npy", "uncuts.npy", "circles.npy", "weirds.npy"
TRANS_PS, COVERAGE_RAW, COVERAGE = "trans_ps.npy", "coverage1.npy", "coverage.npy"
DENSITY, CONTACTS, CHROM_SIZES = "density_map.npy", "contacts.npz", "chromosome_sizes.npy"
DENSITY_BIN, DENSITY_MASK = "density_bin_size.npy", "density_mask.npy"
PS_COUNTS, MAPPABILITY, COVERAGE_KEPT = "ps_counts.npz", "mappability.npy", "coverage_kept.npy"

# Same colours as hicberg/plot.py
DISPLAY_NAMES = "display_names.npy"   # {identifier: readable name} (statistics_v2, from the FASTA headers)
_NAMES = {}


def set_display_names(stats_dir):
    """Readable sequence names (chr1 ... chrX, Mitochondria) for the titles, if the statistics
    folder has them; identifiers otherwise."""
    global _NAMES
    p = Path(stats_dir) / DISPLAY_NAMES if stats_dir else None
    _NAMES = np.load(p, allow_pickle=True).item() if p is not None and p.is_file() else {}


def _n(c):
    return _NAMES.get(c, c)


LAWS = (
    (UNCUTS,  "+- (uncut)",   "salmon"),
    (CIRCLES, "-+ (circle)",  "limegreen"),
    (WEIRDS,  "++/-- (weird)", "royalblue"),
)
ORIENT_COLORS = {"+-": "salmon", "-+": "limegreen", "++": "royalblue", "--": "navy"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dict(folder, name):
    """Load a dictionary saved with np.save, or None if the file is missing."""
    path = Path(folder) / name
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=True).item()


def _chromosomes(main, ref=None):
    """Chromosomes of main (file order), then those only present in ref."""
    names = list(main)
    if ref is not None:
        names += [c for c in ref if c not in main]
    return names


def _positive(x, y):
    """Points that can be drawn on a log-log scale."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    keep = (x > 0) & (y > 0) & np.isfinite(y)
    return x[keep], y[keep]


# ---------------------------------------------------------------------------
# P(s)
# ---------------------------------------------------------------------------

def plot_ps(stats_dir, output_pdf, compare_dir=None, labels=("v2", "reference")):
    """
    One page per chromosome: P(s) of the 3 event types, log-log.
    With compare_dir, the reference laws are overlaid as lines.
    """
    set_display_names(stats_dir)
    xs = load_dict(stats_dir, XS)
    laws = {f: load_dict(stats_dir, f) for f, _, _ in LAWS}
    xs_ref = load_dict(compare_dir, XS) if compare_dir else None
    laws_ref = {f: load_dict(compare_dir, f) for f, _, _ in LAWS} if compare_dir else None

    with PdfPages(output_pdf) as pdf:
        for chrom in _chromosomes(xs, xs_ref):
            fig, ax = plt.subplots(figsize=(8, 7))

            for f, label, color in LAWS:
                if chrom in xs:
                    ax.plot(*_positive(xs[chrom], laws[f][chrom]), "o", ms=4, color=color,
                            label=f"{label} {labels[0]}" if compare_dir else label)
                if xs_ref is not None and chrom in xs_ref:
                    ax.plot(*_positive(xs_ref[chrom], laws_ref[f][chrom]), "-", lw=1.5,
                            color=color, alpha=0.8, label=f"{label} {labels[1]}")

            missing = [lab for lab, d in zip(labels, (xs, xs_ref)) if d is not None and chrom not in d]
            title = f"P(s) — {_n(chrom)}" + (f"  (absent in {', '.join(missing)})" if missing else "")
            if not any(len(l.get_xdata()) for l in ax.get_lines()):
                # no pair on this sequence (e.g. small unmappable scaffold): skip the page
                plt.close(fig)
                continue
            ax.set(xscale="log", yscale="log", title=title,
                   xlabel="Genomic distance (bp, log bins)",
                   ylabel="Number of events / number of position pairs")
            ax.grid(True, which="major", alpha=0.4)
            ax.legend(fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)


def plot_ps_ratio(stats_dir, compare_dir, output_pdf, labels=("v2", "reference")):
    """
    One page per chromosome: ratio of the P(s) laws (v2 / reference) on the
    common distance bins. A ratio of 1 means identical laws.
    """
    xs, xs_ref = load_dict(stats_dir, XS), load_dict(compare_dir, XS)
    laws = {f: load_dict(stats_dir, f) for f, _, _ in LAWS}
    laws_ref = {f: load_dict(compare_dir, f) for f, _, _ in LAWS}
    common = [c for c in xs if c in xs_ref]

    summary = []
    with PdfPages(output_pdf) as pdf:
        for chrom in common:
            fig, ax = plt.subplots(figsize=(8, 5))
            row = [chrom]
            for f, label, color in LAWS:
                x, ratio = _ratio_on_common_bins(xs[chrom], laws[f][chrom],
                                                 xs_ref[chrom], laws_ref[f][chrom])
                ax.plot(x, ratio, "o-", ms=3, lw=0.8, color=color, label=label)
                row.append(np.nanmedian(ratio) if len(ratio) else np.nan)
            summary.append(row)

            ax.axhline(1, color="black", lw=1)
            ax.set(xscale="log", yscale="log", ylim=(0.5, 2),
                   title=f"P(s) ratio {labels[0]} / {labels[1]} — {_n(chrom)}",
                   xlabel="Genomic distance (bp, log bins)", ylabel="Ratio")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)

        # Summary page: median ratio per chromosome and per law
        fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(summary) + 1.5)))
        y = np.arange(len(summary))
        for k, (_, label, color) in enumerate(LAWS):
            ax.plot([r[k + 1] for r in summary], y, "o", color=color, label=label)
        ax.axvline(1, color="black", lw=1)
        ax.set_yticks(y, [r[0] for r in summary])
        ax.invert_yaxis()
        ax.set(xscale="log", xlim=(0.5, 2), xlabel=f"Median ratio {labels[0]} / {labels[1]}",
               title="Median P(s) ratio per chromosome")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return summary


def _ratio_on_common_bins(x1, y1, x2, y2):
    x1, y1 = np.asarray(x1), np.asarray(y1, dtype=float)
    x2, y2 = np.asarray(x2), np.asarray(y2, dtype=float)
    x, i1, i2 = np.intersect1d(x1[:len(y1)], x2[:len(y2)], return_indices=True)
    a, b = y1[i1], y2[i2]
    keep = (x > 0) & (a > 0) & (b > 0)
    return x[keep], a[keep] / b[keep]


# ---------------------------------------------------------------------------
# Orientations (v2 only)
# ---------------------------------------------------------------------------

def plot_orientations(stats_dir, output_pdf, min_pairs=50):
    """
    Fraction of each orientation (+-, -+, ++, --) vs distance, all
    chromosomes pooled (first page) then per chromosome. Expected: 25% each
    at large distance; excess of +- (dangling ends) and -+ (self-circles)
    at short distance.
    """
    set_display_names(stats_dir)
    path = Path(stats_dir) / PS_COUNTS
    if not path.is_file():
        print(f"  {PS_COUNTS} not found in {stats_dir}: orientations plot skipped")
        return False

    z = np.load(path)
    orientations = list(z["orientations"])
    xs = load_dict(stats_dir, XS)
    chroms = [c for c in xs if c in z.files]

    # Pooled: chromosomes share the same xs prefix
    n = max(len(xs[c]) for c in chroms)
    x_all = next(xs[c] for c in chroms if len(xs[c]) == n)
    pooled = np.zeros((len(orientations), n))
    for c in chroms:
        pooled[:, :z[c].shape[1]] += z[c]

    with PdfPages(output_pdf) as pdf:
        for name, x, counts in [("all chromosomes", x_all, pooled)] + \
                               [(c, xs[c], z[c]) for c in chroms]:
            total = counts.sum(axis=0)
            keep = (total >= min_pairs) & (x > 0)
            if not keep.any():
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for k, o in enumerate(orientations):
                ax.plot(x[keep], counts[k, keep] / total[keep], "o-", ms=3, lw=0.8,
                        color=ORIENT_COLORS.get(o), label=o)
            ax.axhline(0.25, color="black", lw=1, ls="--")
            ax.set(xscale="log", ylim=(0, 1), title=f"Pair orientations — {_n(name)}",
                   xlabel="Genomic distance (bp, log bins)",
                   ylabel=f"Fraction of pairs (bins with ≥ {min_pairs} pairs)")
            ax.grid(True, which="major", alpha=0.4)
            ax.legend(fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def plot_coverage(stats_dir, output_pdf, bin_size=2000, compare_dir=None,
                  labels=("v2", "reference")):
    """
    One page per chromosome: raw coverage (points) + smoothed (line), and the
    fraction of mappable positions per bin (right axis) if available.
    """
    set_display_names(stats_dir)
    raw, smooth = load_dict(stats_dir, COVERAGE_RAW), load_dict(stats_dir, COVERAGE)
    smooth_ref = load_dict(compare_dir, COVERAGE) if compare_dir else None
    mappability = load_dict(stats_dir, MAPPABILITY)
    kept = load_dict(stats_dir, COVERAGE_KEPT)

    with PdfPages(output_pdf) as pdf:
        for chrom in _chromosomes(smooth, smooth_ref):
            fig, ax = plt.subplots(figsize=(11, 5))
            if chrom in raw and kept is not None and chrom in kept:
                x, k = np.arange(len(raw[chrom])), kept[chrom]
                ax.plot(x[k], raw[chrom][k], "o", ms=1.5, color="cornflowerblue",
                        label=f"raw {labels[0]} — bins kept", rasterized=True)
                ax.plot(x[~k], raw[chrom][~k], "o", ms=2.5, color="red",
                        label=f"raw {labels[0]} — bins excluded (interpolated)", rasterized=True)
            elif chrom in raw:
                ax.plot(raw[chrom], "o", ms=1, color="cornflowerblue",
                        label=f"raw {labels[0]}", rasterized=True)
            if chrom in smooth:
                ax.plot(smooth[chrom], color="darkblue", lw=1,
                        label=f"final {labels[0]} (coverage.npy)")
            if smooth_ref is not None and chrom in smooth_ref:
                ax.plot(smooth_ref[chrom], color="orange", lw=1, alpha=0.8,
                        label=f"final {labels[1]}")
            ax.set(title=f"Coverage — {_n(chrom)} — bins of {bin_size} bp",
                   xlabel="Bin number", ylabel="Number of reads")
            ax.grid(True, alpha=0.4)
            handles, labels_ = ax.get_legend_handles_labels()
            if mappability is not None and chrom in mappability:
                ax2 = ax.twinx()
                ax2.fill_between(np.arange(len(mappability[chrom])), mappability[chrom],
                                 step="mid", color="grey", alpha=0.25, lw=0)
                ax2.set(ylim=(0, 1.05), ylabel="Fraction of mappable positions")
                handles.append(plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.25))
                labels_.append("mappability")
            ax.legend(handles, labels_, fontsize=8, markerscale=4)
            pdf.savefig(fig)
            plt.close(fig)


def plot_effective_mappability(stats_dir, output_pdf, step=0.05):
    """
    Calibration of the effective mappable fraction e(f) of the run (statistics_v2.
    effective_mappability_curve): top, local coverage ratio of each coverage bin (coverage / mean
    coverage of the fully mappable bins within +-50 kb) against its mappable fraction f (2-D
    histogram), mean per class of f, calibrated curve e(f) (effective_mappability.tsv) and the
    identity e = f (mappability track taken as is); bottom, number of bins per class of f.
    Returns False if the files are missing.
    """
    import statistics_v2 as st
    raw, frac = load_dict(stats_dir, COVERAGE_RAW), load_dict(stats_dir, MAPPABILITY)
    sizes = load_dict(stats_dir, CHROM_SIZES)
    tsv = Path(stats_dir) / "effective_mappability.tsv"
    if raw is None or frac is None or sizes is None or not tsv.is_file():
        return False
    curve = np.loadtxt(tsv, skiprows=1, ndmin=2)
    r, f = st.local_coverage_ratios(raw, frac, sizes)
    if len(r) == 0:
        return False
    edges = np.arange(0, 1 + step / 2, step)
    k = np.clip(np.digitize(f, edges[1:-1]), 0, len(edges) - 2)
    n = np.bincount(k, minlength=len(edges) - 1)
    mean_r = np.bincount(k, weights=r, minlength=len(edges) - 1) / np.maximum(n, 1)
    mean_f = np.bincount(k, weights=f, minlength=len(edges) - 1) / np.maximum(n, 1)
    good = n >= 20

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=(3, 1), sharex=True)
    h, xe, ye = np.histogram2d(f, r, bins=(np.linspace(0, 1, 101), np.linspace(0, 2, 101)))
    h = np.where(h > 0, h, np.nan)
    ax.pcolormesh(xe, ye, np.log10(h).T, cmap="Greys", vmin=0, rasterized=True)
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.5, color="#5f5e5a", label="e = f (mappability track as is)")
    ax.plot(curve[:, 0], curve[:, 1], lw=2.5, color="#2a78d6", label="e(f) used (monotone fit)", zorder=3)
    ax.plot(mean_f[good], mean_r[good], "o", ms=8, color="#eb6834", mec="white", mew=1.5,
            label="mean ratio per class of f", zorder=4)
    for q in (0.2, 0.5, 0.8):
        e = np.interp(q, curve[:, 0], curve[:, 1])
        ax.annotate(f"e({q}) = {e:.2f}", (q, e), xytext=(-10, 22), textcoords="offset points",
                    fontsize=9, color="#2a2a28", ha="right", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color="#5f5e5a", lw=0.8))
    ax.set(xlim=(0, 1), ylim=(0, 2), ylabel="coverage / coverage of the fully mappable bins (±50 kb)",
           title=f"Effective mappable fraction e(f), calibrated on the unique pairs of the run\n"
                 f"{len(r):,} coverage bins; grey: number of bins (log scale)")
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    axh.bar(edges[:-1], n, width=step, align="edge", color=np.where(good, "#2a78d6", "#c8c7c2"),
            edgecolor="white", lw=1)
    axh.set(yscale="log", xlabel="mappable fraction f of the bin (mappability track)",
            ylabel="bins")
    axh.grid(color="#e6e5e0", lw=0.6, axis="y")
    for a in (ax, axh):
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Trans P(s)
# ---------------------------------------------------------------------------

def _trans_matrix(trans_ps, chroms):
    m = np.array([[trans_ps.get((a, b), trans_ps.get((b, a), np.nan)) for b in chroms]
                  for a in chroms], dtype=float)
    np.fill_diagonal(m, np.nan)   # intra: not a trans law
    return m


def trans_oe(m, n_iter=200):
    """
    Observed / expected of a chromosome x chromosome trans matrix (diagonal ignored): iterative
    balancing (ICE-like) to equal marginals, then divided by the mean. Every per-chromosome
    factor (mappable length, copy number, coverage, digestion) is absorbed by the balancing, so
    the trans P(s) can be balanced directly instead of the counts. 1 = no preference.
    Chromosomes without any trans value (excluded) get NaN.
    """
    x = np.where(np.isfinite(m) & (m > 0), m, 0.0).astype(np.float64)
    np.fill_diagonal(x, 0.0)
    ok = x.sum(axis=1) > 0
    b = np.where(ok, 1.0, 0.0)
    for _ in range(n_iter):
        s = (x * b[:, None] * b[None, :]).sum(axis=1)
        mean = s[ok].mean()
        b[ok] /= np.sqrt(s[ok] / mean)
    bal = x * b[:, None] * b[None, :]
    off = ~np.eye(len(x), dtype=bool) & ok[:, None] & ok[None, :] & (bal > 0)
    oe = np.full(x.shape, np.nan)
    oe[off] = bal[off] / bal[off].mean()
    return oe


def plot_trans_ps(stats_dir, output_pdf, compare_dir=None, labels=("v2", "reference")):
    """Heatmap of log10 trans P(s) (the absolute rates used for the re-attribution weights), then
    the balanced observed / expected (relative contact preferences, display only); with
    compare_dir, side by side + scatter."""
    set_display_names(stats_dir)
    tps = load_dict(stats_dir, TRANS_PS)
    chroms = list(load_dict(stats_dir, XS))
    m = _trans_matrix(tps, chroms)

    with PdfPages(output_pdf) as pdf:
        panels = [(labels[0], m)]
        if compare_dir:
            m_ref = _trans_matrix(load_dict(compare_dir, TRANS_PS), chroms)
            panels.append((labels[1], m_ref))

        with np.errstate(divide="ignore", invalid="ignore"):
            logs = [np.log10(np.where(p > 0, p, np.nan)) for _, p in panels]
        vmin = np.nanmin([np.nanmin(l) for l in logs])
        vmax = np.nanmax([np.nanmax(l) for l in logs])

        fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 7.5), squeeze=False)
        for ax, (label, _), lm in zip(axes[0], panels, logs):
            im = opaque_imshow(ax, lm, "Wistia", vmin, vmax)
            ax.set_xticks(range(len(chroms)), [_n(c) for c in chroms], rotation=90, fontsize=7)
            ax.set_yticks(range(len(chroms)), [_n(c) for c in chroms], fontsize=7)
            ax.set_title(f"log10 trans P(s) — {label}\ncontacts per pair of mappable positions: "
                         f"used for the re-attribution weights", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # relative contacts: observed / expected after balancing (display only, not in the weights)
        oes = [(lab, trans_oe(p)) for lab, p in panels]
        with np.errstate(divide="ignore", invalid="ignore"):
            loe = [np.log2(o) for _, o in oes]
        allv = np.concatenate([l[np.isfinite(l)] for l in loe]) if any(np.isfinite(l).any() for l in loe) else np.array([1.0])
        lim = float(np.clip(np.percentile(np.abs(allv), 98), 0.3, 2.0))
        fig, axes = plt.subplots(1, len(oes), figsize=(8 * len(oes), 7.5), squeeze=False)
        for ax, (label, _), lo in zip(axes[0], oes, loe):
            im = opaque_imshow(ax, lo, "RdBu_r", -lim, lim, bad="lightgrey")
            ax.set_xticks(range(len(chroms)), [_n(c) for c in chroms], rotation=90, fontsize=7)
            ax.set_yticks(range(len(chroms)), [_n(c) for c in chroms], fontsize=7)
            ax.set_title(f"log2 observed / expected trans contacts — {label}\nrelative contacts "
                         f"(balanced with ICE): display only, not used for the weights", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, label="log2 O/E")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if compare_dir:
            iu = np.triu_indices(len(chroms), k=1)
            a, b = m[iu], panels[1][1][iu]
            keep = (a > 0) & (b > 0)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.loglog(b[keep], a[keep], "o", ms=4, alpha=0.7)
            lims = [min(a[keep].min(), b[keep].min()), max(a[keep].max(), b[keep].max())]
            ax.plot(lims, lims, "k-", lw=1)
            ax.set(xlabel=f"trans P(s) {labels[1]}", ylabel=f"trans P(s) {labels[0]}",
                   title=f"Trans P(s): {labels[0]} vs {labels[1]} (one point per chromosome pair)")
            ax.grid(True, which="both", alpha=0.3)
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Density maps
# ---------------------------------------------------------------------------

def _block_downsample(blk, factor):
    """nan-mean over factor × factor cells of a (rectangular) block."""
    if factor <= 1:
        return np.asarray(blk, dtype=np.float64)
    r, c = -(-blk.shape[0] // factor), -(-blk.shape[1] // factor)
    p = np.full((r * factor, c * factor), np.nan)
    p[:blk.shape[0], :blk.shape[1]] = blk
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmean(p.reshape(r, factor, c, factor), axis=(1, 3))


def _genome_density(density, chroms=None, max_pixels=None):
    """
    Assemble the {(a, b): block} density map into one genome-wide matrix
    (chromosomes `chroms`, default all), each block downsampled by the same
    factor first (at most max_pixels rows): the full matrix of a large genome
    does not fit in memory. Returns (matrix, names, offsets, factor).
    """
    names = [a for a, b in density if a == b]
    if chroms is not None:
        names = [c for c in names if c in set(chroms)]
    total = sum(density[(c, c)].shape[0] for c in names)
    factor = max(1, -(-total // max_pixels)) if max_pixels else 1
    sizes = [-(-density[(c, c)].shape[0] // factor) for c in names]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    g = np.full((offsets[-1], offsets[-1]), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if (a, b) in density:
                blk = _block_downsample(density[(a, b)], factor)
                g[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]] = blk
                g[offsets[j]:offsets[j + 1], offsets[i]:offsets[i + 1]] = blk.T
    return g, names, offsets, factor


def _downsample(g, factor):
    """nan-mean over factor × factor blocks."""
    if factor <= 1:
        return g
    n = -(-g.shape[0] // factor) * factor
    p = np.full((n, n), np.nan)
    p[:g.shape[0], :g.shape[1]] = g
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmean(p.reshape(n // factor, factor, n // factor, factor), axis=(1, 3))


def plot_density(maps, output_pdf, max_pixels=2000, zoom_pairs=None, vlim=2.0, chroms=None):
    """
    maps: [(label, density_map.npy path), ...] shown side by side.
    Page 1: genome-wide log2(density) (red = more contacts than expected);
    next pages: zoom on chromosome pairs (default: the first 3 trans pairs).
    """
    loaded = [(lab, np.load(p, allow_pickle=True).item()) for lab, p in maps]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")

    def show(ax, mat, title, ticks=None):
        with np.errstate(divide="ignore", invalid="ignore"):
            im = opaque_imshow(ax, np.log2(mat), cmap, -vlim, vlim, bad="lightgrey")
        ax.set_title(title, fontsize=9)
        if ticks:
            pos, lab = ticks
            ax.set_xticks(pos, [_n(x) for x in lab], rotation=90, fontsize=6)
            ax.set_yticks(pos, [_n(x) for x in lab], fontsize=6)
        return im

    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(1, len(loaded), figsize=(7 * len(loaded), 7), squeeze=False)
        for ax, (lab, d) in zip(axes[0], loaded):
            g, names, offsets, factor = _genome_density(d, chroms, max_pixels)
            im = show(ax, g, f"{lab} — log2 density (genome)",
                      ((offsets[:-1] + offsets[1:]) / 2, names))
            for o in offsets[1:-1]:
                ax.axhline(o - 0.5, color="black", lw=0.3)
                ax.axvline(o - 0.5, color="black", lw=0.3)
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, label="log2 density")
        pdf.savefig(fig)
        plt.close(fig)

        first = loaded[0][1]
        if zoom_pairs is None:
            zoom_pairs = [k for k in first if k[0] != k[1]
                          and (chroms is None or (k[0] in chroms and k[1] in chroms))][:3]
        for pair in zoom_pairs:
            fig, axes = plt.subplots(1, len(loaded), figsize=(6 * len(loaded), 5), squeeze=False)
            for ax, (lab, d) in zip(axes[0], loaded):
                if pair in d:
                    im = show(ax, d[pair], f"{lab} — {_n(pair[0])} × {_n(pair[1])}")
                    ax.set(xlabel=f"{_n(pair[1])} bin", ylabel=f"{_n(pair[0])} bin")
            fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, label="log2 density")
            pdf.savefig(fig)
            plt.close(fig)


def plot_density_matrices(stats_dir, output_dir, bin_size=2000, max_trans_pairs=200,
                          vlim=1.5, vlim_raw=3.0, raw_bin_intra=1, raw_bin_trans=5,
                          chroms=None, min_bins=5):
    """
    One page per chromosome (density_intra.pdf) and per pair of chromosomes
    (density_trans.pdf, at most max_trans_pairs), with 2 matrices side by side,
    real proportions, axes in kb, one colour bar each:
      - raw obs / exp (not smoothed, same mask and mappable-length weights as
        the density), aggregated over raw_bin × raw_bin bins (sum obs / sum
        exp): 1 (2 kb) in intra, 5 (10 kb) in trans where 2-kb pixels hold
        0 or 1 contact; log2 range ±vlim_raw
      - the density used by the re-attribution; log2 range ±vlim
    Only the sequences with >= min_bins density bins get pages (5: Wolbachia, 7 bins of 200 kb, kept; small scaffolds: 1-2 bins of
    200 kb in the mosquito, unreadable pages that filled the 200 trans pages).
    Needs density_map.npy and contacts.npz (statistics_v2.py).
    """
    set_display_names(stats_dir)
    import itertools
    from scipy import sparse
    import density_v2 as dn

    stats_dir, output_dir = Path(stats_dir), Path(output_dir)
    density = load_dict(stats_dir, DENSITY)
    sizes = load_dict(stats_dir, CHROM_SIZES)
    upper = sparse.load_npz(stats_dir / CONTACTS)
    matrix = (upper + upper.T - sparse.diags(upper.diagonal())).tocsr()
    all_names = [a for a, b in density if a == b]
    n_bins = {c: density[(c, c)].shape[0] for c in all_names}
    offsets = dict(zip(all_names,
                       np.concatenate([[0], np.cumsum([n_bins[c] for c in all_names])[:-1]])))
    names = [c for c in all_names if chroms is None or c in set(chroms)]
    small = [c for c in names if n_bins[c] < min_bins]
    if small:
        print(f"  density_intra / trans.pdf: {len(small)} sequences with < {min_bins} density bins "
              f"left out ({', '.join(_n(c) for c in small[:5])}{', ...' if len(small) > 5 else ''})")
        names = [c for c in names if n_bins[c] >= min_bins]
    if (stats_dir / DENSITY_BIN).is_file():
        bin_size = int(np.load(stats_dir / DENSITY_BIN))

    mask = load_dict(stats_dir, DENSITY_MASK)
    if mask is not None:                        # bad bins and weights at the density bin
        bad = {c: mask[c][0] for c in names}
        weights = {c: mask[c][1] for c in names}
    else:                                       # older runs: density at the coverage bin
        kept, frac = load_dict(stats_dir, COVERAGE_KEPT), load_dict(stats_dir, MAPPABILITY)
        bad = {c: (~kept[c][:n_bins[c]] if kept is not None and c in kept
                   else np.zeros(n_bins[c], bool)) for c in names}
        weights = {}
        for c in names:
            n = n_bins[c]
            if frac is not None and c in frac:
                f = np.zeros(n)
                f[:min(n, len(frac[c]))] = frac[c][:n]
                weights[c] = f * np.clip(sizes[c] - np.arange(n) * bin_size, 0, bin_size) / bin_size
            else:
                weights[c] = None

    def raw_obs_exp(a, b, k):
        blk = matrix[offsets[a]:offsets[a] + n_bins[a], offsets[b]:offsets[b] + n_bins[b]]
        blk = blk.toarray().astype(np.float64)
        blk[bad[a], :] = np.nan
        blk[:, bad[b]] = np.nan
        return dn.obs_exp_binned(blk, a == b, weights[a], weights[b], k)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")
    kb = bin_size / 1000

    def page(pdf, a, b):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize, to_rgb
        k = raw_bin_intra if a == b else raw_bin_trans
        # raw map aggregated to <= ~600 pixels per side: at 50 kb a whole human chromosome has
        # 0-1 contact per far pixel and showed nothing but the zero colour
        k = max(k, int(np.ceil(max(sizes[a], sizes[b]) / bin_size / 600)))
        extent = (0, sizes[b] / 1e3, sizes[a] / 1e3, 0)          # axes in kb
        raw = raw_obs_exp(a, b, k)
        with np.errstate(divide="ignore", invalid="ignore"):
            zr = np.log2(raw[np.isfinite(raw) & (raw > 0)])
        lim_raw = float(np.clip(np.percentile(np.abs(zr), 98), 1.0, vlim_raw)) if zr.size else vlim_raw
        panels = [(f"raw obs / exp {kb * k:g} kb (not smoothed)", raw, lim_raw),
                  (f"density (unbiased, adaptive Gaussian), {kb:g} kb", density[(a, b)], vlim)]
        # very different sizes (mosquito chromosome x Wolbachia: 160x): real proportions leave a
        # line of pixels; the short axis is stretched instead (said in the title)
        stretch = max(sizes[a], sizes[b]) > 8 * min(sizes[a], sizes[b])
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"wspace": 0.45})
        for ax, (lab, mat, lim) in zip(axes, panels):
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.log2(mat)
            norm = Normalize(-lim, lim)
            rgb = cmap(norm(np.clip(np.where(np.isfinite(z), z, 0.0), -lim, lim)))[..., :3]
            rgb[mat == 0] = to_rgb("white")                 # zero contact
            rgb[~np.isfinite(mat)] = to_rgb("lightgrey")     # excluded bins
            ax.imshow(rgb, extent=extent, aspect="auto" if stretch else "equal", interpolation="none")
            ax.set_title(lab, fontsize=9)
            ax.set(xlabel=f"{_n(b)} (kb)", ylabel=f"{_n(a)} (kb)")
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                         fraction=0.046 if stretch else 0.046 * min(1.0, sizes[a] / sizes[b]),
                         pad=0.03, label="log2")
        title = _n(a) if a == b else f"{_n(a)} × {_n(b)}"
        fig.suptitle(f"{title} — grey = excluded bins, white = 0 contact"
                     + (" — not to scale (short axis stretched)" if stretch else ""), fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(output_dir / "density_intra.pdf") as pdf:
        for c in names:
            page(pdf, c, c)
    pairs = list(itertools.combinations(names, 2))
    if len(pairs) > max_trans_pairs:
        print(f"  density_trans.pdf: first {max_trans_pairs} of {len(pairs)} chromosome pairs")
        pairs = pairs[:max_trans_pairs]
    with PdfPages(output_dir / "density_trans.pdf") as pdf:
        for a, b in pairs:
            page(pdf, a, b)


def plot_contact_maps(pairs_files, chrom_sizes, output_pdf, bin_size=2000, genome_bin=20000,
                      regions=None, chromosomes=None, cmap="afmhot_r", mappability=None):
    """
    Contact maps of several pairs files side by side (e.g. unrescued /
    rescued full / rescued standard), log10(count + 1), one colour scale per
    page shared by all the panels (0 → 99.5th percentile of the first map),
    with a 1D track under each map, sharing its x axis:
      - first panel: fraction of mappable positions per bin (mappability =
        statistics_v2 mappability.npy dict at bin_size), if given
      - other panels: coverage (reads per bin) before (= first panel, grey)
        and after (this panel, colour)
    Pages: whole genome at genome_bin; each chromosome (chromosomes: subset)
    at bin_size; regions = [(chrom, start_kb, end_kb), ...] at bin_size.
    pairs_files: [(label, path), ...]; chrom_sizes: {chrom: length}.
    """
    from scipy import sparse
    import density_v2 as dn

    names = list(chrom_sizes)
    n_bins, offsets = dn.genome_bins(chrom_sizes, bin_size)
    mats = [(lab, dn.contact_matrix(path, chrom_sizes, bin_size)) for lab, path in pairs_files]
    # coverage = reads per bin: row sum + diagonal (a pair inside one bin has 2 reads there)
    covs = [np.asarray(m.sum(axis=1)).ravel() + m.diagonal() for _, m in mats]
    mapp = None
    if mappability is not None:
        mapp = np.concatenate([np.pad(np.asarray(mappability.get(c, np.zeros(n)))[:n],
                                      (0, max(0, n - len(mappability.get(c, [])))))
                               for c, n in zip(names, n_bins)])

    # aggregation bin_size → genome_bin, chromosome by chromosome
    k = max(1, genome_bin // bin_size)
    n_coarse = [-(-n // k) for n in n_bins]
    starts = np.concatenate([[0], np.cumsum(n_coarse)[:-1]])
    coarse = np.concatenate([starts[i] + np.arange(n) // k for i, n in enumerate(n_bins)])
    agg = sparse.csr_matrix((np.ones(len(coarse)), (np.arange(len(coarse)), coarse)))

    def page(pdf, title, sel, coarse_view=False, x_kb=None, ticks=None):
        """sel: genome bin indices (fine) of the region shown."""
        if coarse_view:
            blocks = [(agg.T @ m @ agg).toarray() for _, m in mats]
            tracks_cov = [agg.T @ c for c in covs]
            track_map = (agg.T @ mapp) / np.maximum(agg.T @ np.ones(len(mapp)), 1) if mapp is not None else None
            x = np.arange(blocks[0].shape[0]) + 0.5
            extent = (0, blocks[0].shape[1], blocks[0].shape[0], 0)
        else:
            blocks = [m[sel][:, sel].toarray() for _, m in mats]
            tracks_cov = [c[sel] for c in covs]
            track_map = mapp[sel] if mapp is not None else None
            x = x_kb
            step = x_kb[1] - x_kb[0] if len(x_kb) > 1 else 1
            extent = (x_kb[0] - step / 2, x_kb[-1] + step / 2, x_kb[-1] + step / 2, x_kb[0] - step / 2)
        logs = [np.log10(b + 1.0) for b in blocks]
        vmax = np.percentile(logs[0], 99.5) or 1.0
        n = len(blocks)
        # 1.5 × the 99th percentile: extreme peaks (rDNA) are clipped so the rest stays readable
        cov_max = 1.5 * max(np.percentile(t, 99) for t in tracks_cov) or 1.0

        # fixed layout (inches): square maps, tracks of exactly the same width
        # below them, colour bar in its own column (it must not shrink the maps)
        side, track_h, gap, vgap = 4.4, 1.1, 0.9, 0.12
        left, right, bottom, top = 0.8, 1.0, 0.7, 0.6
        fig_w = left + n * side + (n - 1) * gap + right
        fig_h = bottom + track_h + vgap + side + top
        fig = plt.figure(figsize=(fig_w, fig_h))
        frac = lambda x0, y0, w, h: [x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h]
        map_y = bottom + track_h + vgap
        for i, ((lab, _), z) in enumerate(zip(mats, logs)):
            x0 = left + i * (side + gap)
            ax = fig.add_axes(frac(x0, map_y, side, side))
            im = opaque_imshow(ax, z, cmap, 0, vmax, extent=extent, aspect="auto")
            ax.set_title(lab, fontsize=10)
            ax.tick_params(labelbottom=False, labelsize=7)
            tr = fig.add_axes(frac(x0, bottom, side, track_h), sharex=ax)
            if i == 0 and track_map is not None:
                tr.fill_between(x, track_map, step="mid", color="grey", alpha=0.6, lw=0)
                tr.set(ylim=(0, 1.05))
                tr.set_ylabel("mappability", fontsize=8)
            else:
                tr.plot(x, tracks_cov[0], color="grey", lw=0.7, label="before")
                if i > 0:
                    tr.plot(x, tracks_cov[i], color="darkred", lw=0.7, label="after")
                    tr.legend(fontsize=6, loc="upper right")
                tr.set(ylim=(0, cov_max))
                tr.set_ylabel("coverage", fontsize=8)
            tr.tick_params(labelsize=7)
            tr.set_xlim(extent[0], extent[1])
            if ticks:
                tr.set_xticks(ticks[0], ticks[1], rotation=90, fontsize=6)
                ax.set_yticks(ticks[0], ticks[1], fontsize=6)
                for o in ticks[2]:
                    ax.axhline(o, color="grey", lw=0.3)
                    ax.axvline(o, color="grey", lw=0.3)
                    tr.axvline(o, color="grey", lw=0.3)
            else:
                tr.set_xlabel("kb", fontsize=8)
                ax.set_ylabel("kb", fontsize=8)
        cax = fig.add_axes(frac(left + n * side + (n - 1) * gap + 0.2, map_y, 0.15, side))
        fig.colorbar(im, cax=cax, label="log10(contacts + 1)")
        fig.suptitle(title, fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(output_pdf) as pdf:
        ends = np.cumsum(n_coarse)
        mids = (np.concatenate([[0], ends[:-1]]) + ends) / 2
        page(pdf, f"Whole genome — {genome_bin // 1000} kb bins", None, coarse_view=True,
             ticks=(mids, names, ends[:-1]))
        for i, c in enumerate(names):
            if chromosomes is not None and c not in chromosomes:
                continue
            sel = np.arange(offsets[i], offsets[i] + n_bins[i])
            page(pdf, f"{_n(c)} — {bin_size // 1000} kb bins", sel,
                 x_kb=(np.arange(n_bins[i]) + 0.5) * bin_size / 1000)
        for c, lo, hi in (regions or []):
            i = names.index(c)
            b0, b1 = int(lo * 1000 // bin_size), min(int(-(-hi * 1000 // bin_size)), n_bins[i])
            page(pdf, f"{_n(c)} {lo}-{hi} kb — {bin_size // 1000} kb bins", offsets[i] + np.arange(b0, b1),
                 x_kb=(np.arange(b0, b1) + 0.5) * bin_size / 1000)
    return mats


def plot_all(stats_dir, output_dir=None, compare_dir=None, bin_size=2000,
             labels=("v2", "reference"), plot_chroms=None):
    set_display_names(stats_dir)
    output_dir = Path(output_dir or stats_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    done = []

    plot_ps(stats_dir, output_dir / "ps.pdf", compare_dir, labels)
    done.append("ps.pdf")
    if compare_dir:
        summary = plot_ps_ratio(stats_dir, compare_dir, output_dir / "ps_ratio.pdf", labels)
        done.append("ps_ratio.pdf")
    if plot_orientations(stats_dir, output_dir / "orientations.pdf"):
        done.append("orientations.pdf")
    plot_coverage(stats_dir, output_dir / "coverage.pdf", bin_size, compare_dir, labels)
    done.append("coverage.pdf")
    try:
        if plot_effective_mappability(stats_dir, output_dir / "effective_mappability.pdf"):
            done.append("effective_mappability.pdf")
    except Exception as err:      # a plot must not stop the pipeline
        print(f"  [warning] effective_mappability.pdf not made: {err}")
    plot_trans_ps(stats_dir, output_dir / "trans_ps.pdf", compare_dir, labels)
    done.append("trans_ps.pdf")
    if (Path(stats_dir) / DENSITY).is_file():
        maps = [(labels[0], Path(stats_dir) / DENSITY)]
        if compare_dir and (Path(compare_dir) / DENSITY).is_file():
            maps.append((labels[1], Path(compare_dir) / DENSITY))
        plot_density(maps, output_dir / "density.pdf", chroms=plot_chroms)
        done.append("density.pdf")
        if (Path(stats_dir) / CONTACTS).is_file():
            plot_density_matrices(stats_dir, output_dir, bin_size, chroms=plot_chroms)
            done += ["density_intra.pdf", "density_trans.pdf"]

    print(f"Plots saved in {output_dir}: {', '.join(done)}")
    if compare_dir:
        print(f"\nMedian P(s) ratio {labels[0]} / {labels[1]}:")
        print(f"  {'chromosome':<16}{'uncut':>8}{'circle':>8}{'weird':>8}")
        for chrom, *r in summary:
            print(f"  {chrom:<16}" + "".join(f"{v:>8.3f}" for v in r))


def main():
    parser = argparse.ArgumentParser(description="PDF plots of P(s), coverage and trans P(s).")
    parser.add_argument("stats_dir", help="Folder with the outputs of statistics_v2.py")
    parser.add_argument("--output_dir", default=None, help="Where to save the PDFs (default: stats_dir)")
    parser.add_argument("--compare_dir", default=None,
                        help="Reference statistics folder to overlay (e.g. previous hicberg version)")
    parser.add_argument("--bin_size", type=int, default=2000, help="Coverage bin size (for titles)")
    parser.add_argument("--labels", nargs=2, default=["v2", "v1"], metavar=("MAIN", "REF"),
                        help="Legend labels (default: v2 v1)")
    args = parser.parse_args()

    plot_all(args.stats_dir, args.output_dir, args.compare_dir, args.bin_size, tuple(args.labels))


if __name__ == "__main__":
    main()
