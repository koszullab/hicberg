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


XS, UNCUTS, CIRCLES, WEIRDS = "xs.npy", "uncuts.npy", "circles.npy", "weirds.npy"
TRANS_PS, COVERAGE_RAW, COVERAGE = "trans_ps.npy", "coverage1.npy", "coverage.npy"
PS_COUNTS, MAPPABILITY = "ps_counts.npz", "mappability.npy"

# Same colours as hicberg/plot.py
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
            title = f"P(s) — {chrom}" + (f"  (absent in {', '.join(missing)})" if missing else "")
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
                   title=f"P(s) ratio {labels[0]} / {labels[1]} — {chrom}",
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
            ax.set(xscale="log", ylim=(0, 1), title=f"Pair orientations — {name}",
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
    raw, smooth = load_dict(stats_dir, COVERAGE_RAW), load_dict(stats_dir, COVERAGE)
    smooth_ref = load_dict(compare_dir, COVERAGE) if compare_dir else None
    mappability = load_dict(stats_dir, MAPPABILITY)

    with PdfPages(output_pdf) as pdf:
        for chrom in _chromosomes(smooth, smooth_ref):
            fig, ax = plt.subplots(figsize=(11, 5))
            if chrom in raw:
                ax.plot(raw[chrom], "o", ms=1, color="cornflowerblue",
                        label=f"raw {labels[0]}", rasterized=True)
            if chrom in smooth:
                ax.plot(smooth[chrom], color="darkblue", lw=1,
                        label=f"smoothed {labels[0]}")
            if smooth_ref is not None and chrom in smooth_ref:
                ax.plot(smooth_ref[chrom], color="orange", lw=1, alpha=0.8,
                        label=f"smoothed {labels[1]}")
            ax.set(title=f"Coverage — {chrom} — bins of {bin_size} bp",
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


# ---------------------------------------------------------------------------
# Trans P(s)
# ---------------------------------------------------------------------------

def _trans_matrix(trans_ps, chroms):
    m = np.array([[trans_ps.get((a, b), trans_ps.get((b, a), np.nan)) for b in chroms]
                  for a in chroms], dtype=float)
    np.fill_diagonal(m, np.nan)   # intra: not a trans law
    return m


def plot_trans_ps(stats_dir, output_pdf, compare_dir=None, labels=("v2", "reference")):
    """Heatmap of log10 trans P(s); with compare_dir, side by side + scatter."""
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
            im = ax.imshow(lm, cmap="Wistia", interpolation="none", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(chroms)), chroms, rotation=90, fontsize=7)
            ax.set_yticks(range(len(chroms)), chroms, fontsize=7)
            ax.set_title(f"log10 trans P(s) — {label}")
            fig.colorbar(im, ax=ax, fraction=0.046)
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

def plot_all(stats_dir, output_dir=None, compare_dir=None, bin_size=2000,
             labels=("v2", "reference")):
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
    plot_trans_ps(stats_dir, output_dir / "trans_ps.pdf", compare_dir, labels)
    done.append("trans_ps.pdf")

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
