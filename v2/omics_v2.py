#!/usr/bin/env python3
"""
omics_v2.py

Paired-end "omics" data other than Hi-C (MNase-seq, ChIP-seq, ATAC-seq, MFA /
whole-genome sequencing...): signal tracks before and after the
re-attribution of the ambiguous pairs.

The pipeline is the Hi-C one (classification, P(s) and coverage laws,
re-attribution, mode ps by default); for such data the intra "P(s)" of the
proper pairs is the fragment-length distribution, so a read with several copies
is placed at the copy at fragment distance from its mate (mate rescue). The
fragments inside identical copies cannot be resolved: ps splits them evenly
(best for uniform signals: MNase-seq benchmark); mode standard weights them by the
coverage of their neighbourhood (one EM step): best for peaked ChIP-seq (cohesin
benchmark: peaks back on the right copy, contrast exaggerated). DEVELOPMENT.md.

Instead of the Hi-C matrices, this stage writes, from unrescued.pairs and
rescued.pairs (5' positions and strands of both reads):
  - signal_unrescued.bw, signal_rescued.bw: fragment coverage (each fragment
    from the 5' end of its forward read to the 5' end of its reverse read,
    as bamCoverage --extendReads in the first version, which built paired BAM
    files for that), fragments between min_length and max_length only
  - rescued_fraction.bw: share of the rescued signal brought by re-attributed
    pairs (0 = unique pairs only, 1 = only re-attributed pairs): where the
    reconstructed signal is an estimate rather than an observation
  - omics_stats.txt: pairs kept / discarded (orientation, length, trans)
  - omics_plots.pdf: fragment-length distributions, signal before / after
    along each plotted chromosome with the rescued fraction

CLI:
  python3 omics_v2.py unrescued.pairs rescued.pairs chrom.sizes output_dir \
      --min_length 100 --max_length 1000
"""

import argparse
from pathlib import Path

import numpy as np


def read_chrom_sizes(path):
    return {l.split()[0]: int(l.split()[1]) for l in Path(path).read_text().splitlines() if l.strip()}


def auto_resolution(chrom_sizes):
    """1 bp up to 200 Mb of genome, else 10 bp (memory: one int32 per bin and chromosome)."""
    return 1 if sum(chrom_sizes.values()) <= 200_000_000 else 10


def fragments(pairs_path, min_length=100, max_length=1000, chunk=2_000_000):
    """
    Yield (chrom, starts, ends) arrays of the proper fragments of a pairs file
    (0-based half-open [start, end)). A pair is a fragment if both reads are on
    the same chromosome, facing each other (forward read 5' upstream of the
    reverse read 5'), with a length (5' to 5', inclusive) in [min_length, max_length].
    Also returns, at the end, the counters (yielded as (None, counters, None)).
    """
    n = dict(pairs=0, fragments=0, trans=0, same_strand=0, outward=0, short=0, long=0)
    buf = {}

    def flush():
        for c, (s, e) in buf.items():
            if s:
                yield c, np.array(s, dtype=np.int64), np.array(e, dtype=np.int64)
        buf.clear()

    k = 0
    with open(pairs_path) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 7)
            n["pairs"] += 1
            if f[1] != f[3]:
                n["trans"] += 1
                continue
            s1, s2 = f[5], f[6].rstrip("\n")
            if s1 == s2:
                n["same_strand"] += 1
                continue
            p1, p2 = int(f[2]), int(f[4])
            fwd, rev = (p1, p2) if s1 == "+" else (p2, p1)
            if fwd > rev:
                n["outward"] += 1
                continue
            length = rev - fwd + 1
            if length < min_length:
                n["short"] += 1
                continue
            if length > max_length:
                n["long"] += 1
                continue
            n["fragments"] += 1
            s, e = buf.setdefault(f[1], ([], []))
            s.append(fwd - 1)
            e.append(rev)
            k += 1
            if k >= chunk:
                yield from flush()
                k = 0
    yield from flush()
    yield None, n, None


def fragment_coverage(pairs_path, chrom_sizes, resolution=1, min_length=100, max_length=1000):
    """
    Fragment coverage per chromosome at `resolution` bp (mean number of fragments
    covering each bp of the bin), fragment-length histogram and counters.
    """
    diff = {c: np.zeros(L // resolution + 2, dtype=np.float64) for c, L in chrom_sizes.items()}
    lengths = np.zeros(max_length + 1, dtype=np.int64)
    counters = None
    for c, s, e in fragments(pairs_path, min_length, max_length):
        if c is None:
            counters = s
            continue
        if c not in diff:
            continue
        np.add.at(lengths, np.minimum(e - s, max_length), 1)
        if resolution == 1:
            np.add.at(diff[c], s, 1.0)
            np.add.at(diff[c], e, -1.0)
        else:
            # exact bp coverage averaged per bin: split the partial first / last bins
            bs, be = s // resolution, e // resolution
            np.add.at(diff[c], bs, 1.0 - (s - bs * resolution) / resolution)
            np.add.at(diff[c], bs + 1, (s - bs * resolution) / resolution)
            np.add.at(diff[c], be, -(1.0 - (e - be * resolution) / resolution))
            np.add.at(diff[c], be + 1, -(e - be * resolution) / resolution)
    cov = {c: np.cumsum(d)[: -(-chrom_sizes[c] // resolution)] for c, d in diff.items()}
    return cov, lengths, counters


def write_bigwig(path, coverage, chrom_sizes, resolution):
    """bigWig of a coverage dict (runs of equal values, zeros skipped)."""
    import pyBigWig
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(c, int(L)) for c, L in chrom_sizes.items()])
    for c, L in chrom_sizes.items():
        v = coverage.get(c)
        if v is None or not len(v):
            continue
        v = np.round(v.astype(np.float64), 4)
        change = np.flatnonzero(np.diff(v) != 0) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(v)]])
        vals = v[starts]
        keep = vals != 0
        starts, ends, vals = starts[keep] * resolution, np.minimum(ends[keep] * resolution, L), vals[keep]
        if len(starts):
            bw.addEntries([c] * len(starts), starts.tolist(), ends=ends.tolist(),
                          values=vals.astype(float).tolist())
    bw.close()


def rebin(v, k):
    n = len(v) // k * k
    out = v[:n].reshape(-1, k).mean(axis=1)
    if n < len(v):
        out = np.append(out, v[n:].mean())
    return out


def omics_tracks(unrescued_pairs, rescued_pairs, chrom_sizes_path, output_dir, min_length=100,
                 max_length=1000, resolution=None, plot_chroms=None, mappability=None,
                 map_bin_size=None):
    """Signal tracks, statistics and plots of the omics mode (see the module docstring)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = read_chrom_sizes(chrom_sizes_path)
    res = resolution or auto_resolution(sizes)
    cov_u, len_u, n_u = fragment_coverage(unrescued_pairs, sizes, res, min_length, max_length)
    cov_r, len_r, n_r = fragment_coverage(rescued_pairs, sizes, res, min_length, max_length)
    write_bigwig(output_dir / "signal_unrescued.bw", cov_u, sizes, res)
    write_bigwig(output_dir / "signal_rescued.bw", cov_r, sizes, res)
    frac = {c: np.divide(cov_r[c] - cov_u[c], cov_r[c], out=np.zeros_like(cov_r[c]),
                         where=cov_r[c] > 0) for c in sizes}
    write_bigwig(output_dir / "rescued_fraction.bw", frac, sizes, res)

    lines = [f"resolution of the tracks : {res} bp",
             f"fragment length          : {min_length}-{max_length} bp (5' to 5')"]
    for lab, n, ln in (("unrescued", n_u, len_u), ("rescued", n_r, len_r)):
        p = max(n["pairs"], 1)
        med = int(np.searchsorted(np.cumsum(ln), ln.sum() / 2)) if ln.sum() else 0
        lines.append(f"{lab:9s}: {n['pairs']:,} pairs, {n['fragments']:,} fragments "
                     f"({100 * n['fragments'] / p:.1f}%, median length {med} bp); discarded: "
                     f"trans {n['trans']:,}, same strand {n['same_strand']:,}, outward {n['outward']:,}, "
                     f"< {min_length} bp {n['short']:,}, > {max_length} bp {n['long']:,}")
    tot_u = sum(v.sum() for v in cov_u.values())
    tot_r = sum(v.sum() for v in cov_r.values())
    lines.append(f"signal rescued / unrescued : {tot_r / max(tot_u, 1e-12):.3f}")
    (output_dir / "omics_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n=== Omics tracks ===\n" + "\n".join(lines))

    chroms = plot_chroms or list(sizes)
    mapp = None
    if mappability is not None and Path(mappability).is_file():
        mapp = np.load(mappability, allow_pickle=True).item()
    with PdfPages(output_dir / "omics_plots.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(max_length + 1)
        for lab, ln, col in (("unrescued", len_u, "black"), ("rescued", len_r, "green")):
            if ln.sum():
                ax.plot(x, ln / ln.sum(), color=col, lw=1, label=f"{lab} ({ln.sum():,} fragments)")
        ax.set(xlabel="fragment length (bp)", ylabel="fraction of fragments",
               title="Fragment-length distribution")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        for c in chroms:
            if c not in sizes:
                continue
            k = max(1, -(-sizes[c] // res) // 3000)              # <= ~3000 points per chromosome
            u, r = rebin(cov_u[c], k), rebin(cov_r[c], k)
            x = (np.arange(len(u)) + 0.5) * k * res / 1e3
            n_tr = 3 if mapp is not None and c in mapp else 2
            fig, ax = plt.subplots(n_tr, 1, figsize=(13, 2.2 * n_tr + 0.6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1, 1][:n_tr]})
            ax[0].plot(x, r, color="green", lw=.7, label="rescued")
            ax[0].plot(x, u, color="black", lw=.7, label="unrescued")
            ax[0].set_ylabel("fragment coverage")
            ax[0].legend(frameon=False, fontsize=8, loc="upper right")
            ax[0].set_title(f"{c} — {k * res / 1e3:g} kb bins", fontsize=10)
            with np.errstate(divide="ignore", invalid="ignore"):
                fr = np.where(r > 0, (r - u) / r, np.nan)
            ax[1].fill_between(x, 0, fr, color="green", alpha=.5, step="mid")
            ax[1].set_ylim(0, 1); ax[1].set_ylabel("rescued\nfraction")
            if n_tr == 3:
                m = np.asarray(mapp[c], dtype=float)
                xm = (np.arange(len(m)) + 0.5) * sizes[c] / len(m) / 1e3
                ax[2].fill_between(xm, 0, m, color="grey", step="mid")
                ax[2].set_ylim(0, 1.05); ax[2].set_ylabel("Mappability")
            ax[-1].set_xlabel(f"{c} (kb)")
            for a in ax:
                a.spines[["top", "right"]].set_visible(False)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return output_dir / "signal_rescued.bw"


def ratio_tracks(ip_dir, input_dir, output_dir=None, bin_size=50, pseudocount=None):
    """
    log2(IP / input) tracks before and after re-attribution, from two omics output
    folders (the IP and its input, both processed by the pipeline with the same
    settings: the input must be re-attributed like the IP, or the ratio at the
    repeats reflects the re-attribution, not the enrichment). Each signal is summed
    in bin_size bins and scaled to the same total (genome-wide); pseudocount
    (default: 1 % of the mean input bin) avoids dividing by empty input bins.
    Writes log2ratio_unrescued.bw and log2ratio_rescued.bw.
    """
    import pyBigWig
    ip_dir, input_dir = Path(ip_dir), Path(input_dir)
    output_dir = Path(output_dir or ip_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("unrescued", "rescued"):
        a = pyBigWig.open(str(ip_dir / f"signal_{kind}.bw"))
        b = pyBigWig.open(str(input_dir / f"signal_{kind}.bw"))
        sizes = a.chroms()
        sa, sb = {}, {}
        for c, L in sizes.items():
            n = -(-L // bin_size)
            for d, bw in ((sa, a), (sb, b)):
                v = np.nan_to_num(np.array(bw.values(c, 0, L), dtype=float))
                d[c] = np.add.reduceat(np.pad(v, (0, n * bin_size - L)), np.arange(0, n * bin_size, bin_size))
        ta, tb = sum(v.sum() for v in sa.values()), sum(v.sum() for v in sb.values())
        pc = pseudocount if pseudocount is not None else 0.01 * tb / sum(len(v) for v in sb.values())
        ratio = {c: np.log2((sa[c] * tb / ta + pc) / (sb[c] + pc)) for c in sizes}
        write_bigwig(output_dir / f"log2ratio_{kind}.bw", ratio, sizes, bin_size)
    return output_dir / "log2ratio_rescued.bw"


def main():
    p = argparse.ArgumentParser(description="Signal tracks of paired-end omics data before / after re-attribution.")
    p.add_argument("unrescued_pairs")
    p.add_argument("rescued_pairs")
    p.add_argument("chrom_sizes")
    p.add_argument("output_dir")
    p.add_argument("--min_length", type=int, default=100)
    p.add_argument("--max_length", type=int, default=1000)
    p.add_argument("--resolution", type=int, default=None, help="bp per value (default: 1, or 10 > 200 Mb)")
    args = p.parse_args()
    omics_tracks(args.unrescued_pairs, args.rescued_pairs, args.chrom_sizes, args.output_dir,
                 args.min_length, args.max_length, args.resolution)


if __name__ == "__main__":
    main()
