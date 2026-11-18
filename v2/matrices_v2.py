#!/usr/bin/env python3
"""
matrices_v2.py

Last step of the pipeline: contact matrices before / after re-attribution.

  1. hicstuff_filter(): optional hicstuff processing of a pairs file
     (removal of non-informative events: uncuts, loops, weirds; PCR duplicates)
  2. build_cool(): fixed-bin cooler matrix (cooler cload pairs) + ICE balancing
     (cooler balance); permissive mad-max for the rescued matrix, as in v1, so
     that the reconstructed repeats are not filtered out again
  3. plot_before_after(): v1-style plots (hicberg/plot.py plot_matrix), one
     page per pair of chromosomes, raw and normalised (balanced) PDFs:
        left : unrescued map; GC content, mappability
        right: rescued map; coverage before / after, Hi-C restriction sites per
               bin (GC content for Micro-C)
"""

import gzip
import itertools
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


# ---------------------------------------------------------------------------
# hicstuff filtering
# ---------------------------------------------------------------------------

MNASE_FRAGMENT = 100     # Micro-C: virtual fragment size (bp) for hicstuff event filtering


def hicstuff_filter(pairs, genome, enzyme, out_dir, threads=4, prefix=None, thresholds=None):
    """
    hicstuff processing of a pairs file (4DN header with chr1 / chr2 and
    #chromsize lines), with the hicstuff library functions called one by one
    (as in hicstuff pipeline --start-stage=pairs --filter --duplicates, but
    without the hicstuff matrix, which is not used and took ~15 min on 47 M
    pairs):
      1. restriction fragment of each read (hicstuff.digest.attribute_fragments)
      2. sort by coordinates (hicstuff.io.sort_pairs)
      3. removal of non-informative events: uncuts, loops, weirds
         (hicstuff.filter.filter_events)
      4. removal of PCR duplicates (hicstuff.pipeline.filter_pcr_dup)
    enzyme: e.g. "DpnII,HinfI"; for Micro-C (MNase, no restriction site) a
    number = virtual fragments of that size (hicstuff fixed-size digestion),
    "mnase" = 100 bp fragments: uncuts, loops and weirds are then detected
    between neighbouring 100 bp segments, as for a frequent cutter.
      thresholds None : uncut / loop thresholds estimated from the library
                        (hicstuff.filter.get_thresholds)
      thresholds (u, l): the same thresholds for another file (rescued pairs
                        must be filtered like the unrescued ones, else the
                        before / after comparison is biased: on yeast hicstuff
                        estimated 19-25 on unrescued but 8-10 on rescued)
    Returns (filtered PCR-free pairs path, (uncut, loop) thresholds used).
    The hicstuff log goes to out_dir/hicstuff_<prefix>/hicstuff.log.
    """
    import logging
    pairs, out_dir = Path(pairs), Path(out_dir)
    prefix = prefix or pairs.name.split(".")[0]
    work = out_dir / f"hicstuff_{prefix}"
    tmp = work / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    log_path = work / "hicstuff.log"
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(levelname)s :: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        import hicstuff.digest as hcd
        import hicstuff.filter as hcf
        import hicstuff.io as hio
        from hicstuff.pipeline import filter_pcr_dup
        idx = work / f"{prefix}.valid_idx.pairs"
        enz = str(enzyme).strip()
        if enz.lower() == "mnase":
            enz = MNASE_FRAGMENT
        elif enz.isdigit():
            enz = int(enz)
        else:
            enz = enz.split(",") if "," in enz else enz
        logging.info("digestion: %s", f"virtual {enz} bp fragments" if isinstance(enz, int) else enz)
        from Bio import SeqIO
        table = {rec.id: hcd.get_restriction_table(rec.seq, enz)
                 for rec in SeqIO.parse(hio.read_compressed(str(genome)), "fasta")}
        hcd.attribute_fragments(str(pairs), str(idx), table)
        hio.sort_pairs(str(idx), str(idx) + ".sorted", keys=["chr1", "pos1", "chr2", "pos2"],
                       threads=threads, tmp_dir=str(tmp))
        Path(str(idx) + ".sorted").replace(idx)
        if thresholds is None:
            thresholds = hcf.get_thresholds(str(idx))
            logging.info("thresholds: uncuts=%d loops=%d", *thresholds)
        else:
            logging.info("thresholds given: uncuts=%d loops=%d", *thresholds)
        use = work / f"{prefix}.valid_idx_filtered.pairs"
        hcf.filter_events(str(idx), str(use), int(thresholds[0]), int(thresholds[1]))
        idx.unlink()
        out = work / f"{prefix}.valid_idx_filtered_pcrfree.pairs"
        filter_pcr_dup(str(use), str(out))
        use.unlink()
    finally:
        root.removeHandler(handler)
        handler.close()
        shutil.rmtree(tmp, ignore_errors=True)
    with open(out) as fh:
        if not any(not line.startswith("#") for line in fh):
            raise RuntimeError(f"hicstuff produced no pairs, see {log_path}")
    return out, tuple(int(t) for t in thresholds)


# ---------------------------------------------------------------------------
# Cooler matrices
# ---------------------------------------------------------------------------

def write_bins(chrom_sizes, bin_size, path):
    """Fixed-size bins as a BED file (chrom, start, end), as cooler expects."""
    with open(path, "w") as fh:
        for c, L in chrom_sizes.items():
            for start in range(0, L, bin_size):
                fh.write(f"{c}\t{start}\t{min(start + bin_size, L)}\n")
    return path


def build_cool(pairs, chrom_sizes, bin_size, out_cool, balance=True, mad_max=5, nproc=4):
    """
    Cooler matrix from a pairs file (plain or .gz; columns readID chr1 pos1
    chr2 pos2 ...), balanced with ICE (cooler balance --mad-max) if balance.
    """
    out_cool = Path(out_cool)
    bins = write_bins(chrom_sizes, bin_size, out_cool.with_suffix(".bins.bed"))
    cmd = ["cooler", "cload", "pairs", "-c1", "2", "-p1", "3", "-c2", "4", "-p2", "5",
           "--assembly", "unknown", str(bins), "-", str(out_cool)]
    opener = gzip.open if str(pairs).endswith(".gz") else open
    with opener(pairs, "rt") as src:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
        for line in src:
            proc.stdin.write(line)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"cooler cload failed for {pairs}")
    Path(bins).unlink()
    if balance:
        subprocess.run(["cooler", "balance", "--force", "--nproc", str(nproc),
                        "--mad-max", str(mad_max), str(out_cool)],
                       check=True, stdout=subprocess.DEVNULL)
    return out_cool


# ---------------------------------------------------------------------------
# GC content
# ---------------------------------------------------------------------------

def iter_fasta(genome, names=None):
    """Yield (name, upper-case sequence) from a FASTA (plain or gzipped)."""
    name, chunks = None, []
    opener = gzip.open if str(genome).endswith(".gz") else open
    with opener(genome, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None and (names is None or name in names):
                    yield name, "".join(chunks).upper()
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip())
    if name is not None and (names is None or name in names):
        yield name, "".join(chunks).upper()


def _bin_sum(mask, bin_size):
    n = -(-len(mask) // bin_size)
    return np.pad(mask, (0, n * bin_size - len(mask))).reshape(n, bin_size).sum(axis=1)


NICE_FACTORS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)


def auto_plot_bin_size(chrom_sizes, chroms, bin_size, max_bins=3000):
    """
    Bin size of the before / after plots: bin_size × the smallest nice factor
    such that the largest plotted chromosome has <= max_bins bins (a dense
    map of a 225 Mb chromosome at 2 kb cannot be drawn). Yeast: 2 kb;
    mosquito (225 Mb): 100 kb.
    """
    l_max = max(chrom_sizes[c] for c in chroms)
    for f in NICE_FACTORS:
        if l_max / (bin_size * f) <= max_bins:
            return bin_size * f
    return bin_size * NICE_FACTORS[-1]


def auto_plot_chroms(chrom_sizes, max_all=30, min_size=1_000_000, max_chroms=20):
    """
    Sequences shown in the before / after plots: all of them if <= max_all,
    else those >= min_size (at most max_chroms, the largest), e.g. the
    chromosomes and large scaffolds / symbionts of a draft assembly.
    """
    if len(chrom_sizes) <= max_all:
        return list(chrom_sizes)
    big = [c for c, L in sorted(chrom_sizes.items(), key=lambda x: -x[1]) if L >= min_size]
    keep = set(big[:max_chroms]) or {max(chrom_sizes, key=chrom_sizes.get)}
    return [c for c in chrom_sizes if c in keep]


def coarsen_cool(cool, factor, out_cool, mad_max=5, nproc=4, balance=True):
    """cooler coarsen (factor × the bin size) + ICE balancing."""
    out_cool = Path(out_cool)
    subprocess.run(["cooler", "coarsen", "-k", str(factor), "-p", str(nproc), "-o", str(out_cool),
                    str(cool)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if balance:
        subprocess.run(["cooler", "balance", "--force", "--nproc", str(nproc),
                        "--mad-max", str(mad_max), str(out_cool)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_cool


def rebin_mean(values, factor):
    """Mean over groups of `factor` consecutive bins (last group may be shorter)."""
    v = np.asarray(values, dtype=np.float64)
    if factor <= 1:
        return v
    n = -(-len(v) // factor)
    p = np.full(n * factor, np.nan)
    p[:len(v)] = v
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmean(p.reshape(n, factor), axis=1)


def gc_content(genome, chrom_sizes, bin_size):
    """{chrom: GC fraction per bin} from a FASTA (N not counted)."""
    out = {}
    for name, seq in iter_fasta(genome, chrom_sizes):
        arr = np.frombuffer(seq.encode(), dtype=np.uint8)
        gc = _bin_sum(np.isin(arr, np.frombuffer(b"GC", dtype=np.uint8)), bin_size)
        acgt = _bin_sum(np.isin(arr, np.frombuffer(b"ACGT", dtype=np.uint8)), bin_size)
        out[name] = np.divide(gc, acgt, out=np.full(len(gc), np.nan), where=acgt > 0)
    return out


IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "N": "[ACGT]", "R": "[AG]", "Y": "[CT]",
         "S": "[CG]", "W": "[AT]", "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
         "H": "[ACT]", "V": "[ACG]"}
COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")


def enzyme_sites(enzymes):
    """
    Recognition sites of the enzymes ('DpnII,HinfI', or sites given directly,
    e.g. 'GATC,GANTC'). Returns [] for a number or 'mnase' (Micro-C: no site).
    """
    sites = []
    for e in str(enzymes).split(","):
        e = e.strip()
        if not e or e.isdigit() or e.lower() == "mnase":
            continue
        if set(e.upper()) <= set(IUPAC):
            sites.append(e.upper())
            continue
        from Bio.Restriction import RestrictionBatch
        sites.append(str(RestrictionBatch([e]).get(e).site).upper())
    return sites


def restriction_sites(genome, chrom_sizes, bin_size, enzymes):
    """
    {chrom: number of restriction sites per bin} for the Hi-C enzyme(s), both
    strands (a site is counted once at its start; overlapping sites counted).
    None if no site (Micro-C / MNase).
    """
    sites = enzyme_sites(enzymes)
    if not sites:
        return None
    motifs = set()
    for site in sites:
        motifs.add(site)
        motifs.add(site.translate(COMPLEMENT)[::-1])          # non-palindromic sites
    pattern = re.compile("(?=(" + "|".join("".join(IUPAC[b] for b in m) for m in sorted(motifs)) + "))")
    out = {}
    for name, seq in iter_fasta(genome, chrom_sizes):
        pos = np.fromiter((m.start() for m in pattern.finditer(seq)), dtype=np.int64)
        out[name] = np.bincount(pos // bin_size, minlength=-(-len(seq) // bin_size)).astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# Before / after plots (v1 style + mappability)
# ---------------------------------------------------------------------------

def plot_before_after(unrescued_cool, rescued_cool, output_dir, mappability=None, gc=None,
                      sites=None, enzyme_label="restriction sites", chrom_pairs=None,
                      intra_only=False, vmin_raw=None, vmax_raw=None,
                      labels=("unrescued", "rescued"), repeat_links=None, min_link_kmers=3,
                      max_arcs=400):
    """
    Two multi-page PDFs (matrices_raw.pdf, matrices_normalised.pdf), one page
    per pair of chromosomes (all pairs, or intra only, or chrom_pairs), as in
    v1 (hicberg/plot.py plot_matrix):
      left : unrescued map; tracks: GC, mappability (shorter); arcs on the left between
             the copies of the repeated sequences (repeat_links, from
             mappability_v2.repeat_links)
      right: rescued map; tracks: coverage before / after (+ median), restriction
             sites per bin (bars; GC if sites is None, e.g. Micro-C)
    Coverage = column sums of the unbalanced block (as v1). mappability, gc:
    {chrom: value per bin} at the bin size of the matrices.
    Maps keep their real proportions; the tracks have exactly the width of
    their map.
    """
    import cooler
    from plots_v2 import opaque_imshow
    unres, res = cooler.Cooler(str(unrescued_cool)), cooler.Cooler(str(rescued_cool))
    names = unres.chromnames
    if chrom_pairs is None:
        chrom_pairs = [(c, c) for c in names] if intra_only else \
            list(itertools.combinations_with_replacement(names, 2))
    kb = unres.binsize / 1000
    cmap = plt.get_cmap("afmhot_r").copy()
    cmap.set_bad("white")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    links = None
    if repeat_links is not None:
        from mappability_v2 import read_repeat_links, repeat_families
        links = read_repeat_links(repeat_links) if not hasattr(repeat_links, "columns") \
            else repeat_links
        links = links[links["n_kmers"] > 0]
        fam_bins, families = repeat_families(links, unres.binsize, min_link_kmers,
                                             dict(unres.chromsizes))
        families.to_csv(output_dir / "repeat_families.tsv", sep="\t", index=False)
        fam_bins.to_csv(output_dir / "repeat_family_bins.tsv", sep="\t", index=False)
        family_of = {(c, int(st) // unres.binsize): int(f)
                     for c, st, f in zip(fam_bins["chrom"], fam_bins["start"], fam_bins["family"])}
        family_label = dict(zip(families["family"].astype(int), families["label"]))
        # the 12 largest families get their own colour (tab10 without its grey,
        # + 3), the others grey
        palette = [c for k, c in enumerate(plt.get_cmap("tab10").colors) if k != 7] + \
            [(0.0, 0.5, 0.5), (0.1, 0.1, 0.45), (0.85, 0.65, 0.0)]

        def family_color(f):
            return palette[f - 1] if 1 <= f <= len(palette) else (0.6, 0.6, 0.6)

    def page_links(a, b):
        """
        Links of the page at the bin size of the maps:
        (pos1_kb, pos2_kb, family, n_kmers).
        """
        sel = links[(links["chrom1"] == a) & (links["chrom2"] == b)]
        if sel.empty:
            return np.zeros((0, 4))
        b1 = sel["start1"].to_numpy() // unres.binsize
        b2 = sel["start2"].to_numpy() // unres.binsize
        n = sel["n_kmers"].to_numpy()
        if a == b:                                       # one arc per pair of bins
            keep = b1 < b2
            b1, b2, n = b1[keep], b2[keep], n[keep]
        key = np.stack([b1, b2], axis=1)
        if not len(key):
            return np.zeros((0, 4))
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        tot = np.bincount(inv.ravel(), weights=n)
        keep = tot >= min_link_kmers
        uniq, tot = uniq[keep], tot[keep]
        if len(tot) > max_arcs:
            top_ = np.argsort(-tot)[:max_arcs]
            uniq, tot = uniq[top_], tot[top_]
        fam = np.array([family_of.get((a, int(x)), 0) for x in uniq[:, 0]], dtype=np.float64)
        out = np.column_stack([(uniq[:, 0] + 0.5) * kb, (uniq[:, 1] + 0.5) * kb, fam, tot])
        return out[np.argsort(-out[:, 2])] if len(out) else out     # largest families on top

    def link_widths(arcs):
        return 0.3 + 0.6 * np.log10(arcs[:, 3]) / max(np.log10(arcs[:, 3].max()), 1) \
            if len(arcs) else np.zeros(0)

    def family_legend(ax, arcs, loc, anchor):
        fams = sorted({int(f) for f in arcs[:, 2]} - {0})
        main = [f for f in fams if f <= len(palette)]
        for f in main:
            ax.plot([], [], color=family_color(f), label=family_label.get(f, f"F{f}"))
        if len(main) < len(fams):
            ax.plot([], [], color=family_color(0), label="other families")
        if fams:
            ax.legend(fontsize=5, loc=loc, bbox_to_anchor=anchor, frameon=False)

    def draw_links(ax, arcs, intra, length_kb):
        """
        Strip left of the map (y = row chromosome), one colour per repeat
        family (repeat_families: the 10 largest, others grey).
        Intra: arcs between the two copies, width ∝ sqrt(distance), line width
        ∝ log(number of shared k-mers). Trans: ticks at the positions of the
        row chromosome that have a copy on the column chromosome.
        """
        from matplotlib.patches import Arc
        ax.set_xlim(1.05, 0)
        lw = link_widths(arcs)
        for k, (y1, y2, f, n) in enumerate(arcs):
            if intra:
                d = abs(y2 - y1)
                w = np.sqrt(d / length_kb)
                ax.add_patch(Arc((0, (y1 + y2) / 2), 2 * w, d, theta1=270, theta2=90,
                                 color=family_color(int(f)), lw=lw[k], alpha=0.7))
            else:
                ax.plot([0, 0.35], [y1, y1], color=family_color(int(f)), lw=lw[k], alpha=0.7)
        ax.set_title("repeats", fontsize=7)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for side_ in ("top", "left", "bottom"):
            ax.spines[side_].set_visible(False)
        family_legend(ax, arcs, "upper left", (-0.15, -0.02))

    def draw_links_trans(ax_top, arcs):
        """
        Trans pages: ticks on the strip above the map at the positions of the
        column chromosome that have a copy on the row chromosome (the left
        strip has those of the row chromosome); same family colours. The map
        itself stays free of annotations.
        """
        ax_top.set_ylim(0, 1.05)
        lw = link_widths(arcs)
        for k, (y1, x2, f, n) in enumerate(arcs):
            ax_top.plot([x2, x2], [0, 0.35], color=family_color(int(f)), lw=lw[k], alpha=0.7)
        ax_top.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for side_ in ("top", "left", "right"):
            ax_top.spines[side_].set_visible(False)

    def track(values, n):
        if values is None:
            return None
        v = np.full(n, np.nan)
        v[:min(n, len(values))] = np.asarray(values, dtype=np.float64)[:n]
        return v

    def page(pdf, a, b, balance):
        m_u = unres.matrix(balance=balance).fetch(a, b).astype(np.float64)
        m_r = res.matrix(balance=balance).fetch(a, b).astype(np.float64)
        raw_u = m_u if not balance else unres.matrix(balance=False).fetch(a, b)
        raw_r = m_r if not balance else res.matrix(balance=False).fetch(a, b)
        cov_u, cov_r = raw_u.sum(axis=0), raw_r.sum(axis=0)
        # number of contacts (pairs) in each block: upper triangle + diagonal in intra
        n_contacts = [int(np.triu(m).sum()) if a == b else int(m.sum()) for m in (raw_u, raw_r)]
        med = np.median(cov_u)
        nb = m_u.shape[1]
        x = (np.arange(nb) + 0.5) * kb
        with np.errstate(divide="ignore", invalid="ignore"):
            z_u, z_r = np.log10(m_u), np.log10(m_r)
        # colour scale shared by the two maps of a page, from the non-zero pixels
        finite = np.concatenate([z_u[np.isfinite(z_u)], z_r[np.isfinite(z_r)]])
        if not finite.size:
            vmin, vmax = 0.0, 1.0
        elif a == b:
            # intra: lower bound = 1st percentile of the non-zero pixels, lowered by
            # 0.5 log × the fraction of empty pixels: in sparse maps (yeast 2 kb:
            # 60-70 % empty) single contacts are coloured (not white like the empty
            # pixels) and the far field / domains stay visible; dense maps (chrMT,
            # 2-20 % empty) keep a tight scale. Upper bound 99.97th percentile
            # (95th / 99th / 99.9th saturated the first diagonals; the max depends on
            # outlier pixels)
            seen = [m[np.isfinite(m)] for m in (m_u, m_r)]
            empty = np.mean(np.concatenate(seen) == 0) if sum(x.size for x in seen) else 0.0
            vmin = vmin_raw if (vmin_raw is not None and not balance) else \
                np.percentile(finite, 1) - 0.5 * empty
            vmax = vmax_raw if (vmax_raw is not None and not balance) else np.percentile(finite, 99.97)
        elif balance:
            vmin, vmax = np.percentile(finite, [5, 99])
        else:                                               # trans: far fewer contacts per pixel
            vmin, vmax = 0.0, max(np.percentile(finite, 99.5), 0.5)

        # fixed layout (inches): real proportions, tracks as wide as their map
        side = 5.0
        h_map = float(np.clip(side * m_u.shape[0] / nb, 0.3 * side, 1.6 * side))
        t_h, gap, vgap, left, right, bottom, top, cb = 0.9, 1.6, 0.35, 0.9, 1.5, 0.7, 0.8, 0.15
        arcs = page_links(a, b) if links is not None else None
        arc_w, arc_h = 0.9, 0.45
        if arcs is not None:
            left += arc_w + 0.1
            if a != b:                                  # strip above the left map
                top += arc_h + 0.1
        n_tracks = 2                    # left: GC, mappability; right: coverage before / after, sites (or GC)
        t_h2 = 0.55                     # height of the 2nd tracks (mappability, sites)
        fig_w = left + 2 * side + gap + right
        fig_h = bottom + n_tracks * (t_h + vgap) + h_map + top
        fig = plt.figure(figsize=(fig_w, fig_h))
        frac = lambda x0, y0, w, h: [x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h]
        map_y = bottom + n_tracks * (t_h + vgap)
        extent = (0, nb * kb, m_u.shape[0] * kb, 0)
        kind = "normalised (ICE)" if balance else "raw"

        for col, (lab, z) in enumerate(((labels[0], z_u), (labels[1], z_r))):
            x0 = left + col * (side + gap)
            ax = fig.add_axes(frac(x0, map_y, side, h_map))
            im = opaque_imshow(ax, z, cmap, vmin, vmax, bad="white", extent=extent, aspect="auto")
            pad = 6 + (72 * (arc_h + 0.1) if arcs is not None and a != b else 0)
            ax.set_title(f"{lab} — {a} × {b} — {kind}, {kb:g} kb bins\n"
                         f"{n_contacts[col]:,} contacts", fontsize=9, pad=pad)
            ax.set_ylabel(f"{a} (kb)", fontsize=8)
            ax.tick_params(labelbottom=False, labelsize=7)
            cax = fig.add_axes(frac(x0 + side + 0.1, map_y, cb, h_map))
            fig.colorbar(im, cax=cax, label="log10 contacts")
            if col == 0 and arcs is not None:
                arc_ax = fig.add_axes(frac(x0 - 0.8 - arc_w, map_y, arc_w, h_map), sharey=ax)
                draw_links(arc_ax, arcs, a == b, m_u.shape[0] * kb)
                if a != b:
                    top_ax = fig.add_axes(frac(x0, map_y + h_map + 0.05, side, arc_h), sharex=ax)
                    draw_links_trans(top_ax, arcs)

            tracks = []
            rs = track(sites.get(b) if sites else None, nb)
            g = track(gc.get(b) if gc else None, nb)
            if col == 0:                        # left: GC content, then mappability (shorter)
                if g is not None:
                    tracks.append(("GC content", [(g, "orange", None)], None))
                mapp = track(mappability.get(b) if mappability else None, nb)
                if mapp is not None:
                    tracks.append(("Mappability", [(mapp, "grey", "fill")], None))
            else:                               # right: coverage, then sites (or GC)
                tracks.append(("Coverage before / after",
                               [(cov_u, "royalblue", "before"), (cov_r, "limegreen", "after")], med))
                if rs is not None:
                    tracks.append((enzyme_label, [(rs, "royalblue", "bar")], None))
                elif g is not None:
                    tracks.append(("GC content", [(g, "orange", None)], None))
            for k, (title, series, median) in enumerate(tracks):
                h = t_h2 if k == 1 else t_h      # 2nd track shorter (mappability / sites)
                y0 = map_y - k * (t_h + vgap) - vgap - h
                tr = fig.add_axes(frac(x0, y0, side, h), sharex=ax)
                for values, color, label in series:
                    if label == "fill":
                        tr.fill_between(x, values, step="mid", color=color, alpha=0.6, lw=0)
                        tr.set_ylim(0, 1.05)
                    elif label == "bar":
                        tr.bar(x, np.nan_to_num(values), width=kb, color=color, lw=0)
                    else:
                        tr.plot(x, values, color=color, lw=0.9, label=label)
                if median is not None:
                    tr.axhline(median, color="black", ls="--", lw=0.8)
                if any(lbl for _, _, lbl in series if lbl not in (None, "fill", "bar")):
                    tr.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
                tr.set_ylabel(title, fontsize=7)
                tr.spines[["top", "right"]].set_visible(False)
                tr.tick_params(labelsize=6, labelbottom=(k == len(tracks) - 1))
                tr.set_xlim(extent[0], extent[1])
                if k == len(tracks) - 1:
                    tr.set_xlabel(f"{b} (kb)", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)

    for balance, name in ((False, "matrices_raw.pdf"), (True, "matrices_normalised.pdf")):
        with PdfPages(output_dir / name) as pdf:
            for a, b in chrom_pairs:
                page(pdf, a, b, balance)
    return [output_dir / "matrices_raw.pdf", output_dir / "matrices_normalised.pdf"]
