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
        left : unrescued map; mappability, GC content
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
                      sites=None, enzyme_label="restriction sites", chrom_pairs=None, intra_only=False, vmin_raw=None, vmax_raw=None,
                      labels=("unrescued", "rescued")):
    """
    Two multi-page PDFs (matrices_raw.pdf, matrices_normalised.pdf), one page
    per pair of chromosomes (all pairs, or intra only, or chrom_pairs), as in
    v1 (hicberg/plot.py plot_matrix):
      left : unrescued map; tracks: mappability, GC
      right: rescued map; tracks: coverage before / after (+ median), restriction
             sites per bin (bars; GC if sites is None, e.g. Micro-C)
    Coverage = column sums of the unbalanced block (as v1). mappability, gc:
    {chrom: value per bin} at the bin size of the matrices.
    Maps keep their real proportions; the tracks have exactly the width of
    their map.
    """
    import cooler
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
        n_tracks = 2                    # left: mappability, GC; right: coverage before / after, sites (or GC)
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
            im = ax.imshow(np.ma.masked_invalid(z), cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="none", extent=extent, aspect="auto")
            ax.set_title(f"{lab} — {a} × {b} — {kind}, {kb:g} kb bins\n"
                         f"{n_contacts[col]:,} contacts", fontsize=9)
            ax.set_ylabel(f"{a} (kb)", fontsize=8)
            ax.tick_params(labelbottom=False, labelsize=7)
            cax = fig.add_axes(frac(x0 + side + 0.1, map_y, cb, h_map))
            fig.colorbar(im, cax=cax, label="log10 contacts")

            tracks = []
            if col == 0:
                mapp = track(mappability.get(b) if mappability else None, nb)
                if mapp is not None:
                    tracks.append(("mappability", [(mapp, "grey", "fill")], None))
            else:
                tracks.append(("coverage before / after",
                               [(cov_u, "royalblue", "before"), (cov_r, "limegreen", "after")], med))
            rs = track(sites.get(b) if sites else None, nb)
            g = track(gc.get(b) if gc else None, nb)
            if col == 1 and rs is not None:
                tracks.append((enzyme_label, [(rs, "royalblue", "bar")], None))
            elif g is not None:
                tracks.append(("GC content", [(g, "orange", None)], None))
            for k, (title, series, median) in enumerate(tracks):
                y0 = map_y - (k + 1) * (t_h + vgap)
                tr = fig.add_axes(frac(x0, y0, side, t_h), sharex=ax)
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
