#!/usr/bin/env python3
"""
mappability_v2.py

Theoretical mappability and its use to normalise the P(s).

1. bowtie2_mappability() : (default) the k-mers of the genome (one every
                           `step` bp) are aligned with bowtie2 in the mode of
                           the pipeline (--very-sensitive-local); a k-mer is
                           mappable if it aligns without XS tag, i.e. exactly
                           the criterion that sends a read to group1. Output:
                           bedgraph, score 1 over [p, p + step) for each
                           unique k-mer start p. No extra dependency.
   run_genmap()          : (alternative) GenMap index + map → per-position
                           bedgraph (score = 1 / number of occurrences of the
                           k-mer starting at this position, with up to E
                           errors). GenMap is no longer actively developed.
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
    python3 mappability_v2.py genome.fa out_dir -K 35 --index bt2_prefix -T 8
    python3 mappability_v2.py genome.fa out_dir -K 35 --method genmap -E 2
"""

import argparse
import gzip
import shutil
import subprocess
import threading
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
# 1b. bowtie2 (default)
# ---------------------------------------------------------------------------

BOWTIE2_MODE = ["--very-sensitive-local"]       # same mode as the first alignment


def iter_fasta(genome):
    """Yield (name, upper-case sequence) from a FASTA (plain or gzipped)."""
    name, chunks = None, []
    opener = gzip.open if str(genome).endswith(".gz") else open
    with opener(genome, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks).upper()
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks).upper()


def _write_kmers(genome, kmer, step, stdin):
    """FASTA of the k-mers starting every `step` bp (k-mers with N skipped)."""
    try:
        for chrom, seq in iter_fasta(genome):
            buf = []
            for p in range(0, len(seq) - kmer + 1, step):
                km = seq[p:p + kmer]
                if "N" in km:
                    continue
                buf.append(f">{chrom}:{p}\n{km}\n")
                if len(buf) >= 100_000:
                    stdin.write("".join(buf))
                    buf = []
            stdin.write("".join(buf))
    finally:
        stdin.close()


def bowtie2_mappability(genome, index, output_dir, kmer, step=10, threads=8):
    """
    Mappability with the aligner of the pipeline. Every `step` bp, the k-mer
    starting there is aligned (bowtie2 --very-sensitive-local, the mode of the
    first alignment); it is mappable if it aligns without XS tag (no second
    alignment: the criterion that sends a read to group1). Unaligned k-mers,
    k-mers with N and ambiguous k-mers are not mappable.
    Output: bedgraph (0-based, half-open), score 1 over [p, min(p + step, L))
    for each mappable k-mer start p, consecutive intervals merged; positions
    absent = not mappable. Read by binned_unique_fraction() like the GenMap
    bedgraph (one k-mer every `step` bp: fraction resolution = step).
    Time (mouse 2.9 Gb, step 10): ~1.5 h for k = 35 (8 threads), ~2.5-4 h for
    k = 100 (16 threads); run once per genome and read length.
    """
    if shutil.which("bowtie2") is None:
        raise RuntimeError("bowtie2 not found in PATH")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"bowtie2_K{kmer}_S{step}.bedgraph"
    tmp = out.with_suffix(".bedgraph.tmp")
    cmd = ["bowtie2", "-f", *BOWTIE2_MODE, "--reorder", "--no-hd", "--no-unal",
           "-p", str(threads), "-x", str(index), "-U", "-"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=open(output_dir / f"{out.stem}.bowtie2.log", "w"),
                            text=True, bufsize=1 << 20)
    writer = threading.Thread(target=_write_kmers, args=(genome, kmer, step, proc.stdin),
                              daemon=True)
    writer.start()
    run = None                                          # [chrom, start, end] being merged
    with open(tmp, "w") as fh:
        for line in proc.stdout:
            # read name "chrom:pos" (pos 0-based start of the k-mer)
            f = line.split("\t", 2)
            if int(f[1]) & 4 or "XS:i:" in line:
                continue
            chrom, p = f[0].rsplit(":", 1)
            p = int(p)
            s, e = p, p + step
            if run is not None and run[0] == chrom and run[2] == s:
                run[2] = e
            else:
                if run is not None:
                    fh.write(f"{run[0]}\t{run[1]}\t{run[2]}\t1\n")
                run = [chrom, s, e]
        if run is not None:
            fh.write(f"{run[0]}\t{run[1]}\t{run[2]}\t1\n")
    writer.join()
    if proc.wait() != 0:
        raise RuntimeError(f"bowtie2 failed, see {output_dir / (out.stem + '.bowtie2.log')}")
    # clip the last interval of each chromosome to its length
    lengths = {c: len(s) for c, s in iter_fasta(genome)}
    with open(tmp) as src, open(out, "w") as dst:
        for line in src:
            c, s, e, v = line.split("\t")
            dst.write(f"{c}\t{s}\t{min(int(e), lengths[c])}\t{v}")
    tmp.unlink()
    return out


# ---------------------------------------------------------------------------
# 1c. Repeat links: which repeated sequences go together
# ---------------------------------------------------------------------------

LINK_COLUMNS = ["chrom1", "start1", "chrom2", "start2", "orientation", "n_kmers"]


def _as_tag(line):
    i = line.find("AS:i:")
    j = line.find("\t", i)
    return int(line[i + 5:j if j > 0 else None])


def repeat_links(genome, index, output_dir, kmer, step=50, resolution=1000, max_copies=100,
                 as_delta=8, threads=8):
    """
    Links between the copies of the repeated sequences. One k-mer every `step`
    bp is aligned with bowtie2 -k max_copies (mode of the pipeline); the
    copies with AS >= best AS - as_delta (default 8: up to ~1 mismatch) are
    kept. For a k-mer starting at p with n copies (2 <= n < max_copies), a
    link is counted from the bin of p to the bin of each other copy, with the
    relative orientation of the copy (same: tandem / direct repeat, opposite:
    inverted repeat). K-mers with >= max_copies copies (saturated: e.g. Alu,
    LINE, satellites) are not linked, only counted.
    Output (TSV, bins of `resolution` bp, both directions present):
        chrom1 start1 chrom2 start2 orientation(+|-) n_kmers
    Returns (path, summary dict).
    """
    if shutil.which("bowtie2") is None:
        raise RuntimeError("bowtie2 not found in PATH")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"repeat_links_K{kmer}_S{step}.tsv"
    cmd = ["bowtie2", "-f", *BOWTIE2_MODE, "-k", str(max_copies), "--reorder", "--no-hd",
           "--no-unal", "-p", str(threads), "-x", str(index), "-U", "-"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=open(output_dir / f"{out.stem}.bowtie2.log", "w"),
                            text=True, bufsize=1 << 20)
    writer = threading.Thread(target=_write_kmers, args=(genome, kmer, step, proc.stdin),
                              daemon=True)
    writer.start()
    counts = {}
    summary = {"kmers_aligned": 0, "unique": 0, "linked": 0, "saturated": 0}

    def flush(name, hits):
        if not hits:
            return
        summary["kmers_aligned"] += 1
        if len(hits) >= max_copies:
            summary["saturated"] += 1
            return
        best = max(h[3] for h in hits)
        copies = [h for h in hits if h[3] >= best - as_delta]
        if len(copies) < 2:
            summary["unique"] += 1
            return
        chrom, p = name.rsplit(":", 1)
        p = int(p)
        # the k-mer itself: the copy at its origin (forward strand)
        own = [h for h in copies if h[0] == chrom and abs(h[1] - p) < kmer and not h[2]]
        if not own:
            return
        own_rev = own[0][2]
        b1 = p // resolution
        linked = False
        for c2, p2, rev, _ in copies:
            if c2 == chrom and abs(p2 - p) < kmer:
                continue
            key = (chrom, b1, c2, p2 // resolution, "-" if rev != own_rev else "+")
            counts[key] = counts.get(key, 0) + 1
            linked = True
        summary["linked"] += linked

    name, hits = None, []
    for line in proc.stdout:
        f = line.split("\t", 4)
        flag = int(f[1])
        if flag & 4:
            continue
        if f[0] != name:
            flush(name, hits)
            name, hits = f[0], []
        hits.append((f[2], int(f[3]) - 1, bool(flag & 16), _as_tag(line)))
    flush(name, hits)
    writer.join()
    if proc.wait() != 0:
        raise RuntimeError(f"bowtie2 failed, see {output_dir / (out.stem + '.bowtie2.log')}")
    with open(out, "w") as fh:
        fh.write("#" + "\t".join(LINK_COLUMNS) + f"\t# resolution={resolution}\n")
        for (c1, b1, c2, b2, o), n in sorted(counts.items()):
            fh.write(f"{c1}\t{b1 * resolution}\t{c2}\t{b2 * resolution}\t{o}\t{n}\n")
    return out, summary


def repeat_families(links, resolution, min_kmers=3, chrom_sizes=None, end_kb=30, seed=0):
    """
    Families of repeated sequences. Graph: nodes = bins (of `resolution` bp)
    holding a copy, edges = links with >= min_kmers shared k-mers, weighted
    by the number of shared k-mers. Families = Louvain communities (networkx;
    fixed seed): connected components chain different families through a few
    shared k-mers (yeast with -k 100: Ty elements and subtelomeres in one
    component of 506 bins); Louvain cuts these weak links. Without networkx:
    connected components. Families are ranked by their number of bins
    (family 1 = largest) and described automatically:
      tandem      : one chromosome, copies within a compact region (e.g. rDNA,
                    CUP1)
      subtelomeric: >= 60 % of the bins within end_kb of a chromosome end
                    (needs chrom_sizes)
      dispersed   : otherwise (e.g. Ty elements)
    Returns (bins DataFrame [chrom, start, family], families DataFrame
    [family, n_bins, n_chroms, kind, label]).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    L = links[links["n_kmers"] > 0].copy()
    L["b1"] = L["start1"] // resolution * resolution
    L["b2"] = L["start2"] // resolution * resolution
    agg = L.groupby(["chrom1", "b1", "chrom2", "b2"], sort=False)["n_kmers"].sum().reset_index()
    agg = agg[(agg["n_kmers"] >= min_kmers)
              & ~((agg["chrom1"] == agg["chrom2"]) & (agg["b1"] == agg["b2"]))]
    cols = ["chrom", "start", "family"]
    fcols = ["family", "n_bins", "n_chroms", "kind", "label"]
    if agg.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=fcols)
    k1 = list(zip(agg["chrom1"], agg["b1"]))
    k2 = list(zip(agg["chrom2"], agg["b2"]))
    nodes = {k: i for i, k in enumerate(dict.fromkeys(k1 + k2))}
    i = np.fromiter((nodes[k] for k in k1), np.int64, len(k1))
    j = np.fromiter((nodes[k] for k in k2), np.int64, len(k2))
    n = len(nodes)
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, w in zip(i, j, agg["n_kmers"].to_numpy()):
            w0 = G[u][v]["weight"] if G.has_edge(u, v) else 0
            G.add_edge(int(u), int(v), weight=w0 + float(w))
        comp = np.empty(n, dtype=np.int64)
        for c, members in enumerate(louvain_communities(G, weight="weight", seed=seed)):
            comp[list(members)] = c
    except ImportError:
        _, comp = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)),
                                       directed=False)
    sizes = np.bincount(comp)
    rank = np.empty_like(sizes)
    rank[np.argsort(-sizes, kind="stable")] = np.arange(1, len(sizes) + 1)
    keys = list(nodes)
    bins = pd.DataFrame({"chrom": [k[0] for k in keys], "start": [k[1] for k in keys],
                         "family": rank[comp]})
    fams = []
    for fam, g in bins.groupby("family"):
        chroms = g["chrom"].unique()
        if chrom_sizes is not None:
            L_ = g["chrom"].map(chrom_sizes).to_numpy()
            near_end = np.minimum(g["start"].to_numpy(), L_ - g["start"].to_numpy()) \
                <= end_kb * 1000
            sub = near_end.mean() >= 0.6
        else:
            sub = False
        span = g["start"].max() - g["start"].min() + resolution
        if len(chroms) == 1 and span <= 3 * len(g) * resolution:
            kind = "tandem"
            label = f"tandem {chroms[0]}:{g['start'].min() // 1000}-{span // 1000 + g['start'].min() // 1000} kb"
        elif sub:
            kind, label = "subtelomeric", f"subtelomeric ({len(chroms)} chr)"
        else:
            kind, label = "dispersed", f"dispersed ({len(chroms)} chr)"
        fams.append((fam, len(g), len(chroms), kind, f"F{fam} {label}"))
    return bins, pd.DataFrame(fams, columns=fcols)


def read_repeat_links(path):
    """DataFrame of the repeat links (columns LINK_COLUMNS)."""
    return pd.read_csv(path, sep="\t", comment="#", header=None, names=LINK_COLUMNS,
                       dtype={"chrom1": str, "chrom2": str})


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
    parser = argparse.ArgumentParser(description="Compute theoretical mappability "
                                                 "(bowtie2 by default, or GenMap).")
    parser.add_argument("genome", help="Genome FASTA")
    parser.add_argument("output_dir", help="Output directory (bedgraph)")
    parser.add_argument("-K", "--kmer", type=int, required=True, help="k-mer length = read length")
    parser.add_argument("--method", choices=("bowtie2", "genmap"), default="bowtie2")
    parser.add_argument("--index", help="bowtie2 index prefix (method bowtie2)")
    parser.add_argument("--step", type=int, default=10,
                        help="One k-mer every STEP bp (method bowtie2, default=10)")
    parser.add_argument("-E", "--errors", type=int, default=2,
                        help="Mismatches allowed (method genmap, default=2)")
    parser.add_argument("-T", "--threads", type=int, default=8, help="Threads (default=8)")
    parser.add_argument("--links_step", type=int, default=50,
                        help="Repeat links: one k-mer every STEP bp (needs --index; 0 = no links)")
    args = parser.parse_args()

    if args.method == "bowtie2":
        if not args.index:
            parser.error("--index is required with --method bowtie2")
        bedgraph = bowtie2_mappability(args.genome, args.index, args.output_dir, args.kmer,
                                       args.step, args.threads)
    else:
        bedgraph = run_genmap(args.genome, args.output_dir, args.kmer, args.errors, args.threads)
    print(f"Mappability bedgraph: {bedgraph}")
    if args.index and args.links_step > 0:
        links, summary = repeat_links(args.genome, args.index, args.output_dir, args.kmer,
                                      args.links_step, threads=args.threads)
        print(f"Repeat links: {links} {summary}")


if __name__ == "__main__":
    main()
