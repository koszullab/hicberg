#!/usr/bin/env python3
"""
reattribute_v2.py

Re-attribution of the ambiguous pairs (group2) to one of their possible
positions, drawn at random with a probability proportional to a propensity
computed from the statistical laws of the unambiguous pairs (statistics_v2.py).

For each group2 chunk (group2_1_chunk{i}.fastq / group2_2_chunk{i}.fastq,
streaming_classifier_v2.py):
  1. bowtie2 --very-sensitive-local --reorder -k K (or -a) on R1 and R2,
     read as a stream (no SAM written on disk), alignments grouped by read
  2. candidates of a read = its alignments within --as_delta of the best
     score (AS), each weighted by its likelihood exp(-lambda × (best AS - AS)),
     lambda from --error_rate (1 mismatch = ×1/300 at 1%). The previous
     version kept all the -k alignments with the same weight (~26% of them
     have at least one more mismatch than the best one on yeast); a first v2
     kept the best-score ones only, which sent every low-complexity read
     (poly-T: 1 perfect locus, hundreds at 1 mismatch) to the same few loci
     (artefactual contact spots); the likelihood weighting spreads them
  3. propensity of every R1 × R2 combination, vectorised over batches of
     pairs, with the same conventions as the statistics (5' positions,
     orientation after ordering by position, circular distances):
        random    1
        ps        P(s) (orientation, distance) or trans P(s)
        standard  P(s) × coverage(R1) × coverage(R2)
        full      P(s) × density(R1, R2)
  4. one combination drawn per pair (weighted, seeded → reproducible)
  5. predicted.pairs: same format as unrescued.pairs + the numbers of
     candidates of R1 and R2

CLI:
  python3 reattribute_v2.py classifier_dir stats_dir output_dir \
      --index genome_index --mode full --max_alignments all --n_workers 4
"""

import argparse
import multiprocessing as mp
import re
import subprocess
import time
from pathlib import Path

import numpy as np

import statistics_v2 as st
from streaming_classifier_v2 import split_sam, fast_outer_5prime

MODES = ("random", "ps", "standard", "full")
_AS = re.compile(r"(?:^|\t)AS:i:(-?\d+)")


# ---------------------------------------------------------------------------
# Alignments
# ---------------------------------------------------------------------------

def bowtie2_command(index, fastq, max_alignments="100", threads=1):
    """bowtie2 command writing SAM to stdout (-k K, or -a for 'all')."""
    k = ["-a"] if str(max_alignments) == "all" else ["-k", str(max_alignments)]
    return ["bowtie2", "--very-sensitive-local", "--reorder", *k, "-p", str(threads),
            "-x", str(index), "-U", str(fastq)]


def read_blocks(lines):
    """
    Group the alignment lines of a SAM stream by read (consecutive with
    --reorder). Yields (read name, [(chrom, 5' position, is_reverse, AS), ...]),
    unmapped alignments left out (empty list if the read is unmapped).
    """
    name, block = None, []
    for line in lines:
        f = split_sam(line)
        if f is None:
            continue
        if f[0] != name:
            if name is not None:
                yield name, block
            name, block = f[0], []
        flag = int(f[1])
        if flag & 4:
            continue
        m = _AS.search(f[11])
        score = int(m.group(1)) if m else 0
        rev = bool(flag & 16)
        block.append((f[2], fast_outer_5prime(f, rev), rev, score))
    if name is not None:
        yield name, block


def best_candidates(block, as_delta=0):
    """Alignments whose score is >= best score - as_delta."""
    if not block:
        return block
    best = max(a[3] for a in block)
    return [a for a in block if a[3] >= best - as_delta]


def score_lambda(error_rate, mismatch_cost=8):
    """
    Log-likelihood cost of one alignment-score point. In bowtie2 local mode a
    high-quality mismatch replaces the +2 match bonus by a -6 penalty (8
    points), and is e / (3 (1 - e)) times less likely than a correct base:
    lambda = ln(3 (1 - e) / e) / 8 (0.71 for e = 1%). error_rate <= 0 →
    infinite (only the best-score alignments count).
    """
    if error_rate <= 0:
        return np.inf
    return np.log(3 * (1 - error_rate) / error_rate) / mismatch_cost


def alignment_weights(scores, lam):
    """Relative likelihood of each alignment of a read: exp(-lambda × (best AS - AS))."""
    scores = np.asarray(scores, dtype=np.float64)
    gap = scores.max() - scores
    if np.isinf(lam):
        return (gap == 0).astype(np.float64)
    return np.exp(-lam * gap)


# ---------------------------------------------------------------------------
# Statistical laws (outputs of statistics_v2.py)
# ---------------------------------------------------------------------------

class Laws:
    """Laws of a statistics folder, stored as flat arrays for vectorised lookups."""

    def __init__(self, stats_dir, mode, circular=None, bin_size=2000):
        d = Path(stats_dir)
        load = lambda name: np.load(d / name, allow_pickle=True).item()
        sizes = load(st.CHROM_SIZES)
        self.names = list(sizes)
        self.code = {c: i for i, c in enumerate(self.names)}
        self.sizes = np.array([sizes[c] for c in self.names], dtype=np.int64)
        self.bin_size = bin_size
        self.mode = mode
        if circular is None:
            circular = read_circular(d)
        self.circular = np.array([c in circular for c in self.names])

        # P(s): law[code, type, xs index], types 0 uncut, 1 circle, 2 weird
        xs = load(st.XS)
        self.xs_glob = max(xs.values(), key=len)
        self.n_xs = np.array([len(xs[c]) for c in self.names])
        self.law = np.zeros((len(self.names), 3, len(self.xs_glob)))
        for t, name in enumerate((st.UNCUTS, st.CIRCLES, st.WEIRDS)):
            law = load(name)
            for c in self.names:
                self.law[self.code[c], t, :len(law[c])] = law[c]
        trans = load(st.TRANS_PS)
        self.trans = np.array([[trans.get((a, b), 0.0) for b in self.names] for a in self.names])

        if mode == "standard":
            cov = load(st.COVERAGE)
            self.cov_n = np.array([len(cov[c]) for c in self.names])
            self.cov_off = np.concatenate([[0], np.cumsum(self.cov_n)[:-1]])
            self.cov = np.concatenate([np.asarray(cov[c], dtype=np.float64) for c in self.names])
        if mode == "full":
            self.density = load(st.DENSITY_MAP)
            self.dens_n = np.array([self.density[(c, c)].shape[0] for c in self.names])

    def ps(self, c1, p1, r1, c2, p2, r2):
        """P(s) of each combination (intra: by orientation and distance; trans: trans P(s))."""
        out = self.trans[c1, c2].astype(np.float64)
        cis = c1 == c2
        if cis.any():
            c, a, b = c1[cis], p1[cis], p2[cis]
            orient = st.pair_orientation(a, b, r1[cis], r2[cis])
            kind = np.minimum(orient, 2)                       # ++ and -- → weird
            d = st.cis_distance(a, b, self.sizes[c], self.circular[c])
            idx = np.minimum(st.attribute_xs(self.xs_glob, d), self.n_xs[c] - 1)
            out[cis] = self.law[c, kind, idx]
        return out

    def coverage(self, c, p):
        b = np.minimum((p - 1) // self.bin_size, self.cov_n[c] - 1)
        return self.cov[self.cov_off[c] + np.maximum(b, 0)]

    def density_of(self, c1, p1, c2, p2):
        """Density of each combination; blocks are stored for chromosome a <= b."""
        b1 = np.clip((p1 - 1) // self.bin_size, 0, self.dens_n[c1] - 1)
        b2 = np.clip((p2 - 1) // self.bin_size, 0, self.dens_n[c2] - 1)
        swap = c1 > c2
        ca, cb = np.where(swap, c2, c1), np.where(swap, c1, c2)
        ba, bb = np.where(swap, b2, b1), np.where(swap, b1, b2)
        out = np.empty(len(c1))
        key = ca * len(self.names) + cb
        for k in np.unique(key):
            sel = key == k
            blk = self.density[(self.names[k // len(self.names)], self.names[k % len(self.names)])]
            out[sel] = blk[ba[sel], bb[sel]]
        return out

    def propensity(self, c1, p1, r1, c2, p2, r2):
        if self.mode == "random":
            return np.ones(len(c1))
        ps = self.ps(c1, p1, r1, c2, p2, r2)
        if self.mode == "ps":
            return ps
        if self.mode == "standard":
            return ps * self.coverage(c1, p1) * self.coverage(c2, p2)
        return ps * self.density_of(c1, p1, c2, p2)


def read_circular(stats_dir):
    """Circular chromosomes recorded in statistics_stats.txt (statistics_v2.py)."""
    path = Path(stats_dir) / "statistics_stats.txt"
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("circular"):
                value = line.split(":", 1)[1].strip()
                return set() if value == "-" else {c.strip() for c in value.split(",")}
    return set()


# ---------------------------------------------------------------------------
# Vectorised draw over a batch of pairs
# ---------------------------------------------------------------------------

def combination_indices(n1, n2):
    """
    For pairs with n1[p] R1 candidates and n2[p] R2 candidates, the index (in
    the concatenated candidate arrays) of R1 and R2 of every combination, and
    the start of each pair in the combination array.
    """
    n1, n2 = np.asarray(n1, dtype=np.int64), np.asarray(n2, dtype=np.int64)
    k = n1 * n2
    starts = np.concatenate([[0], np.cumsum(k)[:-1]])
    pair = np.repeat(np.arange(len(k)), k)
    q = np.arange(k.sum()) - starts[pair]
    i = q // n2[pair]
    j = q % n2[pair]
    start1 = np.concatenate([[0], np.cumsum(n1)[:-1]])
    start2 = np.concatenate([[0], np.cumsum(n2)[:-1]])
    return start1[pair] + i, start2[pair] + j, starts, k


def weighted_draw(weights, starts, sizes, rng):
    """
    One index per segment [starts[p], starts[p] + sizes[p]) of `weights`,
    drawn with probability proportional to the weights (uniform if they are
    all 0 in the segment). Non-finite or negative weights count as 0.
    """
    w = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    cs = np.cumsum(w)
    before = np.concatenate([[0.0], cs])[starts]
    total = np.concatenate([[0.0], cs])[starts + sizes] - before
    u = rng.random(len(starts))
    idx = np.searchsorted(cs, before + u * total, side="right")
    uniform = starts + np.minimum((u * sizes).astype(np.int64), sizes - 1)
    idx = np.where(total > 0, idx, uniform)
    return np.clip(idx, starts, starts + sizes - 1)


# ---------------------------------------------------------------------------
# One chunk
# ---------------------------------------------------------------------------

def reattribute_stream(lines1, lines2, laws, out_fh, rng, as_delta=30, error_rate=0.01,
                       batch_combinations=2_000_000):
    """
    Re-attribute the pairs of two SAM streams (R1, R2) and write the chosen
    combination of each pair to out_fh. Candidates: alignments within as_delta
    of the best score, each weighted by its likelihood (alignment_weights,
    error_rate; 0 → best-score ties only). Returns counters.
    """
    lam = score_lambda(error_rate)
    if np.isinf(lam):
        as_delta = 0
    stats = dict(pairs=0, placed=0, unplaced=0, combinations=0, cand1=0, cand2=0, intra=0)
    batch = []                     # (name, cands1, cands2)
    n_comb = 0

    def flush():
        nonlocal batch, n_comb
        if not batch:
            return
        names = [b[0] for b in batch]
        cands = [b[1] for b in batch], [b[2] for b in batch]
        n1 = np.array([len(c) for c in cands[0]])
        n2 = np.array([len(c) for c in cands[1]])
        arrays = []
        for side in cands:
            flat = [a for c in side for a in c]
            arrays.append((np.array([laws.code.get(a[0], -1) for a in flat], dtype=np.int64),
                           np.array([a[1] for a in flat], dtype=np.int64),
                           np.array([a[2] for a in flat], dtype=bool),
                           np.concatenate([alignment_weights([a[3] for a in c], lam) for c in side])))
        i1, i2, starts, k = combination_indices(n1, n2)
        (c1, p1, r1, a1), (c2, p2, r2, a2) = arrays
        # statistical propensity × likelihood of each of the two alignments
        w = laws.propensity(c1[i1], p1[i1], r1[i1], c2[i2], p2[i2], r2[i2]) * a1[i1] * a2[i2]
        chosen = weighted_draw(w, starts, k, rng)
        s1, s2 = i1[chosen], i2[chosen]
        for p, (name, a, b) in enumerate(zip(names, s1, s2)):
            out_fh.write(f"{name}\t{laws.names[c1[a]]}\t{p1[a]}\t{laws.names[c2[b]]}\t{p2[b]}\t"
                         f"{'-' if r1[a] else '+'}\t{'-' if r2[b] else '+'}\t{n1[p]}\t{n2[p]}\n")
        stats["placed"] += len(batch)
        stats["intra"] += int(np.sum(c1[s1] == c2[s2]))
        stats["combinations"] += int(k.sum())
        stats["cand1"] += int(n1.sum())
        stats["cand2"] += int(n2.sum())
        batch, n_comb = [], 0

    for (name1, b1), (name2, b2) in zip(read_blocks(lines1), read_blocks(lines2)):
        if name1 != name2:
            raise ValueError(f"R1 / R2 out of sync: {name1} vs {name2}")
        stats["pairs"] += 1
        c1, c2 = best_candidates(b1, as_delta), best_candidates(b2, as_delta)
        c1 = [a for a in c1 if a[0] in laws.code]
        c2 = [a for a in c2 if a[0] in laws.code]
        if not c1 or not c2:
            stats["unplaced"] += 1
            continue
        batch.append((name1, c1, c2))
        n_comb += len(c1) * len(c2)
        if n_comb >= batch_combinations:
            flush()
    flush()
    return stats


PAIRS_HEADER = ("## pairs format v1.0\n"
                "#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2 n_candidates1 n_candidates2\n")


def merge_pairs(unrescued, predicted, out_path):
    """rescued.pairs = unrescued.pairs + predicted.pairs (7 standard columns). Returns the number of pairs."""
    n = 0
    with open(out_path, "w") as out:
        out.write("## pairs format v1.0\n#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2\n")
        for path in (unrescued, predicted):
            with open(path) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    f = line.rstrip("\n").split("\t")
                    out.write("\t".join(f[:7]) + "\n")
                    n += 1
    return n


def reattribute_chunk(job):
    """
    Worker: one group2 chunk. job = dict(fastq1, fastq2 | sam1, sam2, out, stats_dir,
    mode, circular, bin_size, index, max_alignments, threads, as_delta, seed).
    Runs bowtie2 on both FASTQ files (stdout read as a stream), or reads SAM files.
    """
    t0 = time.time()
    laws = Laws(job["stats_dir"], job["mode"], job.get("circular"), job["bin_size"])
    rng = np.random.default_rng(job["seed"])
    procs, handles = [], []
    try:
        if job.get("sam1"):
            handles = [open(job["sam1"]), open(job["sam2"])]
            streams = handles
        else:
            for fq in (job["fastq1"], job["fastq2"]):
                log = open(str(job["out"]) + f".{Path(fq).stem}.bowtie2.log", "w")
                handles.append(log)
                procs.append(subprocess.Popen(
                    bowtie2_command(job["index"], fq, job["max_alignments"], job["threads"]),
                    stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1 << 20))
            streams = [p.stdout for p in procs]
        with open(job["out"], "w") as out:
            stats = reattribute_stream(streams[0], streams[1], laws, out, rng, job["as_delta"],
                                       job["error_rate"])
        for p in procs:
            if p.wait() != 0:
                raise RuntimeError(f"bowtie2 failed (see {job['out']}.*.bowtie2.log)")
    finally:
        for h in handles:
            h.close()
    stats["seconds"] = time.time() - t0
    return stats


def reattribute(classifier_dir, stats_dir, output_dir, index=None, mode="full",
                max_alignments="100", as_delta=30, n_workers=1, threads=1, circular=None,
                bin_size=2000, seed=0, sam_pairs=None, error_rate=0.01):
    """
    Re-attribute all the group2 chunks of classifier_dir (in parallel, one
    process per chunk) and write output_dir/predicted.pairs.
    sam_pairs: optional list of (sam1, sam2) already aligned, instead of running bowtie2.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode}, choose among {MODES}")
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = dict(stats_dir=str(stats_dir), mode=mode, circular=circular, bin_size=bin_size,
                index=index, max_alignments=max_alignments, threads=threads, as_delta=as_delta,
                error_rate=error_rate)
    jobs = []
    if sam_pairs:
        for i, (s1, s2) in enumerate(sam_pairs):
            jobs.append(dict(base, sam1=s1, sam2=s2, seed=seed + i,
                             out=output_dir / f"predicted_chunk{i}.pairs"))
    else:
        r1 = sorted(Path(classifier_dir).glob("group2_1_chunk*.fastq"),
                    key=lambda p: int(re.search(r"chunk(\d+)", p.name).group(1)))
        if not r1:
            raise FileNotFoundError(f"No group2_1_chunk*.fastq in {classifier_dir}")
        if index is None:
            raise ValueError("--index (bowtie2 index) is required to align the group2 chunks")
        for i, f1 in enumerate(r1):
            f2 = f1.with_name(f1.name.replace("group2_1_", "group2_2_"))
            jobs.append(dict(base, fastq1=str(f1), fastq2=str(f2), seed=seed + i,
                             out=output_dir / f"predicted_chunk{i}.pairs"))

    if n_workers > 1 and len(jobs) > 1:
        with mp.Pool(min(n_workers, len(jobs))) as pool:
            results = pool.map(reattribute_chunk, jobs)
    else:
        results = [reattribute_chunk(j) for j in jobs]

    with open(output_dir / "predicted.pairs", "w") as out:
        out.write(PAIRS_HEADER)
        for j in jobs:
            with open(j["out"]) as fh:
                for line in fh:
                    out.write(line)
            Path(j["out"]).unlink()

    unrescued = Path(classifier_dir) / "unrescued.pairs" if classifier_dir else None
    if unrescued is not None and unrescued.is_file():
        n_res = merge_pairs(unrescued, output_dir / "predicted.pairs", output_dir / "rescued.pairs")
        print(f"rescued.pairs: {n_res:,} pairs (unrescued + predicted)")

    total = {k: sum(r[k] for r in results) for k in results[0] if k != "seconds"}
    p = max(total["placed"], 1)
    lines = [
        f"mode                  : {mode}",
        f"alignments            : {'-a (all)' if str(max_alignments) == 'all' else f'-k {max_alignments}'}, "
        + (f"candidates within {as_delta} of the best score, weighted by their likelihood "
           f"(error rate {error_rate:g}, {score_lambda(error_rate):.2f} per AS point)"
           if error_rate > 0 else "best-score candidates only"),
        f"pairs                 : {total['pairs']}",
        f"placed                : {total['placed']}",
        f"unplaced              : {total['unplaced']} (a mate without alignment)",
        f"candidates per read   : R1 {total['cand1'] / p:.2f}, R2 {total['cand2'] / p:.2f}",
        f"combinations per pair : {total['combinations'] / p:.2f}",
        f"placed intra          : {total['intra']} ({100 * total['intra'] / p:.1f}%)",
        f"time                  : {time.time() - t0:.0f} s ({n_workers} workers)",
    ]
    (output_dir / "reattribution_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n=== Re-attribution ===\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Re-attribute the ambiguous pairs (group2).")
    parser.add_argument("classifier_dir", help="Output folder of streaming_classifier_v2.py (group2 chunks)")
    parser.add_argument("stats_dir", help="Output folder of statistics_v2.py")
    parser.add_argument("output_dir", help="Output folder (predicted.pairs)")
    parser.add_argument("--index", help="bowtie2 index of the genome")
    parser.add_argument("--mode", choices=MODES, default="full",
                        help="standard = P(s) × coverage, full = P(s) × density (default=full)")
    parser.add_argument("--max_alignments", default="100",
                        help="bowtie2 -k value, or 'all' for -a (default=100)")
    parser.add_argument("--error_rate", type=float, default=0.01,
                        help="Sequencing error rate of the likelihood weight of each alignment, "
                             "exp(-lambda x (best AS - AS)) (default=0.01: 1 mismatch = x1/300). "
                             "0 = best-score alignments only")
    parser.add_argument("--as_delta", type=int, default=30,
                        help="Ignore the alignments more than this AS below the best one "
                             "(default=30: weight < 1e-9 with error_rate 0.01)")
    parser.add_argument("--n_workers", type=int, default=1, help="Chunks processed in parallel")
    parser.add_argument("--threads", type=int, default=1, help="bowtie2 threads per worker")
    parser.add_argument("--circular", default=None,
                        help="Circular chromosomes (default: those of statistics_stats.txt)")
    parser.add_argument("--bin_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    circular = None if args.circular is None else {c for c in args.circular.split(",") if c}
    reattribute(args.classifier_dir, args.stats_dir, args.output_dir, args.index, args.mode,
                args.max_alignments, args.as_delta, args.n_workers, args.threads, circular,
                args.bin_size, args.seed, error_rate=args.error_rate)


if __name__ == "__main__":
    main()
