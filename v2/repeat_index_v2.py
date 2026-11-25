"""
Repeat index: where every highly repeated sequence of the genome lies, computed once.

Why: for a read from a satellite (HSATII, HSat3, alpha satellite...) or a young
interspersed repeat, bowtie2 -k K returns the first K hits it finds, not a
representative sample of the copies. In the human pilot, HSATII reads got 76 % of
their candidates on chr1, 8 % on chr16 and 0.2 % on chr15, while RepeatMasker puts
41 % / 39 % / 13 % of the HSATII copies there: the candidate list is biased, and the
re-attribution can only choose among the candidates it is given.

What: canonical k-mers are sampled with (w, k) minimizers (one k-mer per window of w
consecutive k-mers, chosen by a hash). Minimizers occurring at least `min_count` times
in the genome are kept with the genome bins (bin_size) that carry them and their number
of copies per bin. The memory cost scales with the number of bins a repeat spans, not
with its number of copies: a k-mer present 100 000 times in a 13 Mb array uses at most
13 Mb / bin_size entries, and *all* the copies are represented.

A read whose bowtie2 list is saturated is then looked up: its minimizers give the bins
that share its sequence (all of them), weighted by the number of shared minimizers and
copies, before the usual P(s) x density draw and a local alignment inside the drawn bin.

Only numpy is needed; the genome FASTA is the only input.

Usage:
    python3 repeat_index_v2.py genome.fa out_prefix [--k 31 --w 10 --bin_size 2000 --min_count 50]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

LUT = np.full(256, 4, np.uint8)
for _i, _c in enumerate(b"ACGT"):
    LUT[_c] = _i
    LUT[_c + 32] = _i
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
INVALID = np.uint64(0xFFFFFFFFFFFFFFFF)


def read_fasta(path):
    """Yield (name, sequence bytes) for each record (name = first word of the header)."""
    name, chunks = None, []
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                if name is not None:
                    yield name, b"".join(chunks)
                name, chunks = line[1:].split()[0].decode(), []
            else:
                chunks.append(line.rstrip())
    if name is not None:
        yield name, b"".join(chunks)


def splitmix64(x):
    """Invertible 64-bit mixer (vectorised, wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def kmer_hashes(codes, k):
    """
    Hash of the canonical k-mer starting at each position of `codes` (0-3, 4 = N).
    Returns uint64 array of length len(codes) - k + 1; k-mers with an N get INVALID.
    """
    n = len(codes) - k + 1
    if n <= 0:
        return np.zeros(0, np.uint64)
    c = codes.astype(np.uint64)
    bad = codes == 4
    c[bad] = 0
    fwd = np.zeros(n, np.uint64)
    rev = np.zeros(n, np.uint64)
    for j in range(k):
        fwd = (fwd << np.uint64(2)) | c[j:j + n]
        rev |= (np.uint64(3) - c[j:j + n]) << np.uint64(2 * j)
    h = splitmix64(np.minimum(fwd, rev))
    if bad.any():
        nb = np.convolve(bad.astype(np.int32), np.ones(k, np.int32), "valid") > 0
        h[nb] = INVALID
    return h


def minimizers(h, w):
    """
    Positions of the (w, k) minimizers of a hash array: in every window of w consecutive
    k-mers, the leftmost k-mer with the smallest hash. Returns sorted unique positions
    (INVALID k-mers never selected).
    """
    if len(h) == 0:
        return np.zeros(0, np.int64)
    if len(h) < w:
        p = np.array([int(np.argmin(h))])
    else:
        win = np.lib.stride_tricks.sliding_window_view(h, w)
        p = np.unique(np.arange(len(win)) + win.argmin(axis=1))
    return p[h[p] != INVALID]


def sequence_minimizers(seq, k, w, chunk=20_000_000):
    """(hashes, positions) of the minimizers of a sequence (bytes), chunked to bound memory."""
    codes = LUT[np.frombuffer(seq, np.uint8)]
    hs, ps = [], []
    step = chunk
    for s in range(0, max(1, len(codes) - k + 1), step):
        e = min(len(codes), s + step + k - 1 + w - 1)
        h = kmer_hashes(codes[s:e], k)
        p = minimizers(h, w)        # windows starting in [s, s + step): exactly this chunk's windows
        hs.append(h[p])
        ps.append(p + s)
    p, first = np.unique(np.concatenate(ps), return_index=True)   # a k-mer chosen by both chunks
    return np.concatenate(hs)[first], p


def _aggregate(H, G, min_count):
    """Sorted (keys, offsets, bins, copies, total, n repeated occurrences) of the minimizers of
    one hash range with >= min_count occurrences."""
    order = np.argsort(H, kind="stable")
    H = H[order]
    G = G[order]
    del order
    starts = np.flatnonzero(np.r_[True, H[1:] != H[:-1]]) if len(H) else np.zeros(0, np.int64)
    counts = np.diff(np.r_[starts, len(H)])
    rep = counts >= min_count
    keep = np.repeat(rep, counts)
    H = H[keep]
    G = G[keep]
    n_rep = int(counts[rep].sum())
    del keep
    o = np.lexsort((G, H))
    H = H[o]
    G = G[o]
    del o
    new = np.r_[True, (H[1:] != H[:-1]) | (G[1:] != G[:-1])] if len(H) else np.zeros(0, bool)
    st = np.flatnonzero(new)
    copies = np.diff(np.r_[st, len(H)]).astype(np.uint32)
    kh = H[st]
    bins = G[st]
    kst = np.flatnonzero(np.r_[True, kh[1:] != kh[:-1]]) if len(kh) else np.zeros(0, np.int64)
    keys = kh[kst]
    offsets = np.r_[kst, len(kh)].astype(np.int64)
    total = np.add.reduceat(copies.astype(np.int64), kst).astype(np.uint32) if len(kst) else \
        np.zeros(0, np.uint32)
    return keys, offsets, bins, copies, total, n_rep


def build_repeat_index(genome, out_prefix=None, k=31, w=10, bin_size=2000, min_count=2, log=print,
                       n_parts=8, tmp_dir=None):
    """
    Build the repeat index of a genome FASTA. Returns a dict (also saved as one .npy file per
    array in the directory out_prefix + '.idx', memory-mapped by load_repeat_index, and the
    genome as codes in out_prefix + '.genome.u8'):
      keys     uint64 sorted hashes of the repeated minimizers (>= min_count copies)
      offsets  int64, entries of keys[i] are offsets[i]:offsets[i+1]
      bins     uint32 global bin ids (chrom_offset[chrom] + pos // bin_size)
      copies   uint32 copies of the minimizer in that bin
      total    uint32 total copies of each key in the genome
      chroms, chrom_sizes, chrom_offsets (bins), k, w, bin_size, min_count
    Memory: the minimizers are spilled to n_parts temporary files by the top bits of their
    hash and each part is sorted alone (human, 8 parts: ~3 GB instead of ~18 GB).
    """
    import tempfile, shutil
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="repeat_index_", dir=tmp_dir or (Path(out_prefix).parent
                                                                         if out_prefix else None)))
    shift = np.uint64(64 - int(np.log2(n_parts)))
    fh = [(open(tmp / f"H{i}.bin", "wb"), open(tmp / f"G{i}.bin", "wb")) for i in range(n_parts)]
    chroms, sizes, offs = [], [], [0]
    n_all = 0
    try:
        for name, seq in read_fasta(genome):
            h, p = sequence_minimizers(seq, k, w)
            g = (offs[-1] + p // bin_size).astype(np.uint32)
            part = (h >> shift).astype(np.int64) if n_parts > 1 else np.zeros(len(h), np.int64)
            for i in range(n_parts):
                sel = part == i
                fh[i][0].write(h[sel].tobytes())
                fh[i][1].write(g[sel].tobytes())
            n_all += len(h)
            chroms.append(name)
            sizes.append(len(seq))
            offs.append(offs[-1] + -(-len(seq) // bin_size))
            log(f"  {name}: {len(seq):,} bp, {len(h):,} minimizers ({time.time() - t0:.0f} s)")
        for a_, b_ in fh:
            a_.close(); b_.close()
        K, O, Bn, C, T = [], [], [], [], []
        n_rep, base = 0, 0
        for i in range(n_parts):
            H = np.fromfile(tmp / f"H{i}.bin", np.uint64)
            G = np.fromfile(tmp / f"G{i}.bin", np.uint32)
            keys, offsets, bins, copies, total, nr = _aggregate(H, G, min_count)
            del H, G
            (tmp / f"H{i}.bin").unlink(); (tmp / f"G{i}.bin").unlink()
            K.append(keys); O.append(offsets[:-1] + base); Bn.append(bins); C.append(copies); T.append(total)
            base += len(bins)
            n_rep += nr
            log(f"  part {i + 1}/{n_parts}: {len(keys):,} repeated minimizers ({time.time() - t0:.0f} s)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    idx = dict(keys=np.concatenate(K), offsets=np.r_[np.concatenate(O), base].astype(np.int64),
               bins=np.concatenate(Bn), copies=np.concatenate(C), total=np.concatenate(T),
               chroms=np.array(chroms), chrom_sizes=np.array(sizes, np.int64),
               chrom_offsets=np.array(offs, np.int64), k=np.int64(k), w=np.int64(w),
               bin_size=np.int64(bin_size), min_count=np.int64(min_count))
    log(f"repeat index: {n_all:,} minimizers, {len(idx['keys']):,} repeated (>= {min_count} copies) "
        f"covering {n_rep:,} occurrences ({n_rep / max(n_all, 1):.1%}), {len(idx['bins']):,} "
        f"(minimizer, bin) entries, {time.time() - t0:.0f} s")
    if out_prefix:
        d = Path(str(out_prefix) + ".idx")
        d.mkdir(parents=True, exist_ok=True)
        for key, v in idx.items():
            np.save(d / f"{key}.npy", v)
        write_genome_codes(genome, str(out_prefix) + ".genome.u8")
    for key in ("k", "w", "bin_size", "min_count"):
        idx[key] = int(idx[key])
    return idx


def load_repeat_index(path):
    """Index saved by build_repeat_index: a directory of .npy files (memory-mapped: shared by
    the worker processes through the page cache) or an older .npz file."""
    path = Path(path)
    if path.suffix == ".npz" or (not path.is_dir() and Path(str(path) + ".npz").is_file()
                                 and not Path(str(path) + ".idx").is_dir()):
        npz = path if path.suffix == ".npz" else Path(str(path) + ".npz")
        d = np.load(npz, allow_pickle=False)
        idx = {key: d[key] for key in d.files}
    else:
        d = path if path.is_dir() else Path(str(path) + ".idx")
        idx = {f.stem: np.load(f, mmap_mode="r") for f in d.glob("*.npy")}
        idx["chroms"] = np.asarray(idx["chroms"])
    for key in ("k", "w", "bin_size", "min_count"):
        idx[key] = int(idx[key])
    return idx


def lookup(idx, seq, min_fraction=0.8):
    """
    Candidate bins of a read (bytes): bins sharing at least min_fraction of the best bin's
    number of read minimizers. Returns (bins, n_shared, weight) where weight = mean copies
    of the shared minimizers in the bin (number of positions the read could occupy there).
    Empty arrays when no read minimizer is repeated.
    """
    h, _ = sequence_minimizers(seq, idx["k"], idx["w"])
    h = np.unique(h)
    i = np.searchsorted(idx["keys"], h)
    inside = i < len(idx["keys"])
    i = i[inside]
    i = i[idx["keys"][i] == h[inside]]
    if len(i) == 0:
        return np.zeros(0, np.uint32), np.zeros(0, np.int64), np.zeros(0)
    a, b = idx["offsets"][i], idx["offsets"][i + 1]
    sel = np.concatenate([np.arange(x, y) for x, y in zip(a, b)])
    bins, inv = np.unique(idx["bins"][sel], return_inverse=True)
    shared = np.bincount(inv, minlength=len(bins))
    copies = np.bincount(inv, weights=idx["copies"][sel], minlength=len(bins))
    ok = shared >= min_fraction * shared.max()
    return bins[ok], shared[ok], copies[ok] / shared[ok]


def bin_to_chrom(idx, bins):
    """Global bin ids -> (chromosome index, start bp)."""
    c = np.searchsorted(idx["chrom_offsets"], bins, side="right") - 1
    return c, (bins - idx["chrom_offsets"][c]).astype(np.int64) * idx["bin_size"]


# ---------------------------------------------------------------------------
# Genome codes on disk (shared by the workers through the page cache)
# ---------------------------------------------------------------------------

def write_genome_codes(genome, path):
    """Genome as one flat uint8 file of codes 0-3 (4 = N), chromosomes concatenated.
    Returns {chrom: (offset, length)}; also written to path + '.tsv'."""
    layout, off = {}, 0
    with open(path, "wb") as out:
        for name, seq in read_fasta(genome):
            out.write(LUT[np.frombuffer(seq, np.uint8)].tobytes())
            layout[name] = (off, len(seq))
            off += len(seq)
    with open(str(path) + ".tsv", "w") as fh:
        for name, (o, n) in layout.items():
            fh.write(f"{name}\t{o}\t{n}\n")
    return layout


def _kmer_codes(codes, k):
    """Forward 2-bit value of each k-mer of a code array (no hashing, N -> -1)."""
    n = len(codes) - k + 1
    if n <= 0:
        return np.zeros(0, np.int64)
    c = codes.astype(np.int64)
    bad = c == 4
    c[bad] = 0
    v = np.zeros(n, np.int64)
    for j in range(k):
        v = (v << 2) | c[j:j + n]
    if bad.any():
        v[np.convolve(bad.astype(np.int32), np.ones(k, np.int32), "valid") > 0] = -1
    return v


_INV = np.full(256, ord("N"), np.uint8)              # codes 0-3 -> ACGT, 4 -> N
for _b in b"ACGT":
    _INV[LUT[_b]] = _b


class RepeatIndex:
    """
    Repeat index + genome codes, as used by the re-attribution:
      candidates(seq) -> chromosome names, bin starts and weights of the bins sharing the read
      place(seq, chrom, bin_start, rng) -> exact 5' position and strand of the read in that bin
    """

    def __init__(self, prefix):
        prefix = str(prefix)
        self.idx = load_repeat_index(prefix)
        self.k, self.bin_size = self.idx["k"], self.idx["bin_size"]
        self.chroms = [str(c) for c in self.idx["chroms"]]
        self.genome = np.memmap(prefix + ".genome.u8", dtype=np.uint8, mode="r")
        self.layout = {}
        for line in open(prefix + ".genome.u8.tsv"):
            name, o, n = line.split()
            self.layout[name] = (int(o), int(n))
        self.anchor_k = 21

    def candidates(self, seq, min_fraction=0.8):
        """(chrom names, bin starts, weights) of the bins sharing >= min_fraction of the best
        bin's read minimizers; weight = copies of the shared minimizers in the bin."""
        bins, _, w = lookup(self.idx, seq, min_fraction)
        if len(bins) == 0:
            return [], np.zeros(0, np.int64), np.zeros(0)
        c, start = bin_to_chrom(self.idx, bins)
        return [self.chroms[i] for i in c], start, w

    def reference_segment(self, chrom, pos5, is_reverse, length):
        """Genome sequence (bytes, forward strand) of `length` bp at a read's first alignment,
        5' end pos5 with the conventions of the SAM parsing (forward: 1-based start; reverse:
        0-based exclusive end); b"" outside the sequence. Looked up in the index for the reads
        whose own sequence has no repeated minimizer (an error in a short read kills every k-mer):
        the reference at the first alignment carries the copies (mosquito mito / chr3 NUMT)."""
        if chrom not in self.layout or length <= 0:
            return b""
        off, n = self.layout[chrom]
        a = pos5 - length if is_reverse else pos5 - 1
        a, b = max(0, a), min(n, a + length)
        if b - a < self.k:
            return b""
        return bytes(_INV[np.asarray(self.genome[off + a: off + b])])

    def place(self, seq, chrom, bin_start, rng, quals=None):
        """One exact placement of the read in the bin, drawn among the tied best ones (see
        place_choices). Returns (5' position, is_reverse) or None."""
        ch = self.place_choices(seq, chrom, bin_start, quals)
        return ch[int(rng.integers(len(ch)))] if ch else None

    def place_choices(self, seq, chrom, bin_start, quals=None):
        """[(5' position, is_reverse)] of the best placements of the read in the bin (tandem
        copies: several ties), [] if no anchor."""
        return self.score_bin(seq, chrom, bin_start, quals)[1]

    def score_bin(self, seq, chrom, bin_start, quals=None, min_votes=0.5, anchor=None):
        """
        Best local alignment of the read in [bin_start, bin_start + bin_size) (+ one read
        length on each side), without indels:
          1. exact anchor_k-mer matches vote for a diagonal (genome offset of the read start)
             on each strand; diagonals with >= min_votes x the best number of votes are kept;
          2. on each kept diagonal, the best-scoring segment of the read (maximum subarray)
             with the bowtie2 local scores: +2 per match, mismatch -(2 + 4 min(Q, 40) / 40)
             (-6 without qualities), -1 for an N;
          3. the best-scoring diagonals win (tandem copies: several ties).
        Returns (score, [(5' position with the conventions of the SAM parsing, is_reverse)]);
        (None, []) if no anchor. Score ~ bowtie2 AS of an ungapped alignment.
        anchor: another sequence of the same length (the reference at the read's first alignment)
        whose k-mers give the diagonals, the read itself being scored on them (a short read with
        one error in the middle has no exact anchor_k-mer).
        """
        off, n = self.layout[chrom]
        L = len(seq)
        a = max(0, bin_start - L)
        b = min(n, bin_start + self.bin_size + L)
        ref_codes = np.asarray(self.genome[off + a: off + b])
        ref = _kmer_codes(ref_codes, self.anchor_k)
        r = LUT[np.frombuffer(seq, np.uint8)]
        rc = np.where(r < 4, 3 - r, 4)[::-1]
        if anchor is not None and len(anchor) == L:
            va = LUT[np.frombuffer(anchor, np.uint8)]
            vote_seqs = ((False, va), (True, np.where(va < 4, 3 - va, 4)[::-1]))
        else:
            vote_seqs = ((False, r), (True, rc))
        if quals is None:
            pen = np.full(L, 6)
        else:
            q = np.minimum(np.frombuffer(quals, np.uint8).astype(np.int64) - 33, 40)
            pen = 2 + (4 * q) // 40
        order = np.argsort(ref, kind="stable")
        sref = ref[order]
        cand = []                                       # (votes, diagonal, is_rev)
        for is_rev, qc in vote_seqs:
            v = _kmer_codes(qc, self.anchor_k)
            qi = np.flatnonzero(v >= 0)
            if len(qi) == 0:
                continue
            lo = np.searchsorted(sref, v[qi], "left")
            hi = np.searchsorted(sref, v[qi], "right")
            cnt = hi - lo
            if cnt.sum() == 0:
                continue
            rep = np.repeat(qi, cnt)
            pos = order[np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))]
            diag, votes = np.unique(pos - rep, return_counts=True)
            cand += [(int(x), int(d), is_rev) for d, x in zip(diag, votes)]
        if not cand:
            return None, []
        top = max(c[0] for c in cand)
        best, out = None, []
        for votes, d, is_rev in cand:
            if votes < min_votes * top:
                continue
            qc = rc if is_rev else r
            pq = pen[::-1] if is_rev else pen
            j = np.arange(L)
            g = d + j                                   # positions in ref_codes
            ok = (g >= 0) & (g < len(ref_codes))
            if not ok.any():
                continue
            j, g = j[ok], g[ok]
            rb, qb = ref_codes[g], qc[j]
            sc = np.where((rb == 4) | (qb == 4), -1, np.where(rb == qb, 2, -pq[j]))
            # maximum subarray: best S[e] - min_{s <= e} S[s] over the prefix sums
            S = np.concatenate([[0], np.cumsum(sc)])
            m = np.minimum.accumulate(S[:-1])
            gain = S[1:] - m
            e = int(np.argmax(gain))
            score = int(gain[e])
            st_ = int(np.flatnonzero(S[:e + 1] == m[e])[-1])         # segment = j[st_ .. e]
            first = a + int(g[st_])                     # 0-based start of the aligned part
            end = a + int(g[e]) + 1                     # 0-based exclusive end
            # 5' conventions of the SAM parsing: reverse = 0-based exclusive end, forward = 1-based start
            plc = (end, True) if is_rev else (first + 1, False)
            if best is None or score > best:
                best, out = score, [plc]
            elif score == best and plc not in out:
                out.append(plc)
        return best, out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("genome")
    p.add_argument("out_prefix")
    p.add_argument("--k", type=int, default=31)
    p.add_argument("--w", type=int, default=10)
    p.add_argument("--bin_size", type=int, default=2000)
    p.add_argument("--min_count", type=int, default=2)
    p.add_argument("--n_parts", type=int, default=8, help="Temporary parts (memory / 8 per doubling)")
    a = p.parse_args(argv)
    build_repeat_index(a.genome, a.out_prefix, a.k, a.w, a.bin_size, a.min_count,
                       log=lambda m: print(m, flush=True), n_parts=a.n_parts)


if __name__ == "__main__":
    sys.exit(main())
