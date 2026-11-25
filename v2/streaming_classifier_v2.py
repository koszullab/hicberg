#!/usr/bin/env python3
"""
streaming_classifier_v2.py

Classifies read pairs from two SAM files (bowtie2 --very-sensitive-local --reorder)
into 3 groups:
  - group0 : at least one read is unmapped
  - group1 : both reads are uniquely mapped (no XS tag)
  - group2 : at least one read is ambiguous (XS tag present)

Outputs:
  - group0_names.txt         : read names
  - unrescued.pairs          : pairtools format (group1)
  - group2_1_chunk{i}.fastq  : group2 R1 split by n_cpus
  - group2_2_chunk{i}.fastq  : group2 R2 split by n_cpus
  - group2_hits_chunk{i}.tsv : best alignment of each group2 read, same order as the FASTQ
                              (per mate: chrom, 5' position, strand, AS, unique, NM)
                               (name, then for R1 and R2: chrom, 5' position, strand, AS,
                               unique 0/1), used by the index-only re-attribution
  - chimeric_names.txt       : read names of chimeric reads
  - stats.txt                : summary statistics
  - coverage_all_reads.npz   : (option) 5' positions of all aligned reads (incl. ambiguous,
                               at bowtie2's random best position), 100-bp bins
"""

import os
import sys
import re
import shlex
import argparse
import subprocess
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CIGAR utilities
# ---------------------------------------------------------------------------

def parse_cigar(cigar_str):
    """Parse CIGAR string into list of (op, length) tuples."""
    if cigar_str == '*':
        return []
    return [(op, int(l)) for l, op in re.findall(r'(\d+)([MIDNSHP=X])', cigar_str)]


def get_largest_M_chunk(cigar_tuples, sequence, qualities, min_bp=30):
    """
    Aligned part of a chimeric read: the query between the leading and trailing soft-clips
    (insertions and deletions inside it kept: an indel is not a ligation junction; before
    2026-11-24 only the longest M block was kept, so 60M1I89M lost 61 bp).
    Returns (seq, quals), or (None, None) if fewer than min_bp aligned (M) bases.
    """
    if not cigar_tuples:
        return None, None
    start = cigar_tuples[0][1] if cigar_tuples[0][0] == 'S' else 0
    end_clip = cigar_tuples[-1][1] if len(cigar_tuples) > 1 and cigar_tuples[-1][0] == 'S' else 0
    end = len(sequence) - end_clip
    if sum(l for op, l in cigar_tuples if op == 'M') < min_bp or end <= start:
        return None, None
    return sequence[start:end], qualities[start:end]


def is_chimeric(cigar_tuples, query_length, threshold_clip=30, min_aligned_frac=0.5):
    """
    A read is chimeric if a soft-clip at EITHER end of the alignment is longer than
    threshold_clip, OR if its aligned fraction is below min_aligned_frac.
    Both ends: the CIGAR follows the reference strand (its last operation is the 3' end of a
    forward read but the 5' end of a reverse one), and the ligation junction can be on either
    side of the aligned piece (local mode keeps the longer one). Before 2026-11-24 only the last
    operation was tested (human, clip > 30 bp: read 3' end 16 %, 5' end 11 % on both strands;
    tested 16 % of the forward reads, 11 % of the reverse ones).
    """
    if not cigar_tuples:
        return False

    aligned_bp = sum(l for op, l in cigar_tuples if op == 'M')
    clipped = any(op == 'S' and l > threshold_clip
                  for op, l in (cigar_tuples[0], cigar_tuples[-1]))
    low_frac = (aligned_bp / query_length) < min_aligned_frac

    return clipped or low_frac


# ---------------------------------------------------------------------------
# SAM parsing
# ---------------------------------------------------------------------------

def parse_sam_line(line):
    """
    Parse a SAM line into a lightweight dict.
    Returns None for header lines.
    """
    if line.startswith('@'):
        return None

    fields = line.rstrip('\n').split('\t')
    if len(fields) < 11:
        return None

    flag    = int(fields[1])
    is_unmapped = bool(flag & 4)
    is_reverse  = bool(flag & 16)

    # Tags
    tags = {}
    for t in fields[11:]:
        parts = t.split(':', 2)
        if len(parts) == 3:
            tags[parts[0]] = parts[2]

    # CIGAR
    cigar_str    = fields[5]
    cigar_tuples = parse_cigar(cigar_str)

    # Positions (0-based internally, converted on output)
    try:
        ref_start = int(fields[3]) - 1  # SAM is 1-based → 0-based
    except ValueError:
        ref_start = 0

    # reference_end : ref_start + sum of M/D/N/=/X lengths
    ref_end = ref_start
    for op, length in cigar_tuples:
        if op in ('M', 'D', 'N', '=', 'X'):
            ref_end += length

    sequence  = fields[9]
    qualities = fields[10]

    return {
        'query_name':  fields[0],
        'flag':        flag,
        'chrom':       fields[2],
        'ref_start':   ref_start,   # 0-based
        'ref_end':     ref_end,     # 0-based exclusive (like pysam)
        'mapq':        int(fields[4]),
        'cigar_str':   cigar_str,
        'cigar_tuples': cigar_tuples,
        'sequence':    sequence,
        'qualities':   qualities,
        'is_unmapped': is_unmapped,
        'is_reverse':  is_reverse,
        'has_xs':      'XS' in tags,
    }


def get_outer_5prime(read):
    """
    1-based outer-most 5' position following pairtools convention:
      forward (+) : reference_start + 1
      reverse (-) : reference_end        (0-based exclusive = 1-based inclusive of last aligned base → no +1)
    """
    if read['is_reverse']:
        return read['ref_end']          # 0-based exclusive = 1-based position of last aligned base
    else:
        return read['ref_start'] + 1    # 0-based → 1-based


def get_strand(read):
    return '-' if read['is_reverse'] else '+'


# ---------------------------------------------------------------------------
# Fast path (group0 / group1): minimal parsing
# ---------------------------------------------------------------------------
# Only ~15% of pairs (group2) need the full parse_sam_line() (sequence,
# qualities, CIGAR tuples for chimera detection). For the others, the line is
# split once (tags kept as a single string) and XS is found by substring search.

_REF_CONSUMING = re.compile(r'(\d+)[MDN=X]')
_AS_TAG = re.compile(r"(?:^|\t)AS:i:(-?\d+)")


def split_sam(line):
    """
    Split a SAM alignment line into its 11 mandatory fields + the raw tag
    string (not split). Returns None for header or malformed lines.
    """
    if line.startswith('@'):
        return None
    fields = line.rstrip('\n').split('\t', 11)
    if len(fields) < 11:
        return None
    if len(fields) == 11:
        fields.append('')
    return fields


def has_xs_tag(tags):
    """XS tag present in the raw tag string (tags are tab-separated)."""
    return tags.startswith('XS:') or '\tXS:' in tags


ADAPTER_CORE = "GATCGGAAGAGC"     # Illumina TruSeq / Nextera-free universal adapter, "AGATCGGAAGAGC"


def insert_length(seq, core=ADAPTER_CORE):
    """
    Length of the genomic insert of a read (sequenced orientation): the read
    length, or the position where the Illumina adapter starts (read-through of a
    short fragment). The adapter is AGATCGGAAGAGC...; its first A may be missing
    (adapter dimers), so the core GATCGGAAGAGC is searched and the A before it,
    if any, counted in the adapter.
    """
    i = seq.find(core)
    if i < 0:
        return len(seq)
    return i - 1 if i > 0 and seq[i - 1] == "A" else i


def is_unique(fields, min_mapq=30):
    """
    Read considered uniquely placed: no XS tag (no second alignment), or
    MAPQ >= min_mapq although an XS exists (bowtie2: the second alignment is
    clearly worse; the usual filter of hicstuff and of most Hi-C pipelines).
    min_mapq None: XS criterion only (previous behaviour).
    """
    if not has_xs_tag(fields[11]):
        return True
    return min_mapq is not None and int(fields[4]) >= min_mapq


_NM_TAG = re.compile(r"(?:^|\t)NM:i:(\d+)")


class UniqueVerifier:
    """
    Second look at the reads unique by MAPQ only (XS present, MAPQ >= min_mapq) that align
    poorly (NM >= min_nm): in the large satellites bowtie2 does not extend all the seeds of a
    read and may report a diverged copy whose seeds are rare, with a high MAPQ, while the true
    copy was never examined (human CHM13 Hi-C: chr16 52.18 Mb, 3,151 reads, NM 3.5; chr14 8.27 Mb,
    16,801 reads, NM 5.0; for 47 % of such reads the repeat index finds a better alignment
    elsewhere). Genome-wide this concerns ~4 % of the unique-by-MAPQ reads, 0.2 % of which have a
    better alignment elsewhere. The read's repeated minimizers give its candidate bins (repeat
    index); if one of them (other than the bin of its alignment) scores >= AS - tol (score_bin:
    ungapped local score, bowtie2 scale; tol 6 = one mismatch), the read is ambiguous.
    """

    def __init__(self, repeat_index, min_nm=4, max_bins=10, tol=6, min_fraction=0.3):
        from repeat_index_v2 import RepeatIndex
        self.ri = RepeatIndex(repeat_index) if not hasattr(repeat_index, "score_bin") else repeat_index
        self.min_nm, self.max_bins, self.tol, self.min_fraction = min_nm, max_bins, tol, min_fraction
        self.checked = self.requalified = 0

    def __call__(self, fields):
        """True if the read stays unique."""
        m = _NM_TAG.search(fields[11])
        if m is None or int(m.group(1)) < self.min_nm:
            return True
        a = _AS_TAG.search(fields[11])
        if a is None:
            return True
        self.checked += 1
        seq, qual = fields[9].encode(), fields[10].encode()
        chroms, starts, w = self.ri.candidates(seq, self.min_fraction)
        if len(chroms) == 0:
            return True
        B = self.ri.bin_size
        own_c, own_p = fields[2], int(fields[3])
        need = int(a.group(1)) - self.tol
        for k in np.argsort(-np.asarray(w), kind="stable")[:self.max_bins]:
            c, b = chroms[k], int(starts[k])
            if c == own_c and b - B <= own_p < b + 2 * B:   # its own bin (or a neighbour)
                continue
            sc, _ = self.ri.score_bin(seq, c, b, qual)
            if sc is not None and sc >= need:
                self.requalified += 1
                return False
        return True


def verified_unique(fields, min_mapq=30, verifier=None, requalified=None):
    """is_unique, then for the reads unique by MAPQ only: not in `requalified` (names found by
    find_requalified, the parallel pre-pass), or accepted by the verifier (UniqueVerifier)."""
    if not has_xs_tag(fields[11]):
        return True
    if min_mapq is None or int(fields[4]) < min_mapq:
        return False
    if requalified is not None:
        return fields[0] not in requalified
    return True if verifier is None else verifier(fields)


_WORKER_VERIFIER = None


def _init_verifier(repeat_index, min_nm, max_bins):
    global _WORKER_VERIFIER
    _WORKER_VERIFIER = UniqueVerifier(repeat_index, min_nm=min_nm, max_bins=max_bins)


def _verify_batch(lines):
    """Worker: names of the reads of a batch that become ambiguous, and the number checked."""
    out = []
    for line in lines:
        f = split_sam(line)
        if f is not None and not _WORKER_VERIFIER(f):
            out.append(f[0])
    return out, len(lines)


def find_requalified(sam_path, repeat_index, n_workers, min_mapq=30, min_nm=4, max_bins=10,
                     batch=500):
    """
    Parallel pre-pass of UniqueVerifier over one SAM file: the primary alignments unique by MAPQ
    only (XS present, MAPQ >= min_mapq) with NM >= min_nm are selected by cheap string tests and
    checked by a pool of n_workers processes (the repeat index is memory-mapped, shared).
    Human (2 x 242 M reads, ~2 % of the alignments checked): ~1 h with 12 workers, vs ~5 h in the
    single-process classifier. Returns (set of read names to treat as ambiguous, number checked).
    """
    import multiprocessing as mp

    def batches():
        cur = []
        with open(sam_path) as fh:
            for line in fh:
                if line[0] == "@" or "\tXS:i:" not in line:
                    continue
                f = line.split("\t", 5)
                if int(f[1]) & 2308 or int(f[4]) < min_mapq:         # unmapped / secondary / supplementary
                    continue
                m = _NM_TAG.search(line)
                if m is None or int(m.group(1)) < min_nm:
                    continue
                cur.append(line)
                if len(cur) >= batch:
                    yield cur
                    cur = []
        if cur:
            yield cur

    names, checked = set(), 0
    with mp.Pool(n_workers, initializer=_init_verifier,
                 initargs=(str(repeat_index), min_nm, max_bins)) as pool:
        for out, n in pool.imap_unordered(_verify_batch, batches(), chunksize=1):
            names.update(out)
            checked += n
    return names, checked





def reference_length(cigar_str):
    """Number of reference bases covered by the alignment (M/D/N/=/X)."""
    if cigar_str[:-1].isdigit() and cigar_str[-1] == 'M':   # e.g. '36M'
        return int(cigar_str[:-1])
    return sum(int(n) for n in _REF_CONSUMING.findall(cigar_str))


def fast_outer_5prime(fields, is_reverse):
    """
    Same as get_outer_5prime() from the raw fields:
      forward : POS (already 1-based = reference_start + 1)
      reverse : POS - 1 + reference length (= 0-based exclusive ref_end)
    """
    try:
        pos = int(fields[3])
    except ValueError:
        pos = 1   # same fallback as parse_sam_line (ref_start = 0)
    if is_reverse:
        return pos - 1 + reference_length(fields[5])
    return pos


# ---------------------------------------------------------------------------
# Coverage of all aligned reads
# ---------------------------------------------------------------------------

COVERAGE_ALL = "coverage_all_reads.npz"


def sam_chrom_sizes(sam_path):
    """{chrom: length} from the @SQ lines of a SAM header."""
    sizes = {}
    with open(sam_path) as fh:
        for line in fh:
            if not line.startswith('@'):
                break
            if line.startswith('@SQ'):
                tags = dict(t.split(':', 1) for t in line.rstrip('\n').split('\t')[1:])
                sizes[tags['SN']] = int(tags['LN'])
    return sizes


class AllReadsCoverage:
    """
    5' positions of ALL aligned reads (groups 1 and 2, and the mapped mate of
    group 0), binned at `resolution` bp. A multi-mapping read is counted at the
    position bowtie2 reported, chosen at random among its best alignments:
    on average each copy of a repeat gets its share, which estimates the
    coverage of regions without uniquely mappable reads.
    """

    def __init__(self, chrom_sizes, resolution=100, flush_every=1_000_000):
        self.res = resolution
        self.names = list(chrom_sizes)
        self.n_bins = {c: -(-L // resolution) for c, L in chrom_sizes.items()}
        offsets = np.concatenate([[0], np.cumsum(list(self.n_bins.values()))[:-1]])
        self.offset = dict(zip(self.names, offsets.tolist()))
        self._where = {c: (self.offset[c], self.n_bins[c] - 1) for c in self.names}
        self.counts = np.zeros(int(sum(self.n_bins.values())), dtype=np.int64)
        self.buffer = []
        self.flush_every = flush_every

    def add(self, fields, flag):
        """Count a read; returns its 5' position (None if unmapped), to be reused."""
        if flag & 4:
            return None
        pos = fast_outer_5prime(fields, bool(flag & 16))
        where = self._where.get(fields[2])
        if where is not None:
            b = (pos - 1) // self.res
            self.buffer.append(where[0] + (b if 0 <= b <= where[1] else min(max(b, 0), where[1])))
            if len(self.buffer) >= self.flush_every:
                self.flush()
        return pos

    def flush(self):
        if self.buffer:
            self.counts += np.bincount(self.buffer, minlength=len(self.counts))
            self.buffer = []

    def save(self, path):
        self.flush()
        np.savez_compressed(path, resolution=self.res, **{
            c: self.counts[self.offset[c]:self.offset[c] + self.n_bins[c]] for c in self.names})


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def encode_quals(qualities_str):
    """Return quality string as-is (already ASCII Phred+33 in SAM)."""
    return qualities_str


_REVCOMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def original_orientation(sequence, qualities, is_reverse):
    """
    Read as sequenced. For a read aligned on the reverse strand, SAM SEQ / QUAL
    are reverse-complemented / reversed: they must be put back before writing
    the group2 FASTQ, else the re-alignment sees every read on the + strand
    (strand lost, reverse reads moved to the wrong 5' end; on full AC1 this
    turned the rescued +- / -+ pairs into ++: 5.3 M ++ vs 0.16 M --, and the
    hicstuff "weirds" of rescued x7).
    """
    if not is_reverse:
        return sequence, qualities
    return sequence.translate(_REVCOMP)[::-1], qualities[::-1]


def write_fastq_record(fh, query_name, sequence, qualities):
    fh.write(f"@{query_name}\n{sequence}\n+\n{qualities}\n")


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify(sam1_path, sam2_path, output_dir, n_cpus, min_chunk_bp=30,
             threshold_clip=30, min_aligned_frac=0.5, coverage_all_reads=False,
             dedup=True, min_mapq=30, dedup_tol=3, min_insert=20, verify_unique=None,
             verify_workers=1, exclude_chroms=()):
    """
    coverage_all_reads: also count the 5' positions of all aligned reads
    (coverage_all_reads.npz, ~+25% time), used by statistics_v2.py
    --coverage_all_reads to estimate the coverage of unmappable bins.
    dedup: PCR duplicates removed on the sequences, the same way for the
    unique (group1) and the ambiguous (group2) pairs: one pair kept per
    identical (R1, R2) sequence pair (in the sequenced orientation; reads
    with sequencing errors are not merged). Needed before the
    re-attribution: the copies of a duplicated ambiguous pair are otherwise
    drawn independently, land at different positions and are no longer seen
    as duplicates on their coordinates (mosquito: 46.6 % of the group2 pairs
    were exact duplicates, 57 % of the extra copies ended at another
    position). Memory: one 64-bit hash per mapped pair kept (Python set,
    ~70 bytes per pair: ~3 GB for 40 M pairs).
    min_mapq: a read with an XS tag but MAPQ >= min_mapq is unique (group1
    if its mate is too); None = XS only.
    min_insert: a read whose Illumina adapter starts before this position
    (insert < min_insert bp: adapter dimers, tiny fragments) is treated as
    unmapped (group0): such reads align uniquely to adapter pieces left in some
    assemblies (mosquito chr3 35,436,681: 1,246 false contacts with the whole
    genome). Longer inserts are kept (local alignment clips the adapter). 0 = off.
    verify_unique: repeat index prefix (or a RepeatIndex): the reads unique by MAPQ only but
    poorly aligned are checked against the other copies (UniqueVerifier); None = off (default:
    it only fixes 2 alignment attractors in the human genome). verify_workers > 1: parallel
    pre-pass find_requalified on both SAM files (human ~1 h with 12 workers, vs ~5 h inline).
    exclude_chroms: pairs with a read aligned to one of these sequences are dropped (counted in
    group0, 'excluded'): e.g. chrY of a female sample.
    dedup_tol: then, the unique pairs are also deduplicated on their
    coordinates with this tolerance (dedup_coordinates, as pairtools dedup);
    < 0 = off.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Open outputs ---
    f_group0 = open(output_dir / "group0_names.txt", 'w')
    f_chimeric = open(output_dir / "chimeric_names.txt", 'w')

    # unrescued.pairs header (pairtools format)
    f_pairs = open(output_dir / "unrescued.pairs", 'w')
    # 4DN pairs format (chr1 / chr2 column names and #chromsize lines are
    # required by pairtools / hicstuff)
    f_pairs.write("## pairs format v1.0\n")
    for chrom, length in sam_chrom_sizes(sam1_path).items():
        f_pairs.write(f"#chromsize: {chrom} {length}\n")
    f_pairs.write("#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n")

    # group2 FASTQ chunks
    g2_r1_files = [
        open(output_dir / f"group2_1_chunk{i}.fastq", 'w')
        for i in range(n_cpus)
    ]
    g2_r2_files = [
        open(output_dir / f"group2_2_chunk{i}.fastq", 'w')
        for i in range(n_cpus)
    ]
    g2_hits = [open(output_dir / f"group2_hits_chunk{i}.tsv", 'w') for i in range(n_cpus)]

    # --- Counters ---
    stats = {
        'total_pairs':    0,
        'group0':         0,
        'group1':         0,
        'group2':         0,
        'chimeric_r1':    0,
        'chimeric_r2':    0,
        'dup_group1':     0,
        'dup_group2':     0,
        'adapter':        0,
        'excluded':       0,
        'chunk_index':    0,   # round-robin counter for splitting
    }
    seen = set()             # hashes of the (R1, R2) sequences of the pairs kept

    cov_all = AllReadsCoverage(sam_chrom_sizes(sam1_path)) if coverage_all_reads else None
    excl = set(exclude_chroms or ())
    verifier = req1 = req2 = None
    n_checked = n_requalified = 0
    if verify_unique is not None and verify_workers > 1 and min_mapq is not None:
        import time as _t
        t0 = _t.time()
        req1, c1 = find_requalified(sam1_path, verify_unique, verify_workers, min_mapq)
        req2, c2 = find_requalified(sam2_path, verify_unique, verify_workers, min_mapq)
        n_checked, n_requalified = c1 + c2, len(req1) + len(req2)
        print(f"  unique-by-MAPQ reads checked against the repeat index: {n_checked:,}, "
              f"{n_requalified:,} ambiguous ({_t.time() - t0:.0f} s, {verify_workers} workers)", flush=True)
    elif verify_unique is not None:
        verifier = UniqueVerifier(verify_unique)

    # --- Stream both SAM files simultaneously ---
    with open(sam1_path) as fh1, open(sam2_path) as fh2:

        # Alignment lines only (header / malformed lines skipped), as
        # (raw line, split fields) — fields split once, tags left unsplit
        def alignments(fh):
            for line in fh:
                fields = split_sam(line)
                if fields is not None:
                    yield line, fields

        iter1, iter2 = alignments(fh1), alignments(fh2)

        while True:
            a1 = next(iter1, None)
            a2 = next(iter2, None)

            if a1 is None or a2 is None:
                break

            (line1, f1), (line2, f2) = a1, a2
            flag1, flag2 = int(f1[1]), int(f2[1])
            stats['total_pairs'] += 1
            if cov_all is not None:
                pos1, pos2 = cov_all.add(f1, flag1), cov_all.add(f2, flag2)

            # ----------------------------------------------------------------
            # GROUP 0 — at least one unmapped
            # ----------------------------------------------------------------
            if (flag1 & 4) or (flag2 & 4):
                f_group0.write(f1[0] + '\n')
                stats['group0'] += 1
                continue

            # pair touching an excluded sequence (e.g. chrY of a female sample): dropped
            if excl and (f1[2] in excl or f2[2] in excl):
                f_group0.write(f1[0] + '\n')
                stats['group0'] += 1
                stats['excluded'] += 1
                continue

            # adapter read-through: insert too short to be placed → unmapped
            if min_insert:
                s1 = f1[9].translate(_REVCOMP)[::-1] if flag1 & 16 else f1[9]
                s2 = f2[9].translate(_REVCOMP)[::-1] if flag2 & 16 else f2[9]
                if insert_length(s1) < min_insert or insert_length(s2) < min_insert:
                    f_group0.write(f1[0] + '\n')
                    stats['group0'] += 1
                    stats['adapter'] += 1
                    continue

            u1 = verified_unique(f1, min_mapq, verifier, req1)
            u2 = verified_unique(f2, min_mapq, verifier, req2)
            unique = u1 and u2

            # ----------------------------------------------------------------
            # PCR duplicates: identical (R1, R2) sequences, sequenced orientation;
            # the pair is unordered (the two strands of one ligation product give
            # the same pair with R1 and R2 swapped, as pairtools dedup flips them)
            # ----------------------------------------------------------------
            if dedup:
                s1 = f1[9].translate(_REVCOMP)[::-1] if flag1 & 16 else f1[9]
                s2 = f2[9].translate(_REVCOMP)[::-1] if flag2 & 16 else f2[9]
                key = hash((s1, s2) if s1 <= s2 else (s2, s1))
                if key in seen:
                    stats['dup_group1' if unique else 'dup_group2'] += 1
                    continue
                seen.add(key)

            # ----------------------------------------------------------------
            # GROUP 1 — both uniquely mapped (no XS, or MAPQ >= min_mapq)
            # ----------------------------------------------------------------
            if unique:
                rev1, rev2 = bool(flag1 & 16), bool(flag2 & 16)
                if cov_all is None:
                    pos1, pos2 = fast_outer_5prime(f1, rev1), fast_outer_5prime(f2, rev2)
                f_pairs.write(
                    f"{f1[0]}\t"
                    f"{f1[2]}\t{pos1}\t"
                    f"{f2[2]}\t{pos2}\t"
                    f"{'-' if rev1 else '+'}\t{'-' if rev2 else '+'}\n"
                )
                stats['group1'] += 1
                continue

            # ----------------------------------------------------------------
            # GROUP 2 — at least one ambiguous (XS present): full parsing
            # ----------------------------------------------------------------
            r1, r2 = parse_sam_line(line1), parse_sam_line(line2)
            stats['group2'] += 1

            # Process r1
            seq1  = r1['sequence']
            qual1 = r1['qualities']
            chim1 = False

            if r1['has_xs'] and is_chimeric(
                r1['cigar_tuples'], len(seq1), threshold_clip, min_aligned_frac
            ):
                best_seq, best_quals = get_largest_M_chunk(
                    r1['cigar_tuples'], seq1, qual1, min_chunk_bp
                )
                if best_seq is not None:
                    seq1  = best_seq
                    qual1 = best_quals
                    chim1 = True

            # Process r2
            seq2  = r2['sequence']
            qual2 = r2['qualities']
            chim2 = False

            if r2['has_xs'] and is_chimeric(
                r2['cigar_tuples'], len(seq2), threshold_clip, min_aligned_frac
            ):
                best_seq, best_quals = get_largest_M_chunk(
                    r2['cigar_tuples'], seq2, qual2, min_chunk_bp
                )
                if best_seq is not None:
                    seq2  = best_seq
                    qual2 = best_quals
                    chim2 = True

            # Log chimeric read names
            if chim1 or chim2:
                f_chimeric.write(r1['query_name'] + '\n')
                if chim1:
                    stats['chimeric_r1'] += 1
                if chim2:
                    stats['chimeric_r2'] += 1

            # Write to group2 FASTQ chunks, reads in their sequenced orientation
            seq1, qual1 = original_orientation(seq1, qual1, r1['is_reverse'])
            seq2, qual2 = original_orientation(seq2, qual2, r2['is_reverse'])
            chunk_idx = stats['chunk_index'] % n_cpus
            stats['chunk_index'] += 1
            write_fastq_record(g2_r1_files[chunk_idx], r1['query_name'], seq1, qual1)
            write_fastq_record(g2_r2_files[chunk_idx], r2['query_name'], seq2, qual2)
            hits = []
            for f, uq in ((f1, u1), (f2, u2)):
                rev = bool(int(f[1]) & 16)
                m = _AS_TAG.search(f[11])
                nm = _NM_TAG.search(f[11])
                hits.append(f"{f[2]}\t{fast_outer_5prime(f, rev)}\t{'-' if rev else '+'}\t"
                            f"{m.group(1) if m else 0}\t{int(uq)}\t{nm.group(1) if nm else 0}")
            g2_hits[chunk_idx].write(f"{r1['query_name']}\t" + "\t".join(hits) + "\n")

    if cov_all is not None:
        cov_all.save(output_dir / COVERAGE_ALL)

    # --- Close all files ---
    f_group0.close()
    f_chimeric.close()
    f_pairs.close()
    for fh in g2_r1_files + g2_r2_files + g2_hits:
        fh.close()

    # --- PCR duplicates of the unique pairs on their coordinates (± dedup_tol bp) ---
    if dedup and dedup_tol is not None and dedup_tol >= 0:
        _, stats['dup_coord'] = dedup_coordinates(output_dir / "unrescued.pairs", dedup_tol,
                                                  tmp_dir=output_dir)
        stats['group1'] -= stats['dup_coord']

    # --- Write stats ---
    with open(output_dir / "stats.txt", 'w') as fs:
        fs.write(f"total_pairs : {stats['total_pairs']}\n")
        fs.write(f"group0      : {stats['group0']}  "
                 f"({100*stats['group0']/max(stats['total_pairs'],1):.1f}%)\n")
        fs.write(f"group1      : {stats['group1']}  "
                 f"({100*stats['group1']/max(stats['total_pairs'],1):.1f}%)\n")
        fs.write(f"group2      : {stats['group2']}  "
                 f"({100*stats['group2']/max(stats['total_pairs'],1):.1f}%)\n")
        if dedup:
            for g in ("group1", "group2"):
                d = stats[f"dup_{g}"]
                fs.write(f"{g} dup. : {d}  ({100*d/max(d + stats[g], 1):.1f}% of the {g} pairs: "
                         f"identical R1 + R2 sequences, removed)\n")
        if 'dup_coord' in stats:
            fs.write(f"group1 coord. dup. : {stats['dup_coord']}  (same chromosomes and strands, "
                     f"both positions within {dedup_tol} bp of a kept pair, removed)\n")
        if min_insert:
            fs.write(f"adapter     : {stats['adapter']}  (a read with < {min_insert} bp before the "
                     f"Illumina adapter: counted in group0)\n")
        if exclude_chroms:
            fs.write(f"excluded    : {stats['excluded']}  (a read aligned to {', '.join(sorted(excl))}: "
                     f"counted in group0)\n")
        fs.write(f"chimeric_r1 : {stats['chimeric_r1']}\n")
        fs.write(f"chimeric_r2 : {stats['chimeric_r2']}\n")
        if verifier is not None:
            n_checked, n_requalified = verifier.checked, verifier.requalified
        if verify_unique is not None:
            fs.write(f"unique by MAPQ, NM >= 4, checked against the repeat index : "
                     f"{n_checked} ; requalified ambiguous (a copy as good elsewhere) : "
                     f"{n_requalified}\n")

    # Print stats to stdout
    print("\n=== Classification stats ===")
    with open(output_dir / "stats.txt") as fs:
        print(fs.read())


def dedup_coordinates(pairs_path, tol=3, tmp_dir=None):
    """
    Remove the pairs whose two 5' positions are both within `tol` bp of an earlier
    pair with the same chromosomes and strands (as pairtools dedup --max-mismatch,
    default 3). Catches the PCR "jackpots" missed by the sequence dedup: copies of one
    molecule that differ by sequencing errors and by 1-4 bp of local-alignment clipping
    (mosquito: 104 copies of one chr1 x chr3 pair, at 5 positions within 4 bp, made a
    density hotspot). The pair is unordered: its two ends are compared in a canonical
    order (smaller chromosome / position first), so that a copy with R1 and R2 swapped
    is found (as pairtools flips the pairs; mosquito: 12 % of the unique pairs were such
    swapped copies). The lines themselves are written unchanged. The file is sorted
    externally (GNU sort, any size) and rewritten in place. Returns (pairs read, pairs
    removed).
    """
    pairs_path = Path(pairs_path)
    header, n_in, n_dup = [], 0, 0
    with open(pairs_path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            header.append(line)
    tmp = pairs_path.with_suffix(".sorted.tmp")
    env = dict(os.environ, LC_ALL="C")
    # canonical ends prepended (6 columns), sorted on them, stripped when written
    flip = ("awk 'BEGIN{FS=OFS=\"\\t\"} /^#/{next} "
            "{if ($2 < $4 || ($2 == $4 && $3+0 <= $5+0)) print $2,$3,$6,$4,$5,$7,$0; "
            "else print $4,$5,$7,$2,$3,$6,$0}' " + shlex.quote(str(pairs_path)))
    cmd = ["sort", "-t", "\t", "-k1,1", "-k4,4", "-k3,3", "-k6,6", "-k2,2n", "-k5,5n",
           "-S", "2G", "-o", str(tmp)]
    if tmp_dir:
        cmd += ["-T", str(tmp_dir)]
    src = subprocess.Popen(["bash", "-c", flip], stdout=subprocess.PIPE, env=env)
    subprocess.run(cmd, stdin=src.stdout, check=True, env=env)
    if src.wait() != 0:
        raise RuntimeError(f"canonical ordering of {pairs_path} failed")
    out_path = pairs_path.with_suffix(".dedup.tmp")
    with open(tmp) as fh, open(out_path, "w") as out:
        out.writelines(header)
        group, window = None, []
        for line in fh:
            f = line.split("\t", 6)
            key = (f[0], f[3], f[2], f[5])
            p1, p2 = int(f[1]), int(f[4])
            n_in += 1
            if key != group:
                group, window = key, []
            window = [w for w in window if p1 - w[0] <= tol]
            if any(abs(p2 - w[1]) <= tol for w in window):
                n_dup += 1
                continue
            window.append((p1, p2))
            out.write(f[6])
    tmp.unlink()
    out_path.replace(pairs_path)
    return n_in, n_dup


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classify Hi-C/Micro-C read pairs from two SAM files."
    )
    parser.add_argument("sam1",       help="SAM file for read 1")
    parser.add_argument("sam2",       help="SAM file for read 2")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "--n_cpus", type=int, default=1,
        help="Number of CPUs (splits group2 FASTQ into n chunks, default=1)"
    )
    parser.add_argument(
        "--min_chunk_bp", type=int, default=30,
        help="Minimum aligned bp to keep a chimeric chunk (default=30)"
    )
    parser.add_argument(
        "--threshold_clip", type=int, default=30,
        help="Soft-clip length threshold to call a read chimeric (default=30)"
    )
    parser.add_argument(
        "--min_aligned_frac", type=float, default=0.5,
        help="Minimum aligned fraction to call a read chimeric (default=0.5)"
    )
    parser.add_argument(
        "--coverage_all_reads", action="store_true",
        help="Also save the 5' positions of all aligned reads (incl. ambiguous) in "
             "100-bp bins (coverage_all_reads.npz, ~+25%% time), for "
             "statistics_v2.py --coverage_all_reads"
    )

    parser.add_argument(
        "--min_mapq", type=int, default=30,
        help="A read with an XS tag but MAPQ >= this value is unique (default=30; "
             "-1 = XS criterion only)"
    )
    parser.add_argument("--exclude_chroms", default="",
                        help="Comma-separated sequences: pairs with a read aligned to them are dropped")
    parser.add_argument("--verify_workers", type=int, default=1,
                        help="Processes of the --verify_unique check (> 1: parallel pre-pass)")
    parser.add_argument("--verify_unique", default=None,
                        help="Repeat index prefix (repeat_index_v2.py): reads unique by MAPQ only but "
                             "poorly aligned (NM >= 4) become ambiguous if a copy as good exists elsewhere")
    args = parser.parse_args()

    classify(
        sam1_path         = args.sam1,
        sam2_path         = args.sam2,
        output_dir        = args.output_dir,
        n_cpus            = args.n_cpus,
        min_chunk_bp      = args.min_chunk_bp,
        threshold_clip    = args.threshold_clip,
        min_aligned_frac  = args.min_aligned_frac,
        coverage_all_reads = args.coverage_all_reads,
        min_mapq          = None if args.min_mapq < 0 else args.min_mapq,
        verify_unique     = args.verify_unique,
        verify_workers    = args.verify_workers,
        exclude_chroms    = [x for x in args.exclude_chroms.split(",") if x],
    )


if __name__ == "__main__":
    main()
