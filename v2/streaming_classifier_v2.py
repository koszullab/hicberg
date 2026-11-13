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
  - chimeric_names.txt       : read names of chimeric reads
  - stats.txt                : summary statistics
  - coverage_all_reads.npz   : (option) 5' positions of all aligned reads (incl. ambiguous,
                               at bowtie2's random best position), 100-bp bins
"""

import sys
import re
import argparse
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
    Extract the largest M block from a chimeric read.
    Returns (seq, quals) of the largest M block, or (None, None) if none >= min_bp.
    """
    query_pos = 0
    best_seq   = None
    best_quals = None
    best_len   = 0

    for op, length in cigar_tuples:
        if op == 'M':
            if length > best_len:
                best_len   = length
                best_seq   = sequence[query_pos:query_pos + length]
                best_quals = qualities[query_pos:query_pos + length]
            query_pos += length
        elif op in ('S', 'I'):
            query_pos += length
        # D, N, H do not consume query

    if best_len < min_bp:
        return None, None

    return best_seq, best_quals


def is_chimeric(cigar_tuples, query_length, threshold_clip=30, min_aligned_frac=0.5):
    """
    A read is chimeric if it has a significant 3' soft-clip
    OR if its aligned fraction is below min_aligned_frac.
    """
    if not cigar_tuples:
        return False

    aligned_bp   = sum(l for op, l in cigar_tuples if op == 'M')
    last_op, last_len = cigar_tuples[-1]

    has_3prime_clip = (last_op == 'S' and last_len > threshold_clip)
    low_frac        = (aligned_bp / query_length) < min_aligned_frac

    return has_3prime_clip or low_frac


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


def write_fastq_record(fh, query_name, sequence, qualities):
    fh.write(f"@{query_name}\n{sequence}\n+\n{qualities}\n")


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify(sam1_path, sam2_path, output_dir, n_cpus, min_chunk_bp=30,
             threshold_clip=30, min_aligned_frac=0.5, coverage_all_reads=False):
    """
    coverage_all_reads: also count the 5' positions of all aligned reads
    (coverage_all_reads.npz, ~+25% time), used by statistics_v2.py
    --coverage_all_reads to estimate the coverage of unmappable bins.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Open outputs ---
    f_group0 = open(output_dir / "group0_names.txt", 'w')
    f_chimeric = open(output_dir / "chimeric_names.txt", 'w')

    # unrescued.pairs header (pairtools format)
    f_pairs = open(output_dir / "unrescued.pairs", 'w')
    f_pairs.write("## pairs format v1.0\n")
    f_pairs.write("#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2\n")

    # group2 FASTQ chunks
    g2_r1_files = [
        open(output_dir / f"group2_1_chunk{i}.fastq", 'w')
        for i in range(n_cpus)
    ]
    g2_r2_files = [
        open(output_dir / f"group2_2_chunk{i}.fastq", 'w')
        for i in range(n_cpus)
    ]

    # --- Counters ---
    stats = {
        'total_pairs':    0,
        'group0':         0,
        'group1':         0,
        'group2':         0,
        'chimeric_r1':    0,
        'chimeric_r2':    0,
        'chunk_index':    0,   # round-robin counter for splitting
    }

    cov_all = AllReadsCoverage(sam_chrom_sizes(sam1_path)) if coverage_all_reads else None

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

            # ----------------------------------------------------------------
            # GROUP 1 — both uniquely mapped (no XS)
            # ----------------------------------------------------------------
            if not has_xs_tag(f1[11]) and not has_xs_tag(f2[11]):
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
            chunk_idx = stats['chunk_index'] % n_cpus
            stats['chunk_index'] += 1

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

            # Write to group2 FASTQ chunks
            write_fastq_record(g2_r1_files[chunk_idx], r1['query_name'], seq1, qual1)
            write_fastq_record(g2_r2_files[chunk_idx], r2['query_name'], seq2, qual2)

    if cov_all is not None:
        cov_all.save(output_dir / COVERAGE_ALL)

    # --- Close all files ---
    f_group0.close()
    f_chimeric.close()
    f_pairs.close()
    for fh in g2_r1_files + g2_r2_files:
        fh.close()

    # --- Write stats ---
    with open(output_dir / "stats.txt", 'w') as fs:
        fs.write(f"total_pairs : {stats['total_pairs']}\n")
        fs.write(f"group0      : {stats['group0']}  "
                 f"({100*stats['group0']/max(stats['total_pairs'],1):.1f}%)\n")
        fs.write(f"group1      : {stats['group1']}  "
                 f"({100*stats['group1']/max(stats['total_pairs'],1):.1f}%)\n")
        fs.write(f"group2      : {stats['group2']}  "
                 f"({100*stats['group2']/max(stats['total_pairs'],1):.1f}%)\n")
        fs.write(f"chimeric_r1 : {stats['chimeric_r1']}\n")
        fs.write(f"chimeric_r2 : {stats['chimeric_r2']}\n")

    # Print stats to stdout
    print("\n=== Classification stats ===")
    with open(output_dir / "stats.txt") as fs:
        print(fs.read())


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
    )


if __name__ == "__main__":
    main()
