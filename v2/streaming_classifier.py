#!/usr/bin/env python3
"""
streaming_classifier.py

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
"""

import sys
import re
import argparse
from pathlib import Path


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
      reverse (-) : reference_end        (pysam ref_end is already exclusive → no +1)
    """
    if read['is_reverse']:
        return read['ref_end']          # already 1-based exclusive = last aligned bp
    else:
        return read['ref_start'] + 1    # 0-based → 1-based


def get_strand(read):
    return '-' if read['is_reverse'] else '+'


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
             threshold_clip=30, min_aligned_frac=0.5):

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

    # --- Stream both SAM files simultaneously ---
    with open(sam1_path) as fh1, open(sam2_path) as fh2:

        iter1 = (parse_sam_line(l) for l in fh1)
        iter2 = (parse_sam_line(l) for l in fh2)

        # Skip None (header lines)
        def next_read(it):
            r = None
            while r is None:
                r = next(it, 'EOF')
                if r == 'EOF':
                    return None
                if r is None:
                    continue
            return r

        while True:
            r1 = next_read(iter1)
            r2 = next_read(iter2)

            if r1 is None or r2 is None:
                break

            stats['total_pairs'] += 1

            # ----------------------------------------------------------------
            # GROUP 0 — at least one unmapped
            # ----------------------------------------------------------------
            if r1['is_unmapped'] or r2['is_unmapped']:
                f_group0.write(r1['query_name'] + '\n')
                stats['group0'] += 1
                continue

            # ----------------------------------------------------------------
            # GROUP 1 — both uniquely mapped (no XS)
            # ----------------------------------------------------------------
            if not r1['has_xs'] and not r2['has_xs']:
                pos1    = get_outer_5prime(r1)
                pos2    = get_outer_5prime(r2)
                strand1 = get_strand(r1)
                strand2 = get_strand(r2)
                f_pairs.write(
                    f"{r1['query_name']}\t"
                    f"{r1['chrom']}\t{pos1}\t"
                    f"{r2['chrom']}\t{pos2}\t"
                    f"{strand1}\t{strand2}\n"
                )
                stats['group1'] += 1
                continue

            # ----------------------------------------------------------------
            # GROUP 2 — at least one ambiguous (XS present)
            # ----------------------------------------------------------------
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

    args = parser.parse_args()

    classify(
        sam1_path         = args.sam1,
        sam2_path         = args.sam2,
        output_dir        = args.output_dir,
        n_cpus            = args.n_cpus,
        min_chunk_bp      = args.min_chunk_bp,
        threshold_clip    = args.threshold_clip,
        min_aligned_frac  = args.min_aligned_frac,
    )


if __name__ == "__main__":
    main()
