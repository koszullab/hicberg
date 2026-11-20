"""The fast path (group0/group1) must give the same results as parse_sam_line()."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streaming_classifier_v2 as sc  # noqa: E402


def sam_line(flag, pos, cigar, tags, seq_len=150):
    seq, qual = "A" * seq_len, "I" * seq_len
    fields = ["read1", str(flag), "chr1", str(pos), "42", cigar, "*", "0", "0", seq, qual]
    return "\t".join(fields + tags) + "\n"


CASES = [
    # flag, pos, cigar, tags
    (0,  1000, "150M",            ["AS:i:-5", "XN:i:0", "YT:Z:UU"]),
    (16, 1000, "150M",            ["AS:i:-5", "YT:Z:UU"]),
    (16, 1000, "20S100M30S",      ["AS:i:180", "XS:i:170", "YT:Z:UU"]),
    (0,  1000, "40S110M",         ["AS:i:200"]),
    (16, 5000, "60M2D40M3I47M",   ["XS:i:100", "AS:i:150"]),
    (16, 5000, "50M1000N100M",    ["AS:i:150"]),
    (16, 5000, "70=1X79=",        []),                         # no tag at all
    (0,  7,    "5H145M",          ["XS:i:3"]),                 # XS as first tag
    (16, 12,   "100M50S",         ["AS:i:1", "ZS:Z:XS:i:2"]),  # 'XS:' inside another tag value
]


@pytest.mark.parametrize("flag, pos, cigar, tags", CASES)
def test_fast_path_matches_full_parse(flag, pos, cigar, tags):
    line = sam_line(flag, pos, cigar, tags)
    full = sc.parse_sam_line(line)
    fields = sc.split_sam(line)

    assert fields[0] == full["query_name"] and fields[2] == full["chrom"]
    assert sc.has_xs_tag(fields[11]) == full["has_xs"]
    is_rev = bool(int(fields[1]) & 16)
    assert is_rev == full["is_reverse"]
    assert sc.fast_outer_5prime(fields, is_rev) == sc.get_outer_5prime(full)


def test_split_sam_skips_header_and_malformed():
    assert sc.split_sam("@SQ\tSN:chr1\tLN:1000\n") is None
    assert sc.split_sam("too\tfew\tfields\n") is None
    assert sc.split_sam(sam_line(0, 1, "150M", []))[11] == ""


def test_reference_length():
    assert sc.reference_length("36M") == 36
    assert sc.reference_length("20S100M30S") == 100
    assert sc.reference_length("60M2D40M3I47M") == 149
    assert sc.reference_length("50M1000N100M") == 1150
    assert sc.reference_length("70=1X79=") == 150


def test_all_reads_coverage_counts_5prime_of_mapped_reads(tmp_path):
    cov = sc.AllReadsCoverage({"chr1": 1050}, resolution=100)
    for flag, pos, cigar in [(0, 1, "150M"),        # forward, 5' = 1     → bin 0
                             (16, 1, "150M"),       # reverse, 5' = 150   → bin 1
                             (0, 1001, "50M"),      # forward, 5' = 1001  → bin 10 (last, 50 bp wide)
                             (4, 500, "*")]:        # unmapped → ignored
        cov.add(sc.split_sam(sam_line(flag, pos, cigar, ["XS:i:5"])), flag)
    path = tmp_path / "cov.npz"
    cov.save(path)
    z = np.load(path)
    assert int(z["resolution"]) == 100 and len(z["chr1"]) == 11
    assert z["chr1"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_sam_chrom_sizes(tmp_path):
    sam = tmp_path / "x.sam"
    sam.write_text("@HD\tVN:1.0\n@SQ\tSN:chrI\tLN:230490\n@SQ\tSN:chrII\tLN:822297\n"
                   + sam_line(0, 1, "150M", []))
    assert sc.sam_chrom_sizes(sam) == {"chrI": 230490, "chrII": 822297}


def test_original_orientation():
    assert sc.original_orientation("AACGT", "ABCDE", False) == ("AACGT", "ABCDE")
    assert sc.original_orientation("AACGT", "ABCDE", True) == ("ACGTT", "EDCBA")


def test_group2_fastq_keeps_the_sequenced_orientation(tmp_path):
    # pair 1: R1 ambiguous on the reverse strand, R2 unique forward.
    # SAM SEQ of a reverse read is the reverse complement of the sequenced read:
    # the FASTQ must hold the read as sequenced, so that the re-alignment
    # finds it on the reverse strand again.
    sequenced = "ACGTTGCAAGGCTTACCGATAGCTAGGCTTAACGATCGATCGGATCCA"     # 48 bp
    sam_seq = sequenced.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    qual = "".join(chr(33 + 10 + k % 30) for k in range(48))
    head = "@HD\tVN:1.0\n@SQ\tSN:chr1\tLN:100000\n"
    r1 = "\t".join(["p1", "16", "chr1", "500", "1", "48M", "*", "0", "0", sam_seq, qual[::-1],
                    "AS:i:96", "XS:i:96"]) + "\n"
    r2 = "\t".join(["p1", "0", "chr1", "9000", "42", "48M", "*", "0", "0", "A" * 48, "I" * 48,
                    "AS:i:96"]) + "\n"
    (tmp_path / "1.sam").write_text(head + r1)
    (tmp_path / "2.sam").write_text(head + r2)
    sc.classify(tmp_path / "1.sam", tmp_path / "2.sam", tmp_path / "out", n_cpus=1)
    fq1 = (tmp_path / "out" / "group2_1_chunk0.fastq").read_text().splitlines()
    fq2 = (tmp_path / "out" / "group2_2_chunk0.fastq").read_text().splitlines()
    assert fq1[1] == sequenced and fq1[3] == qual
    assert fq2[1] == "A" * 48


def test_group2_pcr_duplicates_removed_before_rescue(tmp_path):
    head = "@HD\tVN:1.0\n@SQ\tSN:chr1\tLN:100000\n"
    seqA, seqB = "ACGT" * 12, "TTGCA" * 9 + "ACG"
    def line(name, seq, flag=0, pos=500, xs=True):
        tags = ["AS:i:96", "XS:i:96"] if xs else ["AS:i:96"]
        return "\t".join([name, str(flag), "chr1", str(pos), "1", f"{len(seq)}M", "*", "0", "0",
                          seq, "I" * len(seq)] + tags) + "\n"
    r1 = [line("p1", seqA), line("p2", seqA), line("p3", seqA)]          # p2 = PCR duplicate of p1
    r2 = [line("p1", seqB, pos=9000, xs=False), line("p2", seqB, pos=9000, xs=False),
          line("p3", "G" * 48, pos=9000, xs=False)]                       # p3: different R2
    (tmp_path / "1.sam").write_text(head + "".join(r1))
    (tmp_path / "2.sam").write_text(head + "".join(r2))
    sc.classify(tmp_path / "1.sam", tmp_path / "2.sam", tmp_path / "out", n_cpus=1)
    names = (tmp_path / "out" / "group2_1_chunk0.fastq").read_text().splitlines()[::4]
    assert names == ["@p1", "@p3"]
    assert "group2 dup. : 1" in (tmp_path / "out" / "stats.txt").read_text()
    sc.classify(tmp_path / "1.sam", tmp_path / "2.sam", tmp_path / "keep", n_cpus=1, dedup=False)
    assert len((tmp_path / "keep" / "group2_1_chunk0.fastq").read_text().splitlines()) == 12


def test_is_unique_mapq():
    f = sc.split_sam(sam_line(0, 100, "50M", ["AS:i:-5", "XS:i:-30"]))
    f[4] = "42"
    assert sc.is_unique(f, 30) and not sc.is_unique(f, None)
    f[4] = "1"
    assert not sc.is_unique(f, 30)
    g = sc.split_sam(sam_line(0, 100, "50M", ["AS:i:-5"]))
    g[4] = "0"
    assert sc.is_unique(g, 30)                    # no XS: unique whatever the MAPQ


def test_group1_pcr_duplicates_removed_on_sequences(tmp_path):
    head = "@HD\tVN:1.0\n@SQ\tSN:chr1\tLN:100000\n"
    seqA, seqB = "ACGT" * 12, "TTGCA" * 9 + "ACG"
    def line(name, seq, flag=0, pos=500):
        return "\t".join([name, str(flag), "chr1", str(pos), "42", f"{len(seq)}M", "*", "0", "0",
                          seq, "I" * len(seq), "AS:i:96"]) + "\n"
    rc = lambda x: x.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    # u1, u2: same molecule (u2 seen on the other strand in the SAM: same sequenced reads)
    r1 = [line("u1", seqA), line("u2", seqA), line("u3", seqA)]
    r2 = [line("u1", rc(seqB), 16, 9000), line("u2", rc(seqB), 16, 9000), line("u3", "G" * 48, 0, 9000)]
    (tmp_path / "1.sam").write_text(head + "".join(r1))
    (tmp_path / "2.sam").write_text(head + "".join(r2))
    sc.classify(tmp_path / "1.sam", tmp_path / "2.sam", tmp_path / "out", n_cpus=1)
    rows = [l.split("\t")[0] for l in (tmp_path / "out" / "unrescued.pairs").read_text().splitlines()
            if not l.startswith("#")]
    assert sorted(rows) == ["u1", "u3"]          # file sorted by the coordinate dedup
    assert "group1 dup. : 1" in (tmp_path / "out" / "stats.txt").read_text()


def test_dedup_coordinates_tolerance(tmp_path):
    p = tmp_path / "u.pairs"
    rows = ["## pairs format v1.0\n", "#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n"]
    # a jackpot: 5 copies within 2 bp; an unrelated pair 10 bp away; another strand; another chrom
    for k, (a, b) in enumerate([(1000, 5000), (1001, 5000), (1000, 5002), (999, 4999), (1002, 5001)]):
        rows.append(f"j{k}\tchrA\t{a}\tchrB\t{b}\t+\t+\n")
    rows += ["x\tchrA\t1012\tchrB\t5000\t+\t+\n", "y\tchrA\t1000\tchrB\t5000\t-\t+\n",
             "z\tchrA\t1000\tchrC\t5000\t+\t+\n"]
    p.write_text("".join(rows))
    n_in, n_dup = sc.dedup_coordinates(p, tol=3)
    assert (n_in, n_dup) == (8, 4)
    kept = [l for l in p.read_text().splitlines() if not l.startswith("#")]
    assert len(kept) == 4 and p.read_text().startswith("## pairs format")


def test_swapped_mates_are_duplicates(tmp_path):
    """The two strands of one ligation product give the same pair with R1 and R2 swapped."""
    head = "@HD\tVN:1.0\n@SQ\tSN:chr1\tLN:100000\n"
    seqA, seqB = "ACGT" * 12, "TTGCA" * 9 + "ACG"
    def line(name, seq, pos):
        return "\t".join([name, "0", "chr1", str(pos), "42", f"{len(seq)}M", "*", "0", "0",
                          seq, "I" * len(seq), "AS:i:96"]) + "\n"
    # u2 = u1 with R1 / R2 swapped (sequences and positions)
    r1 = [line("u1", seqA, 500), line("u2", seqB, 9000)]
    r2 = [line("u1", seqB, 9000), line("u2", seqA, 500)]
    (tmp_path / "1.sam").write_text(head + "".join(r1))
    (tmp_path / "2.sam").write_text(head + "".join(r2))
    sc.classify(tmp_path / "1.sam", tmp_path / "2.sam", tmp_path / "out", n_cpus=1)
    rows = [l for l in (tmp_path / "out" / "unrescued.pairs").read_text().splitlines()
            if not l.startswith("#")]
    assert len(rows) == 1


def test_dedup_coordinates_swapped_mates(tmp_path):
    p = tmp_path / "u.pairs"
    p.write_text("## pairs format v1.0\n#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n"
                 "a\tchr1\t100\tchr2\t500\t+\t-\n"
                 "b\tchr2\t501\tchr1\t102\t-\t+\n"      # a swapped, within 2 bp → duplicate
                 "c\tchr2\t500\tchr1\t100\t+\t-\n"      # swapped but other strands → kept
                 "d\tchr1\t300\tchr1\t200\t+\t+\n"
                 "e\tchr1\t201\tchr1\t299\t+\t+\n")     # d swapped → duplicate
    assert sc.dedup_coordinates(p, tol=3) == (5, 2)
    kept = sorted(l.split("\t")[0] for l in p.read_text().splitlines() if not l.startswith("#"))
    assert kept[0] == "a" and "c" in kept and len(kept) == 3
    assert "chr2\t500\tchr1\t100" in p.read_text()             # lines written unchanged


def test_insert_length_adapter():
    assert sc.insert_length("ACGT" * 9) == 36                                   # no adapter
    assert sc.insert_length("GATCGGAAGAGCACACGTCTGAACTCCAGTCACCG") == 0          # adapter dimer
    assert sc.insert_length("TTTTTTTTTTAGATCGGAAGAGCACACGT") == 10              # 10 bp insert + AGATCGG...
    assert sc.insert_length("C" * 69 + "AGATCGGAAGAGCACACGTCTGAACTCCAGTC") == 69


def test_adapter_reads_go_to_group0(tmp_path):
    head = "@HD\tVN:1.0\n@SQ\tSN:chr1\tLN:100000\n"
    def line(name, seq, pos):
        return "\t".join([name, "0", "chr1", str(pos), "42", f"{len(seq)}M", "*", "0", "0",
                          seq, "I" * len(seq), "AS:i:60"]) + "\n"
    adapter = "GATCGGAAGAGCACACGTCTGAACTCCAGTCACCG"
    long_insert = "TTGCA" * 6 + "AGATCGGAAG"                                      # 30 bp insert: kept
    r1 = [line("a", adapter, 500), line("b", long_insert, 900)]
    r2 = [line("a", "ACGTA" * 7, 9000), line("b", "CCGTA" * 7, 9500)]
    (tmp_path / "1.sam").write_text(head + "".join(r1))
    (tmp_path / "2.sam").write_text(head + "".join(r2))
    sc.classify(tmp_path / "1.sam", tmp_path / "2.sam", tmp_path / "out", n_cpus=1)
    stats = (tmp_path / "out" / "stats.txt").read_text()
    assert "adapter     : 1" in stats and "group1      : 1" in stats
