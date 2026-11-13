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
