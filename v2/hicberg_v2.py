#!/usr/bin/env python3
"""
hicberg_v2.py — HiCberg v2 pipeline.

Stages (--start_stage / --exit_stage to run a part only):
  index        bowtie2 index of the genome (skipped if --index is given or exists)
  mappability  GenMap theoretical mappability (k = read length; --no_mappability to skip)
  align        bowtie2 --very-sensitive-local --reorder on R1 and R2
  classify     streaming_classifier_v2: group0 / group1 (unrescued.pairs) / group2 chunks
  stats        statistics_v2: P(s), coverage, density + checkpoint plots
               → stop here (--exit_stage stats) to check the laws, then --start_stage rescue
  rescue       reattribute_v2: group2 re-aligned (-k / -a), pairs drawn → rescued.pairs
  matrices     optional hicstuff filtering, cooler matrices (ICE), before / after plots

Example (yeast):
  python3 hicberg_v2.py genome.fa R1.fq.gz R2.fq.gz -o out --cpus 16 \\
      --circular plasmid2micron,chrMT --max_alignments all --hicstuff --enzyme DpnII,HinfI
"""

import argparse
import gzip
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

STAGES = ("index", "mappability", "align", "classify", "stats", "rescue", "matrices")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fasta_sizes(genome):
    """{chrom: length} from a FASTA (plain or gzipped), in file order."""
    sizes, name = {}, None
    opener = gzip.open if str(genome).endswith(".gz") else open
    with opener(genome, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
                sizes[name] = 0
            elif name is not None:
                sizes[name] += len(line.strip())
    return sizes


def read_length(fastq, n=2000):
    """Longest read among the first n reads of a FASTQ (plain or gzipped)."""
    opener = gzip.open if str(fastq).endswith(".gz") else open
    longest = 0
    with opener(fastq, "rt") as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                longest = max(longest, len(line.strip()))
            if i >= 4 * n:
                break
    return longest


def run_stage(name, args):
    return STAGES.index(args.start_stage) <= STAGES.index(name) <= STAGES.index(args.exit_stage)


def main():
    p = argparse.ArgumentParser(description="HiCberg v2: rescue multi-mapping reads in Hi-C / Micro-C data.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("genome", help="Reference genome (FASTA)")
    p.add_argument("fastq1", help="R1 reads (FASTQ, may be gzipped)")
    p.add_argument("fastq2", help="R2 reads")
    p.add_argument("-o", "--output_dir", required=True)
    p.add_argument("--index", help="Existing bowtie2 index prefix (else built in output_dir/index)")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--start_stage", choices=STAGES, default=STAGES[0])
    p.add_argument("--exit_stage", choices=STAGES, default=STAGES[-1])
    p.add_argument("--circular", default="", help="Circular chromosomes, e.g. plasmid2micron,chrMT")
    p.add_argument("--bin_size", type=int, default=2000)
    # mappability
    p.add_argument("--no_mappability", action="store_true", help="Do not use GenMap (trapezoid areas)")
    p.add_argument("--genmap_errors", type=int, default=2)
    # statistics
    p.add_argument("--density_n_min", default="200", help="Density threshold, or 'auto'")
    # rescue
    p.add_argument("--mode", choices=("random", "ps", "standard", "full"), default="full")
    p.add_argument("--max_alignments", default="100", help="bowtie2 -k value, or 'all' (-a)")
    p.add_argument("--error_rate", type=float, default=0.01)
    p.add_argument("--as_delta", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    # matrices
    p.add_argument("--hicstuff", action="store_true",
                   help="Filter non-informative events and PCR duplicates with hicstuff")
    p.add_argument("--enzyme", default=None, help="Restriction enzyme(s) of the Hi-C, e.g. DpnII,HinfI "
                                                  "(a number or 'mnase' for Micro-C); used by hicstuff "
                                                  "and for the sites-per-bin track of the plots")
    p.add_argument("--intra_only", action="store_true",
                   help="Before / after plots for intra-chromosomal maps only (default if > 30 chromosomes)")
    p.add_argument("--keep_sam", action="store_true", help="Keep the SAM files of the first alignment")
    args = p.parse_args()

    if args.hicstuff and not args.enzyme:
        p.error("--hicstuff needs --enzyme")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirs = {k: out / k for k in ("index", "mappability", "alignments", "classifier",
                                 "statistics", "rescue", "matrices")}
    t_start = time.time()

    # chromosome sizes from the FASTA (used by every stage)
    chrom_sizes_file = out / "chrom.sizes"
    if not chrom_sizes_file.is_file():
        sizes = fasta_sizes(args.genome)
        chrom_sizes_file.write_text("".join(f"{c}\t{L}\n" for c, L in sizes.items()))
    sizes = {l.split()[0]: int(l.split()[1]) for l in chrom_sizes_file.read_text().splitlines()}
    log(f"{len(sizes)} chromosomes, {sum(sizes.values()):,} bp")

    # --- index ---
    index = args.index or str(dirs["index"] / "genome")
    if run_stage("index", args) and not args.index and not Path(index + ".1.bt2").is_file() \
            and not Path(index + ".1.bt2l").is_file():
        dirs["index"].mkdir(exist_ok=True)
        log("Building the bowtie2 index ...")
        subprocess.run(["bowtie2-build", "--threads", str(args.cpus), args.genome, index],
                       check=True, stdout=subprocess.DEVNULL)

    # --- mappability ---
    bedgraph = None
    if not args.no_mappability:
        k = read_length(args.fastq1)
        bedgraph = dirs["mappability"] / f"genmap_K{k}_E{args.genmap_errors}.bedgraph"
        if run_stage("mappability", args) and not bedgraph.is_file():
            import mappability_v2 as mp
            log(f"GenMap mappability (k = {k}, {args.genmap_errors} errors) ...")
            bedgraph = mp.run_genmap(args.genome, dirs["mappability"], k, args.genmap_errors, args.cpus)

    # --- align ---
    sam1, sam2 = dirs["alignments"] / "1.sam", dirs["alignments"] / "2.sam"
    if run_stage("align", args):
        dirs["alignments"].mkdir(exist_ok=True)
        for fq, sam in ((args.fastq1, sam1), (args.fastq2, sam2)):
            log(f"bowtie2 {Path(fq).name} ...")
            with open(str(sam) + ".log", "w") as lg:
                subprocess.run(["bowtie2", "--very-sensitive-local", "--reorder", "-p", str(args.cpus),
                                "-x", index, "-U", str(fq), "-S", str(sam)], check=True, stderr=lg)

    # --- classify ---
    n_chunks = max(1, args.cpus // 4)
    if run_stage("classify", args):
        import streaming_classifier_v2 as sc
        log("Classifying the pairs ...")
        sc.classify(sam1, sam2, dirs["classifier"], n_cpus=n_chunks)
        if not args.keep_sam:
            sam1.unlink(missing_ok=True)
            sam2.unlink(missing_ok=True)

    # --- stats ---
    if run_stage("stats", args):
        import statistics_v2 as st
        log("Statistical laws ...")
        st.generate_statistics(dirs["classifier"] / "unrescued.pairs", chrom_sizes_file,
                               dirs["statistics"], bin_size=args.bin_size, circular=args.circular,
                               mappability=bedgraph,
                               density_n_min=args.density_n_min if args.density_n_min == "auto"
                               else int(args.density_n_min))
        if args.exit_stage == "stats":
            log(f"Checkpoint: check {dirs['statistics'] / 'plots'}, then run again with "
                f"--start_stage rescue")

    # --- rescue ---
    if run_stage("rescue", args):
        import reattribute_v2 as ra
        log("Re-attribution of the ambiguous pairs ...")
        ra.reattribute(dirs["classifier"], dirs["statistics"], dirs["rescue"], index, args.mode,
                       args.max_alignments, args.as_delta, n_workers=n_chunks,
                       threads=max(1, args.cpus // n_chunks), bin_size=args.bin_size,
                       seed=args.seed, error_rate=args.error_rate)

    # --- matrices ---
    if run_stage("matrices", args):
        import matrices_v2 as mx
        dirs["matrices"].mkdir(exist_ok=True)
        unrescued, rescued = dirs["classifier"] / "unrescued.pairs", dirs["rescue"] / "rescued.pairs"
        suffix = ""
        if args.hicstuff:
            log("hicstuff filtering (non-informative events, PCR duplicates) ...")
            unrescued, thresholds = mx.hicstuff_filter(unrescued, args.genome, args.enzyme,
                                                       dirs["matrices"], args.cpus, "unrescued")
            log(f"  thresholds estimated on unrescued: uncuts {thresholds[0]}, loops {thresholds[1]}"
                " → applied to rescued too")
            rescued, _ = mx.hicstuff_filter(rescued, args.genome, args.enzyme, dirs["matrices"],
                                            args.cpus, "rescued", thresholds=thresholds)
            suffix = ".filtered"
        log("Cooler matrices (ICE balancing) ...")
        cool_u = mx.build_cool(unrescued, sizes, args.bin_size,
                               dirs["matrices"] / f"unrescued{suffix}.cool", mad_max=5, nproc=args.cpus)
        cool_r = mx.build_cool(rescued, sizes, args.bin_size,
                               dirs["matrices"] / f"rescued{suffix}.cool", mad_max=1000, nproc=args.cpus)
        log("Before / after plots ...")
        mapp_file = dirs["statistics"] / "mappability.npy"
        mapp = np.load(mapp_file, allow_pickle=True).item() if mapp_file.is_file() else None
        gc = mx.gc_content(args.genome, sizes, args.bin_size)
        sites = mx.restriction_sites(args.genome, sizes, args.bin_size, args.enzyme) \
            if args.enzyme else None
        mx.plot_before_after(cool_u, cool_r, dirs["matrices"] / "plots", mappability=mapp, gc=gc,
                             sites=sites, enzyme_label=f"{args.enzyme} sites" if args.enzyme else "",
                             intra_only=args.intra_only or len(sizes) > 30,
                             labels=(f"unrescued{suffix}", f"rescued{suffix}"))

    log(f"Done in {(time.time() - t_start) / 60:.1f} min — {out}")


if __name__ == "__main__":
    main()
