import time
import glob, sys, shutil
from glob import glob
from shutil import which, rmtree
from os.path import join
from pathlib import Path
import subprocess as sp
import multiprocessing
from multiprocessing import Process
from functools import partial
import logging
import numpy as np

import hicberg.align as hal
import hicberg.io as hio
import hicberg.utils as hut
import hicberg.plot as hpl
import hicberg.statistics as hst
import hicberg.omics as hom
from hicberg import logger

UNRESCUED_MATRIX = "unrescued.pairs.filtered.cool"
RESTRICTION_MAP = "restriction_map.npy"

def check_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None

def pipeline(
        name : str = "sample", 
        start_stage : str = "fastq",
        exit_stage : str = "None", 
        genome : str = None, 
        index = None,
        fq_for : str = None, 
        fq_rev : str = None, 
        sensitivity : str = "very-sensitive-local",
        max_alignment : int = None, 
        trim5 : int = 0,
        mapq : int = 30, 
        enzyme  : list[str] = ["DpnII", "HinfI"],
        circular : str = "", 
        rate : float = 1.0, 
        distance : int = 1000, 
        bin_size : int = 2000, 
        nb_chunks : int = 1,
        mode : str = "standard",
        kernel_size : int = 11, 
        deviation : float = 0.5,  
        verbose : bool = False,
        cpus : int = 4, 
        output_dir : str = None, 
        force : bool = False, 
        blacklist : str = None) -> None :

    args = locals()

    if not check_tool("bowtie2"):
        logger.error("bowtie2 is not available on your system.")
        raise ValueError("bowtie2 is not available on your system.")

    if not check_tool("samtools"):
        logger.error("samtools is not available on your system.")
        raise ValueError("samtools is not available on your system.")
    
    if not check_tool('bedtools'):
        logger.error("bedtools is not available on your system, intall it\
                     if you want to use omics mode.")
        # raise ValueError("bedtools is not available on your system.")
    
    if not check_tool('bedGraphToBigWig'):
        logger.error("bedGraphToBigWig is not available on your system, intall it\
                     if you want to use omics mode.")
        # raise ValueError("bedGraphToBigWig is not available on your system.")
        
    logger.info(f"Hicberg command used : {' '.join(sys.argv)}")

    if mode not in ["random", "ps", "coverage", "density", "standard", "full", "omics", "d1d2"]:
        logger.error(f"The provided mode {mode} must be: random ps coverage density standard full omics d1d2.")
        raise IOError(f"The provided mode {mode} must be: random ps coverage density standard full omics d1d2.")   
        
    if start_stage not in ["fastq", "bam", "groups", "build", "stats", "rescue", "final"]:
        logger.error(f"The provided start_stage {start_stage} must be: fastq bam groups build stats rescue final.")
        raise IOError(f"The provided start_stage {start_stage} must be: fastq bam groups build stats rescue final.")
        
    if exit_stage not in ["None", "bam", "groups", "build", "stats", "rescue", "final"]:
        logger.error(f"The provided exit_stage {exit_stage} must be: None bam groups build stats rescue final.")
        raise IOError(f"The provided exit_stage {exit_stage} must be: None bam groups build stats rescue final.")
            
    if fq_for == fq_rev :
        logger.error(f"The two provided inputs {fq_for} and {fq_rev} files must be different.")
        raise IOError(f"The two provided inputs {fq_for} and {fq_rev} files must be different.")
        
    logger.info("Start Hicberg pipeline, welcome everyone :)")

    stages = {"fastq": 0, "bam": 1, "groups": 2, "build": 3, "stats": 4, "rescue": 5, "final": 6}
    out_stage = {"None": None,  "bam": 1, "groups": 2, "build": 3, "stats": 4, "rescue": 5, "final": 6}
    
    start_stage_n = stages[start_stage]  # start_stage as variable of command line - default to "fastq" -> 0
    exit_stage_n  = out_stage[exit_stage]
    
    # Keep track of the arguments used
    for arg in args:
        logger.info("%s: %s", arg, args[arg])

    # Check if the output directory exists
    output_folder = Path(output_dir, name).as_posix()

    # Reformat blacklisted genomic regions if provided
    if blacklist is not None:
        blacklist = hut.format_blacklist(blacklist = blacklist)
        print(f"reformatted blacklist : {blacklist}")

    if start_stage_n  < 1 : 
        output_folder = hio.create_folder(sample_name = name, output_dir = output_dir, force = force)

        hut.get_chromosomes_sizes(genome = genome, output_dir = output_folder)
        hut.get_bin_table(bin_size = bin_size, output_dir = output_folder)

        if index is None:
            index = hal.hic_build_index(genome = genome, output_dir = output_folder, cpus = cpus, verbose = verbose)

        hal.hic_align(index = index, fq_for = fq_for, fq_rev = fq_rev, 
                      sensitivity = sensitivity, 
                      max_alignment = max_alignment, 
                      trim5 = trim5, 
                      output_dir = output_folder, 
                      cpus = cpus, 
                      verbose = True)
        hal.hic_view(cpus = cpus, output_dir = output_folder, verbose = True)  # convertion into bam
        hal.hic_sort(cpus = cpus, output_dir = output_folder, verbose = True)  # sort of bam files

    if exit_stage_n  == 1:
        logger.info(f"Ending Hicberg pipeline at {exit_stage}")
        return
    
    if start_stage_n  < 2:
        logger.info("Starting reads classification.")
        hut.classify_reads(mapq = mapq, output_dir = output_folder)

    if exit_stage_n  == 2:
        logger.info(f"Ending HiCBERG pipeline at {exit_stage}")
        return
    
    if start_stage_n  < 3:
        hio.build_pairs(output_dir = output_folder)
        hio.build_matrix(pairs='unrescued.pairs', cpus = cpus, balance = True, output_dir = output_folder)
        hio.hicstuff_process(pairs='unrescued.pairs', cpus = cpus, enzyme=enzyme, genome = genome, output_dir = output_folder)    # hicstuff process 
        hio.build_matrix(pairs='unrescued.pairs.filtered', cpus = cpus, balance = True, mad_max=5, output_dir = output_folder)

    if exit_stage_n  == 3:
        logger.info(f"Ending Hicberg pipeline at {exit_stage}")
        return
    
    if start_stage_n < 4:
        restriction_map = hst.get_restriction_map(genome = genome, enzyme = enzyme, output_dir = output_folder)
        # hst.get_dist_frags(genome = genome, restriction_map = restriction_map, circular = circular, rate = rate, output_dir = output_folder)
        hst.log_bin_genome(genome = genome, output_dir = output_folder)
        
        p1 = Process(target = hst.generate_intra_ps, kwargs = dict(circular = circular, blacklist = blacklist, output_dir = output_folder))
        p2 = Process(target = hst.generate_trans_ps, kwargs = dict(output_dir = output_folder))
        p3 = Process(target = hst.generate_coverages, kwargs = dict(genome = genome, bin_size = bin_size, output_dir = output_folder))
        
        for process in [p1, p2, p3]:
            process.start()

        for process in [p1, p2, p3]:
            process.join()
        
        if mode in ["d1d2", "full"]:   
            hst.generate_density_map2(cooler_file = UNRESCUED_MATRIX, threads  = cpus, output_dir  = output_folder)
            
        if mode in ["density", "full"]:
            hst.generate_d1d2, kwargs = dict(output_dir = output_folder)
                     
    if exit_stage_n  == 4:
        logger.info(f"Ending Hicberg pipeline at {exit_stage}")
        return

    if start_stage_n  < 5:
        restriction_map = hio.load_dictionary(Path(output_folder) / RESTRICTION_MAP)
        hut.chunk_bam(nb_chunks = nb_chunks, output_dir = output_folder)   # can last a few hours

        # Get chunks as lists
        forward_chunks, reverse_chunks = hut.get_chunks(output_dir = output_folder)
        
        # Reattribute reads in parallel, the most important step of the algo
        with multiprocessing.Pool(processes=cpus) as pool:
            pool.map(
                partial(
                    hst.reattribute_reads,
                    mode=mode,
                    circular=circular,
                    restriction_map=restriction_map,
                    output_dir=output_folder
                ),
                zip(forward_chunks, reverse_chunks)
            )

        hio.merge_predictions(output_dir = output_folder, clean = True, cpus = cpus)

        # Delete chunks
        folder_to_delete = Path(output_folder) / 'chunks'
        rmtree(folder_to_delete)

        hio.build_pairs(mode = True, output_dir = output_folder)
        hio.hicstuff_process(pairs='rescued.pairs', cpus = cpus, enzyme=enzyme, genome = genome, output_dir = output_folder)
        hio.build_matrix(pairs='rescued.pairs.filtered', cpus = cpus, balance = True, mad_max=1000, output_dir = output_folder)
        
        if mode == "omics":
            hom.preprocess_pairs(pairs_file = "rescued.pairs", threshold  = distance, output_dir = output_folder)
            hom.format_chrom_sizes(chromosome_sizes = "chromosome_sizes.npy", output_dir = output_folder)
            hom.get_bed_coverage(chromosome_sizes = "chromosome_sizes.bed", pairs_file = "preprocessed_pairs.pairs", output_dir = output_folder)
            hom.get_bedgraph(bed_coverage  = "coverage.bed", output_dir  = output_folder)
            hom.bedgraph_to_bigwig(bedgraph_file = "coverage.bedgraph", chromosome_sizes = "chromosome_sizes.txt", output_dir = output_folder)
    
    if exit_stage_n  == 5:
        logger.info(f"Ending Hicberg pipeline at {exit_stage}")
        return
    
    if start_stage_n  <= 6:
        logger.info("Start plotting results")
        p1 = Process(target = hpl.plot_laws, kwargs = dict(output_dir = output_folder))
        p2 = Process(target = hpl.plot_trans_ps, kwargs = dict(output_dir = output_folder))
        p3 = Process(target = hpl.plot_coverages, kwargs = dict(bin_size = bin_size, output_dir = output_folder))
        p4 = Process(target = hpl.plot_couple_repartition, kwargs = dict(output_dir = output_folder))
        p5 = Process(target = hpl.plot_matrix, kwargs = dict(genome = genome, output_dir = output_folder))

        for process in [p1, p2, p3, p4, p5]:
            process.start()
            
        for process in [p1, p2, p3, p4, p5]:
            process.join()

        if mode in ["d1d2", "full"]:   
            hpl.plot_d1d2, kwargs = dict(output_dir = output_folder)
            
        if mode in ["density", "full"]:
            hpl.plot_density, kwargs = dict(output_dir = output_folder)

        logger.info(f"Results plotted in {output_folder}")

    if exit_stage_n == 6:
        logger.info(f"Ending Hicberg pipeline at {exit_stage}")
        return
    
    logger.info(f"Tidying : {output_folder}")
    hio.tidy_folder(output_dir = output_folder)
    logger.info("End of the Hicberg pipeline.")
     
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
    
    # copy of the log file into the output directory     
    with open('name_of_log.txt', 'r') as f:
        log_filename = f.read().strip()

    shutil.copy(log_filename, output_folder)
        

