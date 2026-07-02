import subprocess as sp
from pathlib import Path
import numpy as np

import hicberg.io as hio
from hicberg import logger


def preprocess_pairs(pairs_file : str = "rescued.pairs", threshold : int = 1000, output_dir : str = None) -> None:
    """
    Preprocess pairs file to remove pairs that are not in the same chromosome or are greater than a threshold.
    Retain columns are : chromosome, start, end, count.

    Parameters
    ----------
    pairs_file : str, optional
        Path to the pairs file, by default "rescued.pairs"
    threshold : int, optional
        Threshold distance beyond which pairs will not be kept, by default 1000
    output_dir : str, optional
        Path where the formatted pairs will be saved, by default None
    """    

    output_dir_path = Path(output_dir)
    if not output_dir_path.is_dir():
        raise IOError(f"Output directory {output_dir} not found. Please provide a valid path.")

    pairs_path = Path(output_dir, pairs_file)

    if not pairs_path.is_file():
            
        raise IOError(f"Pairs file {pairs_path} not found. Please provide a valid path.")
    
    pairs_handler = open(pairs_path, "r")

    processed_pairs_path = Path(output_dir_path , "preprocessed_pairs.pairs")

    with open(processed_pairs_path, "w") as f_out:

        for line in pairs_handler:

            if line.startswith("#"):
                continue

            read_id, chromosome_for, position_for, chromosome_rev, position_rev, strand_for, strand_rev = line.split("\t")

            if chromosome_for != chromosome_rev or np.abs(int(position_rev) - int(position_for)) < threshold:
                continue

            else: 

                if int(position_for) < int(position_rev):

                    f_out.write(f"{chromosome_for}\t{position_for}\t{position_rev}\t1\n")
                else :

                    f_out.write(f"{chromosome_for}\t{position_rev}\t{position_for}\t1\n")

    pairs_handler.close()

    logger.info(f"Formated paris saved at {processed_pairs_path}")

def format_chrom_sizes(chromosome_sizes : str = "chromosome_sizes.npy", output_dir : str = None) -> None:
    """
    Format chromosome sizes to bed and txt format.
    - bed format : chrom, start, end
    - txt format : chrom, size
    Parameters
    ----------
    chrom_sizes : str, optional
        Path to chromosome sizes file (.npy), by default "chromosome_sizes.npy"
    output_dir : str, optional
        Path where the formatted chromosome sizes will be saved, by default None

    """    
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_dir():
        raise IOError(f"Output directory {output_dir} not found. Please provide a valid path.")

    chrom_size_path = Path(output_dir, chromosome_sizes)

    if not chrom_size_path.is_file():
            
        raise IOError(f"Pairs file {chrom_size_path.name} not found. Please provide a valid path.")
    
    chrom_size = hio.load_dictionary(chrom_size_path)

    chrom_size_bed_path = Path(output_dir_path / "chromosome_sizes.bed")
    chrom_size_txt_path = Path(output_dir_path / "chromosome_sizes.txt")
    

    with open(chrom_size_bed_path, 'w') as f_out:

        for k, v in chrom_size.items():
            f_out.write(f'{k}\t0\t{v}\n')

    f_out.close()

    with open(chrom_size_txt_path, 'w') as f_out:

        for k, v in chrom_size.items():
            f_out.write(f'{k}\t{v}\n')

    f_out.close()

    logger.info(f"Formated chromosome sizes saved at {chrom_size_bed_path} and {chrom_size_txt_path}")

def get_bed_coverage(chromosome_sizes : str = "chromosome_sizes.bed", pairs_file : str = "preprocessed_pairs.pairs", output_dir : str = None) -> None:
    """
    Get bed coverage from pairs file (using bedtools).

    Parameters
    ----------
    chromosome_sizes : str, optional
        Path to chromsomes sizes files (.bed format), by default "chromosome_sizes.bed"
    pairs_file : str, optional
        Path to processed pairs files (columns : chrom, start, end, count), by default "preprocessed_pairs.pairs"
    output_dir : str, optional
        Path where the coverage (.bed) will be saved, by default None
    """    

    output_dir_path = Path(output_dir)
    if not output_dir_path.is_dir():
        raise IOError(f"Output directory {output_dir} not found. Please provide a valid path.")

    chrom_size_path = Path(output_dir, chromosome_sizes)

    pairs_path = Path(output_dir, pairs_file)

    if not chrom_size_path.is_file():
            
        raise IOError(f"Pairs file {chrom_size_path} not found. Please provide a valid path.")
    
    if not pairs_path.is_file():
                
        raise IOError(f"Pairs file {pairs_path} not found. Please provide a valid path.")
    
    bed_coverage_path = Path(output_dir_path , "coverage.bed")
    
    bedtools_cmd = f"bedtools coverage -a {str(chrom_size_path)} -b {str(pairs_path)} -d"

    with open(bed_coverage_path, "w") as f_out:

        sp.run(bedtools_cmd, shell=True, stdout=f_out)

    f_out.close()

    logger.info(f"Saved data coverage at {bed_coverage_path}")

def get_bedgraph(bed_coverage : str = "coverage.bed", output_dir : str = None) -> None:
    """
    Convert bed coverage to bedgraph format.
    Format is : chrom, start, end, count.
    Start and end are different by 1bp (end  = start + 1).

    Parameters
    ----------
    bed_coverage : str, optional
        Path to coverage (.bed), by default "coverage.bed"
    output_dir : str, optional
        Path where the coverage (.bedgraph) will be saved, by default None
    """    
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_dir():
        raise IOError(f"Output directory {output_dir} not found. Please provide a valid path.")

    bed_coverage_path = Path(output_dir, bed_coverage)

    if not bed_coverage_path.is_file():
            
        raise IOError(f"Pairs file {bed_coverage_path.name} not found. Please provide a valid path.")
    
    bed_handler = open(bed_coverage_path, "r")

    bedgraph_coverage_path = Path(output_dir_path, "coverage.bedgraph")

    with open(bedgraph_coverage_path, "w") as f_out:

        for line in bed_handler:

            chromosome, start, end, index, count = line.split("\t")

            if end == index:
                continue

            f_out.write(f"{chromosome}\t{int(index)}\t{int(index) + 1}\t{count}")

    f_out.close()
    bed_handler.close()
    
def bedgraph_to_bigwig(bedgraph_file : str = "coverage.bedgraph", chromosome_sizes : str = "chromosome_sizes.txt", output_dir : str = None) -> None:
    """
    Convert bedgraph to bigwig format.

    Parameters
    ----------
    bedgraph_file : str, optional
        Path to coverage (.bedgraph), by default "coverage.bedgraph"
    chromosome_sizes : str, optional
        Path to chromosome sizes file (chrom_id, size), by default "chromosome_sizes.txt"
    output_dir : str, optional
        [description], by default None

    Raises
    ------
    IOError
        [description]
    IOError
        [description]
    IOError
        [description]
    """    
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_dir():
        raise IOError(f"Output directory {output_dir} not found. Please provide a valid path.")

    bedgraph_coverage_path = Path(output_dir, bedgraph_file)
    if not bedgraph_coverage_path.is_file():
        raise IOError(f"Pairs file {bedgraph_coverage_path.name} not found. Please provide a valid path.")
    
    chromosome_sizes_path = Path(output_dir, chromosome_sizes)
    if not bedgraph_coverage_path.is_file():
        raise IOError(f"Pairs file {chromosome_sizes_path.name} not found. Please provide a valid path.")
    
    output_bigwig_path = Path(output_dir, "signal.bw")
    
    bedgraphtobigwig_cmd = f"bedGraphToBigWig {bedgraph_coverage_path} {chromosome_sizes_path} {output_bigwig_path}"

    sp.run([bedgraphtobigwig_cmd], shell = True)

    logger.info(f"Saved data in BigWig format at {output_bigwig_path}")
    

# alternative process (no need to install bedgraph or bedGraphToBigWig)

def concatenate_bam(bam_file1 : str = "group1.1.bam", 
                    bam_file2 : str = "group2.1.rescued.bam", 
                    cpus : int = 8, 
                    output_dir : str = None,
                    output_name : str = "groups1_2.1.bam") -> None:
    """
    Concatenate 2 bam files.

    Parameters
    ----------
    bam_file1 : str, optional
        Path to bam file 1 by default group1.1.bam
    bam_file2 : str, optional
        Path to bam file 2 by default group2.1.rescued.bam
    cpus : int, optional
        Number of cpus to use for the merging, by default 8    
    output_dir : str, optional
        Output directory, by default None
    output_name : str, optional
        Output name, by default groups1_2.1.bam

    """
    
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_dir():
        raise IOError(f"Output directory {output_dir} not found.")
    
    output_name = Path(output_dir, output_name)
    
    bam_file1 = Path(output_dir, bam_file1)
    bam_file2 = Path(output_dir, bam_file2)   
        
    cmd = f"samtools merge -@ {cpus} {output_name} {bam_file1} {bam_file2}"
    sp.run([cmd], shell = True)
        
    logger.info(f"Saved data in BigWig format at {output_dir}")           
          
     
def bam_to_bigwig(bam_file1 : str = "groups1_2.1.bam", bam_file2 : str = "groups1_2.2.bam", 
                  dist_min_omics : int = 100, dist_max_omics : int = 1000,
                  cpus : int = 8, 
                  output_dir : str = None,
                  output_name : str = "signal.bw") -> None:
    """
    Convert bam files to a bigwig file (.bw).

    Parameters
    ----------
    bam_file1 : str, optional
        Path to bam file 1 by default groups1_2.1.bam
    bam_file2 : str, optional
        Path to bam file 2 by default groups1_2.2.bam 
    dist_min_omics : int, optional
        Minimal distance between reads to keep, by default 100    
    dist_max_omics : int, optional
        Maximal distance between reads to keep, by default 1000    
    cpus : int, optional
        Number of cpus to use for the merging, by default 8    
    output_dir : str, optional
        Output directory, by default None
    output_name : str, optional
        Output name, by default signal.bw
    """    
    
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_dir():
        raise IOError(f"Output directory {output_dir} not found.")

    output_bigwig_path = Path(output_dir, output_name)
    bam_file1 = Path(output_dir, bam_file1)
    bam_file2 = Path(output_dir, bam_file2)
    
    # concatenation of both mates
    cmd = f"samtools merge -f  -@ {cpus} combined.bam {bam_file1} {bam_file2}"
    sp.run([cmd], shell = True)
    
    # sort according to read name
    cmd = f"samtools sort -N -@ {cpus} -o combined_nsorted.bam combined.bam"
    sp.run([cmd], shell = True)
    
    # fix mates
    cmd = f"samtools fixmate -m -@ {cpus} combined_nsorted.bam combined_fixmate.bam"   # in.bam  out.bam 
    sp.run([cmd], shell = True)
    
    # sort according to genomic position
    cmd = f"samtools sort -@ {cpus} -o combined_fixmate.sorted.bam combined_fixmate.bam"
    sp.run([cmd], shell = True)
    
    cmd = f"samtools index -@ {cpus} combined_fixmate.sorted.bam"
    sp.run([cmd], shell = True)
    
    # creation of the bw file     
    cmd = f"bamCoverage -b combined_fixmate.sorted.bam -o {output_bigwig_path} --binSize 1 --extendReads --normalizeUsing CPM --numberOfProcessors {cpus} --skipNonCoveredRegions"
    sp.run([cmd], shell = True)
    
    logger.info(f"Saved data in BigWig format at {output_bigwig_path}")
    
    
    

# bam_to_bigwig(bam_file1 = "/media/axel/EVO/benchmark_part2/hicberg_testing/out_SRR5399542/alignments/group1.1.bam", bam_file2  = "/media/axel/EVO/benchmark_part2/hicberg_testing/out_SRR5399542/alignments/group1.2.bam", 
#                   output_dir = "/media/axel/EVO/benchmark_part2/hicberg_testing/out_SRR5399542/alignments/",
#                   output_name  = "signal_unrescued.bw")  

    
# bam_to_bigwig(bam_file1 = "/media/axel/EVO/benchmark_part2/hicberg_testing/out_SRR5399542/alignments/groups1_2.1.bam", bam_file2  = "/media/axel/EVO/benchmark_part2/hicberg_testing/out_SRR5399542/alignments/groups1_2.2.bam", 
#                   output_dir = "/media/axel/EVO/benchmark_part2/hicberg_testing/out_SRR5399542/alignments/",
#                   output_name  = "signal_rescued.bw")     
    
    
    
   
    
    
    
    