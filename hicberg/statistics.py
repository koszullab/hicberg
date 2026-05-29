import time
import sys
from os import getcwd
from os.path import join
from pathlib import Path
import uuid
import multiprocessing as mp
from functools import partial

import itertools

import numpy as np
from numpy.random import choice
from scipy.spatial.distance import pdist
from scipy.ndimage import gaussian_filter
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

import pysam
from Bio import SeqIO, Restriction
from Bio.Restriction import *

import cooler
import hicberg.utils as hut
import hicberg.io as hio
from hicberg import logger
from concurrent.futures import ThreadPoolExecutor
from functools import partial


lowess = sm.nonparametric.lowess

DIST_FRAG = "dist.frag.npy"
XS = "xs.npy"
COVERAGE_DICO = "coverage.npy"
D1D2 = "d1d2.npy"
UNCUTS = "uncuts.npy"
WEIRDS = "weirds.npy"
CIRCLES = "circles.npy"
TRANS_PS = "trans_ps.npy"
RESTRICTION_MAP = "restriction_map.npy"
DENSITY_MAP = "density_map.npy"

#  General functions
def get_restriction_map(genome : str = None, enzyme : str = "DpnII", output_dir : str = None) -> dict[str, np.ndarray[int]]:
    """
    Get ordered restriction map (including 0 and n) from a chromosome sequence.
    Return a dictionary where keys are chromosomes names and values are restrictions sites positions.

    Parameters
    ----------
    genome : str, optional
        Path to the genome to digest, by default None, by default None
    enzyme : list[str], optional
        Enzyme or list of enzyme to digest the genome with. 
        If integer passed, micro-C mode using Mnase is used, and the integer correspond 
        to the size of nucleosomal fragment, by default None, by default ["DpnII","HinfI"]
    output_dir : str, optional
        Path to the folder where to save the plot, by default None

    Returns
    -------
    dict
        Dictionary of the product of digestion where keys are chromosomes names and values are restrictions sites positions.
    """

    logger.info("Generating restriction map...")
    genome_path = Path(genome)

    if not genome_path.is_file():
        raise FileNotFoundError(f"Genome file {genome} not found. Please provide a valid path to a genome file.")

    if output_dir is None:
        output_path = Path(getcwd())

    else : 
        output_path = Path(output_dir)
    
    restriction_map_dictionary = dict()

    liste_enzymes = [e.strip() for e in enzyme.split(',')]
    
    if len(liste_enzymes) == 1 and liste_enzymes[0].isnumeric():
        enzyme = int(liste_enzymes[0])
        print(f"Enabled Micro-c with cut size: {enzyme}")

        for seq_record in SeqIO.parse(genome, "fasta"):
            restriction_map = np.arange(0, len(seq_record.seq), enzyme)
            restriction_map = np.insert(
                restriction_map,
                [len(restriction_map)],
                [len(seq_record.seq)],
            )
            restriction_map_dictionary[seq_record.id] = restriction_map

    elif type(enzyme) == str or type(enzyme) == list or type(enzyme) == tuple:
        restriction_batch = Restriction.RestrictionBatch()

        for enz in liste_enzymes:
            print(enz+" added")
            restriction_batch.add(enz)

        # parse sequence from fasta file
        for seq_record in SeqIO.parse(genome, "fasta"):
            # Get restriction map from the restriction batch.
            restriction_map = restriction_batch.search(seq_record.seq)

            # Convert dictionary values to numpy array
            restriction_map_array = np.sort(
                np.array([pos for sites in restriction_map.values() for pos in sites])
            )
            restriction_map_array = np.insert(
                restriction_map_array,
                [0, len(restriction_map_array)],
                [0, len(seq_record.seq)],
            )

            restriction_map_dictionary[seq_record.id] = restriction_map_array


    np.save(output_path / RESTRICTION_MAP, restriction_map_dictionary)
    logger.info(f"Saved restriction map at : {output_path}")
    return restriction_map_dictionary

def generate_xs(chromosome_size : int, base : float = 1.1) -> np.ndarray[int]:
    """
    Generate xs array for computing P(s). Return xs array which is log space.

    Parameters
    ----------
    chromosome_size : int
        Size of the chromosome to be binned in bp.
    base : float, optional
        Base of the log space., by default 1.1

    Returns
    -------
    np.ndarray[int]
        Array of log bins related to the chromosome.
    """

    n_bins = np.divide(np.log1p(chromosome_size), np.log(base)).astype(int)
    xs = np.unique(np.logspace(0, n_bins, num=n_bins + 1, base=base, dtype=int))
    
    # Add 0 to the beginning of the array (allow distance 0 to be counted in the first bin)
    xs = np.insert(xs, 0, 0)

    return xs
    # # TODO : recent correction to fuse bins n-1 and n    (to delete ?)
    # return np.delete(xs, -2)

def log_bin_genome(genome :str, base : float = 1.1, output_dir : str = None) -> dict[str, np.ndarray[int]]:
    
    logger.info("Start log binning of genome...")
    genome_path = Path(genome)

    if not genome_path.is_file():
        raise FileNotFoundError(f"Genome file {genome} not found. Please provide a valid path to a genome file.")
    
    if output_dir is None:
        folder_path = Path(getcwd())

    else:
        folder_path = Path(output_dir)

    genome_parser = SeqIO.parse(genome, "fasta")
    xs_dict = {seq_record.id : generate_xs(chromosome_size = len(seq_record.seq), base = base) for seq_record in genome_parser}

    np.save(folder_path / XS, xs_dict)

    logger.info(f"Log binning of genome {genome} saved in {folder_path / XS}.")

def attribute_xs(xs : np.ndarray[int], distance : int) -> int:
    """
    Attibute genomic distance to the corresponding log bin of xs.

    Parameters
    ----------
    xs : np.ndarray[int]
        Array containing the log bins.
    distance : int
        Genomic distance in bp.

    Returns
    -------
    int
        Index of the corresponding bins where distance has to be attributed.
    """

    idx = np.searchsorted(xs, distance, side="right") - 1
    
    return idx if distance > 0 else 0 

# not used anymore:
def get_dist_frags(genome : str = None, restriction_map : dict = None, circular : str = "", rate : float = 1.0, output_dir : str = None) -> None:
    """
    Get the distribution of fragments' distance from a genome distribution.

    Parameters
    ----------
    genome : str, optional
        Path to the genome, by default None
    restriction_map : dict, optional
        Restriction map saved as a dictionary like chrom_name : list of restriction sites' position, by default None
    circular : str, optional
        Name of the chromosomes to consider as circular, by default ""
    rate : float, optional
        Set the proportion of restriction sites to consider. Avoid memory overflow when restriction maps are very dense, by default 1.0
    output_dir : str, optional
        Path to the folder where to save the dictionary, by default None

    Returns
    -------
    dict
        Dictionary of sub-sampled restriction map with keys as chromosome names and values as lists of restriction sites' position.
    """
    logger.info("Start generating distribution of fragments' distance...")

    if output_dir is None:
        folder_path = Path(getcwd())

    else:
       folder_path = Path(output_dir)

    if (rate <= 0.0) or (rate > 1.0):
        raise ValueError("Subsampling rate must be between 0.0 and 1.0.")

    genome_path = Path(genome)

    if not genome_path.is_file():
        raise FileNotFoundError(f"Genome file {genome} not found. Please provide a valid path to a genome file.")
    
    dist_frag = dict()
    xs = dict()
    
    if rate != 1.0:
        restriction_map = hut.subsample_restriction_map(restriction_map = restriction_map, rate = rate)

    for seq_record in SeqIO.parse(genome, "fasta"):
        seq_name = seq_record.id

        if seq_record.id in circular:
            map_size = restriction_map[seq_name].shape[0]

            forward_distances = pdist(
                np.reshape(restriction_map[seq_name], (map_size, 1))
            )
            max_size_vector = np.full(
                forward_distances.shape, np.max(restriction_map[seq_name])
            )
            backward_distances = max_size_vector - forward_distances
            pairwise_distances = np.minimum(forward_distances, backward_distances)
            pairwise_distances = np.delete(
                pairwise_distances, np.where(pairwise_distances == 0)
            )

            # freeing memory
            del forward_distances
            del backward_distances
            del max_size_vector

        else :

            pairwise_distances = pdist(
                np.reshape(
                    restriction_map[seq_name],
                    (len(restriction_map[seq_name]), 1),
                )
            )
            pairwise_distances = np.delete(
                pairwise_distances, np.where(pairwise_distances == 0)
            )

        # Computing xs
        xs[seq_name] = generate_xs(len(seq_record.seq), base=1.1)
        dist_frag[seq_name] = np.zeros(xs[seq_name].shape)
        # Parse distances
        for distance in pairwise_distances:
            dist_frag[seq_name][attribute_xs(xs[seq_name], distance)] += 1

    # Save dictionaries
    np.save(folder_path / DIST_FRAG, dist_frag)
    logger.info(f"Saved restriction map at : {folder_path / DIST_FRAG}")

#------------------------------------------------------------------------------
#  Functions to generate statistical tendancies from the unrescued part of the data 
def generate_intra_ps(forward_bam_file : str = "group1.1.bam", reverse_bam_file : str = "group1.2.bam", 
                 xs : str = "xs.npy", chrom_sizes : str = "chromosome_sizes.npy", circular : str = "", 
                 blacklist : str = None, output_dir : str = None) -> None:
    """
    Generates the different events distribution from read pairs alignment.

    Parameters
    ----------
    forward_bam_file : str, optional
        Path to forward .bam alignment file, by default None, by default group1.1.bam, by default "group1.1.bam", by default "group1.1.bam"
    reverse_bam_file : str, optional
        Path to reverse .bam alignment file, by default None, by default group1.1.bam, by default "group1.1.bam", by default "group1.2.bam"
    xs : str, optional
        Path to the dictionary containing the xs values, by default "xs.npy"
    dist_frag : str, optional
        Path to the dictionary containing the inter-fragment distances, by default "dist.frag.npy"
    circular : str, optional
        Name of the chromosomes to consider as circular, by default ""
    output_dir : str, optional
        Path to the folder where to save the dictionary, by default None, by default None, by default None
    """    

    logger.info("Start generating different types of events distribution (uncuts, circles, weirds)...")

    if output_dir is None:
        output_path = Path(getcwd())

    else:
        output_path = Path(output_dir)

    forward_bam_path = Path(output_path, forward_bam_file)
    reverse_bam_path = Path(output_path, reverse_bam_file)

    if not forward_bam_path.is_file():       
        raise FileNotFoundError(f"Forward .bam file {forward_bam_file} not found.")
    
    if not reverse_bam_path.is_file():
        raise FileNotFoundError(f"Reverse .bam file {reverse_bam_file} not found.")
    
    #Load xs
    xs = hio.load_dictionary(output_path / XS)
    # dist_frag = hio.load_dictionary(output_path / dist_frag)
    chrom_size_dict = hio.load_dictionary(output_path / chrom_sizes)

    # Create placeholders for the dictionaries
    weirds = {seq_name : np.zeros(xs.get(seq_name).shape) for seq_name in xs.keys()}
    uncuts = {seq_name : np.zeros(xs.get(seq_name).shape) for seq_name in xs.keys()}
    circles = {seq_name : np.zeros(xs.get(seq_name).shape) for seq_name in xs.keys()}

    # Create placeholder for area to divide logbins counts
    trapezoids_area = {seq_name : np.zeros(xs.get(seq_name).shape) for seq_name in xs.keys()}

    # Compute areas of trapezoids
    for chrom in xs.keys():
        xs_ = xs[chrom]
        chrom_size_ = chrom_size_dict[chrom]

        trapezoids_area[chrom] = [(2 * chrom_size_ - xs_[j+1] - xs_[j]) * (xs_[j+1] - xs_[j]) * 0.5 for j in range(len(xs_) - 1)]
        trapezoids_area[chrom].append( (((chrom_size_ - xs_[-1]) ** 2) / 2))

    forward_bam_handler, reverse_bam_handler = pysam.AlignmentFile(forward_bam_path, "rb"), pysam.AlignmentFile(reverse_bam_path, "rb")

    for forward_read, reverse_read in zip(forward_bam_handler, reverse_bam_handler):
                # TODO : Add blacklisting system to avoid counting reads from blacklisted regions
                # if not is_blacklisted(forward_read, reverse_read): --> TO BE ADDED TO UTILS
                #    continue
                # blacklisting system per coordinates or per file as a list of coordinates (BED file)
        # if  hut.is_blacklisted(read_forward = forward_read, read_reverse = reverse_read, blacklist = blacklist):
        #     print(f"Blacklisted read pair : {forward_read.query_name}")
        #     continue

        if hut.is_intra_chromosome(forward_read, reverse_read):
            
            if hut.is_uncut(forward_read, reverse_read):
                uncuts[forward_read.reference_name][
                    attribute_xs(
                        xs.get(forward_read.reference_name),
                        hut.get_cis_distance(forward_read, reverse_read, circular) + 1,
                    )
                ] += 1

            if hut.is_circle(forward_read, reverse_read):
                circles[forward_read.reference_name][
                    attribute_xs(
                        xs.get(forward_read.reference_name),
                        hut.get_cis_distance(forward_read, reverse_read, circular) + 1,
                    )
                ] += 1
                     
            if hut.is_weird(forward_read, reverse_read):
                weirds[forward_read.reference_name][
                    attribute_xs(
                        xs.get(forward_read.reference_name),
                        hut.get_cis_distance(forward_read, reverse_read, circular) + 1,
                    )
                ] += 1

    forward_bam_handler.close()
    reverse_bam_handler.close()
    
    # division by the trapezoids_area: 
    for chromosome in xs.keys():
        uncuts[chromosome] = np.divide(
            uncuts[chromosome],
            trapezoids_area.get(chromosome),
            out=np.zeros_like(uncuts[chromosome]),
            where=trapezoids_area.get(chromosome) != 0,
        )

        circles[chromosome] = np.divide(
            circles[chromosome],
            trapezoids_area.get(chromosome),
            out=np.zeros_like(circles[chromosome]),
            where=trapezoids_area.get(chromosome) != 0,
        )
       
        weirds[chromosome] = np.divide(
            weirds[chromosome],
            np.multiply(trapezoids_area.get(chromosome), 2),
            out=np.zeros_like(weirds[chromosome]),
            where=trapezoids_area.get(chromosome) != 0,
        )


    # Post-processing using fitting procedure      
    smoothed_uncuts =  {seq_name : hut.fitting_ps(xs.get(seq_name),uncuts.get(seq_name)) for seq_name in uncuts.keys()}
    smoothed_circles = {seq_name : hut.fitting_ps(xs.get(seq_name),circles.get(seq_name)) for seq_name in circles.keys()}
    smoothed_weirds =  {seq_name : hut.fitting_ps(xs.get(seq_name),weirds.get(seq_name)) for seq_name in weirds.keys()}
    
    np.save(output_path / UNCUTS, smoothed_uncuts)
    np.save(output_path / CIRCLES, smoothed_circles)
    np.save(output_path / WEIRDS, smoothed_weirds)
    
    logger.info(f"Saved {UNCUTS} {CIRCLES} and {WEIRDS} in {output_path}")

def generate_trans_ps(matrix : str = "unrescued.pairs.cool", chrom_sizes : str = "chromosome_sizes.npy", output_dir : str = None) -> None:
    """
    Generate pseudo-P(s) while considering reads originating from different chromosomes.
    Pseudo-P(s) are computed as the number of interactions between two chromosomes divided by the product of the length of these two chromosomes.

    Parameters
    ----------
    matrix : str, 
        Path to a cooler matrix, by default "unrescued.pairs.cool"
    chrom_sizes : str, optional
        Path to the chromosome sizes dictionary, by default "chromosome_sizes.npy"
    output_dir : str, optional
        Path to the folder where to save the dictionary, by default None.

    """    

    logger.info("Start generating trans-P(s).")

    if output_dir is None:
        output_path = Path(getcwd())

    else:
        output_path = Path(output_dir)

    chrom_size_dict = hio.load_dictionary(output_path / chrom_sizes)
    matrix_path = Path(output_path, matrix)

    if not matrix_path.is_file():
        raise FileNotFoundError(f"Matrix file {matrix} not found. Please provide a valid path to a matrix file.")
        
    matrix = cooler.Cooler(matrix_path.as_posix())
    chromosome_sets = itertools.product((chrom_size_dict.keys()), repeat=2)
    chromosomes_detected = matrix.chromnames
    
    trans_ps = {}
    all_interaction_matrix = np.zeros((len(chrom_size_dict.keys()) ** 2, 1))
    n_frags_matrix = np.zeros((len(chrom_size_dict.keys()) ** 2, 1))

    for idx, s in enumerate(chromosome_sets):     
        if s[0] in chromosomes_detected and s[1] in chromosomes_detected :
            all_interactions = matrix.matrix(balance=False).fetch(s[0], s[1]).sum()
        else : 
            all_interactions = 0  # we never found this chr or scaffold in the unrescued part of the data 
            
        n_frags = chrom_size_dict.get(s[0]) * chrom_size_dict.get(s[1])    #  areas of each chr in fact   
        
        trans_ps[s] = np.divide(all_interactions, np.multiply(n_frags, 4)) # Multiplied by 4 to balance 4 configurations of reads orientation (++/+-/-+/--)
        
        all_interaction_matrix[idx] = all_interactions
        n_frags_matrix[idx] = n_frags

    np.save(output_path / TRANS_PS, trans_ps)

    logger.info(f"Trans P(s) saved in {output_path}.")

def generate_coverages(genome : str = None, bin_size : int = 2000, forward_bam_file : str = "group1.1.bam", reverse_bam_file : str = "group1.2.bam", output_dir : str = None) -> None:
    """
    Take a genome and both forward and reverse bam files for unambiguous group and return a dictionary containing the coverage in terms of reads over chromosomes.

    Parameters
    ----------
    genome : str, optional
        Path to the genome file to get coverage on, by default None
    bin_size : int, optional
        Size of the desired bin, by default 2000
    forward_bam_file : str, optional
        Path to forward .bam alignment file, by default None, by default group1.1.bam
    reverse_bam_file : str, optional
        Path to reverse .bam alignment file, by default None, by default group1.2.bam
    output_dir : str, optional
        Path to the folder where to save the classified alignment files, by default None, by default None
    """        
    
    logger.info("Start generating coverages.")

    if output_dir is None:
        output_path = Path(getcwd())

    else:
        output_path = Path(output_dir)
    
    genome_path = Path(genome)

    if not genome_path.is_file():          
        raise FileNotFoundError(f"Genome file {genome} not found. Please provide a valid path to a genome file.")   
    
    genome_parser = SeqIO.parse(genome, "fasta")

    genome_coverages = {seq_record.id : np.zeros(np.round(np.divide(len(seq_record.seq), bin_size) + 1).astype(int)) for seq_record in genome_parser}

    forward_bam_path = Path(output_dir, forward_bam_file)
    reverse_bam_path = Path(output_dir, reverse_bam_file)

    if not forward_bam_path.is_file():
        raise FileNotFoundError(f"Forward .bam file {forward_bam_file} not found. Please provide a valid path to a forward .bam file.")
    
    if not reverse_bam_path.is_file():
        raise FileNotFoundError(f"Reverse .bam file {reverse_bam_file} not found. Please provide a valid path to a reverse .bam file.")
    
    forward_bam_handler, reverse_bam_handler = pysam.AlignmentFile(forward_bam_path, "rb"), pysam.AlignmentFile(reverse_bam_path, "rb")

    for forward_read, reverse_read in zip(forward_bam_handler, reverse_bam_handler):
        genome_coverages[forward_read.reference_name][np.divide(forward_read.reference_start, bin_size).astype(int)] += 1
        genome_coverages[reverse_read.reference_name][np.divide(reverse_read.reference_start, bin_size).astype(int)] += 1

    # close files
    forward_bam_handler.close()
    reverse_bam_handler.close()

    # Smooth coverages
    smoothed_coverages = {seq_name : hut.mad_smoothing(coverage) for seq_name, coverage in genome_coverages.items()}

    np.save(output_path / COVERAGE_DICO, smoothed_coverages)

    logger.info(f"Coverage dictionary saved in {output_path}")

def generate_d1d2(forward_bam_file : str = "group1.1.bam", reverse_bam_file : str = "group1.2.bam", 
                  restriction_map : str = "restriction_map.npy", output_dir : str = None) -> None:
    """
    Compute d1d2 distance law with the given alignments and restriction map.

    Parameters
    ----------
    forward_bam_file : str, optional
        Path to forward .bam alignment file, by default None, by default group1.1.bam, by default "group1.1.bam"
    reverse_bam_file : str, optional
        Path to reverse .bam alignment file, by default None, by default group1.1.bam, by default "group1.2.bam"
    restriction_map : str, optional
        Restriction map saved as a dictionary like chrom_name : list of restriction sites' position, by default "dist.frag.npy"
    output_dir : str, optional
        Path to the folder where to save the dictionary, by default None, by default None
    """    
    
    logger.info("Start generating d1d2 law...")
    
    if output_dir is None:
        output_path = Path(getcwd())

    else:
        output_path = Path(output_dir)

    forward_bam_path = Path(output_path, forward_bam_file)
    reverse_bam_path = Path(output_path, reverse_bam_file)

    if not forward_bam_path.is_file():       
        raise FileNotFoundError(f"Forward .bam file {forward_bam_file} not found. Please provide a valid path to a forward .bam file.")
    
    if not reverse_bam_path.is_file():
        raise FileNotFoundError(f"Reverse .bam file {reverse_bam_file} not found. Please provide a valid path to a reverse .bam file.")
    
    forward_bam_handler, reverse_bam_handler = pysam.AlignmentFile(forward_bam_path, "rb"), pysam.AlignmentFile(reverse_bam_path, "rb")

    # Ensure that the restriction map is a dictionary to be loaded
    try:
        restriction_map = hio.load_dictionary(output_path / restriction_map)

    except : 
        print("Restriction map not found.")
        pass

    list_d1d2 = []  # list containing all the (d1+d2) i.e estimated sizes of the fragment to sequence

    for forward_read, reverse_read in zip(forward_bam_handler, reverse_bam_handler):

        r_sites_forward_read = restriction_map[forward_read.reference_name]
        r_sites_reverse_read = restriction_map[reverse_read.reference_name]

        if forward_read.flag == 0 or forward_read.flag == 256:
            index = np.searchsorted(r_sites_forward_read, forward_read.reference_start, side="right")
            distance_1 = np.subtract(r_sites_forward_read[index], forward_read.reference_start)

        elif forward_read.flag == 16 or forward_read.flag == 272:
            index = np.searchsorted(r_sites_forward_read, forward_read.reference_end, side="left")
            distance_1 = np.abs(np.subtract(forward_read.reference_end, r_sites_forward_read[index]))

        if reverse_read.flag == 0 or reverse_read.flag == 256:
            index = np.searchsorted(r_sites_reverse_read, reverse_read.reference_start, side="right")  # right
            distance_2 = np.subtract(r_sites_reverse_read[index], reverse_read.reference_start)

        elif reverse_read.flag == 16 or reverse_read.flag == 272:
            index = np.searchsorted(r_sites_reverse_read, reverse_read.reference_end, side="left")  # left
            distance_2 = np.abs(np.subtract(reverse_read.reference_end, r_sites_reverse_read[index]))

        # Correction for uncuts with no restriction sites inside
        if forward_read.reference_name == reverse_read.reference_name and np.add(distance_1, distance_2) > np.abs(np.subtract(reverse_read.reference_start, forward_read.reference_start)):
            list_d1d2.append(np.abs(np.subtract(reverse_read.reference_start, forward_read.reference_start)))

        else:
            list_d1d2.append(np.add(distance_1, distance_2))

    histo, bins = np.histogram(list_d1d2, int(max(list_d1d2)))  # here histogram 
    np.save(output_path / D1D2, histo)

    logger.info(f"Saved d1d2 law at : {output_path / D1D2}")

def generate_density_map2(cooler_file : str = "unrescued.pairs.filtered.cool", threads : int = 2, output_dir : str = None) -> None:
    """
    Create density map from a Hi-C matrix. Return a dictionary where keys are chromosomes names and values are density maps.

    Parameters
    ----------
    cooler_file : str
        [description], by default unrescued.pairs.filtered.cool
    threads : int, optional
        [description], by default 2
    output_dir : str, optional
        [description], by default cwd

    """
    logger.info("Start generating density map...")

    if output_dir is None:
        output_path = Path(getcwd())

    else : 
        output_path = Path(output_dir)

    matrix_path = output_path / cooler_file

    if not matrix_path.is_file():
        raise FileNotFoundError(f"Matrix file {matrix_path} not found. Please provide a valid path to matrix file.")

    #Load cooler file
    matrix = hio.load_cooler(matrix = matrix_path)

    #Get chromosomes names
    chromosomes = matrix.chromnames

    #Get possible chromosomes couples
    chromosomes_couples = list(itertools.combinations_with_replacement(chromosomes, 2))

    # Get chromsomes maps
    # chromosomes_maps = [matrix.matrix(balance = True).fetch(chrom1, chrom2) for chrom1, chrom2 in chromosomes_couples]

    pool = mp.Pool(processes=threads)
    results = pool.map(partial(hut.get_local_density, str(matrix_path), nan_threshold  = False),chromosomes_couples)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    results_dict =  {key : value for key, value in results}

    # for chrom_pair in results_dict.copy().keys():
    #     if chrom_pair[0] == chrom_pair[1]:
    #         pass

    #     else :
    #         results_dict[(chrom_pair[1], chrom_pair[0])]  = results_dict[chrom_pair].T

    np.save(output_path / DENSITY_MAP, results_dict)

    logger.info(f"Saved density maps at : {output_path}")

# -----------------------------------------------------------------------------
#  Functions to assign values from statistical tendancies

def get_intra_ps(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment, xs : dict, weirds :  dict, uncuts : dict, circles : dict, circular : str = "") -> float:
    """
    Take two reads and return the P(s) value depending on event type (intra-chromosomal case only).

    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read to compare with the reverse read.
    read_reverse : pysam.AlignedSegment
        Reverse read to compare with the forward read.
    xs : dict
        Dictionary containing log binning values for each chromosome.
    weirds : dict
        Dictionary containing number of weird events considering distance for each chromosome.
    uncuts : dict
        Dictionary containing number of uncuts events considering distance for each chromosome.
    circles : dict
        Dictionary containing number of circles events considering distance for each chromosome.
    circular : str, optional
        Name of the chromosomes to consider as circular, by default None, by default "".

    Returns
    -------
    float
        P(s) of the pair considering the event type.
    """    
    
    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("Reads are not coming from the same pair.")

    if not hut.is_intra_chromosome(read_forward, read_reverse):
        raise ValueError("Reads are not intra-chromosomal.")
    
    if hut.is_uncut(read_forward, read_reverse):
        dist=hut.get_cis_distance(read_forward, read_reverse, circular)
        propensity =  uncuts[read_forward.reference_name][attribute_xs(xs[read_forward.reference_name],dist)]

    elif hut.is_circle(read_forward, read_reverse):
        dist=hut.get_cis_distance(read_forward, read_reverse, circular)
        propensity = circles[read_forward.reference_name][attribute_xs(xs[read_forward.reference_name],dist)]
    
    elif hut.is_weird(read_forward, read_reverse):
        dist=hut.get_cis_distance(read_forward, read_reverse, circular)
        propensity =  weirds[read_forward.reference_name][attribute_xs(xs[read_forward.reference_name],dist)]
    
    return propensity

def get_trans_ps(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment, trans_ps : dict) -> float:
    """
    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read to compare with the reverse read.
    read_reverse : pysam.AlignedSegment
        Reverse read to compare with the forward read.
    trans_ps : dict
        Dictionary of trans-chromosomal P(s)

    Returns
    -------
    float
        Trans P(s) value

    """    
    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("Reads are not coming from the same pair.")
    
    if hut.is_intra_chromosome(read_forward, read_reverse):
        raise ValueError("Reads are not inter-chromosomal.")
    
    return trans_ps[tuple(sorted([read_forward.reference_name, read_reverse.reference_name]))] 

def get_coverages(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment, coverage : dict, bin_size : int) -> float:
    """
    Get the coverage of a pair of reads.

    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read to compare with the reverse read.
    read_reverse : pysam.AlignedSegment
        Reverse read to compare with the forward read.
    coverage : dict
        Dictionary containing the coverage of each chromosome.
    bin_size : int
        Size of the desired bin.

    Returns
    -------
    float
        Product of the coverages of the pair of reads.
    """    

    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("Reads are not coming from the same pair.")
    
    if (read_forward.flag == 16 or read_forward.flag == 272) and (read_reverse.flag == 0 or read_reverse.flag == 256):
        return (
            coverage[read_forward.reference_name][int(read_forward.reference_end / bin_size)]
            * coverage[read_reverse.reference_name][int(read_reverse.reference_start / bin_size)]
        )

    elif (read_forward.flag == 0 or read_forward.flag == 256) and (read_reverse.flag == 16 or read_reverse.flag == 272):
        return (
            coverage[read_forward.reference_name][int(read_forward.reference_start / bin_size)]
            * coverage[read_reverse.reference_name][int(read_reverse.reference_end / bin_size)]
        )

    elif (read_forward.flag == 16 or read_forward.flag == 272) and (read_reverse.flag == 16 or read_reverse.flag == 272):
        return (
            coverage[read_forward.reference_name][int(read_forward.reference_end / bin_size)]
            * coverage[read_reverse.reference_name][int(read_reverse.reference_end / bin_size)]
        )

    else:
        return (
            coverage[read_forward.reference_name][int(read_forward.reference_start / bin_size)]
            * coverage[read_reverse.reference_name][int(read_reverse.reference_start / bin_size)]
        )

def get_d1d2(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment, restriction_map : dict = None, d1d2 : np.array = None) -> int:
    """
    Get the d1d2 value of a pair of reads.

    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read to compare with the reverse read.
    read_reverse : pysam.AlignedSegment
        Reverse read to compare with the forward read.
    restriction_map : dict, optional
        Restriction map saved as a dictionary like chrom_name : list of restriction sites' position, by default None
    d1d2 : np.array, optional
        Distribution of d1d2 values, by default None

    Returns
    -------
    int
        propensity of d1d2 value
    """

    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("Reads are not coming from the same pair.")
    
    # Get appropriate restriction sites vecctors in dicitionary
    r_sites_read_for = restriction_map[read_forward.reference_name]
    r_sites_read_rev = restriction_map[read_reverse.reference_name]

    if read_forward.flag == 0 or read_forward.flag == 256:
        index = np.searchsorted(r_sites_read_for, read_forward.reference_start, side="right")
        distance_1 = np.subtract(r_sites_read_for[index], read_forward.pos)

    if read_forward.flag == 16 or read_forward.flag == 272:
        index = np.searchsorted(r_sites_read_for, read_forward.reference_end, side="left")
        distance_1 = np.abs(np.subtract(read_forward.reference_end, r_sites_read_for[index]))

    if read_reverse.flag == 0 or read_reverse.flag == 256:
        index = np.searchsorted(r_sites_read_rev, read_reverse.reference_start, side="right")  # right
        distance_2 = np.subtract(r_sites_read_rev[index], read_reverse.reference_start)

    if read_reverse.flag == 16 or read_reverse.flag == 272:
        index = np.searchsorted(r_sites_read_rev, read_reverse.reference_end, side="left")  # left
        distance_2 = np.abs(np.subtract(read_reverse.reference_end, r_sites_read_rev[index]))

    # Correction for uncuts with no restriction sites inside
    if read_forward.reference_name == read_reverse.reference_name and np.add(distance_1, distance_2) > np.abs(np.subtract(read_reverse.reference_start, read_forward.reference_start)):
        distance = np.abs(np.subtract(read_reverse.reference_start, read_forward.reference_start))

    else:
        distance = np.add(distance_1, distance_2)
         
    if distance < len(d1d2) :
        propensity_d1d2 = d1d2[distance]
    else : 
        propensity_d1d2 = 1   #  which means that we found this d1d2 just one time
        
    return propensity_d1d2

def get_density(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment, density_map : dict[(str, str) : np.array], bin_size : int = 2000) -> float:
    """
    Get density from density map dictionary.

    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read to get density from.
    read_reverse : pysam.AlignedSegment
        Reverse read to get density from.
    density_map : dict
        Dictionary containing density maps for each chromosome couple.
    bin_size : int, optional
        Resolution of the matrix on which density map has been computed, by default 2000

    Returns
    -------
    float
        Density corresponding to the pair of reads.

    """    
    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("Reads are not coming from the same pair.")

    if hut.is_reverse(read_forward):
        position_for = int(read_forward.reference_end // bin_size)

    elif not hut.is_reverse(read_forward):
        position_for = int(read_forward.reference_start // bin_size)

    if hut.is_reverse(read_reverse):
        position_rev = int(read_reverse.reference_end // bin_size)

    elif not hut.is_reverse(read_reverse):
        position_rev = int(read_reverse.reference_start // bin_size)

    if (read_forward.reference_name, read_reverse.reference_name) in density_map.keys():
        couple_density = density_map.get((read_forward.reference_name, read_reverse.reference_name))[position_for, position_rev]
    else :
        couple_density = density_map.get((read_reverse.reference_name, read_forward.reference_name))[position_rev, position_for]
        
    return couple_density

# -----------------------------------------------------------------------------

def _get_intra_ps_optimized(read_forward: pysam.AlignedSegment,
                             read_reverse: pysam.AlignedSegment,
                             xs: dict, weirds: dict, uncuts: dict,
                             circles: dict, circular: str = "") -> float:
    """
    Version optimisée de get_intra_ps :
    - Distance calculée une seule fois
    - bin_index calculé une seule fois
    - Vérifications redondantes supprimées
    - Bug fix : propensity toujours définie
    """

    chrom     = read_forward.reference_name
    # ✅ Distance et bin calculés une seule fois
    dist      = hut.get_cis_distance(read_forward, read_reverse, circular)
    bin_index = attribute_xs(xs[chrom], dist)

    # ✅ Flags simplifiés par opérations bit à bit
    read_forward, read_reverse = hut.get_ordered_reads(read_forward, read_reverse)
    forward_flag = read_forward.flag
    reverse_flag = read_reverse.flag

    is_uncut_pair  = (forward_flag & 16 == 0)  and (reverse_flag & 16 == 16)
    is_circle_pair = (forward_flag & 16 == 16) and (reverse_flag & 16 == 0)
    is_weird_pair  = not is_uncut_pair and not is_circle_pair

    if is_uncut_pair:
        return uncuts[chrom][bin_index]
    elif is_circle_pair:
        return circles[chrom][bin_index]
    elif is_weird_pair:
        return weirds[chrom][bin_index]
    else:
        # ✅ Bug fix : cas non géré explicitement
        raise ValueError(
            f"Read pair {read_forward.query_name} on {chrom} "
            f"cannot be classified as uncut, circle or weird."
        )



def compute_propensity(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment, restriction_map : dict = None, xs : dict = None, 
                       weirds : dict = None, uncuts : dict = None, circles : dict = None, circular : str = "", trans_ps : dict = None,  
                       coverage : dict = None, bin_size : int = 2000, d1d2 : dict = None, density_map : dict = None,  mode : str = "standard") -> float:
    """
    Compute propensity for read pair to be selected among all plausible pairs related to multi-mapping reads.

    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read to compare with the reverse read.
    read_reverse : pysam.AlignedSegment
        Reverse read to compare with the forward read.
    restriction_map : dict, optional
        Restriction map saved as a dictionary like chrom_name : list of restriction sites' position, by default None
    xs : dict
        Dictionary containing log binning values for each chromosome.
    weirds : dict
        Dictionary containing number of weird events considering distance for each chromosome.
    uncuts : dict
        Dictionary containing number of uncuts events considering distance for each chromosome.
    circles : dict
        Dictionary containing number of circles events considering distance for each chromosome.
    circular : str, optional
        Name of the chromosomes to consider as circular, by default None, by default "".
    trans_ps : dict
        Dictionary of trans-chromosomal P(s)
    coverage : dict
        Dictionary containing the coverage of each chromosome.
    bin_size : int
        Size of the desired bin, by default 2000
    d1d2 : np.array, optional
        Distribution of d1d2 values, by default None
    density : dict
        Dictionary of contact density by chromosome couple.
    mode : str, optional
        Mode to use to compute propensity among, here "standard"

    Returns
    -------
    float
        Propensity to use for read couple drawing
    """

    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("Reads are not coming from the same pair.")
    
    if hut.is_intra_chromosome(read_forward, read_reverse):
        ps = get_intra_ps(read_forward,read_reverse,xs,weirds,uncuts,circles,circular)
    else:  
        ps = get_trans_ps(read_forward, read_reverse, trans_ps)

    cover = get_coverages(read_forward, read_reverse, coverage, bin_size=bin_size)

    return ps * cover




def compute_propensity_mode(read_forward: pysam.AlignedSegment,
                            read_reverse: pysam.AlignedSegment,
                            restriction_map: dict = None,
                            xs: dict = None,
                            weirds: dict = None,
                            uncuts: dict = None,
                            circles: dict = None,
                            circular: str = "",
                            trans_ps: dict = None,
                            coverage: dict = None,
                            bin_size: int = 2000,
                            d1d2: dict = None,
                            density_map: dict = None,
                            mode: str = "standard") -> float:
    """
    Compute propensity for read pair to be selected among all plausible pairs
    related to multi-mapping reads.

    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read to compare with the reverse read.
    read_reverse : pysam.AlignedSegment
        Reverse read to compare with the forward read.
    restriction_map : dict, optional
        Restriction map saved as a dictionary like chrom_name : list of restriction sites' position, by default None
    xs : dict
        Dictionary containing log binning values for each chromosome.
    weirds : dict
        Dictionary containing number of weird events considering distance for each chromosome.
    uncuts : dict
        Dictionary containing number of uncuts events considering distance for each chromosome.
    circles : dict
        Dictionary containing number of circles events considering distance for each chromosome.
    circular : str, optional
        Name of the chromosomes to consider as circular, by default "".
    trans_ps : dict
        Dictionary of trans-chromosomal P(s)
    coverage : dict
        Dictionary containing the coverage of each chromosome.
    bin_size : int
        Size of the desired bin, by default 2000
    d1d2 : dict, optional
        Distribution of d1d2 values, by default None
    density_map : dict, optional
        Dictionary containing density maps per chromosome couples, by default None
    mode : str, optional
        Mode to use to compute propensity, by default "standard"

    Returns
    -------
    float
        Propensity to use for read couple drawing
    """

    # ✅ Vérification unique de l'appariement des reads
    if read_forward.query_name != read_reverse.query_name:
        raise ValueError(
            f"Reads are not from the same pair: "
            f"{read_forward.query_name} vs {read_reverse.query_name}"
        )

    # ✅ Mode random : retour immédiat
    if mode == "random":
        return 1.0

    # ✅ Détermination intra/trans UNE SEULE FOIS
    is_intra = hut.is_intra_chromosome(read_forward, read_reverse)

    # ✅ Calcul ps UNE SEULE FOIS si nécessaire selon le mode
    if mode in ["ps", "standard", "omics", "full"]:
        if is_intra:
            ps = _get_intra_ps_optimized(read_forward, read_reverse, xs, weirds, uncuts, circles, circular)
        else:
            ps = get_trans_ps(read_forward, read_reverse, trans_ps)

    # ✅ Calcul coverage UNE SEULE FOIS si nécessaire selon le mode
    if mode in ["coverage", "standard", "full","omics"]:
        cover = get_coverages(read_forward, read_reverse, coverage, bin_size=bin_size)

    # ✅ Calcul density UNE SEULE FOIS si nécessaire selon le mode
    if mode in ["density", "full"]:
        density_val = get_density(read_forward, read_reverse, density_map=density_map)
    
    # ✅ Calcul d1d2 UNE SEULE FOIS si nécessaire selon le mode
    if mode in ["d1d2"]:
        d1d2_val = get_d1d2(read_forward, read_reverse, restriction_map, d1d2)    

    # ✅ Dispatch sans re-calculs
    if mode == "ps":
        return ps

    elif mode == "coverage":
        return cover

    elif mode == "d1d2":
        return d1d2_val

    elif mode == "density":
        return density_val

    elif mode in ["standard", "omics"]:
        return ps * cover

    # elif mode == "one_enzyme":
    #     return ps * cover * d1d2_val

    elif mode == "full":
        return ps * cover * density_val

    else:
        raise ValueError(
            f"Unknown mode: {mode}. "
            f"Available modes: random, ps, coverage, density, standard, full, omics, d1d2")
    
 
def draw_read_couple(propensities : np.array) -> int:
    """
    Draw an index respecting distribution of propensities. 
    This function is used to draw a couple of reads considering the propensity of each couple.

    Parameters
    ----------
    propensities : np.array
        Array containing all the propensities of each couple of reads.

    Returns
    -------
    int
        Index of the couple of reads drawn.
    """
    
    propensities = [0 if x is None       else x for x in propensities]   # to replace None by 0   
    propensities = [0 if str(x) == 'Nan' else x for x in propensities]   # to replace Nan by 0   
    propensities = [0 if str(x) == 'nan' else x for x in propensities]   # to replace nan by 0   

    xk = np.arange(len(propensities))

    if  np.sum(propensities) > 0: 
        try:
            pk = np.divide(propensities, np.sum(propensities))
        
        except:
            print(f"pk : {pk}")
            print(f"propensities : {propensities}")

    elif np.sum(propensities) <= 0:
        try : 
            pk = np.full(xk.shape, np.divide(1, len(propensities)))
        except :              
                print(f"pk : {pk}")
                print(f"propensities : {propensities}")

    else : 
        logger.error(f"Propensities : {propensities}")

    index = choice(xk, p=pk)   # random pick into a distribution :)
    return index

# Heart of the algo 

def reattribute_reads(reads_couple: tuple[str, str] = ("group2.1.bam", "group2.2.bam"),
                      restriction_map: dict = None,
                      xs: dict = "xs.npy",
                      weirds: dict = "weirds.npy",
                      uncuts: dict = "uncuts.npy",
                      circles: dict = "circles.npy",
                      circular: str = "",
                      trans_ps: dict = "trans_ps.npy",
                      coverage: dict = "coverage.npy",
                      bin_size: int = 2000,
                      d1d2: dict = "d1d2.npy",
                      density_map: dict = "density_map.npy",
                      mode: str = "standard",
                      output_dir: str = None) -> None:

    if output_dir is None:
        output_path = Path(getcwd())
    else:
        output_path = Path(output_dir)

    # Chargement des dictionnaires
    xs       = hio.load_dictionary(output_path / xs)
    uncuts   = hio.load_dictionary(output_path / uncuts)
    circles  = hio.load_dictionary(output_path / circles)
    weirds   = hio.load_dictionary(output_path / weirds)
    trans_ps = hio.load_dictionary(output_path / trans_ps)
    coverage = hio.load_dictionary(output_path / coverage)

    d1d2    = None
    density = None

    if mode in ["d1d2"]:
        d1d2 = hio.load_dictionary(output_path / "d1d2.npy")
    elif mode in ["density", "full"]:
        density = hio.load_dictionary(output_path / "density_map.npy")


    forward_bam_path, reverse_bam_path = Path(reads_couple[0]), Path(reads_couple[1])
    file_id = time.time()
    id_for  = uuid.uuid4()
    id_rev  = uuid.uuid4()

    forward_bam_handler = pysam.AlignmentFile(forward_bam_path, "rb")
    reverse_bam_handler = pysam.AlignmentFile(reverse_bam_path, "rb")

    forward_out_bam_handler = pysam.AlignmentFile(
        output_path / f"forward_{id_for}_{file_id}_predicted.bam", "wb", template=forward_bam_handler)
    reverse_out_bam_handler = pysam.AlignmentFile(
        output_path / f"reverse_{id_rev}_{file_id}_predicted.bam", "wb", template=reverse_bam_handler)

    # ✅ Validation du mode UNE SEULE FOIS avant la boucle
    VALID_MODES = {"random", "ps", "coverage", "density", "standard", "full","omics","d1d2"}
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {VALID_MODES}")

    # ✅ Pre-builder la fonction partielle UNE SEULE FOIS
    # Evite de passer les gros dicts comme arguments à chaque appel
    _compute_propensity = partial(
        compute_propensity_mode,
        restriction_map = restriction_map,
        xs              = xs,
        weirds          = weirds,
        uncuts          = uncuts,
        circles         = circles,
        circular        = circular,
        trans_ps        = trans_ps,
        coverage        = coverage,
        bin_size        = bin_size,
        d1d2            = d1d2,
        density_map     = density,
        mode            = mode
    )

    forward_generator = hut.bam_iterator(forward_bam_path)
    reverse_generator = hut.bam_iterator(reverse_bam_path)

    # ✅ ThreadPoolExecutor créé UNE SEULE FOIS pour tout le traitement
    # On utilise les threads (et non multiprocessing) car on est déjà dans un pool de process
    # Si compute_propensity_mode utilise numpy → libère le GIL → threads efficaces
    with ThreadPoolExecutor() as executor:

        for forward_block, reverse_block in zip(forward_generator, reverse_generator):

            n_for = len(forward_block)
            n_rev = len(reverse_block)

            combinations = list(itertools.product(forward_block, reverse_block))

            # ✅ Calcul parallèle des propensités via threads
            propensities = list(executor.map(
                lambda combo: _compute_propensity(
                    read_forward  = combo[0],
                    read_reverse  = combo[1]
                ),
                combinations
            ))

            selected_couple_index = draw_read_couple(propensities)
            selected_read_forward, selected_read_reverse = combinations[selected_couple_index]

            selected_read_forward.set_tag("XL", n_for)
            selected_read_reverse.set_tag("XL", n_rev)

            forward_out_bam_handler.write(selected_read_forward)
            reverse_out_bam_handler.write(selected_read_reverse)

    forward_bam_handler.close()
    reverse_bam_handler.close()
    forward_out_bam_handler.close()
    reverse_out_bam_handler.close()

    logger.info(f"Predictions written in {output_path}")


