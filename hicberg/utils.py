import time
import uuid
import subprocess as sp
from glob import glob
import shutil as sh
from os import getcwd, mkdir
from pathlib import Path
import multiprocessing
from functools import partial
import itertools
from scipy.stats import pearsonr
from typing import Iterator, Tuple, List

import numpy as np
from numpy.random import choice
import pandas as pd
import scipy.stats as st
from scipy.stats import median_abs_deviation
from scipy.ndimage import generic_filter
from scipy.ndimage import label, find_objects
from scipy.sparse import csr_matrix

import pysam
from Bio import SeqIO
import cooler

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import hicberg.io as hio
import hicberg.statistics as hst
from hicberg import logger
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.optimize import isotonic_regression

def generate_gaussian_kernel(size : int = 1, sigma : int = 2) -> np.array:
    """
    Generate a 2D Gaussian kernel of a given size and standard deviation.

    Parameters
    ----------
    size : int, optional
        Size of the kernel, by default 1
    sigma : int, optional
        Standard deviation to use for the kernel build, by default 2

    Returns
    -------
    np.array
        2D Gaussian kernel.
    """    

    x = np.linspace(-sigma, sigma, size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)

    return kern2d / kern2d.sum()

def distance_law(matrix, detectable_bins=None, max_dist=None, smooth=True, fun=np.nanmean):
    """
    Computes genomic distance law by averaging over each diagonal in the upper
    triangle matrix. If a list of detectable bins is provided, pixels in
    missing bins will be excluded from the averages. A maximum distance can be
    specified to define how many diagonals should be computed.

    parameters
    ----------
    matrix: scipy.sparse.csr_matrix
        the input matrix to compute distance law from.
    detectable_bins : numpy.ndarray of ints
        An array of detectable bins indices to consider when computing
        distance law.
    max_dist : int
        Maximum distance from diagonal, in number of bins in which to compute
        distance law
    smooth : bool
        Whether to use isotonic regression to smooth the distance law.
    fun : callable
        A function to apply on each diagonal. Defaults to mean.

    Returns
    -------
    dist: np.ndarray
        the output genomic distance law.

    example
    -------
        >>> m = np.ones((3,3))
        >>> m += np.array([1,2,3])
        >>> m
        array([[2., 3., 4.],
               [2., 3., 4.],
               [2., 3., 4.]])
        >>> distance_law(csr_matrix(m))
        array([3. , 3.5, 4. ])

    """
    mat_n = matrix.shape[0]
    if max_dist is None:
        max_dist = mat_n
    n_diags = min(mat_n, max_dist + 1)
    dist = np.zeros(mat_n)
    if detectable_bins is None:
        detectable_bins = np.array(range(mat_n))

    for diag in range(n_diags):
        # Find detectable which fall in diagonal
        detect_mask = np.zeros(mat_n, dtype=bool)
        detect_mask[detectable_bins] = 1
        # Find bins which are detectable in the diagonal (intersect of
        # hori and verti)
        detect_mask_h = detect_mask[: (mat_n - diag)]
        detect_mask_v = detect_mask[mat_n - (mat_n - diag) :]
        detect_mask_diag = detect_mask_h & detect_mask_v
        detect_diag = matrix.diagonal(diag)[detect_mask_diag]
        dist[diag] = fun(detect_diag[detect_diag > 0])
    # Smooth the curve using isotonic regression: Find closest approximation
    # with the condition that point n+1 cannot be higher than point n.
    # (i.e. contacts can only decrease when increasing distance)
    # if smooth and mat_n > 2:
        # ir = IsotonicRegression(increasing=False)
        # dist = ir.fit_transform(range(len(dist)), dist)
        
        # to adapt
        # dist = isotonic_regression(dist, increasing=False)
        # dist[~np.isfinite(dist)] = 0

    return dist

def detrend(matrix,detectable_bins=None,max_dist=None,smooth=False,fun=np.nanmean,max_val=10):
    """
    Detrends a Hi-C matrix by the distance law.
    The input matrix should have been normalised beforehandand.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The normalised intrachromosomal Hi-C matrix to detrend.
    detectable_bins : tuple
        Tuple containing a list of detectable rows and a list of columns on
        which to perform detrending. Poorly interacting indices have been
        excluded.
    max_dist : int
        Maximum number of bins from the diagonal at which to compute trend.
    smooth : bool
        Whether to use isotonic regression to smooth the trend.
    fun : callable
        Function to use on each diagonal to compute the trend.
    max_val : float or None
        Maximum value in the detrended matrix. Set to None to disable

    Returns
    -------
    numpy.ndarray :
        The detrended Hi-C matrix.
    """
    matrix = matrix.tocsr()
    y = distance_law(
        matrix,
        detectable_bins=detectable_bins,
        max_dist=max_dist,
        smooth=smooth,
        fun=fun,
    )
    y[np.isnan(y)] = 0.0
    # Detrending by the distance law
    clean_mat = matrix.tocoo()
    # clean_mat.data /= y_savgol[abs(clean_mat.row - clean_mat.col)]
    try:
        clean_mat.data = clean_mat.data / y[abs(clean_mat.row - clean_mat.col)]
    # If no nonzero value in matrix, do nothing
    except TypeError:
        pass
    clean_mat = clean_mat.tocsr()
    if max_val is not None:
        clean_mat[clean_mat >= max_val] = 1
    return clean_mat

def get_bad_bins(matrix : np.array = None, n_mads : int = 2) -> np.array:
    """
    Detect bad bins (poor interacting bins) in a normalized Hi-C matrix and return their indexes.
    Bins where the nan sum of interactions is zero  are considered as bad bins.

    Parameters
    ----------
    matrix : Normalized Hi-C matrix to detect bad bins from, by default None.
        
    n_mads : int, optional
        Number of median absolute deviations to set poor interacting bins threshold, by default 2

    Returns
    -------
    np.array
        Indexes of bad bins.
    """   

    # Cis case
    if matrix.shape[0] == matrix.shape[1]:

        nan_sum_bins = np.nansum(matrix, axis = 0)
        bad_indexes = np.where(nan_sum_bins == 0)

        return (bad_indexes)
    
    # Trans case
    else :

        x_sum_bins = np.nansum(matrix, axis = 0)
        y_sum_bins = np.nansum(matrix, axis = 1)

        x_bad_indexes = np.where(x_sum_bins == 0)
        y_bad_indexes = np.where(y_sum_bins == 0) 

        return (x_bad_indexes, y_bad_indexes)
 
# not used anymore    
def nan_conv(matrix : np.array = None, kernel : np.array = None, nan_threshold : bool = False) -> np.array:
    """
    Custom convolution function that takes into account nan values when convolving.
    Used to compute the local density of a Hi-C matrix.

    Parameters
    ----------
    matrix : np.array, optional
        Hi-C matrix to detect bad bins from, by default None
    kernel : np.array, optional
        Kernel to use for convolution (dimension must be odd), by default None
    nan_threshold : bool, optional
        Set wether or not convolution return nan if not enough value are caught, by default False

    Returns
    -------
    np.array
        Convolution product of the matrix and the kernel.
    """

    mat_cp = matrix.copy().astype(float)
    half_kernel = (kernel.shape[0] // 2)
    density_threshold = (kernel.shape[0] - half_kernel + 1) ** 2

    # Cis case
    if matrix.shape[0] == matrix.shape[1]:

        for i in range(half_kernel , (mat_cp.shape[0] - half_kernel), 1): 
            for j in range(half_kernel , (mat_cp.shape[1] - half_kernel), 1): 

                patch = matrix[i - (half_kernel) : (i +  half_kernel + 1) , j - (half_kernel) : j + half_kernel + 1]

                # Disrupt the kernel if too many nan values
                if nan_threshold:
                    nb_nan = np.count_nonzero(np.isnan(patch))
                    if nb_nan > density_threshold:

                        mat_cp[i, j] = np.nan
                        continue

                masked_patch = np.ma.MaskedArray(patch, mask = np.isnan(patch))
                mat_cp[i, j] = np.ma.average(masked_patch, weights = kernel)

    # Trans case
    else : 

        mean_value = np.nanmean(matrix)

        for i in range(half_kernel , (mat_cp.shape[0] - half_kernel), 1): 
            for j in range(half_kernel , (mat_cp.shape[1] - half_kernel), 1): 

                patch = matrix[i - (half_kernel) : (i +  half_kernel + 1) , j - (half_kernel) : j + half_kernel + 1]
                nb_nan = np.count_nonzero(np.isnan(patch))

                # Disrupt the kernel if too many nan values
                if nan_threshold:
                    if nb_nan > density_threshold:

                        mat_cp[i, j] = np.nan
                        continue

                masked_patch = np.ma.MaskedArray(patch, mask = np.isnan(patch))
                conv = np.ma.average(masked_patch, weights = kernel)

                if np.isnan(conv):

                    mat_cp[i, j] = mean_value
                else:
                    mat_cp[i, j] = conv

    return mat_cp

def nanmean_filter(values):
    vals = values[~np.isnan(values)]
    return np.mean(vals) if len(vals) > 0 else np.nan

def iterative_fill(Z, patch_size=10, max_iter=10):
    Z_filled = Z.copy()
    for i in range(max_iter):
        nan_count = np.isnan(Z_filled).sum()
        if nan_count == 0:
            break
        Z_new = generic_filter(Z_filled, nanmean_filter, size=patch_size, mode='mirror')
        # On ne remplit que les NaN à chaque itération
        Z_filled[np.isnan(Z_filled)] = Z_new[np.isnan(Z_filled)]
        print(f"Iteration {i+1}: {nan_count} NaN restants")
    return Z_filled

def estimate_nan_patch_sizes(Z, border_ignore=True, border=1):
    nan_mask = np.isnan(Z)
    
    # Optionnel : ignorer les NaN collés aux bords
    if border_ignore:
        core_mask = np.ones_like(Z, dtype=bool)
        core_mask[:border, :] = False
        core_mask[-border:, :] = False
        core_mask[:, :border] = False
        core_mask[:, -border:] = False
        nan_mask = nan_mask & core_mask
    
    labeled, n = label(nan_mask)
    if n == 0:
        print("Aucune zone NaN interne détectée.")
        return None, None, None, None
    
    regions = find_objects(labeled)
    sizes = [(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) for sl in regions]

    mean_h = np.mean([h for h, _ in sizes])
    mean_w = np.mean([w for _, w in sizes])
    max_h  = np.max([h for h, _ in sizes])
    max_w  = np.max([w for _, w in sizes])

    print(f"Taille moyenne des trous : {mean_h:.1f} x {mean_w:.1f}")
    print(f"Taille max des trous : {max_h} x {max_w}")

    patch_small = int(np.clip(min(mean_h, mean_w) / 2, 3, 20))
    patch_large = int(np.clip(max(max_h, max_w) * 0.8, 10, 100))
    print(f"Patch_small ≈ {patch_small}, patch_large ≈ {patch_large}")

    return patch_small, patch_large, (mean_h, mean_w), (max_h, max_w)

def get_local_density(cooler_file : str = None, chrom_name : tuple = (None, None), nan_threshold : bool = False) -> np.array:
    """
    Create density map from a Hi-C matrix. Return a dictionary where keys are chromosomes names and values are density maps.
    Density is obtained by getting the local density of each pairwise bin using a gaussian kernel convolution.

    Parameters
    ----------
    cooler_file : str, optional
        Path to Hi-C matrix (or sub-matrix) to get density from, by default None
    chrom_name : tuple, optional
        Tuple containing the sub-matrix to fetch, by default (None, None)
    nan_threshold : bool, optional
        Set wether or not convolution return nan if not enough value are caught, by default None

    Returns
    -------
    np.array
        Density contact map.
    """

    #Load cooler file
    chr1=chrom_name[0]
    chr2=chrom_name[1]
    mat = cooler.Cooler(cooler_file).matrix(balance = True).fetch(chr1, chr2)

    # Detrending of p(s)
    if chr1==chr2:
        mat = csr_matrix(mat)
        mat= mat.tocoo()
        mat=detrend(mat)
        mat=mat.todense()
    
    nan_mask = np.isnan(mat)
    labeled, n = label(nan_mask)  # detection of empty areas
    regions = find_objects(labeled)
    
    nan_by_row = np.isnan(mat).mean(axis=1)
    nan_by_col = np.isnan(mat).mean(axis=0)
    
    approx_band_height = np.mean([
        np.sum(nan_by_row > 0.5)  # lines with at least 50% of Nan
    ])
    approx_band_width = np.mean([
        np.sum(nan_by_col > 0.5)
    ])
    
    # print(f"Largeur bandes ~{approx_band_width:.1f}, hauteur bandes ~{approx_band_height:.1f}")
    
    patch_small = int(np.clip(min(approx_band_height, approx_band_width)/2, 5, 15))
    patch_large = int(np.clip(max(approx_band_height, approx_band_width)*1.5, 15, 80))
    
    if chr1==chr2:
        patch_large =  15
    
    print(f"Patch_small ≈ {patch_small}, patch_large ≈ {patch_large}")
    
    mat_tmp = iterative_fill(mat, patch_size=patch_small, max_iter=5)
    mat_filled = generic_filter(mat_tmp, nanmean_filter, size=patch_large, mode='mirror')
    
    # replacement of zeros with the non zero minimum
    min_non_zero= np.min(mat_filled[mat_filled !=0])
    mat_filled[mat_filled ==0]= min_non_zero
    
    if chr1==chr2:
        np.fill_diagonal(mat_filled, 1)   # we stay neutral on the inner diagonale 
    
    if chr1!=chr2:
        mat_filled = mat_filled / np.mean(mat_filled)

    return (chrom_name, mat_filled)

def get_chromosomes_sizes(genome : str = None, output_dir : str = None) -> None:
    """
    Generate a dictionary save in .npy format where keys are chromosome name and value are size in bp.

    Parameters
    ----------
    genome : str, optional
        Path to the genome, by default None

    output_dir : str, optional
        Path to the folder where to save the dictionary, by default None
    """

    logger.info(f"Start getting chromosome sizes")
    genome_path = Path(genome)

    if not genome_path.is_file():
        raise IOError(f"Genome file {genome_path.name} not found. Please provide a valid path.")

    if output_dir is None:
        folder_path = Path(getcwd())

    else:
        folder_path = Path(output_dir)

    chrom_sizes = {}

    output_file = folder_path / "chromosome_sizes.npy"

    for rec in SeqIO.parse(genome_path,"fasta"):
        chrom_sizes[rec.id] = len(rec.seq)

    np.save(output_file, chrom_sizes)

    logger.info(f"Chromosome sizes have been saved in {output_file}")

def get_bin_table(chrom_sizes_dict : str = "chromosome_sizes.npy", bin_size : int = 2000, output_dir : str = None) -> None:
    """
    Create bin table containing start and end position for fixed size bin per chromosome.

    Parameters
    ----------
    chrom_sizes_dict : str
        Path to a dictionary containing chromosome sizes as {chromosome : size} saved in .npy format. By default chromosome_sizes.npy
    bin_size : int
        Size of the desired bin, by default 2000.
    output_dir : str, optional
        Path to the folder where to save the dictionary, by default None
    """

    logger.info(f"Start getting bin table")
    chrom_sizes_dict_path = Path(output_dir, chrom_sizes_dict)

    if not chrom_sizes_dict_path.is_file():
        raise IOError(f"Genome file {chrom_sizes_dict_path.name} not found. Please provide a valid path.")

    if output_dir is None:
        folder_path = Path(getcwd())

    else:

        folder_path = Path(output_dir)

    output_file = folder_path / "fragments_fixed_sizes.txt"

    chrom_size_dic = np.load(chrom_sizes_dict_path, allow_pickle=True).item()
    chr_count = 0

    with open (output_file, "w") as f_out:
        
        for chrom, length in zip(chrom_size_dic.keys(), chrom_size_dic.values()):

            curr_chr, curr_length = chrom, length
            chr_count += 1

            if (curr_length % bin_size) == 0:
                    interval_end = curr_length
            else:
                interval_end = (int((curr_length + bin_size) / bin_size)) * bin_size

                for val in range(0, interval_end, bin_size):
                    curr_start = val

                    if val + bin_size > curr_length:

                        curr_end = curr_length
                    else:
                        curr_end = val + bin_size
                    if (chr_count > 1) or (val > 0):
                        f_out.write("\n")
                    f_out.write(
                        str(curr_chr)
                        + "\t"
                        + str(curr_start)
                        + "\t"
                        + str(int(curr_end))
                        + "\t"
                    )

        # close the output fragment file
        f_out.close()

def is_ambiguous(read : pysam.AlignedSegment) -> bool:
    """
    Check if read from pysam AlignmentFile is mapping more than once along the genome.

    Parameters
    ----------
    read : pysam.AlignedSegment
        pysam AlignedSegment object.

    Returns
    -------
    bool
        True if the read is ambiguous i.e. mapping to more than one position.
    """    

    if "XS" in [x[0] for x in read.get_tags()]:
        return True

    else:
        return False

def is_poor_quality(read : pysam.AlignedSegment, mapq : int) -> bool:
    """
    Check if read from pysam AlignmentFile is under mapping quality threshold

    Parameters
    ----------
    read : pysam.AlignedSegment
        pysam AlignedSegment object.
    mapq : int
        Mapping quality threshold.

    Returns
    -------
    bool
        True if the read quality is below mapq threshold.
    """    
    if read.mapping_quality < mapq:
        return True

    else:
        return False

def is_unmapped(read : pysam.AlignedSegment) -> bool:
    """
    Check if read from pysam AlignmentFile is unmapped

    Parameters
    ----------
    read : pysam.AlignedSegment
        pysam AlignedSegment object.

    Returns
    -------
    bool
        True if the read is unmapped.
    """    
    if read.flag == 4:
        return True
    else:
        return False

def is_reverse(read : pysam.AlignedSegment) -> bool:
    """
    Check if read from pysam AlignmentFile is reverse

    Parameters
    ----------
    read : pysam.AlignedSegment
        pysam AlignedSegment object.

    Returns
    -------
    bool
        True if the read is reverse.
    """ 

    if read.flag == 16 or read.flag == 272:
        return True
    else:
        return False

def classify_reads(bam_couple: tuple[str, str] = ("1.sorted.bam", "2.sorted.bam"), 
                   chromosome_sizes: str = "chromosome_sizes.npy", 
                   mapq: int = 30, 
                   output_dir: str = None) -> None:
    """
    Classification of pairs of reads in 2 different groups:
        Group 0) --> (Unmappable) - files :group0.1.bam and group0.2.bam
        Group 1) --> (Uniquely Mapped  Uniquely Mapped) - files :group1.1.bam and group1.2.bam
        Group 2) --> (Uniquely Mapped with Multi Mapped) or (Multi Mapped with  Multi Mapped).- files : group2.1.bam and group2.2.bam

    Parameters
    ----------
    bam_couple : tuple[str, str]
        Tuple containing the paths to the forward and reverse alignment files. By default ("1.sorted.bam", "2.sorted.bam")
    chromosome_sizes : str, optional
        Path to a chromosome size dictionary save in .npy format, by default chromosome_sizes.npy
    mapq : int, optional
        Minimal read quality under which a Hi-C read pair will not be kept, by default 30
    output_dir : str, optional
        Path to the folder where to save the classified alignment files, by default None
    """

    # ✅ Bug fix : vérifier output_dir AVANT de l'utiliser
    if output_dir is None:
        output_dir = Path(getcwd())
    else:
        output_dir = Path(output_dir)

    forward_bam_file_path = Path(output_dir, bam_couple[0])
    reverse_bam_file_path = Path(output_dir, bam_couple[1])
    chromosome_sizes_path = Path(output_dir, chromosome_sizes)

    if not forward_bam_file_path.is_file():
        raise IOError(f"Forward alignment file {forward_bam_file_path.name} not found. Please provide a valid path.")
    if not reverse_bam_file_path.is_file():
        raise IOError(f"Reverse alignment file {reverse_bam_file_path.name} not found. Please provide a valid path.")
    if not chromosome_sizes_path.is_file():
        raise IOError(f"Chromosome sizes file {chromosome_sizes_path.name} not found. Please provide a valid path.")

    chromosome_sizes_dic = hio.load_dictionary(chromosome_sizes_path)

    forward_bam_file = pysam.AlignmentFile(forward_bam_file_path, "rb")
    reverse_bam_file = pysam.AlignmentFile(reverse_bam_file_path, "rb")

    forward_header = forward_bam_file.header
    reverse_header = reverse_bam_file.header

    forward_bam_file_iter = bam_iterator(forward_bam_file_path)
    reverse_bam_file_iter = bam_iterator(reverse_bam_file_path)

    unmapped_bam_file_foward = pysam.AlignmentFile(output_dir / "group0.1.bam", "wb", template=forward_bam_file, header=forward_header)
    unmapped_bam_file_reverse = pysam.AlignmentFile(output_dir / "group0.2.bam", "wb", template=reverse_bam_file, header=reverse_header)
    uniquely_mapped_bam_file_foward = pysam.AlignmentFile(output_dir / "group1.1.bam", "wb", template=forward_bam_file, header=forward_header)
    uniquely_mapped_bam_file_reverse = pysam.AlignmentFile(output_dir / "group1.2.bam", "wb", template=reverse_bam_file, header=reverse_header)
    multi_mapped_bam_file_foward = pysam.AlignmentFile(output_dir / "group2.1.bam", "wb", template=forward_bam_file, header=forward_header)
    multi_mapped_bam_file_reverse = pysam.AlignmentFile(output_dir / "group2.2.bam", "wb", template=reverse_bam_file, header=reverse_header)

    nb_unmapped_couples, nb_multi_mapped_couples, nb_unique_couples = 0, 0, 0
    nb_unmapped_reads_forward, nb_unmapped_reads_reverse = 0, 0
    nb_uniquely_mapped_reads_forward, nb_uniquely_mapped_reads_reverse = 0, 0
    nb_multi_mapped_reads_forward, nb_multi_mapped_reads_reverse = 0, 0

    for forward_block, reverse_block in zip(forward_bam_file_iter, reverse_bam_file_iter):

        # ✅ Classification O(N+M) au lieu de O(N×M)
        unmapped_couple = any(
            is_unmapped(r) for r in itertools.chain(forward_block, reverse_block)
        )

        multi_mapped_couple = False
        if not unmapped_couple:
            multi_mapped_couple = any(
                is_ambiguous(r) or is_poor_quality(r, mapq)
                for r in itertools.chain(forward_block, reverse_block)
            )

        # ✅ Sélection des handlers et compteurs UNE SEULE FOIS par bloc
        if unmapped_couple:
            out_for, out_rev = unmapped_bam_file_foward, unmapped_bam_file_reverse
            nb_unmapped_couples += 1
            nb_unmapped_reads_forward += len(forward_block)
            nb_unmapped_reads_reverse += len(reverse_block)
            add_tag = False

        elif multi_mapped_couple:
            out_for, out_rev = multi_mapped_bam_file_foward, multi_mapped_bam_file_reverse
            nb_multi_mapped_couples += 1
            nb_multi_mapped_reads_forward += len(forward_block)
            nb_multi_mapped_reads_reverse += len(reverse_block)
            add_tag = True

        else:
            out_for, out_rev = uniquely_mapped_bam_file_foward, uniquely_mapped_bam_file_reverse
            nb_unique_couples += 1
            nb_uniquely_mapped_reads_forward += len(forward_block)
            nb_uniquely_mapped_reads_reverse += len(reverse_block)
            add_tag = True

        # ✅ Écriture sans if-elif-else répétés dans la boucle
        for forward_read in forward_block:
            if add_tag:
                forward_read.set_tag("XG", chromosome_sizes_dic[forward_read.reference_name])
            out_for.write(forward_read)

        for reverse_read in reverse_block:
            if add_tag:
                reverse_read.set_tag("XG", chromosome_sizes_dic[reverse_read.reference_name])
            out_rev.write(reverse_read)

    # Closing files
    forward_bam_file.close()
    reverse_bam_file.close()
    
    unmapped_bam_file_foward.close()
    unmapped_bam_file_reverse.close()
    uniquely_mapped_bam_file_foward.close()
    uniquely_mapped_bam_file_reverse.close()
    multi_mapped_bam_file_foward.close()
    multi_mapped_bam_file_reverse.close()

    logger.info(f"Number of unmapped couples     : {nb_unmapped_couples}")
    logger.info(f"Number of unique couples       : {nb_unique_couples}")
    logger.info(f"Number of multi-mapped couples : {nb_multi_mapped_couples}")
    logger.info(f"Number of multi alignments in forward file : {nb_multi_mapped_reads_forward}")
    logger.info(f"Number of multi alignments in reverse file : {nb_multi_mapped_reads_reverse}")

    # Cleaning files after classification
    forward_bam_file_path.unlink()
    reverse_bam_file_path.unlink()

def is_intra_chromosome(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment) -> bool:
    """
    Return True if two reads of a pair came from the same chromosome.

    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read of the pair.
    read_reverse : pysam.AlignedSegment
        Reverse read of the pair.

    Returns
    -------
    bool
        True if the pair is intra-chromosomic, False otherwise.
    """    

    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("Reads are not coming from the same pair.")

    if read_forward.reference_name == read_reverse.reference_name:
        return True
    else:
        return False 

def get_ordered_reads(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment) -> Tuple[pysam.AlignedSegment, pysam.AlignedSegment]:
    """
    Returns the ordered pair of reads in the same chromosome as the two reads .

    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read to compare with the reverse read.
    read_reverse : pysam.AlignedSegment
        Reverse read to compare with the forward read.

    Returns
    -------
    Tuple[pysam.AlignedSegment, pysam.AlignedSegment]
        The ordered pair of reads in the same chromosome as the two reads.
    """

    if read_forward.query_name != read_reverse.query_name:       
        raise ValueError("The two reads must come from the same pair.")
        
    if is_reverse(read_forward):
        forward_start = read_forward.reference_end
    
    elif not is_reverse(read_forward):
        forward_start = read_forward.reference_start

    if is_reverse(read_reverse):
        reverse_start = read_reverse.reference_end

    elif not is_reverse(read_reverse):
        reverse_start = read_reverse.reference_start


    if forward_start <= reverse_start:   
        return (read_forward, read_reverse)
    
    elif forward_start > reverse_start:
        return (read_reverse, read_forward)

def is_weird(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment) -> bool:
    """
    Check if two reads are forming a weird pattern .

    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read of the pair
    read_reverse : pysam.AlignedSegment
        Reverse read of the pair

    Returns
    -------
    bool
        True if the two reads are forming a weird pattern, False otherwise.
    """
    
    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("The two reads must be mapped on the same chromosome.")

    read_forward, read_reverse = get_ordered_reads(read_forward, read_reverse)

    if (
        (read_forward.flag == read_reverse.flag == 0)
        or (read_forward.flag == read_reverse.flag == 16)
        or (read_forward.flag == read_reverse.flag == 272)
        or (read_forward.flag == read_reverse.flag == 256)
        or (read_forward.flag == 256 and read_reverse.flag == 0)
        or (read_forward.flag == 0 and read_reverse.flag == 256)
        or (read_forward.flag == 16 and read_reverse.flag == 272)
        or (read_forward.flag == 272 and read_reverse.flag == 16)
    ):
        return True
    
    else:
        return False

def is_uncut(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment) -> bool:
    """
    Check if two reads are forming an uncut pattern .

    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read of the pair
    read_reverse : pysam.AlignedSegment
        Reverse read of the pair

    Returns
    -------
    bool
        True if the two reads are forming an uncut pattern, False otherwise.
    """
        
    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("The two reads must be mapped on the same chromosome.")
    
    read_forward, read_reverse = get_ordered_reads(read_forward, read_reverse)

    if (
        (read_forward.flag == 0 and read_reverse.flag == 16)
        or (read_forward.flag == 256 and read_reverse.flag == 16)
        or (read_forward.flag == 0 and read_reverse.flag == 272)
        or (read_forward.flag == 256 and read_reverse.flag == 272)
    ):
        return True
    else:
        return False

def is_circle(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment) -> bool:
    """
    Check if two reads are forming a loop pattern .

    Parameters
    ----------
    read_forward : pysam.AlignedSegment
        Forward read of the pair
    read_reverse : pysam.AlignedSegment
        Reverse read of the pair

    Returns
    -------
    bool
        True if the two reads are forming a loop pattern, False otherwise.
    """

    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("Reads are not coming from the same pair")

    read_forward, read_reverse = get_ordered_reads(read_forward, read_reverse)

    if (
        (read_forward.flag == 16 and read_reverse.flag == 0)
        or (read_forward.flag == 272 and read_reverse.flag == 0)
        or (read_forward.flag == 16 and read_reverse.flag == 256)
        or (read_forward.flag == 272 and read_reverse.flag == 256)
    ):
        return True
    else:
        return False



def get_cis_distance(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment, circular : str = "") -> int:
    """
    Calculate the distance between two reads in the same pairwise alignment .

    Parameters
    ----------
    read_forward : pysam.aligned_segment
        Forward read of the pair
    read_reverse : pysam.AlignedSegment
        Reverse read of the pair
    circular : str, optional
        Name of the chromosomes to consider as circular, by default None, by default "".
        Can be several chrms e.g: chrM,plasmid2micron 

    Returns
    -------
    int
        Genomic distance separating the two reads (bp).

    """  
    # convertion into a list 
    if circular is not None:
        circular = [e.strip() for e in circular.split(',')]
    
    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("Reads are not coming from the same pair")

    if is_intra_chromosome(read_forward, read_reverse):
        read_forward, read_reverse = get_ordered_reads(read_forward, read_reverse)

        if is_weird(read_forward, read_reverse):
            distance = np.abs(np.subtract(read_forward.reference_start, read_reverse.reference_start))

        elif is_uncut(read_forward, read_reverse):
            distance = np.abs(np.subtract(read_forward.reference_start, read_reverse.reference_end))

        elif is_circle(read_forward, read_reverse):
            distance = np.abs(np.subtract(read_forward.reference_end, read_reverse.reference_start))

        # circular case
        if circular is not None:
            if read_forward.reference_name in circular:
                clockwise_distance = distance
                anti_clockwise_distance = np.subtract(read_forward.get_tag("XG"), distance)
                distance = np.min([clockwise_distance, anti_clockwise_distance])

        return distance


def bam_iterator(bam_file: str = None) -> Iterator[List[pysam.AlignedSegment]]:
    """
    Returns an iterator for the given SAM/BAM file (must be query-sorted).
    In each call, the alignments of a single read are yielded as a list of
    pysam.AlignedSegment objects sharing the same query name.

    Parameters
    ----------
    bam_file : str
        Path to alignment file in .sam or .bam format.

    Yields
    -------
    Iterator[List[pysam.AlignedSegment]]
        Yields a list of pysam.AlignedSegment objects sharing the same query name.
    """

    bam_path = Path(bam_file)

    if not bam_path.is_file():
        raise IOError(f"BAM file {bam_path.name} not found. Please provide a valid path.")

    with pysam.AlignmentFile(bam_path, "rb") as bam_handler:

        alignments = bam_handler.fetch(until_eof=True)

        # ✅ Gestion du fichier vide
        try:
            current_aln = next(alignments)
        except StopIteration:
            logger.warning(f"BAM file {bam_path.name} is empty.")
            return

        current_read_name = current_aln.query_name
        block = [current_aln]  # ✅ Style simplifié

        while True:
            try:
                next_aln = next(alignments)
                next_read_name = next_aln.query_name

                if next_read_name != current_read_name:
                    yield block
                    current_read_name = next_read_name
                    block = [next_aln]  # ✅ Style simplifié
                else:
                    block.append(next_aln)

            except StopIteration:
                break

        # Yield last block
        yield block


def block_counter(forward_bam_file: str, reverse_bam_file: str) -> Tuple[int, int]:
    """
    Return as a tuple the number of blocks in the forward and reverse bam files.

    Parameters
    ----------
    forward_bam_file : str
        Path to forward .bam alignment file.
    reverse_bam_file : str
        Path to reverse .bam alignment file.

    Returns
    -------
    Tuple[int, int]
        Number of blocks in the forward and reverse bam files.
    """

    forward_bam_path = Path(forward_bam_file)
    reverse_bam_path = Path(reverse_bam_file)

    if not forward_bam_path.is_file():
        raise IOError(f"BAM file {forward_bam_path.name} not found. Please provide a valid path.")

    if not reverse_bam_path.is_file():
        raise IOError(f"BAM file {reverse_bam_path.name} not found. Please provide a valid path.")

    # ✅ Comptage indépendant pour détecter les fichiers déséquilibrés
    nb_blocks_for = sum(1 for _ in bam_iterator(forward_bam_path))
    nb_blocks_rev = sum(1 for _ in bam_iterator(reverse_bam_path))

    # ✅ Vérification de la cohérence entre les deux fichiers
    if nb_blocks_for != nb_blocks_rev:
        raise ValueError(
            f"Forward and reverse BAM files have different number of blocks: "
            f"{nb_blocks_for} vs {nb_blocks_rev}. Files may be corrupted or mismatched."
        )

    return (nb_blocks_for, nb_blocks_rev)


def chunk_bam(forward_bam_file: str = "group2.1.bam", 
              reverse_bam_file: str = "group2.2.bam", 
              nb_chunks: int = 2, 
              output_dir: str = None) -> None:

    logger.info("Start chunking BAM files")

    if output_dir is None:
        output_dir = Path(getcwd())
    else:
        output_dir = Path(output_dir)

    chunks_path = output_dir / "chunks"
    if chunks_path.is_dir():
        sh.rmtree(chunks_path)
    mkdir(chunks_path)

    forward_bam_path = Path(output_dir, forward_bam_file)
    reverse_bam_path = Path(output_dir, reverse_bam_file)

    if not forward_bam_path.is_file():
        raise IOError(f"BAM file {forward_bam_path.name} not found.")
    if not reverse_bam_path.is_file():
        raise IOError(f"BAM file {reverse_bam_path.name} not found.")

    # ✅ Comptage rapide via samtools (appel C, pas d'itération Python)
    # pysam.view("-c") équivaut à "samtools view -c" → beaucoup plus rapide
    n_alignments = int(pysam.view("-c", str(forward_bam_path)).strip())
    target_chunk_size = n_alignments // nb_chunks

    logger.info(f"Total alignments: {n_alignments}, target chunk size: {target_chunk_size}")

    forward_bam_handler = pysam.AlignmentFile(forward_bam_path, "rb")
    reverse_bam_handler = pysam.AlignmentFile(reverse_bam_path, "rb")

    output_chunk_for = chunks_path / "chunk_for_%d.bam"
    output_chunk_rev = chunks_path / "chunk_rev_%d.bam"

    chunk_index   = 0
    current_count = 0

    # Ouverture du premier chunk
    outfile_for = pysam.AlignmentFile(
        str(output_chunk_for) % chunk_index, "wb", template=forward_bam_handler
    )
    outfile_rev = pysam.AlignmentFile(
        str(output_chunk_rev) % chunk_index, "wb", template=reverse_bam_handler
    )

    # ✅ Une seule passe, écriture immédiate, changement de chunk à la volée
    for forward_block, reverse_block in zip(
        bam_iterator(forward_bam_path), 
        bam_iterator(reverse_bam_path)
    ):
        # Écriture immédiate
        for read in forward_block:
            outfile_for.write(read)
        for read in reverse_block:
            outfile_rev.write(read)

        current_count += len(forward_block)

        # ✅ Changement de chunk quand la taille cible est atteinte
        if current_count >= target_chunk_size and chunk_index < nb_chunks - 1:
            outfile_for.close()
            outfile_rev.close()

            chunk_index   += 1
            current_count  = 0

            outfile_for = pysam.AlignmentFile(
                str(output_chunk_for) % chunk_index, "wb", template=forward_bam_handler
            )
            outfile_rev = pysam.AlignmentFile(
                str(output_chunk_rev) % chunk_index, "wb", template=reverse_bam_handler
            )

    outfile_for.close()
    outfile_rev.close()

    forward_bam_handler.close()
    reverse_bam_handler.close()

    logger.info(f"Chunks saved in {chunks_path}")

def subsample_restriction_map(restriction_map : dict = None, rate : float = 1.0) -> dict[str, np.ndarray[int]]:
    """
    Subsample a restriction map by a given rate.

    Parameters
    ----------
    restriction_map : dict, optional
        Restriction map saved as a dictionary like chrom_name : list of restriction sites' position, by default None
    rate : float, optional
        Set the proportion of restriction sites to consider. Avoid memory overflow when restriction maps are very dense, by default 1.0

    Returns
    -------
    dict[str, np.ndarray[int]]
        Dictionary of sub-sampled restriction map with keys as chromosome names and values as lists of restriction sites' position.

    """

    if (0.0 > rate) or (rate > 1.0):
        raise ValueError("Sub-sampling rate must be between 0.0 and 1.0.")
    

    subsampled_restriction_map = {}

    for chromosome in restriction_map:
            
        if int(len(restriction_map.get(str(chromosome))) * rate) < 5:

            subsampled_restriction_map[str(chromosome)] = restriction_map[str(chromosome)]

            continue

        size_sample = int(len(restriction_map.get(str(chromosome))) * rate)

        subsampled_restriction_map[str(chromosome)] = np.random.choice(
            restriction_map.get(str(chromosome)), size_sample, replace=False
        )
        
        subsampled_restriction_map[str(chromosome)] = np.sort(subsampled_restriction_map[str(chromosome)])

        if subsampled_restriction_map[str(chromosome)][0] != 0:
            subsampled_restriction_map[str(chromosome)][0] = 0

        if (
            subsampled_restriction_map[str(chromosome)][-1]
            != restriction_map.get(str(chromosome))[-1]
        ):
            subsampled_restriction_map[str(chromosome)][-1] = restriction_map.get(str(chromosome))[-1]        

    return subsampled_restriction_map

def max_consecutive_nans(vector : np.ndarray) -> int:
    """
    Return the maximum number of consecutive NaN values in a vector.

    Parameters
    ----------
    vector : np.ndarray
        Vector to get the maximum number of consecutive NaN values from.

    Returns
    -------
    int
        Number of maximum consecutive NaN values.
    """

    mask = np.concatenate(([False], np.isnan(vector), [False]))
    if ~mask.any():
        return 0
    else:
        idx = np.nonzero(mask[1:] != mask[:-1])[0]
        return (idx[1::2] - idx[::2]).max()

def mad_smoothing(vector : np.ndarray[int] = None, window_size : int | str = "auto", nmads : int = 1) -> np.ndarray[int]:
    """
    Apply MAD smoothing to an vector .

    Parameters
    ----------
    vector : np.ndarray[int], optional
        Data to smooth, by default None
    window_size : int or str, optional
        Size of the window to perform mean sliding average in. Window is center on current value as [current_value - window_size/2] U [current_value + window_size/2], by default "auto"
    nmads : int, optional
        number of median absolute deviation to use, by default 1

    Returns
    -------
    np.ndarray[int]
        MAD smoothed vector.
    """

    mad = median_abs_deviation(vector)
    threshold = np.median(vector) - nmads * mad
    # threshold = 0
    imputed_nan_data = np.where(vector <= threshold, np.nan, vector)

    if window_size == "auto":
        # due to centered window, selected windows for rolling mean is :
        # [window_size / 2 <-- center_value --> window_size / 2]
        window_size = (max_consecutive_nans(imputed_nan_data) * 2) + 1

    averaged_data = (
        pd.Series(imputed_nan_data)
        .rolling(window=window_size, min_periods=1, center=True)
        .apply(lambda x: np.nanmean(x))
        .to_numpy()
    )
    
    averaged_data[averaged_data < 0] = 0
    
    return averaged_data

# not used anymore
def replace_consecutive_zeros_with_mean(vector : np.ndarray[float]) -> np.ndarray[float]:
    """
    Replace consecutive zeros in a vector with the mean of the flanking values.

    Parameters
    ----------
    vector : np.ndarray[float]
        Array to replace consecutive zeros in.

    Returns
    -------
    np.ndarray[float]
        Array with consecutive zeros replaced by the mean of the flanking values.
    """    
    
    # Initialize variables
    start = None
    end = None
    i = 0
    
    # Iterate through the array
    while i < len(vector):
        # Check for the start of a sequence of zeros
        if vector[i] == 0 and start is None:
            start = i
        # Check for the end of a sequence of zeros
        elif vector[i] != 0 and start is not None:
            end = i
            # Calculate the mean of the values flanking the sequence of zeros
            mean_value = (vector[start-1] + vector[end]) / 2 if start > 0 else vector[end]
            # Replace zeros with the mean value
            vector[start:end] = mean_value
            # Reset start and end for the next sequence
            start = None
            end = None
        i += 1
    
    # Handle case where sequence of zeros goes till the end of the array
    if start is not None:
        mean_value = vector[start-1] if start > 0 else 0  # Use the preceding value or 0 if at the start
        vector[start:] = mean_value
    
    return vector

def fitting_ps(x : np.ndarray[float], y : np.ndarray[float]) -> np.ndarray[float]:
    """
    Fit ps curves to have estimates notably of last points.

    Parameters
    ----------
    x : np.ndarray[float]
        Array of x (genomic distances log binned)
    y : np.ndarray[float]
        Array containing the number of events in function of x       

    Returns
    -------
    np.ndarray[float]
        Array with corrected values based on a fit procedure. 
    """   
    # Initialize variables
    y_original = y.copy()
    x_original = x.copy()
    # ========================================
    # 1. FILTRAGE DES DONNÉES POUR LE FIT
    # ========================================
    mask_fit = (x > 1000) & (y>0) # on ne fitte que x > 1000
    mask_unfitted = ~mask_fit  # points non-fittés
    
    x_fit = x[mask_fit]
    y_fit = y[mask_fit]
    
    if len(y_fit) >= 2:   # we start fit only with a minimum number of points 
        # ========================================
        # 2. FIT LOWESS INITIAL (en log-log)
        # ========================================
        frac = 0.40
        yhat_log = sm.nonparametric.lowess(
            np.log10(y_fit),
            np.log10(x_fit),
            frac=frac,
            return_sorted=False
        )
        residuals = np.log10(y_fit) - yhat_log
        
        # ========================================
        # 3. DÉTECTION LOCALE DES OUTLIERS
        # ========================================
        window = 30
        k = 1.5
        
        z = np.zeros_like(residuals)
        for i in range(len(residuals)):
            i0 = max(0, i - window // 2)
            i1 = min(len(residuals), i + window // 2)
            local_std = np.std(residuals[i0:i1])
            local_mean = np.mean(residuals[i0:i1])
            z[i] = (residuals[i] - local_mean) / (local_std + 1e-12)
        
        mask_good = np.abs(z) < k
        
        # ========================================
        # 4. TEST SUR LES CHUTES FINALES
        # ========================================
        y_lowess_lin = 10 ** yhat_log
        n_tail = min(5, len(x_fit))
        
        for i in range(1, n_tail + 1):
            idx = -i
            if idx < 0:
                rel_dev = (y_fit[idx] - y_lowess_lin[idx]) / y_lowess_lin[idx]
                if rel_dev < -0.6:
                    mask_good[idx] = False
        
        # ========================================
        # 5. REFIT LOWESS SANS OUTLIERS
        # ========================================
        yhat_clean = sm.nonparametric.lowess(
            np.log10(y_fit[mask_good]),
            np.log10(x_fit[mask_good]),
            frac=frac,
            return_sorted=True
        )
        x_clean = 10 ** yhat_clean[:, 0]
        y_clean = 10 ** yhat_clean[:, 1]
        
    
        # ========================================
        # 6. CRÉATION DES VECTEURS CORRIGÉS (VERSION ROBUSTE)
        # ========================================
        print(f"\n📊 Interpolation :")
        print(f"   Nombre de points x_clean : {len(x_clean)}")
        
        # Vérifier les doublons ou points mal formés
        if len(x_clean) != len(np.unique(x_clean)):
            print("   ⚠️ Doublons détectés dans x_clean !")
            unique_idx = np.unique(x_clean, return_index=True)[1]
            x_clean = x_clean[unique_idx]
            y_clean = y_clean[unique_idx]
            print(f"   → Nettoyé : {len(x_clean)} points uniques")
        
        # Choisir le type d'interpolation selon le nombre de points
        if len(x_clean) < 4:
            kind = 'linear'
            print(f"   Mode : LINEAR (trop peu de points)")
        elif len(x_clean) < 10:
            kind = 'linear'
            print(f"   Mode : LINEAR (sécurité)")
        else:
            kind = 'cubic'
            print(f"   Mode : CUBIC")
        
        try:
            f_interp = interp1d(
                x_clean, y_clean,
                kind=kind,
                bounds_error=False,
                fill_value='extrapolate'
            )
        except Exception as e:
            print(f"   ⚠️ Erreur interpolation {kind} : {e}")
            print(f"   Fallback sur LINEAR...")
            f_interp = interp1d(
                x_clean, y_clean,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
    
        # Évaluer la prédiction LOWESS sur tous les x
        y_pred_all = f_interp(x)
        
        # Évaluer seulement sur les x fittés
        y_fit_pred_full = f_interp(x_fit)
        
        # SI l'extrapolation donne des valeurs bizarres en fin,
        # on la remplace par la pente linéaire en fin de courbe
        x_clean_max = x_clean[-1]
        mask_beyond = x_fit > x_clean_max
        
        if np.any(mask_beyond):
            n_ref_local = 5
            slope_end = np.polyfit(np.log10(x_clean[-n_ref_local:]), 
                                   np.log10(y_clean[-n_ref_local:]), 1)[0]
            intercept_end = np.log10(y_clean[-1]) - slope_end * np.log10(x_clean[-1])
            y_fit_pred_full[mask_beyond] = 10 ** (slope_end * np.log10(x_fit[mask_beyond]) + intercept_end)
            
            # Aussi pour y_pred_all au-delà
            mask_beyond_all = x > x_clean_max
            y_pred_all[mask_beyond_all] = 10 ** (slope_end * np.log10(x[mask_beyond_all]) + intercept_end)
        
        # ========================================
        # 7. VECTEURS FINAUX
        # ========================================
        # Corriger les points fittés
        y_fit_corrected = np.where(mask_good, y_fit, y_fit_pred_full)
        
        # VECTEUR FINAL : points non-fittés (intacts) + points fittés corrigés
        y_final = np.concatenate([y_original[mask_unfitted], y_fit_corrected])
        x_final = np.concatenate([x_original[mask_unfitted], x_fit])
        
        # Trier par x pour que ce soit cohérent
        sort_idx = np.argsort(x_final)
        x_final = x_final[sort_idx]
        y_final = y_final[sort_idx]
    
    else :
        y_final= y_original
    
    
    return y_final

def get_chunks(output_dir : str = None) -> tuple([List[str], List[str]]):
    """
    Return a tuple containing the paths to the forward and reverse chunks.

    Parameters
    ----------
    output_dir : str, optional
        Path to get chunks from, by default None

    Returns
    -------
    tuple([List[str], List[str]]
        Tuple containing the paths to the forward and reverse chunks.
    """

    forward_chunks = sorted(glob(output_dir + '/chunks/chunk_for_*.bam'))
    reverse_chunks = sorted(glob(output_dir + '/chunks/chunk_rev_*.bam'))

    return (forward_chunks, reverse_chunks)

def is_empty_alignment(alignment_file : str) -> bool:
    """
    Check if an alignment file is empty.
    If empty, return True, else return False.

    Parameters
    ----------
    alignment_file : str
        Path to the alignment file to check.

    Returns
    -------
    bool
        Return True if the file is empty, False otherwise.
    """    
    try:
        # Open the SAM/BAM file
        with pysam.AlignmentFile(alignment_file, "rb") as alignment:
            # Attempt to fetch the first read
            try:
                alignment.__next__()
                # If we can fetch a read, the file is not empty
                return False
            except StopIteration:
                # If StopIteration is raised, the file is empty
                return True
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return True  # Assuming file is "empty" if it doesn't exist

def format_blacklist(blacklist : str = None) -> dict[str, Tuple[int, int]]:
    """
    Format a blacklist file into a dictionary.

    Parameters
    ----------
    blacklist : str, optional
        Path to the blacklist file. If set to -1 this is equivalent to None for workflow managers purpose, by default None

    Returns
    -------
    dict[str, Tuple[int, int]]
        Dictionary with chromosome name as key and tuple of start and end as value.
    """

    if blacklist is None or blacklist == "-1":
        return None
    
    if not isinstance(blacklist, str):
        raise TypeError("Blacklist should be a string")

    if not Path(blacklist).exists():
        pieces = blacklist.split(',')
        chromosomes_found = np.unique([blacklist.split(':')[0] for p in pieces])
        indexes_dict = {chrom: 0 for chrom in chromosomes_found}

        result = {}
        for piece in pieces:
            key, value = piece.split(':')
            if key in result:
                index = indexes_dict[key]
                new_key = f'{key}_{index}'
                indexes_dict[key] += 1

            else : 
                new_key = key
            result[new_key] = tuple([int(x) for x in value.split('-')])

        return result
    
    else:
        result = {}
        with open(blacklist, 'r') as f:
            for line in f:
                chrom, start, end = line.split()
                if chrom in result:
                    index = 0
                    while f"{chrom}_{index}" in result:
                        index += 1
                    result[f"{chrom}_{index}"] = (int(start), int(end))
                else:
                    result[chrom] = (int(start), int(end))
        return result

def is_blacklisted(read_forward : pysam.AlignedSegment, read_reverse : pysam.AlignedSegment, blacklist : dict[str, Tuple[int, int]] = None) -> bool:
    """
    Check if a read pair is blacklisted based on a list of coordinates.

    Parameters
    ----------
    read_forward : pysam.AlignmentSegment
        Forward read of the pair.
    read_reverse : pysam.AlignedSegment
        Reverse read of the pair.
    blacklist : dict[str, Tuple[int, int]]
        Blacklist of coordinates to check against. Chromsome name as key and tuple of start and end as value.
        Chromosome names should be formatted as 'chr1_A', 'chr1_B', etc. With A and B being the index of the coordinates to blacklist in a given chromosome.

    Returns
    -------
    bool
        True if the read pair is blacklisted, False otherwise.
    """

    if blacklist is None:
        return False
    
    if read_forward.query_name != read_reverse.query_name:
        raise ValueError("Reads are not coming from the same chromosome")
    
    forward_start = read_forward.reference_start if not is_reverse(read_forward) else read_forward.reference_end
    reverse_start = read_reverse.reference_start if not is_reverse(read_reverse) else read_reverse.reference_end

    forward_check = [low < forward_start < high and read_forward.reference_name == chrom.split("_")[0] for chrom, (low, high) in blacklist.items()]
    reverse_check = [low < reverse_start < high and read_reverse.reference_name == chrom.split("_")[0] for chrom, (low, high) in blacklist.items()]

    return any([f_check or r_check for f_check, r_check in zip(forward_check, reverse_check)])


# Benchamrk analysis functions :

def pearson_score(original_matrix : cooler.Cooler, rescued_matrix : cooler.Cooler , markers : list[int]) -> float:
    """
    Compute Pearson correlation between concatenated matrix bins which have been deleted and reconstructed.
    
    Parameters
    ----------
    original_matrix : cooler.Cooler
        Cooler object containing the original matrix.
    rescued_matrix : cooler.Cooler
        Cooler object containing the reconstructed matrix.
    markers : list[int]
        List of markers to consider. Markers are the bins which have been deleted.

    Returns
    -------
    float
        Pearson correlation between original and reconstructed matrix.
    """    
    ori_matrix = original_matrix.matrix(balance=False)[:]
    reco_matrix = rescued_matrix.matrix(balance=False)[:]

    ori_vector = ori_matrix[markers]
    reco_vector = reco_matrix[markers]

    pearson_score = pearsonr(ori_vector.flatten(), reco_vector.flatten())

    return pearson_score[0]

def get_top_pattern(file : str = None, top : int = 10, chromosome : str = None) -> pd.DataFrame:
    """
    Get top patterns from a dataframe

    Parameters
    ----------
    df : pd.DataFrame, optional
        Dataframe containing patterns given by Chromosight, by default None
    top : int, optional
        Percentage of top patterns to get, by default 10
    chromosome : str, optional
        Chromosome to consider, by default None

    Returns
    -------
    pd.DataFrame
        Dataframe containing top percentage patterns.
    """
    df = pd.read_csv(file, sep = "\t", header = 0)
    top_factor = (df.shape[0] * top) // 100

    if chromosome is not None:
        df = df.query(f"chrom1 == '{chromosome}' and chrom2 == '{chromosome}'")
    df_top = df.sort_values(by='score', ascending=False).head(top_factor).reset_index(drop=True)

    return df_top








