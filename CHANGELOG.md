# CHANGELOG

All significant changes made to this project will be documented in this file (document started 25th of nov 2025).


## [1.0.2] - 2026-02-25

### Added
- Addition of conditions: Nan or None values are replaced with zeros for propensities in the draw_read_couple function.
- Adding plot of normalised matrices
- Adding of mad_max parameter in the build_matrix function (for cooler balance to have all bins after normalisation)
- Adding copy of the log file into the output repository
- Adding in the function mad_smoothing:   averaged_data[averaged_data < 0] = 0
- Adding by default working directory as output repository  (default = '.' in all click options)
- Creation of the hicstuff_process function to process like hicstuff the pairs files before building the matrices in cool format
- Adding of or output_dir == '.':  to the create_folder function and minor changes in the logger.info.
- Adding of x ticks in the plots of contact maps
- Adding --trim5 option (Trim <int> bases from 5' (left) end of each read before alignment (default: 0) )
- Adding of checks for start_stage exit_stage mode entered by the user

### Changed
- Change the plots of inter contacts map. Computation with dist_frag was used now we plot what is in the object TRANS_PS.
- Change the names of the variable loops into circles and patterns into events.
- Change the minimum mapping quality to 30 instead of 35 to be coherent with default hicstuff pipeline
- Change standard mode (ps and coverage) by default instead of full
- Change the computation of the number of multi-mapped couples
- Change bowtie2 default mode from very-sensitive to very-sensitive-local
- Change nan values with masked matrice and resolution for matrices pdf
- Change minor things like labels and colors in the plots of matrices
- Make bedtools and bedGraphToBigWig tools optional to continue the pipeline
- Change name function is_duplicated to is_ambiguous
- Change of forward_read.pos (which was deprecated) into forward_read.reference_start
- Change of forward_read.start +1  to be coherent with hicstuff
- Simplification of the function build_matrix
- Change of names: group1.pairs to unrescued.pairs and all_group.pairs to rescued.pairs
  and unrescued.pairs.cool, rescued.pairs.cool for the matrices to have consistent names
- Rewritting of compute_propensity function (notably removal of multiple conditions on the chosen mode)
- Change name compute_density into generate_density_map2  (not used by default)
- Move of pearson_score function into utils module and modifition of its calling hut.pearson_score
- Move of get_top_pattern function into utils module (a function with this name is also present in eval module)
- Change names get_patterns into generate_intra_ps and get_pair_ps into get_intra_ps
- Change variable name bins into bin_size (bins was also used for the bins table file)
- Change of get_density (compute once the density map per possible couple of chr)
- Use of itertools.combinations_with_replacement used in generate_density_map2 to compute once the density map for chr couples
- Change of get_intra_ps function (more readable)
- In the functions get_d1d2 and generate_d1d2, change read_forward.pos (which was deprecated) into read_forward.reference_start
- In the function generate_coverages, change read_forward.pos (which was deprecated) into read_forward.reference_start
- Change variable name max-alignment into max_alignment for consistency
- in the function generate_trans_ps, change : chromosomes_detected = matrix.chromnames to compute only for chr already detected in the rescued data
- add of replacement of zeros by non zeros minimums in the get_local_density function
- change the function isotonic_regression from scipy package instead of sklearn package (not used by default, to be tested)
- change of the 3 functions bam_iterator block_counter and chunk_bam because chunk_bam uses a lot of RAM and crashes for big files
- Change of the function classify_reads (faster)
- Change compute_propensity_mode  (Simplification)
- Change omics mode (use of sam tools and basecoverage from deeptools)

### Fixed
- Corrected bug on reads directionalities writting in the function build_pairs of io.py to apply filters
- Correction of the function is_poor_quality in utils which took reads with mapq = 0
- Correction of the function get_restriction_map which did not manage a list of several enzymes (req: restriction_map is not used in standard mode) and change in the click option
- Correction of the circular option to enter serveral chromosomes (conversion of circular string variable into list)
- Change of get_local_density function (use of generic_filter from scipy and several iterations)
- Change get_d1d2 function to tackle cases where d1d2 was not found in the unrescued part of the data
- Change of the generate_intra_ps to tackle last points that were outliers, adding fitting_ps function (to have good estimates of this behavior for last points)
- Correction of the plot function of d1d2 (did 2 times the histogram)
- Adding of the circular parameter at the function compute_propensity_mode in the reattribute_reads function
- Adding of the circular parameter at the function hst.reattribute_reads in the pipeline function


### Removed
- Removal of the computation of possibles genomic distances with the restriction table
- Removal of is_unqualitative function which was not used and misleading
- Removal of sum_mat_bins function which was not used
- Removal of --zero-based option in cooler construction of matrice from pairs file to be coherent with hicstuff
- Removal of instructions that assigned a value of 1 when the propensity for ps was zero
- Removal of instructions that assigned a value of 0 when the coverage was negative
- Removal of CLR variable (not used anymore) in plot module
- Removal of generate_density_map_backup and generate_density_map functions
- Removal of detrend_matrix function and use the one from hicstuff instead and adding of distance_law and detrend functions from hicstuff
- Removal of one_enzyme mode


## [1.0.1] - 2025-06-01
- Update for release of version 1.0.1 for Zenodo archiving purpose.


## [1.0.0] - 2025-01-01

### Initial release
- Initial release of the project.
