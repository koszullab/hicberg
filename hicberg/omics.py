import subprocess as sp
from pathlib import Path
import numpy as np

import hicberg.io as hio
from hicberg import logger
import logging
import tempfile
from collections import defaultdict

logger = logging.getLogger(__name__)

import pysam
from collections import defaultdict

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


   

# alternative process (no need to install bedgraph or bedGraphToBigWig)

def concatenate_bam(bam_file1 : str = "group1.1.bam", 
                    bam_file2 : str = "group2.1.rescued.bam", 
                    cpus : int = 8, 
                    output_dir : str = None,
                    output_name : str = "groups1_and_2.1.bam") -> None:
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
        Output name, by default groups1_and_2.1.bam

    """
    
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_dir():
        raise IOError(f"Output directory {output_dir} not found.")
    
    output_name = Path(output_dir, output_name)
    
    bam_file1 = Path(output_dir, bam_file1)
    bam_file2 = Path(output_dir, bam_file2)   
        
    cmd = f"samtools merge -@ {cpus} -f {output_name} {bam_file1} {bam_file2}"
    sp.run([cmd], shell = True)
        
    logger.info(f"Saved data at {output_dir}")           
          
     

def reconstruct_paired_bam(bam_file1: str, 
                           bam_file2: str, 
                           output_bam: str = None ) -> None:
    """
    Reconstruit un BAM paired-end à partir de deux BAM alignés en single-end.
    Suppose que bam_file1 contient les R1 et bam_file2 les R2, avec les mêmes noms de query.
    """

    # Ouvrir le premier BAM pour récupérer le header
    with pysam.AlignmentFile(bam_file1, "rb") as bam1:
        header_dict = bam1.header.to_dict()

    # Ajouter un tag @PG pour documenter la transformation
    pg_entries = header_dict.get("PG", [])
    if isinstance(pg_entries, dict):  # parfois pysam retourne un dict unique
        pg_entries = [pg_entries]
    pg_entries.append({
        "ID": "reconstruct_paired",
        "PN": "reconstruct_paired",
        "CL": f"reconstruct_paired_bam({bam_file1}, {bam_file2}, {output_bam})",
        "VN": "1.0"
    })
    header_dict["PG"] = pg_entries

    # Recréer le header
    new_header = pysam.AlignmentHeader.from_dict(header_dict)

    with (
        pysam.AlignmentFile(bam_file1, "rb") as bam1,
        pysam.AlignmentFile(bam_file2, "rb") as bam2,
        pysam.AlignmentFile(output_bam, "wb", header=new_header) as out
    ):

        # Indexer les R1 par nom de read
        r1_by_name = defaultdict(list)
        for read in bam1:
            r1_by_name[read.query_name].append(read)

        paired_count = 0
        unpaired_r2 = 0
        unpaired_r1 = 0
        diff_chr = 0

        for r2 in bam2:
            mates = r1_by_name.get(r2.query_name, [])

            if not mates:
                unpaired_r2 += 1
                continue

            # Prendre le premier R1 disponible avec ce nom
            r1 = mates.pop(0)
            if not mates:
                del r1_by_name[r2.query_name]

            # Vérifier qu'ils sont sur le même chromosome
            if r1.reference_id != r2.reference_id:
                diff_chr += 1
                # continue

            # Coordonnées
            r1_start = r1.reference_start
            r2_start = r2.reference_start
            r1_end = r1.reference_end
            r2_end = r2.reference_end

            if r1_end is None or r2_end is None:
                unpaired_r2 += 1
                continue

            leftmost_start = min(r1_start, r2_start)
            rightmost_end = max(r1_end, r2_end)
            tlen = rightmost_end - leftmost_start

            # Orientation conventionnelle paired-end : R1 forward, R2 reverse
            # On garde l'orientation d'origine
            r1_is_reverse = r1.is_reverse
            r2_is_reverse = r2.is_reverse
            is_proper = (not r1_is_reverse) and r2_is_reverse

            # --- Mettre à jour R1 ---
            r1.flag = 0x1      # PAIRED
            r1.flag |= 0x40    # READ1
            if is_proper:
                r1.flag |= 0x2  # PROPER_PAIR
            if r1_is_reverse:
                r1.flag |= 0x10  # READ_REVERSE
            if r2_is_reverse:
                r1.flag |= 0x20  # MATE_REVERSE

            r1.next_reference_id = r2.reference_id
            r1.next_reference_start = r2.reference_start
            r1.template_length = tlen if r1_start == leftmost_start else -tlen

            # --- Mettre à jour R2 ---
            r2.flag = 0x1      # PAIRED
            r2.flag |= 0x80    # READ2
            if is_proper:
                r2.flag |= 0x2  # PROPER_PAIR
            if r2_is_reverse:
                r2.flag |= 0x10  # READ_REVERSE
            if r1_is_reverse:
                r2.flag |= 0x20  # MATE_REVERSE

            r2.next_reference_id = r1.reference_id
            r2.next_reference_start = r1.reference_start
            r2.template_length = tlen if r2_start == leftmost_start else -tlen

            out.write(r1)
            out.write(r2)
            paired_count += 1

        # Éventuellement écrire les R1 restants sans R2 (singletons)
        for remaining in r1_by_name.values():
            unpaired_r1 += len(remaining)
            # Optionnel : les écrire en single-end
            # for r1 in remaining:
            #     out.write(r1)

    print(f"{bam_file1}")
    print(f"{bam_file2}")
    print(f"Paired reads written: {paired_count}")
    print(f"R1 without R2 mate: {unpaired_r1}")
    print(f"R2 without R1 mate: {unpaired_r2}")
    print(f"R1/R2 on different chromosomes: {diff_chr}")





def bam_to_bigwig(
    bam_file1: str = "group1.1.bam",
    bam_file2: str = "group1.2.bam",
    dist_min_omics: int = 100,
    dist_max_omics: int = 1000,
    cpus: int = 8,
    output_dir: str = None,
    output_bam: str = "group.1.2.repaired.bam", 
    output_name: str = "signal.bw",
) -> None:
    """
    Convert two BAM files to a BigWig file (.bw).
    """

    output_dir_path = Path(output_dir)
    if not output_dir_path.is_dir():
        raise IOError(f"Output directory {output_dir} not found.")

    # Chemins d'entrée : absolus si fournis, sinon relatifs à output_dir
    bam1 = Path(bam_file1) if Path(bam_file1).is_absolute() else output_dir_path / bam_file1
    bam2 = Path(bam_file2) if Path(bam_file2).is_absolute() else output_dir_path / bam_file2
    bam_file = Path(output_dir, output_bam)
    bw_file = Path(output_dir, output_name)

    for b in (bam1, bam2):
        if not b.is_file():
            raise FileNotFoundError(f"BAM file not found: {b}")

    # pairing the 2 files:
    reconstruct_paired_bam(
            bam_file1 = bam1,
            bam_file2 = bam2,
            output_bam = bam_file,
        )
    
    # Trier et indexer (bamCoverage a besoin d'un BAM trié par coordonnées)
    sorted_bam = bam_file.with_suffix(".sorted.bam")
    pysam.sort("-@", "8", "-o", str(sorted_bam), str(bam_file))
    pysam.index(str(sorted_bam))

    print(f"Sorted output: {sorted_bam}")
    
    # Conversion en BigWig
    sp.run(
        [
            "bamCoverage",
            "-b", str(sorted_bam),
            "-o", str(bw_file),
            "--binSize", "1",
            "--extendReads",
            "--minFragmentLength", str(dist_min_omics),
            "--maxFragmentLength", str(dist_max_omics),
            "--normalizeUsing", "None",
            "--numberOfProcessors", str(cpus),
            "--skipNonCoveredRegions",
        ],
        check=True,
    )

    logger.info(f"Saved data in BigWig format at {output_name}")    




       
   
    
    
    
    