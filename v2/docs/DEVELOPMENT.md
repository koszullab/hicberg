# HiCberg Human-Scale — Development Notes

## Context

HiCberg is a pipeline to rescue multi-mapping reads in Hi-C / Micro-C data.
The current version works well on small genomes with low repeat copy numbers (~50).
This development branch targets **human genome scale** with:
- Near-T2T diploid assembly of GM12878 (Platinum Pedigree, AWS)
- Micro-C data from GM12878 (Dovetail 800M reads or Cleopatra ultra-deep)
- Repeat elements with thousands of copies (LINE-1, Alu...)
- bowtie2 alignment with k=100

---

## Architecture decisions

### Reference genome

- **GM12878 near-T2T diploid phased assembly** (Platinum Pedigree CEPH 1463 consortium)
  - Available on AWS Open Data: `s3://platinum-pedigree-data/`
  - ~63.3% of chromosomes are near-T2T for generations G1-G3
  - Used in CiFi (Nature Communications 2025) for phasing long-read chromatin data
- Alternative for development/testing: **T2T-CHM13v2.0** (GCA_009914755.4)
- For validation: **yeast genome** (small, well-annotated, known P(s) curves)

---

## Pipeline overview (v2 branch)

```
FASTQ_1  FASTQ_2
   |         |
   ▼         ▼
bowtie2 --very-sensitive-local --reorder (NO -k, NO trim)
   |         |
   ▼         ▼
 1.sam     2.sam
      \   /
       ▼ ▼
  streaming_classifier_v2.py
       |
  ┌────┴─────────────┐
  │                  │
group0            group1              group2
unmapped       unrescued.pairs    group2_1_chunk{i}.fastq
names.txt      (pairtools fmt)    group2_2_chunk{i}.fastq
                     |            chimeric_names.txt
                     ▼                 |
             statistics_v2.py          ▼
         P(s), trans P(s), coverage   bowtie2 -k 100 --very-sensitive-local
         (density: TODO)              (pipe stdout → never written to disk)
                     |                 |
                     ▼                 |
           plots/*.pdf (plots_v2.py)   |
                     |                 |
          ══ CHECKPOINT: check the ══  |
          ══ laws, stop if wrong   ══  |
                     |                 |
                     └───────┬─────────┘
                             ▼
                     reattribute_reads()
                     (multi-scale vectorised)
                             |
                             ▼
                      predicted.pairs
```

The statistical laws are plotted as soon as they are computed (before the
re-attribution), so that a wrong signal (P(s), coverage, density) can be
spotted and the run stopped before the costly re-attribution step.

---

## Module 1: streaming_classifier_v2.py

**File:** `streaming_classifier_v2.py` (root of the `v2` branch)

### Key decisions

**Why no BAM sort:**
bowtie2 `--reorder` guarantees reads come out in FASTQ input order →
pairs are always consecutive in the SAM → no need for samtools sort.
This saves: SAM→BAM conversion + samtools sort (huge time/disk savings at human scale).

**Why no `-5 100` trim:**
Empirical test on 10,000 reads showed:
- With trim   : 2567 reads with XS (all clean repeats, 100%)
- Without trim : 5555 reads with XS → 3173 repeats (57%) + 1344 chimeric (24%) + 1038 5'clip (19%)
The trim was eliminating 606 genuine multi-mappers to filter 1344 chimeras.
Solution: use `--very-sensitive-local` (handles soft-clipping natively) + post-alignment CIGAR filter.

**Ambiguity criterion: XS tag (not MAPQ)**
- `XS` present → bowtie2 found at least one alternative alignment → truly ambiguous
- MAPQ < 30 alone is unreliable: a uniquely-mapping read with heavy soft-clip can have low MAPQ
- With `-k 100`, MAPQ is set to 255 (undefined) → completely unusable as criterion

**Chimeric read detection:**
```python
is_chimeric = (last_cigar_op == 'S' and last_cigar_len > 30) OR (aligned_bp / query_length < 0.5)
```

**Chimeric + ambiguous reads:**
A read can be BOTH chimeric AND a genuine repeat.
→ Extract largest M block from CIGAR (regardless of position in read)
→ Send this pure chunk to group2 FASTQ for second alignment
→ Log read name in chimeric_names.txt for later processing

**Why largest M block (not 5'-most):**
- Hi-C/Micro-C ligation junction can occur anywhere in the read
- Both sides of the junction can be on a repeat
- Taking largest block maximises information for discriminating repeat copies

### Classification logic

```
pair (r1, r2)
    │
    ├─ r1 unmapped OR r2 unmapped  →  group0: group0_names.txt
    │
    ├─ XS absent on r1 AND r2      →  group1: unrescued.pairs
    │
    └─ XS present on r1 OR r2      →  group2
           │
           └─ for each ambiguous read:
                  if chimeric → extract largest M chunk (≥30 bp)
                                log name in chimeric_names.txt
                  else        → keep full sequence
           → write to group2_1_chunk{i}.fastq / group2_2_chunk{i}.fastq
```

### Position convention in unrescued.pairs

**pairtools standard:**
```python
forward (+) : pos = reference_start + 1   # 0-based → 1-based
reverse (-) : pos = reference_end          # 0-based exclusive in pysam = 1-based inclusive of last aligned base
```


**Note:** The previous hicberg version computed distances directly from BAM objects via `get_cis_distance()` + `get_ordered_reads()`, which correctly handled forward/reverse positions using pysam. In the new version, since distances are computed from `unrescued.pairs`, the correct outer-most 5' position must be encoded at write time via `get_outer_5prime()`. Both approaches give identical results — only the moment where the forward/reverse distinction is applied changes.

**Validation:** confirmed on read VH01685:169:AAFWFNHM5:1:1101:24101:27694
```
r1 flag=0  (forward) : ref_start=30108081 → pos1 = 30108082
r2 flag=16 (reverse) : ref_end=30108134   → pos2 = 30108134
distance = 52 bp  (correct fragment size)
```

### Performance

Fast path: only group2 pairs (~15%) go through the full `parse_sam_line()`
(sequence, qualities, CIGAR tuples). group0/group1 lines are split once
(tags kept unsplit), XS found by substring search, 5' position computed from
POS + CIGAR reference length. Outputs strictly identical (byte-for-byte on yeast).

| Dataset | Before | Fast path | RAM |
|---------|--------|-----------|-----|
| Yeast 10M pairs (36 bp) | 2 min 29 s | 1 min 11 s | 12 MB |
| Human 500M pairs (150 bp), extrapolated | ~2-4 h | ~1-1.5 h | ~12 MB |

Single core. Next step (pipeline level): split FASTQ into N chunks and stream
`bowtie2 | classifier` per chunk → no SAM on disk (~450 GB for human) and
time divided by N.

### Parallelisation

Group2 FASTQ is split into `n_cpus` chunks at creation time (round-robin per pair).
This replaces the BAM chunking step in the current pipeline which was slow.
Chunks are balanced by pairs (never split a pair across chunks).

### CLI usage

```bash
python3 streaming_classifier_v2.py \
    1.sam 2.sam ./output_dir \
    --n_cpus 20 \
    --min_chunk_bp 30 \
    --threshold_clip 30 \
    --min_aligned_frac 0.5
```

### Outputs

| File | Content |
|------|---------|
| `group0_names.txt` | read names with ≥1 unmapped mate |
| `unrescued.pairs` | pairtools format, group1 pairs |
| `group2_1_chunk{i}.fastq` | group2 R1, split in n_cpus chunks |
| `group2_2_chunk{i}.fastq` | group2 R2, split in n_cpus chunks |
| `chimeric_names.txt` | read names of chimeric reads sent to group2 |
| `stats.txt` | counts per group + chimeric |

---

## Module 2: statistics_v2.py (P(s) + coverage done, density TODO)

**File:** `statistics_v2.py` — tests in `tests/test_statistics_v2.py`

Single streaming pass over `unrescued.pairs` (pandas chunks + `np.bincount`).
Yeast 7.7M pairs: 8 s, 0.8 GB RAM → human ~500M pairs estimated ~10 min.
Output file names and normalisations identical to the previous version
(`xs.npy`, `uncuts.npy`, `circles.npy`, `weirds.npy`, `trans_ps.npy`,
`coverage1.npy`, `coverage.npy`) + raw counts of the 4 orientations in `ps_counts.npz`.
Smoothing reuses `fitting_ps` / `mad_smoothing2` from `hicberg.utils` (temporary dependency).

```bash
python3 statistics_v2.py unrescued.pairs ./stats \
    --chrom_sizes 1.sam          # @SQ header, or chrom.sizes / .fai
    --circular plasmid2micron \
    --bin_size 2000
```

The PDF plots of the laws are generated at the end in `./stats/plots/`
(`--no_plots` to disable).

**Conventions (must be reused by reattribute_reads):**
- distance `d = |pos2 - pos1|` on the outer 5' positions of the pairs file
  (`min(d, L - d)` on circular chromosomes), same `d` for building and lookup
- coverage bin `= (pos - 1) // bin_size`
- orientation defined after ordering reads by position (leftmost first)

**Circular chromosomes (`--circular plasmid2micron,chrMT`):**
- distance `d = min(|pos2 - pos1|, L - |pos2 - pos1|)` → `d ≤ L/2`
- normalisation: a circle has `L` position pairs at every distance `d ≤ L/2`
  (vs `L - d` for a linear chromosome) → area = `L × bin width`, 0 beyond `L/2`
- if chrMT is not declared circular, its P(s) shows an artificial upturn
  near `d ≈ L` (contacts across the origin of the linearised sequence)
- previous version: same circular distance (`classify_reads()` overwrites
  the bowtie2 `XG` tag with the chromosome length, used by `get_cis_distance()`),
  but linear trapezoid areas → v1 vs v2 P(s) differ on circular chromosomes
  by the ratio of the areas, increasingly with distance.

**Yeast check:** beyond 5 kb, the 4 orientations are at 25.0% ± 0.1% each;
below 1 kb, `+-` = 69% (dangling ends).

**Validation vs v1 (yeast AC1 10M, 2026-11-12)** — plots in
`/media/axel/EVO/data_for_v2/yeast/stats_v2/compare_v1/`:
- group1: 7,709,374 (v2) vs 7,708,946 (v1) pairs; same `xs` bins
- P(s), 16 linear chromosomes: median ratio v2/v1 = 1.000 for the 3 laws.
  Median deviation < 0.2% beyond 1 kb, 0% at all distances for circles.
- < 1 kb: distance convention. v1 used one formula per orientation + 1 when
  building (but not at lookup); vs v2: uncut offset 2 bp, weird 1 bp,
  circle 0 bp → large relative deviations only in the 1-bp-wide bins < 100 bp.
- isolated ~10% jumps beyond 1 kb: **`fitting_ps` outlier detection**
  (z-score > 1.5) flips points between versions although raw values agree
  within ~1% → unstable smoothing, likely cause of the P(s) tail fluctuations
  seen on reconstructed maps → to replace.
- coverage: +0.006% reads; v1 counted reverse reads at `reference_start`
  (v2: 5' end, which is also what v1 used at lookup) → median 0.5% per bin;
  v1 has 1 extra empty bin on some chromosomes (`round(L/bin + 1)`).
  Smoothed coverage: 45/6115 bins differ > 20%, all at chromosome ends with
  0 reads (values extrapolated by `mad_smoothing2`) → mappability mask needed.
- chrMT / plasmid2micron: median ratio 0.92-0.99, expected (circular
  normalisation, v1 used linear trapezoids).

### P(s) — probability of contact vs genomic distance

Computed from `unrescued.pairs` (not BAM).

**4 orientations to track separately:**

| Orientation | Name | Biology |
|-------------|------|---------|
| `+/-` inward  | is_uncut  | true Hi-C contacts → classic decaying P(s) |
| `-/+` outward | is_circle | re-ligation artefacts → peak at short distance |
| `++` same     | is_weird  | artefacts |
| `--` same     | is_weird  | artefacts |

**Important:** do NOT filter artefacts before reconstruction.
P(s) must be computed on ALL pairs (artefacts included) so that the
reconstructed group2 carries the same biases → filter AFTER reconstruction.

**Distance calculation:**
The previous version computed a different formula per orientation from BAM
coordinates (`get_cis_distance()`). In v2, `pos` in the pairs file is already
the outer 5' position of each read, so a single formula covers all cases:
`d = |pos2 - pos1|`. It differs from the old one by at most ~2 bp (for `--` pairs,
the old code used `reference_start` → differs by the difference of the two
aligned lengths).
Short distances are real biological artefacts (dangling ends, re-ligations),
not calculation errors.

### Coverage (smoothed 1D)

- Computed per chromosome from group1 alignments
- Bin size: TBD (current code uses 2 kb)
- Smoothing: Gaussian (as in current code)
- Storage: numpy array per chromosome

### Density map (2D)

- For human genome: 23×23 = 529 sub-matrices
- Strategy: `np.memmap` shared between workers to avoid RAM explosion
- Computed by diagonal bands (intra-chromosomal contacts dominate)
- LRU cache for repeated lookups on same bin pairs

### Plots: plots_v2.py

Multi-page PDFs (one page per chromosome) from any statistics folder (v1 or v2,
same file names). `--compare_dir` overlays a reference folder (previous version).

```bash
python3 plots_v2.py ./stats                              # v2 only
python3 plots_v2.py ./stats --compare_dir old/statistics # v2 vs v1
```

| File | Content |
|------|---------|
| `ps.pdf` | P(s) uncut / circle / weird, log-log (v1 as lines if compared) |
| `ps_ratio.pdf` | P(s) ratio v2 / v1 per chromosome + summary page (median ratios) |
| `orientations.pdf` | fraction of +-, -+, ++, -- vs distance (should reach 25% each) |
| `coverage.pdf` | raw + smoothed coverage |
| `trans_ps.pdf` | trans P(s) heatmap (+ scatter v2 vs v1) |

### Mappability: mappability_v2.py (P(s) done, coverage/density TODO)

```bash
# once per genome and read length (yeast: 10 s, 100 MB RAM)
python3 mappability_v2.py genome.fa ./mappability -K 35 -E 2 -T 8
python3 statistics_v2.py unrescued.pairs ./stats ... \
    --mappability ./mappability/genmap_K35_E2.bedgraph   # --map_resolution 100
```

- mask = per bin, **fraction of positions whose k-mer is unique**
  (GenMap score ≥ `--min_mappability`, default 1). Not the mean score: a read
  from a 2-copy position has an XS tag → group2 → never observed in group1.
- P(s) area = number of pairs of mappable positions per distance bin =
  autocorrelation of the mask (FFT, `--map_resolution` 100 bp; human chr1 →
  2.5M points). Mask = 1 everywhere → exactly the trapezoids (tested).
  `C[0] = Σ f` (a position paired with itself), circular autocorrelation
  for circular chromosomes.
- trans P(s): `contacts / (4 × mappable L1 × mappable L2)`
- `mappability.npy`: mask at the coverage bin size (2 kb), plotted with the
  coverage (grey, right axis)

**Yeast AC1 (K=35, E=2): 91.6% of the genome mappable.**
- P(s) level × 1.04 (chrXI) to × 1.40 (chrI): the trapezoid underestimated
  the P(s) differently on each chromosome (chrI: small, large subtelomeric
  repeats) → matters for re-attribution between chromosomes.
- roughness of raw log P(s) beyond 100 kb (std of second differences):
  0.131 → 0.064, Poisson counting noise = 0.053 → most chromosomes are now at
  the noise floor (chrI 0.68 → 0.13, chrXVI 0.17 → 0.04). Still above: chrXII
  (rDNA), chrX, chrXI, chrIII.
- coverage: unmappable bins = coverage drops (subtelomeres); `mad_smoothing2`
  extrapolates over them → next step.

---

## Module 3: reattribute_reads (TODO)

### Context

With k=100: max 100×100 = 10,000 combinations per block.
Real problem: millions of blocks to process.
Mode: `full` = ps × density

### Multi-scale filtering strategy

```
Block of ambiguous reads (N_for × N_rev ≤ 10,000)
        │
  FILTER 1 — Chromosomal  O(N+M)
  Eliminate trans pairs with P(trans) ≈ 0
        │
  FILTER 2 — Low-resolution coverage  O(N'+M')
  Coarse bins (50 kb): keep top-K candidates per arm
        │
  FILTER 3 — Vectorised numpy  O(N''×M'')
  ps × density as np.outer on survivors
  Weighted random draw
        │
  Selected read pair → predicted.pairs
```

### Vectorisation of propensities

```python
# Coverage: outer product → N×M matrix in one operation
cov_for = np.array([coverage[c][b] for c, b in zip(for_chroms, for_bins)])
cov_rev = np.array([coverage[c][b] for c, b in zip(rev_chroms, rev_bins)])
cov_matrix = np.outer(cov_for, cov_rev)

# P(s): group by chromosome pair, then vectorise per group
# → avoids looping over all N×M combinations
```

---

## Validation strategy

1. Run new pipeline on **yeast** (10M reads, known results)
2. Compare P(s) curves: new vs old pipeline
3. Compare density maps: new vs old pipeline
4. Compare rescued contact matrices on repeat regions
5. If validation passes → run on human GM12878 Micro-C

### Test data locations

```
tests/data/yeast/
    1.sam                   # 10M reads R1
    2.sam                   # 10M reads R2
    expected/
        unrescued.pairs     # from current hicberg pipeline
        ps_curve.npy        # reference P(s)
        density_map.npy     # reference density

tests/data/human/
    chr21_1.sam             # reduced dataset for development
    chr21_2.sam
```

---

## Environment

- OS: Linux Mint (Debian/Ubuntu based)
- Python: via conda (base environment)
- Key dependencies: pysam, numpy, scipy
- Alignment: bowtie2
- Reference: T2T-CHM13v2.0 (GCA_009914755.4) for development,
             GM12878 near-T2T for final benchmark

---

## Git workflow

```
main branch          → current hicberg (untouched)
v2 branch            → this development (human-scale)
```

Commit convention:
```
[classifier] fix chimeric chunk extraction
[stats] add P(s) by orientation
[rescue] vectorise propensity computation
[test] validate on yeast 10M reads
```

---

## Open questions / TODO

- [ ] Validate streaming_classifier_v2.py on 10M yeast reads
- [ ] Decide bin size for coverage (keep 2kb or change?)
- [ ] Decide P(s): 4 separate curves or 1 global + orientations as QC?
- [ ] Handle multi-chunk reads (2 M blocks) → saved in multi_chunk.bam for later
- [ ] Score gap threshold for XS: use AS-XS gap to reclassify some ambiguous as unique?
- [ ] Benchmark reattribute_reads on chr21 before full human genome
- [x] Theoretical mappability (GenMap) for the P(s) normalisation (mappability_v2.py)
- [ ] Mappability for the coverage (correct by mask fraction, unmappable bins
      interpolated?) and the density map (was implemented in v1 locally, Nov 2026,
      lost before being pushed)
- [ ] Replace `fitting_ps` (unstable outlier replacement); raw P(s) with
      mappability are close to the Poisson noise → e.g. merge log bins until
      enough pairs, or LOWESS weighted by counts, without point replacement

