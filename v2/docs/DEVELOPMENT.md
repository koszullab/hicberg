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

### P(s) smoothing: `pool` (default) replaces `fitting_ps`

`fitting_ps` (LOWESS + replacement of points with |z-score| > 1.5 by the fit)
is unstable (±10% jumps for tiny input changes, see validation) and even
increases the roughness: beyond 100 kb, raw 0.084 → fitting_ps 0.094. It also
replaces points with > 1000 pairs (up to 22%). Kept as `--ps_smoothing fitting_ps`
to reproduce v1.

`pool` (`pooled_law`, `smooth_ps_laws`) — no point is ever replaced by a fit:
0. below `--ps_pool_distance` (1 kb, same limit as `fitting_ps` in v1): raw
   values, zeros and fluctuations included. The law changes from one bp to
   the next there and zeros are information (no ++/-- pair below 19 bp on
   chrI, ~12 expected with a pooled plateau). A first version pooled these
   bins → flat plateau giving a never-observed 10-bp configuration the same
   weight as d ≈ 10 kb → fixed.
1. beyond `--ps_merge_distance` (5 kb) the 4 orientations are equivalent
   (25% each, check orientations.pdf) → one law for uncut/circle/weird from
   all the pairs over 4 × area (4× more counts)
2. consecutive distance bins pooled (short → long) until ≥ `--ps_min_pairs`
   (100) pairs per group (Poisson error ≤ 10%); last incomplete group merged
   back; a rich bin alone in its group keeps its exact raw value
3. group values interpolated in log-log between group centres, then rescaled
   (3 iterations) so that each group gets exactly its observed pairs
   (Σ area × law = Σ counts). Without this, the group value (area-weighted
   mean of a decaying law) is not the law at the group centre: +15% at 10 kb,
   ×1.9 at 400 kb on simulated power laws. After: no bias (+1% ± 5% over 50 draws)
4. ends: flat before the first group centre (short distances are not a power
   law), log-log slope of the last segment beyond the last centre

Yeast AC1 (with mappability), beyond 100 kb: roughness raw 0.084, fitting_ps
0.094, **pool 0.049**; rich bins (≥ 1000 pairs) unchanged (0.0%).
The tail of small chromosomes rises (last bin / P(s) at L/2: chrI ×3.4,
chrIII ×3.3, chrXI ×2.4, most others ×1.0-1.5): present in the raw data, few
pairs; could be telomere-telomere contacts (Rabl) → to check.

### Is the trapezoid (position-pair) normalisation biased? (yeast AC1, 2026-11-13)

The area = number of position pairs at distance d gives a probability per
position pair, which is what is compared between candidates. It assumes all
positions are equivalent; general form: area(d) = Σ_i w(i) w(i+d) (FFT, same
code as the mappability mask). Two checks:

1. **Coverage autocorrelation.** With propensity = P(s) × cov(i) × cov(j), the
   consistent denominator is Σ cov(i) cov(i+d). Ratio current / coverage-weighted
   (observed group1 coverage at 100 bp, CV 0.39 vs Poisson 0.09), normalised
   on 10-100 kb: current P(s) overestimated by **6% at 100 bp, 5% at 200 bp,
   1% at 500 bp, < 0.3% beyond 1 kb** (16 chromosomes, max 7%). Negligible
   compared with the 5 orders of magnitude of the P(s) → keep the current
   areas. To re-measure on human (CNV, GC: coverage varies at larger scales).
2. **Restriction sites (DpnII + HinfI).** 22% of read 5' ends are within 5 bp
   of a site (5.5% for random mappable positions, ×4), depleted at 5-30 bp.
   Not soft-clipped junctions (0.8% soft-clipped) → reads truly starting at a
   fragment end. But the enrichment is the **same in every pair category**
   (uncut < 1 kb 24%, 1-10 kb 22%, > 10 kb 21%, trans 22%) and identical
   between copies of a repeat (same sequence) → cancels in the P(s) shape
   and in the comparison of candidates. The (slow) restriction-site-based
   count of potential events is not needed. Its effect at short distances is
   already included in check 1.
3. **Very high P(s) at 1-3 bp (uncut +-)** come from the counts, not from the
   normalisation: genome-wide 94 / 816 / **5314** pairs at d = 1 / 2 / 3 bp,
   then ~5-15 per bp up to 30 bp. 99.3% of the d = 3 bp pairs sit on a
   restriction site: both reads start at the two 5' ends of the same cut
   site, offset by the overhang (DpnII 4 nt → d = 3; HinfI 3 nt → d = 2).
   Real (artefactual) events; for a multi-mapping pair of this kind the d = 3
   configuration has the same weight in every repeat copy → favours keeping
   both reads in the same copy. Could also be filtered upstream.

Scripts (one-off, not in the repo): `/media/axel/EVO/data_for_v2/yeast/analyses/`
(`analyse_normalisation.py`, `analyse_sites_par_categorie.py`, `analyse_softclip.py`).

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

### Density map: density_v2.py (step 1 done: v1 reproduced)

v1 (`get_local_density`, per chromosome pair, 2 kb, from the **unbalanced**
cooler): bad bins of the balancing → NaN; intra: obs / mean of the *positive*
values of each diagonal (≥ 10 → 1); trans: obs / block mean; zeros → 1;
NaN filled by nan-mean 15×15 (5 iterations); nan-mean 45×45; diagonal → 1.

v2: same algorithm, from the pairs file, with nan-mean windows computed as
`uniform_filter(values) / uniform_filter(defined)` (identical to v1's
`generic_filter` + Python callback, tested; O(1) per pixel).
- contact matrix built from v1's `unrescued.pairs` = v1 cooler, pixel by
  pixel (bin = (pos - 1) // bin_size, as `cooler cload pairs`)
- density with v1's mask: **identical to v1's density_map.npy** (max abs
  difference 1.3e-14 over the 171 blocks), **1.8 s** vs ~2 min (8 cores)

**Bias found in v1 (yeast AC1):** intra density is almost flat (5-95%:
0.93-1.07), but **trans median = 1.84**. Cause: zeros → 1 is not neutral on
sparse blocks. ~87% of trans pixels are 0 at 2 kb; after obs / mean the
non-zero pixels average 1 / 0.13 ≈ 7.5, so replacing zeros by 1 gives a mean
≈ 1 + 0.87. In mode `full` (P(s) × density) trans candidates are favoured
~×1.8 over intra ones, and the factor depends on the sparsity (depth,
resolution): close to ×2 for human at 2 kb. Intra avoids it only because the
expected is the mean of the positive values (zeros excluded), which in turn
erases real depletions.

**Use:** mode `full` = P(s) × density only (no coverage: the density is
computed on the unbalanced matrix, so it already contains the 1D effects).
For yeast its main role is the centromere clustering in trans (a read is
more likely to come from a centromere than from the middle of an arm).

**Step 2 (yeast AC1)** — `density_v2.py --method unbiased`: zeros kept as 0,
expected = diagonal mean (intra) / block mean (trans) zeros included, no cap.
Mask = mappability (`coverage_kept.npy`, 622 bins) instead of the balancing
(522 bins). Medians over the 120 pairs of nuclear chromosomes:

| version | median intra | median trans | CEN-CEN | CEN / trans median |
|---|---|---|---|---|
| A v1 (file) | 1.00 | 1.84 | 5.62 | 3.06 |
| B v2 pairs + mappability mask + v1 method | 1.00 | 1.84 | 5.67 | 3.08 |
| C v2 pairs + mappability mask + unbiased | 1.02 | **0.95** | 5.10 | **5.34** |
| raw counts, same 45×45 window | | 1.00 | 5.20 | |

The mask changes almost nothing; the unbiased obs / exp gives a neutral
trans background and the raw centromere contrast. Smoothing window
(unbiased): CEN-CEN density = raw enrichment at every scale, but the peak is
sharp: ×24.8 at 10 kb, ×16.7 at 22 kb, ×10.0 at 42 kb, ×7.2 at 62 kb, ×5.1
at 90 kb (v1 window). Noise on the arms (CV): 0.58 / 0.36 / 0.28 / 0.24 /
0.22, Poisson part 0.53 / 0.24 / 0.13 / 0.09 / 0.06 (the rest is real
structure). → idea: adaptive window (smallest one with ≥ N contacts), sharp
near centromeres / diagonal, wide on sparse arms.

Coverage: low-coverage mappable bins are now **kept** by default
(`--coverage_min_ratio 0`): a low coverage can be real (e.g. few restriction
sites) and must be reconstructed faithfully; filtering / normalisation come
after the reconstruction.

Interpolation anchors: only bins with ≥ half a bin of mappable bp. End of
chrI: the last bin (490 bp, 88% mappable) is really under-covered (706 R1
reads / 2 kb of *all* aligned reads, vs ~1580 on the arm; only 9% with XS:
chromosome-end effect, not ambiguity). Used as an anchor it pulled the 26 kb
of interpolated subtelomere (bins 102-114) from 2062 down to 912, whereas all
the aligned reads (ambiguous ones included) show ~0.75× the arm level there.
Now it keeps its own value (823) and the gap stays at 2062.

**Option: unmappable bins from all aligned reads.** `streaming_classifier_v2.py
--coverage_all_reads` (+25% time: 71 → 89 s on yeast) saves the 5' positions
of all aligned reads in 100-bp bins (`coverage_all_reads.npz`); an ambiguous
read is counted at the position bowtie2 reported, chosen at random among its
best alignments, so each copy of a repeat gets its share on average.
`statistics_v2.py --coverage_all_reads coverage_all_reads.npz`: excluded bins
= all-reads coverage × (median group1 / all-reads ratio on fully mappable
bins, 0.92 on yeast). Excluded bins / chromosome median: interpolation 0.88
(Q5-Q95 0.71-1.11), all reads 0.95 (0.51-1.42). Default stays interpolation.

Step 3: human scale (banded intra at fine resolution, coarse trans).

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
  extrapolated over them, and even produced **negative coverages** (down to
  -5923 on chrVIII, line fitted on a few neighbours and extended).

**Coverage with mappability (`correct_coverage`)**, replaces `mad_smoothing2`:
1. corrected = raw × 2000 / (fraction × effective width): the last, shorter
   bin of each chromosome is scaled, not excluded
2. bin kept if fraction (of its own width) ≥ 0.8 (`--coverage_min_fraction`)
   and ≥ 100 mappable bp (`--coverage_min_bp`, guard against tiny remnants)
3. low outliers excluded: corrected < 0.5 × median of the 10 closest kept
   bins on each side (`--coverage_min_ratio`; closest *kept* bins, so that
   an isolated bin after an unmappable stretch is not compared to itself)
4. excluded bins: linear interpolation between kept bins (periodic if
   circular, constant beyond the ends) → never negative

Yeast: 632 / 6139 bins excluded (10%); final coverage / chromosome median
between 0.45 and 1.75, no negative value. `coverage_kept.npy` = bins kept,
shown in blue (kept) / red (excluded) on the coverage plots.
Check: coverage ∝ mappable fraction holds for fraction ≥ 0.6 (ratio 0.92-1.00),
not below 0.2 (median 0.37).

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
- [x] Replace `fitting_ps` (unstable outlier replacement) → `--ps_smoothing pool`
      (default), see "P(s) smoothing" in Module 2

