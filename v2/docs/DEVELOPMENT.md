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
structure).

**Adaptive window** (`--smoothing adaptive`, default): for each pixel, the
smallest window among 1/3/5/11/21/31/45 bins (2-90 kb; 1 and 3 added later, see
below) holding ≥ `--n_min` raw contacts, nan-mean of obs / exp over it. Yeast
(first version, windows 5-45):

| smoothing | trans median | CEN-CEN | arm noise (CV) | window CEN / arms |
|---|---|---|---|---|
| fixed 45 | 0.95 | ×5.1 | 0.22 | 90 kb / 90 kb |
| adaptive n_min=50 | 0.95 | ×24.8 | 0.25 | 10 kb / 42 kb |
| adaptive n_min=100 | 0.95 | ×16.9 | 0.23 | 22 kb / 62 kb |
| **adaptive n_min=200** | 0.95 | **×16.5** | 0.22 | 22 kb / 90 kb |

Centromere peak 3.3× sharper (= raw ×17.1 at 22 kb) with the same arm noise;
4.3 s for the genome.

Are the fine fluctuations real? Split-half test: pairs split at random in two
halves, fluctuations = log2 adaptive − log2 fixed 45 computed on each half
(same windows as the full map), correlation between halves (→ reproducible
part of the full map by Spearman-Brown 2r/(1+r)):

| zone | n_min=100 | n_min=200 (default) |
|---|---|---|
| intra 6-100 kb | 0.78 (≈ 88%) | 0.83 (≈ 91%) |
| intra 100-400 kb | 0.49 (≈ 66%) | 0.54 (≈ 70%) |
| trans | 0.45 (≈ 62%) | 0.84 (≈ 91%) |

→ n_min = 200: same centromere peak, fluctuations ~90% real.

**Threshold choice.**
- A cross-validation criterion (density of one half predicts the counts of
  the other, Poisson deviance) chose n_min 100 intra / 50 trans: sharper
  (trans centromere ×24.6) but visibly noisier (only ~50% of the extra trans
  fluctuations real) → rejected.
- **`--density_n_min auto`** (`density_v2.reproducible_n_min`): smallest
  n_min, separately intra and trans, whose fine fluctuations (log2 adaptive −
  log2 45-bin window, on each of two random halves of the pairs, halves at
  n_min / 2) are reproducible: part = 2r / (1 + r) ≥ `--density_reproducibility`
  (default **0.8**); grid 25/50/100/150/200/300/400. Recomputed per dataset:
  deeper data or stronger structure → lower threshold, finer maps.
  Yeast (intra pixels ≥ 3 bins from the diagonal):

  | n_min | 100 | 150 | 200 | 300 | 400 |
  |---|---|---|---|---|---|
  | intra | 71% | 76% | 79.5% | **85%** | 88% |
  | trans | 62% | 79% | **91%** | 97% | 98% |

  → intra 300, trans 200. Without `auto`: 200 / 200
  (`--density_n_min`, `--density_n_min_trans`). Cost of `auto` on yeast:
  whole statistics step 153 s, 3.6 GB RAM (65 s without). Human: calibrate on
  a subset of chromosomes (step 3). The intra value is dominated by the far
  pixels (most numerous); a per-distance-band threshold would be possible.
- **Windows of 1 and 3 bins** added: the window is already chosen pixel by
  pixel, and near the diagonal 1-3 bins hold enough contacts (chrIV,
  n_min 200: 3×3 enough for 95% of pixels at 2-4 kb, 92% at 4-10 kb, 46% at
  10-20 kb; 1×1 for 30% at 2-4 kb) → 2-6 kb resolution along the diagonal,
  unchanged beyond ~20 kb.
Intra (v1 almost flat, 5-95%: 0.93-1.07): the adaptive map shows the Rabl
X pattern centred on each centromere, the rDNA separation on chrXII and
local structure near the diagonal (correlation with the raw obs / exp 0.31-0.41
vs 0.20-0.22 for v1). Yeast has no real TAD at 2 kb: domains to check on human.
Square textures = shape of the box windows around isolated high pixels
(not the filled bad bins); Gaussian windows would round them (optional). Maps also show the isolation of the rDNA
(chrXII) in trans (low density band), hidden by the inflated v1 background.

**Mappable-length weights** (`--mappability stats/mappability.npy`):
expected(i, j) = E_d × w_i × w_j with w = mappable length / bin_size, E_d =
Σ obs / Σ w_i w_j on each diagonal (trans: on the block). Same logic as the
P(s) areas and the coverage. End of chrI: the last bin (490 bp, 431 bp
mappable) had obs / exp 0.14 only because of its size, and pulled the fill
of the 26-kb unmappable subtelomere down to 0.77; with the weights the end
stays at 0.95-0.97. Elsewhere: median |Δ log2| 0.009 (99%: 0.08), CEN-CEN
×16.3. w = 1 → identical to the unweighted computation (tested).

```bash
python3 density_v2.py unrescued.pairs ./density --chrom_sizes 1.sam \
    --mappability ./stats/mappability.npy \
    --coverage_kept ./stats/coverage_kept.npy      # unbiased + adaptive (default)
# v1: --method v1 --smoothing fixed --bad_bins_cooler unrescued.pairs.cool
```
`plots_v2.plot_density()` → `density.pdf` (genome log2 map + zooms, side by side).

**In the pipeline:** `statistics_v2.py` now computes the density by default
(unbiased, adaptive window, n_min 200 / 200 or `auto`, mappability mask and
weights when `--mappability` is given; `--no_density`, `--density_n_min`,
`--density_n_min_trans`, `--density_reproducibility`), saves
`density_map.npy` and `contacts.npz` (contact matrix at bin_size, upper
triangle), and the checkpoint plots include, for each chromosome
(`density_intra.pdf`) and each pair of chromosomes (`density_trans.pdf`, ≤ 200
pairs), the raw obs / exp (not smoothed, same mask and weights; 2 kb intra,
sum obs / sum exp over 5×5 bins = 10 kb in trans where 2-kb pixels hold 0 or 1
contact; log2 ±3) next to the density (log2 ±1.5), real proportions, axes in
kb, one colour bar each. Grey = excluded bins, darkest blue = zero contact. Yeast: whole
statistics step with density and plots 65 s, 1.6 GB RAM. Human: contacts.npz
and per-pair pages to rethink with the banded storage (step 3).

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

## Pipeline CLI: hicberg_v2.py

```bash
python3 hicberg_v2.py genome.fa R1.fq.gz R2.fq.gz -o out --cpus 16 \
    --circular plasmid2micron,chrMT --max_alignments all \
    --hicstuff --enzyme DpnII,HinfI          # [--index existing_prefix]
# checkpoint: --exit_stage stats, check out/statistics/plots, then --start_stage rescue
```

Stages: `index` (bowtie2-build unless `--index`) → `mappability` (k = read
length read from R1; `--mappability bowtie2` (default) | `genmap` | `none` |
path of a bedgraph; + repeat links, `--repeat_links_step`) → `align` (bowtie2
--very-sensitive-local --reorder, SAM deleted after classify unless
`--keep_sam`) → `classify` → `stats` (+ checkpoint plots) → `rescue` →
`matrices`. Chromosome sizes from the FASTA (`out/chrom.sizes`).
Group2 chunks = cpus / 4, re-aligned and re-attributed in parallel.

**matrices_v2.py** (last stage):
- `--hicstuff`: hicstuff filtering (as `hicstuff pipeline --start-stage=pairs
  --filter --duplicates`) on unrescued.pairs and rescued.pairs (non-informative events: uncuts,
  loops, weirds; PCR duplicates), in `matrices/hicstuff_<prefix>/`.
  Needs the 4DN pairs header: columns **chr1 / chr2** (our first files used
  chrom1 / chrom2: pairtools sort failed and the PCR-free file was empty) and
  `#chromsize` lines → fixed in the classifier and reattribute_v2.
  Yeast group1: 1.81M pairs discarded (uncuts 1.39M, loops 0.41M, weirds
  10.5k), 5% PCR duplicates, 5.59M remaining; 5.5 min.
  **Same thresholds for both files:** hicstuff estimates its uncut / loop
  thresholds per library; on yeast it chose 19 / 25 fragments on unrescued
  but 8 / 10 on rescued, so the rescued pairs were less filtered (1.23M uncuts
  removed vs 1.39M from a file they contain) and before / after was biased.
  Now the thresholds estimated on unrescued are applied to rescued.
  **hicstuff library calls instead of `hicstuff pipeline`** (2026-11-16): the
  pipeline always built its own cooler matrix, unused here (~15 min per file
  on the 47 M pairs of full AC1). `hicstuff_filter` now calls the same steps
  one by one: `digest.attribute_fragments`, `io.sort_pairs`,
  `filter.get_thresholds` (unrescued) / given thresholds (rescued),
  `filter.filter_events`, `pipeline.filter_pcr_dup` (events filtered before
  duplicates, as hicstuff). Checked on 1.5 M yeast pairs: same thresholds
  (19 / 22) and identical 1,130,918 output pairs, 45 s vs 82 s. Micro-C
  (`--enzyme mnase`, or a number): hicstuff fixed-size digestion into virtual
  fragments (100 bp by default), so that uncuts / loops / weirds between
  neighbouring segments are filtered as with a frequent cutter, then PCR
  duplicates.
- cooler matrices at bin_size (`cooler cload pairs`) + ICE (`cooler balance`),
  mad-max 5 for unrescued, 1000 for rescued (as v1: the reconstructed repeats
  must not be filtered out again).
- `plot_before_after`: v1-style pages (hicberg/plot.py plot_matrix), one page
  per pair of chromosomes (intra only if > 30 chromosomes or `--intra_only`),
  `matrices_raw.pdf` and `matrices_normalised.pdf` (ICE): left unrescued map
  + GC, **Mappability** (shorter track); right rescued map + coverage before / after (median
  of before dashed, legend outside on the right), **number of Hi-C restriction
  sites per bin** (bars, `--enzyme`, e.g. DpnII + HinfI; GC for Micro-C): the
  bins that stay poor in reads after rescue are bins with few sites (yeast,
  2 kb: bins with 0-5 sites have 0.49× the median coverage; the 11 bins below
  25 % of the median have 0 site, vs 12 for a typical bin; Spearman 0.46).
  Intra colour scale (raw and ICE): log10 from the 1st percentile of the non-zero
  pixels lowered by 0.5 × the fraction of empty pixels (sparse yeast 2 kb maps,
  60-70 % empty: ≈ -0.33 in raw; dense chrMT: tight scale, ≈ -2.4 to -0.7 in
  ICE as in v1; a fixed offset of 1 washed the far field into orange) to the
  99.97th percentile (95th / 99th / 99.9th
  saturated the first diagonals, the max depends on outlier pixels). At 2 kb
  most non-zero raw pixels hold a single contact (log10 = 0): with vmin = 0
  (v1) they were white like the empty pixels and the far field / domains
  vanished; the v1 vmax 3.5 also washed the maps out. Trans: raw 0-99.5th
  percentile, ICE 5th-99th. Tracks without top / right frame; titles give the number of
  contacts of each block (upper triangle in intra). The normalised
  maps remove the copy-number effect (rDNA, 2µ) that dominates the raw ones.

## Module 3: reattribute_v2.py (first version done)

```bash
python3 reattribute_v2.py classifier_dir stats_dir out_dir --index genome_index \
    --mode full --max_alignments all --n_workers 4 --threads 4   # yeast: -a
```

1. per group2 chunk: `bowtie2 --very-sensitive-local --reorder -k K` (or `-a`,
   `--max_alignments all`, used for yeast) on R1 and R2, stdout read as a
   stream (no SAM on disk), alignments grouped by read.
2. **candidates weighted by their likelihood**: alignments within `--as_delta`
   (30) of the best AS, weight exp(−λ (best AS − AS)), λ = ln(3(1−e)/e) / 8
   per AS point (a high-quality mismatch costs 8 points in local mode),
   `--error_rate` e = 0.01 → λ = 0.71, 1 mismatch = ×1/297; e = 0 → best-score
   alignments only. v1 kept all the -k alignments with the same weight (~26%
   of them score ≥ 1 mismatch below the best on yeast).
   History: a first v2 kept the **best-score alignments only**. It created
   artefactual trans spots with the rDNA (chrXIV 415 kb ×7.7 v1, chrXIV 359 kb
   ×7.6, chrIV 1365 kb ×4.4, chrIV 685 kb, chrVIII 564 kb), partners piled up
   within a few bp: **poly-T reads** (no positional information; 1 perfect
   locus, 230-460 loci at 1 mismatch / 1 clipped base) all sent to the same
   longest poly-T tracts. With 300 near copies at ×1/297 each, the perfect
   copy should get ~50% only: the likelihood weighting does that. Spots after
   weighting: 99 / 197 / 103 / 88 / 83 contacts (best-only 525 / 506 / 346 /
   203 / 168, v1 68 / 67 / 79 / 60 / 55); the real 2µ ↔ rDNA contact (partners
   spread over the plasmid, copy numbers) unchanged (3462, v1 3447).
3. propensity of every R1 × R2 combination, vectorised over batches of pairs
   (combination indices with np.repeat, segmented weighted draw on a global
   cumsum), same conventions as the statistics: `random`, `ps`, `standard`
   = P(s) × cov × cov, `full` = P(s) × density. Seeded draw (one seed per chunk).
4. `predicted.pairs` = unrescued.pairs format + n_candidates1/2.

**Yeast AC1, mode full, -a, likelihood weighting:** 1,524,893 group2 pairs,
all placed, 10.2 / 10.5 candidates per read, 101 combinations per pair, 70.8%
intra (group1 cis: 69.9%), **2 min 54 s** with 4 workers × 4 threads.
Unmappable bins reconstructed at 0.94 × the mappable level (0.93 expected);
predicted-only map vs group1 on mappable pixels: intra 0.544, trans 0.208.
(The numbers below were measured with the first, best-only version.)

Validation vs v1 (same read placed: same chromosome, ±50 bp, both reads):
34% overall (random draws among several candidates: 1×1 68.5%, 2-4 57%,
5-50 20%, > 50 10%). For pairs with a single best combination in v2, v1
places **17.6% of the reads on a copy with a worse score** (median gap 8 ≈
1 mismatch) → 1 − 0.824² ≈ 32% of these pairs differ: the best-score filter
fixes a v1 defect.
Coverage reconstructed (group1 + predicted) in the unmappable bins / median
of the mappable bins of the chromosome: **0.94** (v1 0.95), expected from all
the aligned reads 0.93. chrI subtelomere (bins 102-114) filled with the same
bin-by-bin profile as all the aligned reads.

`rescued.pairs` = unrescued + predicted (written by reattribute_v2).
`plots_v2.plot_contact_maps`: maps side by side (whole genome 20 kb, each
chromosome, zoomed regions), shared colour scale, with a track under each
map: mappability under the first one, coverage before (grey) / after (red)
under the others (y axis 1.5 × 99th percentile: the rDNA peak, 30-40k reads
per 2-kb bin = ~150 copies collapsed on the assembly, is clipped).
Yeast: rDNA, subtelomeres, mating-type cassettes and Ty elements, white
before, are filled; v1 and v2 similar at map scale.

**Modes full vs standard (yeast):** per-pair agreement is not a valid
comparison: two `full` runs with different seeds agree on 42.8% of the pairs
only (random draws among candidates), and full / standard run with the same
seed share their random numbers (78%, inflated). Fidelity to the 2D
organisation instead: Spearman correlation at 20 kb between the map of the
predicted pairs alone and the group1 map, on pixels whose two bins are ≥ 90%
mappable: **full intra 0.540, trans 0.203**; standard 0.519 / 0.145 (seed
noise ~0.002). The density brings the trans organisation (centromere
clustering) that the coverage product cannot → keep `full` as default.

Next: human scale.

### (initial design notes)

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


### Full AC1 run (47.3 M pairs) and the chrII right end (2026-11-16)

Full AC1 (AC1.end1/2.fastq.gz, 47.3 M pairs, 16 CPUs): 89.9 min (old
`hicstuff pipeline` call, ~27 min per file lost on the unused hicstuff
matrix). group0 7.8 %, group1 77.0 %, group2 15.2 % (7.2 M pairs, all placed,
70.8 % intra, 13.4 min). hicstuff thresholds 12 / 15; final matrices 23.7 M
(unrescued) vs 29.7 M (rescued) contacts. Weirds 357 k in rescued vs 50 k in
unrescued (same excess as on 10 M reads, still to explain).

chrII right end under-covered after rescue: in the T2T BY4742 assembly the
last ~25 kb (796.9-822.3 kb) are an inverted duplication of themselves (two
~12.5 kb arms centred at ~808.9 kb; 200 bp probes at 800 / 805 / 812 / 818 kb
have a perfect reverse copy at 817.9 / 812.9 / 805.9 / 799.9 kb). Mappability
~0, so no unrescued reads; after rescue each arm gets 0.45-0.65× the genome
median (unique flank 744-794 kb: ~1.0×), i.e. ~1× in total: AC1 most likely
carries one copy only (this chrII is 822,297 bp vs 813,184 in S288C R64), and
the rescue shares its reads between the two reference copies. Decision: no
copy-number correction for now; the coverage track (before / after) is
enough to warn the user that the assembly may have a different copy number
than the sample (the opposite of the rDNA case).

### Bug: group2 reads lost their strand (excess of weirds in rescued) — fixed

The excess of hicstuff "weirds" in rescued (357 k vs 50 k on full AC1;
83 k vs 10.5 k on 10 M) came from the classifier: for a read aligned on the
reverse strand, SAM SEQ / QUAL are reverse-complemented / reversed, and the
group2 FASTQ was written from them as is. At the re-alignment every read was
back on the + strand: in predicted.pairs all the reads with a single
candidate were "+", and the pairs were 5.34 M ++, 0.85 M +-, 0.85 M -+,
0.16 M -- (short-range same-strand pairs ×4 more frequent than opposite,
vs ×0.28 in unrescued; half of them in the rDNA). Reverse reads were also
placed at their alignment start instead of their 5' end (shift ≤ read
length, 35 bp: negligible at 2 kb, but the hicstuff event filtering of
rescued was wrong). Fix: `original_orientation()` puts the read back as
sequenced before writing the FASTQ. Checked on 50 k pairs of AC1: after
re-alignment the 2,325 group2 R1 reads that were unique all come back on the
same strand and position. The rescued results above (maps, fidelity,
modes) used the wrong strands; at 2 kb they are barely affected, but the
pipeline must be re-run from the alignment (SAMs not kept).

### Mappability with bowtie2 (GenMap optional) and repeat links (2026-11-16)

GenMap is no longer actively developed: the default mappability is now
computed with the aligner of the pipeline (`mappability_v2.bowtie2_mappability`).
One k-mer (k = read length) every `step` bp (default 10) is aligned with
bowtie2 --very-sensitive-local, streamed (stdin / stdout, no file); a k-mer is
mappable if it aligns without XS tag, i.e. exactly the criterion that sends a
read to group1. Bedgraph with score 1 over [p, p + step): same format as the
GenMap one, read by `binned_unique_fraction`. Options: `--mappability
bowtie2|genmap|none|<bedgraph>`, `--mappability_step`.

Yeast (k 35) vs GenMap K35 E2: mappable fraction 91.29 % vs 91.55 %; 2 kb
bins r = 0.999 (21 / 6139 bins differ by > 0.1, all on chrMT: AT-rich, local
mode finds partial second alignments → XS, as for the real reads); 100 bp
bins r = 0.994. Step 10 vs step 1: r = 1.0000 at 2 kb, no bin differs; 34 s
vs 314 s (6 threads). Mouse benchmark (mhaESC T2T, 2.87 Gb, 1 M random
k-mers, machine loaded): k 35 → 55 k k-mers/s (8 threads), k 100 → 15 k/s
(8 threads), 21 k/s (16 threads); whole genome at step 10: ~1.5 h (k 35),
~2.5-4 h (k 100), once per genome and read length. Note: in local mode 36 %
(k 35) but 43 % (k 100) of the mouse k-mers have an XS: a partial hit in a
repeat (Alu / LINE) is enough. For human Micro-C a large part of the reads
will go to group2; a threshold on AS - XS may be needed.

**Repeat links** (`mappability_v2.repeat_links`): one k-mer every 50 bp
aligned with bowtie2 -k 100; copies with AS >= best - 8 (≤ ~1 mismatch); for
a k-mer with 2-99 copies, a link from its 1 kb bin to the bin of each other
copy, with the relative orientation (+ direct / tandem, - inverted); k-mers
with >= 100 copies are saturated (not linked). First with -k 20: the Ty
elements of the chrXII right arm (614 / 672 / 840 kb: 37-40 copies per
k-mer) were saturated and had no arc → -k 100. TSV
`repeat_links_K{k}_S{step}.tsv` (chrom1 start1 chrom2 start2 orientation
n_kmers). Yeast: 3 s; 245 k k-mers, 19.3 k linked, 353 saturated; rDNA
links "+" with the 9.1 kb period, chrII right end "-" (500 / 506 k-mers:
the palindrome). **Families** (`repeat_families`, at the bin size of the
maps): graph bins ↔ links (≥ 3 k-mers, weighted by the shared k-mers),
Louvain communities (networkx, seed 0; connected components as fallback:
with -k 100 they chained Ty and subtelomeres into one family of 506 bins),
ranked by size, described automatically: tandem (one chromosome, compact),
subtelomeric (≥ 60 % of the bins within 30 kb of an end), dispersed. Yeast at
2 kb: F1 dispersed (16 chr, 120 bins), F2 / F3 subtelomeric (16 chr, 78 / 61
bins), F4-F6 dispersed (13-14 chr: Ty sub-families), F8 tandem chrXII
440-512 kb (rDNA), ...
Written to `plots/repeat_families.tsv` and `repeat_family_bins.tsv`.
Plotted as arcs left of the unrescued map, **one colour per family** (the 12
largest, others grey; the orientation stays in the links file), width ∝
sqrt(distance), line width ∝ log(k-mers), ≥ 3 k-mers per
pair of map bins, at most 3000 arcs); trans pages: ticks on a strip left of
the map (positions of the row chromosome with a copy on the column one) and
on a strip above it (positions of the column chromosome), same colours; the
maps themselves stay free of annotations (open circles at the pairs of
copies were tried and dropped). Shows at once the rDNA (tandem
cluster), the chrII palindrome, the subtelomeres (chromosome-end to
chromosome-end) and the Ty elements (long arcs).

### Large / sparse genomes: density bin, shrinkage, coverage threshold (2026-11-16)

Mosquito (Culex quinquefasciatus + Wolbachia wPipMtq, 575 Mb, 59 sequences;
library MM265a, 37 M pairs of 35 bp): only 32 % of the reads align, 85 % of
the others align on yeast (same dual index on all reads: the yeast library
probably got the same index, or yeast DNA was in the tube; to be re-run on a
mosquito + Wolbachia + yeast genome, with the yeast-mosquito chimeric pairs
as a test of when they were mixed). Wolbachia gets 23.6 % of the aligned
reads (1.4 Mb). 43.6 % of the genome uniquely mappable (k 35); group1 3.3 M
pairs (8.8 %), group2 6.6 M (17.8 %).

Scaling changes:
- `--density_bin_size` (statistics / pipeline, default auto) separate from
  the coverage bin: dense blocks of a 225 Mb chromosome at 2 kb would need
  ~1.3 TB. Memory criterion (`auto_density_bin_size`): 100 bytes per cell of
  the largest block (block_density measured ~52 bytes / cell, x2) <= half of
  MemAvailable. Density stored in float32, with its bin size
  (`density_bin_size.npy`) and its mask / mappable weights at that bin
  (`density_mask.npy`); reattribute_v2 looks the density up at its own bin.
- Before / after plots: `cooler coarsen` to `--plot_bin_size` (auto: largest
  plotted sequence <= 3000 bins: yeast 2 kb, mosquito 100 kb), sequences
  `--plot_chroms` (auto: all if <= 30, else those >= 1 Mb, at most 20);
  density plots of the statistics restricted to them and the genome-wide
  density downsampled block by block (the full matrix does not fit).
- P(s) pages of sequences without any pair are skipped (crashed on small
  unmappable scaffolds).

Problems seen on the mosquito statistics and fixes:
- Coverage: bins kept if mappable fraction >= 0.8 → only 20 % of the
  mosquito 2 kb bins (median fraction 0.44), the rest interpolated. Default
  now 0.3 (61 % kept): the coverage is already divided by the mappable
  length, a 30 % mappable bin is unbiased, only noisier.
- Density noise: 0.006 contact per far-field pixel at 25 kb (chr1), the
  45-bin window held ~12 contacts for n_min = 200; the largest-window
  nan-mean of obs / exp gave 0 and values up to 230 (stripes).
  (a) `data_density_bin_size`: from the intra P(s) counts, contact density
  per position pair rho(d) = N(d) / ((L - d) Δd); a 45 × 45 window of bins b
  holds ~ 45² rho b² contacts; smallest nice b with <= 5 % of the intra
  contacts in windows below n_min. Yeast: 0 % at 2 kb → 2 kb; mosquito:
  21 % at 2 kb, 7 % at 25 kb, 4.8 % at 50 kb → 50 kb. The density bin is the
  coarser of the memory and data criteria.
  (b) Shrinkage in `adaptive_filter` where even the largest window has
  < n_min contacts: (O + δ) / (E + δ), δ = n_min - O (the missing contacts
  are taken at density 1, i.e. P(s) only); continuous at O = n_min.
Yeast check (full AC1 unrescued, 36 M pairs): density bin stays 2 kb (0 %
fallback); kept coverage bins 5517 → 5721, coverage r = 0.997 (234 / 6139
bins differ by > 10 %, mostly around unmappable regions); density log2 r =
0.991 (intra median) / 0.995 (trans median) vs the previous maps, the
changes are in the small sparse blocks (2µ × 2µ 4 × 4 bins, chrMT trans:
shrunk towards 1).

**Telomere-telomere contacts (mosquito) — expected pooled in log distance,
larger windows.** The mosquito unrescued map at 1 Mb shows enriched
telomere-telomere contacts, absent from the density (corner ≈ 1.0). Two
causes: (1) the per-diagonal expected — the last diagonals hold a few pixels,
all near the corner, so the expected absorbs the telomere contacts (chr1
corner 3 × 3 Mb obs / exp 1.08 per diagonal, 2.10 with diagonals pooled in
log-distance groups, base 1.1 as the P(s); chr2 2.01, chr3 1.47); (2) the
largest adaptive window (45 bins = 2.25 Mb at 50 kb) holds far fewer than
n_min contacts there (the 3 × 3 Mb corners hold 146-385 contacts).
`expected_unbiased(log_pool=1.1)` (default) and DENSITY_WINDOWS extended to
61, 91, 121, 181 bins (uniform filters: cost ∝ number of windows, not their
size). Mosquito density corners (60 × 60 bins): chr1 1.02 → 2.26, chr2 1.01
→ 3.22, chr3 1.00 → 1.97; far-field median 0.98-0.99. Yeast: chromosome
maps unchanged (log2 r = 0.991 intra / 0.996 trans vs the validated maps,
changes in sparse chrMT / 2µ blocks); yeast telomere corners now visible
too (50 × 50 kb: chrXII 1.05 → 1.51, chrII 1.06 → 1.21, chrIV 1.02 →
1.06), as expected from the Rabl clustering of the telomeres.

**Adaptive Gaussian density (default).** The box windows gave square blocks
(each pixel jumps from one window size to the next). `adaptive_gaussian`:
Gaussian-like kernel (3 box passes: cost independent of the width), widths
σ = 0.5 … 90 bins (factor √2); density = smoothed observed / smoothed
expected (ratio of sums); effective contacts N_eff = smoothed counts /
Σ w² (= the count for a box); each pixel blends the two widths around
N_eff = n_min (geometric, in log N_eff): continuous, no blocks; shrinkage
towards 1 where even σ = 90 is below n_min. Compared on split halves
(n_min / 2 per half), held-out Poisson log-likelihood gain over P(s) only
(density of one half predicting the counts of the other) and split-half
correlation of the log2 densities:

| | box (previous) | Gaussian |
|---|---|---|
| yeast intra 2 kb (9 chr) | +38.4 mnat, r 0.969 | **+51.5**, r 0.951 |
| yeast trans | +209.2, r 0.983 | **+214.7**, r 0.978 |
| mosquito intra 50 kb (3 chr) | +139.0, r 0.917 | **+225.8**, r 0.936 |
| mosquito trans | +134.6, r 0.932 | **+199.5**, r 0.953 |

The held-out likelihood (penalises both noise and over-smoothing) is better
everywhere; the box is slightly more reproducible on yeast intra because it
over-smooths near the diagonal. Visual: no square blocks, structures
continuous, far field with large domains (less shrinkage to 1), telomere
corners kept (`moustic/density_box_vs_gaussian.pdf`). Now the default of
statistics_v2 and of the n_min calibration (`smoothing="gaussian"`).

### PCR duplicates on the sequences, MAPQ >= 30 = unique (2026-11-16)

**Duplicates.** The copies of a PCR-duplicated ambiguous pair were drawn
independently by the re-attribution (mosquito: ~2000 combinations per
pair), so they landed at different positions and hicstuff (coordinates) no
longer saw them: in group2, 46.6 % of the pairs were exact sequence
duplicates and 57 % of the extra copies ended elsewhere, counted as new
contacts (mosquito rescued ×7.9 vs unrescued, chr1 ×11). Now the classifier
keeps one pair per identical (R1, R2) sequence pair (sequenced
orientation), the same way for group1 and group2, before the re-attribution
(`dedup=True`; one 64-bit hash per pair in memory, ~70 bytes: a sorted
on-disk version will be needed for human). hicstuff's coordinate filter is
kept for both (catches duplicates with sequencing errors).

**Unique = no XS, or XS with MAPQ >= 30** (`--min_mapq 30`, -1 = XS only),
as the MAPQ filter of hicstuff, so that the unrescued maps compare with the
standard pipeline; the bowtie2 mappability uses the same criterion
(`bowtie2_K{k}_S{step}_Q30.bedgraph`). Mappable genome: yeast 91.3 → 92.8 %,
mosquito 43.6 → 60.3 %.

Results:

| | yeast AC1 | mosquito MM265a |
|---|---|---|
| group1 / group2 before dedup | 37.31 M / 6.32 M | 3.90 M / 6.01 M |
| sequence duplicates group1 / group2 | 16.3 % / 18.3 % | 72.4 % / 71.5 % |
| kept group1 / group2 | 31.23 M / 5.16 M | 1.08 M / 1.71 M |
| hicstuff PCR duplicates after | 2 % / 2 % | 10 % / 5 % |
| contacts unrescued → rescued | 24.15 M → 28.26 M (×1.17) | 0.63 M → 2.01 M (×3.2) |
| previous run | 23.67 → 28.86 (×1.22) | 0.51 → 4.01 (×7.9) |
| re-attribution | 21 min (3 workers) | 27 min (was 67) |

Yeast check: median rescued / unrescued coverage in well-mappable bins
1.047; unmappable bins (excluding the rDNA) reconstructed at 0.92 × the
median (0.94 before; 0.93 expected). The mosquito library is of low
complexity (~2.8 M distinct molecules aligned on mosquito + Wolbachia out of
37 M pairs); the ×3.2 left is real (1.71 M ambiguous vs 1.08 M unique
molecules, 60 % of the genome uniquely mappable). Density bin for the
mosquito now 200 kb (data criterion with 1.08 M pairs, calibrated on the
45-bin box window: probably too conservative with the Gaussian kernel, to be
recalibrated). Repeat arcs limited to the 400 strongest per page.

### Mosquito + Wolbachia + yeast genome: when were the libraries mixed? (2026-11-18)

Run `moustic/mosq_wolb_yeast/run` (77 sequences, MM265a reads, `--hicstuff --enzyme DpnII,HinfI`,
135 min including the repeat links). Group0 (a mate unaligned) falls from 73 % to 17 %; group1 16.5 M
(23.6 % duplicates), group2 4.5 M (52.1 % duplicates); hicstuff thresholds uncuts 21, loops 15;
11.9 M unrescued / 15.3 M rescued pairs after hicstuff. The mosquito keeps the same pairs: the
recovered reads are yeast.

Unrescued pairs (both mates unique, deduplicated) between organisms:

| | mosq nuc | mosq mito | Wolbachia | yeast nuc | yeast mito |
|---|---|---|---|---|---|
| mosq nuc | 556,221 | 3,961 | 4,947 | 5,784 | 2,098 |
| mosq mito | | 8,385 | 302 | 1 | 0 |
| Wolbachia | | | 502,213 | 18 | 9 |
| yeast nuc | | | | 14,677,658 | 14,996 |

O/E (coverage product): mosq nuc–Wolbachia 0.14, mosq nuc–yeast 0.006, Wolbachia–yeast < 0.0001.
Wolbachia is the internal control: it has as many read ends as the mosquito nucleus and was in the
same tube during ligation; any random process (ligation in a mixed sample, PCR chimeras) would give
~7,000 yeast–Wolbachia pairs, 27 are observed. The ~7,900 yeast–mosquito pairs are clustered
(34 % in 1 % of the mosquito 100 kb bins vs 3.5 % for mosquito trans pairs; 27 % on the A+T-rich
yeast chrMT, which has 5 % of the yeast reads): most likely cross-mapping of A+T-rich sequences,
not ligation products. Conclusion: the yeast library was mixed after ligation (index collision at
pooling / sequencing), as suspected from the shared dual index.

### Hierarchical (coarse-to-fine) draw of the pairs with many combinations (2026-11-18)

Problem: the re-attribution evaluates all n1 × n2 combinations of a pair; with -k 1000 instead of
-k 100 the mean number of candidates per read goes from 45 to 254 on the mosquito (bowtie2 ×8.8) and
the combinations from ~2,000 to ~64,000 per pair (~×30), i.e. hours. -k 100 itself is not biased
(mosquito, 36.5 % of the ambiguous reads have > 100 alignments, but the best AS is missed for 1.4 % of
them and the 100 reported copies are representative: same chromosome shares, same 1 Mb and 5 kb
distributions as -k 1000, r = 0.97 at 5 kb).

`reattribute_v2.hierarchical_draw` (`--max_combinations N`, default 0 = always exact):
- near combinations (same chromosome, < `--near` 100 kb, where P(s) varies steeply and where the
  copies of a repeat often carry both mates): all evaluated exactly (sorted R2 positions per chromosome,
  cost ~ (n1 + n2) log n + number of near combinations). A block average of P(s) would under-weight them
  by up to ~1,000× (candidate distances concentrated at short range, not uniform in the block);
- far combinations: `--n_samples` 256 drawn from a hierarchical proposal — chromosome pair ∝ coarse law
  (trans P(s) × mean density; intra: mean P(s) beyond `near` × mean density) × Σ alignment weights of
  the candidates on each chromosome, then candidates ∝ alignment weights — re-weighted by exact weight /
  proposal probability (importance sampling): unbiased estimate of the far mass, then resampling;
- near vs far drawn ∝ exact near mass vs estimated far mass.

Validation on yeast (all 5.16 M ambiguous pairs, -a, mode full):
- per pair (200 real pairs with > 100 combinations, median 408, max 8,432; 400 draws each): total
  variation to the exact distribution 0.110 (combinations) / 0.101 (10 kb bin pairs) vs 0.112 / 0.103
  for an exact sampler with the same number of draws; no pair worse by > 0.05;
- whole maps, pairs with > 20 combinations, distance to the exact run (seed 0):

| | TV 2 kb | TV 10 kb | TV 100 kb | r log 10 kb | intra | cis < 10 kb | trans pairs |
|---|---|---|---|---|---|---|---|
| exact, seed 1 | 0.1263 | 0.0471 | 0.0105 | 0.958 | 84.39 % | 63.97 % | 425,335 |
| hierarchical > 100 comb. (10.5 % of pairs) | 0.1263 | 0.0473 | 0.0105 | 0.958 | 84.38 % | 63.98 % | 425,520 |
| hierarchical > 20 comb. (52.8 % of pairs) | 0.1265 | 0.0472 | 0.0106 | 0.957 | 84.35 % | 63.97 % | 426,251 |

The hierarchical draw is indistinguishable from re-drawing the exact one with another seed.
Cost: ~3 ms per pair whatever its size (Python loop per pair) vs ~1.3 µs per combination for the batched
exact path → only worth it above ~2,500 combinations (yeast runs: exact 951 s, > 100: 1,071 s,
> 20: 1,570 s). Next: vectorise over batches of pairs, two-resolution density (1 Mb genome-wide + 5 kb
band near the diagonal) for the human genome at 5 kb, test on the mosquito with -k 1000.

### PCR jackpots: coordinate dedup (± 3 bp) and capped outlier pixels in the density (2026-11-18)

Mosquito (MM265a, v3 mitochondrion, mode full): a sharp chr1 ~65 Mb × chr3 ~35 Mb spot in the rescued
1 Mb map, absent from the unrescued one. The density map had a disc of log2 > 3 over ±5 Mb, with a value
of 62 at its centre, whereas the unique pairs gave O/E 4.2 over 2 × 2 Mb and 1.0 over 10 × 10 Mb. Cause:
one 200 kb pixel held 104 unique pairs (expected 0.012), all chr3 35,436,681 × chr1 64,930,896–900, same
strands: copies of a single molecule (PCR jackpot) differing by sequencing errors and 1–4 bp of local-
alignment clipping, so missed by the sequence dedup; hicstuff (exact coordinates) left 5. The density is
estimated before hicstuff: the jackpot alone brought the Gaussian kernel to n_min at σ = 1.1 Mb, and the
re-attribution filled the disc (rescued 1,549 contacts in chr1 60–70 × chr3 30–40 Mb vs 484 in mode
standard). The region also shares sequence with the chr1 centromere (repeat links), probably an assembly
issue, but the spot came from the jackpot.

Fixes:
- `streaming_classifier_v2.dedup_coordinates` (`--dedup_tol 3`, as pairtools dedup --max-mismatch):
  unique pairs removed when both positions are within 3 bp of a kept pair with the same chromosomes and
  strands (external sort, any size). Mosquito: 134,279 of 1,075,612 unique pairs (12.5 %).
- `density_v2.cap_outlier_pixels`: a pixel with > 20 contacts and > 100 × its expected is capped at
  max(20, 100 × expected) before smoothing (near-diagonal and moderate enrichments untouched).

| rescued contacts | chr1 × chr3 spot | centromeres chr1 62–67 × chr2 110–116 Mb |
|---|---|---|
| full, before | 1,549 (1.23 %) | 876 (0.71 %) |
| standard (P(s) × coverage) | 484 (0.40 %) | 281 (0.23 %) |
| full, fixed | 556 (0.44 %) | 804 (0.65 %) |

The fixed full mode removes the artefact and keeps the centromere clustering (visible, weaker, in the
unique pairs), which the standard mode loses; the standard mode also draws stripes (coverage-rich bins
attract the ambiguous reads). Density at the spot centre: 62 → 1.5.

With the v3 strain mitochondrion, O/E mito–chr3 (unrescued) = 0.039, at the level of chr1 / chr2 (0.033 /
0.024): the chr3 NUMT artefacts are gone (reference mito 1.72, v1 0.196, v2 0.071).

### Omics mode: benchmarks with a known truth, ps vs standard (2026-11-18)

`--data omics` (paired-end MNase-seq, ChIP-seq, ATAC-seq...; `omics_v2.py`): same classification and
re-attribution as Hi-C (no density), fragment-coverage bigWig tracks before / after, `rescued_fraction.bw`,
and with `--input_omics` the log2(IP / input) tracks (the input must be processed like the IP). Benchmarks:
a region is copied over another in the genome; the reads of the overwritten region receive the copied
sequence (they keep their name, so the truth is known); truth = the same reads on the original genome.
Figures and scripts: `/media/axel/EVO/data_for_v2/omics_figures/`.

**MNase-seq** (yeast SRR34921858, 10 M pairs, 2 × 51 bp; chrVII 50–58 kb copied over 150–158 kb; nucleosome
signal roughly uniform):

| | border fragments at the right copy | signal / truth source ; target | r 50 bp source ; target |
|---|---|---|---|
| random | 56.8 % | 0.61 ; 0.58 | 0.53 ; 0.71 |
| ps | 100 % | 1.19 ; 1.19 | 0.58 ; 0.79 |
| standard | 100 % | 0.87 ; 1.51 | 0.53 ; 0.77 |
| ceiling (copies indistinguishable) | | | 0.70 ; 0.83 |

The P(s) (fragment-length distribution) places the fragments that cross the border of a copy perfectly.
Fragments inside identical copies cannot be resolved (49 % right copy whatever the mode): ps splits them
evenly; random loses 40 % of the signal (mates placed in different copies are no longer fragments); standard
biases the copies with a coverage prior that is itself reconstructed (no unique read inside the copies).
(The 1.19 excess comes from the simulation: the injected reads have no sequencing error.)

**Cohesin ChIP-seq** (Scc1-HA, nocodazole, W303 T2T; IP CH376 6.2 M pairs, input CH381 4.0 M, 2 × 35 bp;
10 peak windows of 3 kb, log2(IP / input) 2.7–3.3, copied over 10 unenriched windows on other chromosomes;
medians over the 10 pairs):

| | IP signal / truth: peak ; no-signal copy | log2(IP/input) simulated − truth: peak ; no-signal copy |
|---|---|---|
| unrescued | 0.02 ; 0.04 | −1.65 ; −0.56 |
| random | 0.34 ; 1.86 | −0.60 ; +2.05 |
| ps | 0.69 ; 3.71 | −0.63 ; +2.02 |
| standard | 1.18 ; 0.56 | +0.07 ; −0.93 |

ps / random split the peak and create a false peak (+2) on the copy without signal (10 / 10 pairs);
standard puts the peak back on the right copy (its coverage context attracts the IP reads) but exaggerates
the contrast: the input (flat coverage) is split evenly while the IP background goes to the rich copy, so
the no-signal copy ends with a false depletion (6 / 10 pairs, down to −4.3) and 3 / 10 pairs keep a false
peak.

**Recommendation:** ps (default) for roughly uniform signals (MNase-seq, ATAC-seq at large scale, MFA / copy
number); standard for peaked signals (ChIP-seq of factors, cohesin, histone marks with peaks). In both cases
the signal of strictly identical copies is an estimate: check `rescued_fraction.bw`.

### Illumina adapter reads and adapter pieces in the assembly (2026-11-18)

After the coordinate dedup, a thin line remained at chr3 ~35.4 Mb in the mosquito maps. 1,246 unique pairs had
one read starting exactly at chr3:35,436,681 (+), with 626 different partners all over the genome. The reads
are `GATCGGAAGAGCACACGTCTGAACTCCAGTCACCG`: the DpnII site followed by the Illumina TruSeq adapter (fragments
with no insert), and the assembly contains 24–29 bp of this adapter inside a satellite at chr3:35,436,680
(contamination of the assembly). The adapter reads align there uniquely (MAPQ >= 30) and their mates create
false contacts with the whole genome; one PCR-amplified molecule of this kind was the 104-copy jackpot above.

- `streaming_classifier_v2.insert_length` / `--min_insert 20`: a read whose adapter (core GATCGGAAGAGC,
  preceded or not by its A) starts before base 20 is treated as unmapped (group0); longer inserts are kept
  (local alignment clips the adapter). Mosquito: 66,896 pairs removed; pairs at chr3:35,436,681 1,246 → 43;
  contacts of its 100 kb bin 5,422 → 1,790; the line is gone, the centromere clustering is kept.
- `mappability_v2.adapter_sites`: the 19 bp TruSeq read 1 / read 2 adapter starts (both strands) are searched
  in the assembly at the mappability stage → `mappability/adapter_sites.tsv` and a warning. Mosquito
  (VPISU_Cqui_1.0 + wPipMtq): 9 pieces (chr1 114.2; chr2 65.7, 93.0, 204.3, 213.8; chr3 35.4, 89.9, 123.9,
  132.7 Mb); yeast T2T: none.
