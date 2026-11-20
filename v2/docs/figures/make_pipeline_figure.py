"""One-page diagram of the hicberg v2 pipeline (docs/figures/hicberg_v2_pipeline.{png,pdf}).

usage: python3 docs/figures/make_pipeline_figure.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
W, H = 8.27, 11.69                                    # A4 portrait, inches
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# phase colours: (fill, edge)
C = {"prep": ("#E3F0FB", "#2B6CB0"), "cls": ("#E0F4F1", "#1F8A70"), "law": ("#EAF6E2", "#4C9A2A"),
     "res": ("#FDF0E1", "#D9730D"), "mat": ("#F1E9F9", "#7B4BB7"), "in": ("#F4F4F4", "#555555"),
     "idx": ("#FFF6CC", "#B8860B"), "out": ("#FCE8EC", "#C2185B")}


def box(x, y, w, h, kind, title, lines=(), title_size=9.5, size=7.2, step=None, lw=1.4):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw))
    tx = x + 0.12
    if step is not None:
        ax.add_patch(plt.Circle((x + 0.22, y + h - 0.2), 0.14, color=ec))
        ax.text(x + 0.22, y + h - 0.2, str(step), ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold")
        tx = x + 0.44
    ax.text(tx, y + h - 0.2, title, ha="left", va="center", fontsize=title_size, fontweight="bold",
            color=ec)
    for i, l in enumerate(lines):
        ax.text(x + 0.14, y + h - 0.45 - i * 0.165, l, ha="left", va="top", fontsize=size,
                color="#222222")


def arrow(x0, y0, x1, y1, color="#444444", style="-|>", lw=1.3, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------- title
ax.text(W / 2, H - 0.35, "HiCberg v2 — re-attribution of ambiguous Hi-C reads", ha="center",
        va="center", fontsize=15, fontweight="bold", color="#1A1A1A")
ax.text(W / 2, H - 0.62, "hicberg_v2.py   ·   stages: index → mappability → align → classify → stats → "
        "rescue → matrices   (--start_stage / --exit_stage)", ha="center", va="center", fontsize=7.6,
        color="#555555")

# ---------------------------------------------------------------- inputs
yin = H - 1.35
box(0.35, yin, 2.35, 0.55, "in", "Genome FASTA", ["+ --circular molecules"], size=6.8)
box(2.95, yin, 2.35, 0.55, "in", "Hi-C reads R1 / R2", ["FASTQ (gz), paired-end"], size=6.8)
box(5.55, yin, 2.35, 0.55, "in", "Enzymes (optional)", ["e.g. DpnII,HinfI · Micro-C: mnase"], size=6.8)

# ---------------------------------------------------------------- 1-3 preparation
yp = yin - 1.15
box(0.35, yp, 2.35, 0.9, "prep", "index", ["bowtie2-build of the genome", "(skipped if it exists)"], step=1)
box(2.95, yp, 2.35, 0.9, "prep", "mappability", ["k-mers of the read length,", "one every --mappability_step bp",
                                                  "→ mask + weights of the bins"], step=2)
box(5.55, yp, 2.35, 0.9, "prep", "align", ["bowtie2 --very-sensitive-local", "--reorder, R1 and R2",
                                           "aligned separately"], step=3)
for x in (1.52, 4.12, 6.72):
    arrow(x, yin, x, yp + 0.9)
arrow(2.7, yp + 0.45, 2.95, yp + 0.45)
arrow(5.3, yp + 0.45, 5.55, yp + 0.45)

# ---------------------------------------------------------------- 4 classify
yc = yp - 1.55
box(0.35, yc, 7.55, 1.3, "cls", "classify — streaming_classifier_v2", step=4)
gx = [0.55, 2.95, 5.35]
gt = [("group0  ·  discarded", ["a mate unmapped, Illumina adapter", "(< 20 bp before it), chimeric"]),
      ("group1  ·  unique pairs", ["both mates MAPQ ≥ 30", "PCR duplicates removed on", "the sequences → unrescued.pairs"]),
      ("group2  ·  ambiguous pairs", ["≥ 1 mate multi-mapping", "→ FASTQ chunks (one per worker)", "+ best hit of each read"])]
for x, (t, ls) in zip(gx, gt):
    ax.add_patch(FancyBboxPatch((x, yc + 0.1), 2.2, 0.82, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="white", ec=C["cls"][1], lw=0.9))
    ax.text(x + 0.1, yc + 0.8, t, fontsize=7.6, fontweight="bold", color=C["cls"][1], va="center")
    for i, l in enumerate(ls):
        ax.text(x + 0.1, yc + 0.62 - i * 0.16, l, fontsize=6.7, va="center", color="#222222")
arrow(6.72, yp, 6.72, yc + 1.3)

# ---------------------------------------------------------------- 5 stats
ys = yc - 1.75
box(0.35, ys, 4.55, 1.5, "law", "stats — statistical laws from group1", [
    "• intra P(s) by orientation (uncut, circle, weird), pooled",
    "  smoothing, orientations merged beyond 5 kb; trans P(s)",
    "• 1D coverage, corrected by the mappable fraction",
    "• 2D contact density: adaptive Gaussian, mappability mask",
    "  + weights, PCR jackpots capped, refined bins for small",
    "  dense molecules (mito, plasmids)"], step=5, size=6.9)
arrow(4.05, yc + 0.1, 2.6, ys + 1.5, color=C["cls"][1])            # group1 → stats

# repeat index (built once)
box(5.15, ys + 0.15, 2.75, 1.2, "idx", "repeat index (once)", [
    "repeat_index_v2.py: (k = 31, w = 10)",
    "minimizers seen ≥ min_count times,",
    "with their 2 kb bins and copies",
    "→ ALL copies of a repeat"], size=6.7)

# ---------------------------------------------------------------- 6 rescue
yr = ys - 3.05
box(0.35, yr, 7.55, 2.8, "res", "rescue — reattribute_v2: one position drawn per ambiguous pair", step=6)
# candidates
cx, cw = 0.55, 3.55
ax.add_patch(FancyBboxPatch((cx, yr + 1.12), cw, 1.28, boxstyle="round,pad=0.02,rounding_size=0.06",
                            fc="white", ec=C["res"][1], lw=0.9))
ax.text(cx + 0.1, yr + 2.27, "a. candidates of each read", fontsize=7.8, fontweight="bold",
        color=C["res"][1], va="center")
for i, l in enumerate(["--candidates bowtie2: 2nd alignment -k K (or -a),",
                       "  hits within --as_delta of the best score,",
                       "  weighted by their likelihood (--error_rate);",
                       "  saturated lists (-k reached) → repeat index",
                       "--candidates index: no 2nd bowtie2, first hit",
                       "  + index bins (exact scores if ≤ 50 bins)"]):
    ax.text(cx + 0.1, yr + 2.08 - i * 0.155, l, fontsize=6.6, va="center", color="#222222")
# propensity
px, pw = 4.3, 3.4
ax.add_patch(FancyBboxPatch((px, yr + 1.12), pw, 1.28, boxstyle="round,pad=0.02,rounding_size=0.06",
                            fc="white", ec=C["res"][1], lw=0.9))
ax.text(px + 0.1, yr + 2.27, "b. propensity of each R1 × R2 combination", fontsize=7.8,
        fontweight="bold", color=C["res"][1], va="center")
for i, l in enumerate(["--mode full (default):", "   P(s) × density(R1, R2) × weights",
                       "standard:  P(s) × cov(R1) × cov(R2)", "ps:  P(s)    ·    random:  1",
                       "(same 5' / orientation conventions", " as the statistics)"]):
    ax.text(px + 0.1, yr + 2.08 - i * 0.155, l, fontsize=6.6, va="center", color="#222222")
# draw
ax.add_patch(FancyBboxPatch((cx, yr + 0.12), 7.15, 0.85, boxstyle="round,pad=0.02,rounding_size=0.06",
                            fc="white", ec=C["res"][1], lw=0.9))
ax.text(cx + 0.1, yr + 0.83, "c. weighted draw (seeded, reproducible)", fontsize=7.8, fontweight="bold",
        color=C["res"][1], va="center")
for i, l in enumerate(["≤ --max_combinations (5000): exact, vectorised over batches of pairs",
                       "more: hierarchical — near combinations (< 100 kb) exact, far ones by importance sampling",
                       "→ predicted.pairs;   rescued.pairs = unrescued.pairs + predicted.pairs"]):
    ax.text(cx + 0.1, yr + 0.64 - i * 0.165, l, fontsize=6.6, va="center", color="#222222")
arrow(cx + cw / 2, yr + 1.12, cx + cw / 2, yr + 0.97, color=C["res"][1])
arrow(px + pw / 2, yr + 1.12, px + pw / 2, yr + 0.97, color=C["res"][1])
arrow(cx + cw, yr + 1.76, px, yr + 1.76, color=C["res"][1])
# inputs of the rescue
arrow(6.4, yc + 0.1, 7.75, yr + 2.8, color=C["cls"][1], rad=-0.25)          # group2 → rescue
ax.text(7.95, yr + 3.65, "group2", fontsize=6.6, color=C["cls"][1], rotation=-90, va="center")
arrow(2.6, ys, 2.6, yr + 2.8, color=C["law"][1])                             # laws → rescue
ax.text(2.7, ys - 0.12, "laws", fontsize=6.6, color=C["law"][1], va="center")
arrow(6.5, ys + 0.15, 6.0, yr + 2.8, color=C["idx"][1], ls="--")            # index → rescue

# ---------------------------------------------------------------- 7 matrices
ym = yr - 1.65
box(0.35, ym, 4.55, 1.4, "mat", "matrices — Hi-C", [
    "• hicstuff filtering (--hicstuff): uncuts / loops,",
    "  thresholds from unrescued applied to rescued",
    "• cooler matrices + ICE, coarsened for the plots",
    "• before / after maps with tracks: GC, mappability,",
    "  restriction sites, coverage, repeat arcs"], step=7, size=6.9)
box(5.15, ym, 2.75, 1.4, "mat", "omics (--data omics)", [
    "ChIP / MNase-seq, paired-end:",
    "fragment coverage bigWig",
    "before / after, log2(IP / input)",
    "(P(s) = fragment lengths)"], size=6.8)
arrow(2.6, yr, 2.6, ym + 1.4, color=C["res"][1])
arrow(6.5, yr, 6.5, ym + 1.4, color=C["res"][1], ls="--")

# ---------------------------------------------------------------- outputs
yo = ym - 0.8
box(0.35, yo, 7.55, 0.6, "out", "Outputs", [
    "unrescued / rescued .pairs and .cool (filtered), statistics/ (laws + checkpoint plots), "
    "matrices/plots/ (before / after PDF)"], size=6.8)
arrow(2.6, ym, 2.6, yo + 0.6, color=C["mat"][1])
arrow(6.5, ym, 6.5, yo + 0.6, color=C["mat"][1])

# ---------------------------------------------------------------- legend
yl = 0.13
items = [("prep", "preparation"), ("cls", "classification"), ("law", "statistical laws"),
         ("res", "re-attribution"), ("mat", "matrices / tracks"), ("idx", "repeat index")]
for i, (k, t) in enumerate(items):
    x = 0.45 + i * 1.28
    ax.add_patch(FancyBboxPatch((x, yl - 0.07), 0.22, 0.14, boxstyle="round,pad=0.01", fc=C[k][0],
                                ec=C[k][1], lw=1))
    ax.text(x + 0.3, yl, t, fontsize=6.8, va="center", color="#333333")

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"hicberg_v2_pipeline.{ext}", dpi=200)
print("ok", OUT)
