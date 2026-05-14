#!/usr/bin/env bash
# Extract per-position PhyloP conservation scores from bigWig files
# using bigWigAverageOverBed.
#
# Input format (.BW.input.txt): BED-like file with one variant per line.
# Output: tab-separated file with one score column per position.
#
# Tools required:
#   bigWigAverageOverBed (UCSC): https://hgdownload.soe.ucsc.edu/admin/exe/
#
# Reference bigWig files:
#   Human hg38: hg38.phyloP100way.bw  (UCSC, https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/)
#   Mouse mm39: mm39.phyloP35way.bw   (UCSC, https://hgdownload.soe.ucsc.edu/goldenPath/mm39/phyloP35way/)

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────

BIGWIG_BIN="path/to/bigWigAverageOverBed"   # UCSC tool binary

PHYLOP_HUMAN_BW="path/to/hg38.phyloP100way.bw"
PHYLOP_MOUSE_BW="path/to/mm39.phyloP35way.bw"

# Input BED files (one per dataset); each line = one variant position
# Format: chrom  start  end  (BED3, 0-based half-open coordinates)
INPUT_DIR="path/to/input_bed_files"
OUTPUT_DIR="path/to/output_scores"

mkdir -p "$OUTPUT_DIR"

# ── human datasets (hg38) ─────────────────────────────────────────────────────
# Each dataset has a pair of input files:
#   <dataset>.BW.input.txt         — full-window coordinates
#   <dataset>.BW.input.precpos.txt — single-base variant position

HUMAN_DATASETS=(
    "Benign.single.nucleotide.variant.clinvar.Filt"
    "Pathogenic.single.nucleotide.variant.clinvar.Filt"
    "Uncertain_significance.single.nucleotide.variant.clinvar.Filt.Sampled"
)

for DATASET in "${HUMAN_DATASETS[@]}"; do
    INPUT_FULL="${INPUT_DIR}/${DATASET}.BW.input.txt"
    INPUT_POS="${INPUT_DIR}/${DATASET}.BW.input.precpos.txt"

    echo "=== ${DATASET} ==="
    "$BIGWIG_BIN" "$PHYLOP_HUMAN_BW" "$INPUT_POS"  "${OUTPUT_DIR}/${DATASET}.PHYLOP.precpos.tsv"
    "$BIGWIG_BIN" "$PHYLOP_HUMAN_BW" "$INPUT_FULL" "${OUTPUT_DIR}/${DATASET}.PHYLOP.tsv"
done

# ── mouse dataset (mm39) ──────────────────────────────────────────────────────

MOUSE_DATASET="mouse_GRCm39_8categories_5994_noN.BW.input"
MOUSE_INPUT_FULL="${INPUT_DIR}/${MOUSE_DATASET}.txt"

echo "=== mouse GRCm39 ==="
"$BIGWIG_BIN" "$PHYLOP_MOUSE_BW" "$MOUSE_INPUT_FULL" "${OUTPUT_DIR}/${MOUSE_DATASET}.PHYLOP.tsv"

echo "Done. Scores written to: ${OUTPUT_DIR}"
