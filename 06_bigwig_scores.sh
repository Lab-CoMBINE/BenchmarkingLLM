#!/usr/bin/env bash
# Extract per-position conservation scores from bigWig files using bigWigAverageOverBed.
#
# Scores computed:
#   - GERP (dbNSFP GerpRs)
#   - PhastCons (100-way vertebrate alignment)
#   - PhyloP  (100-way vertebrate alignment)
#
# Input format (.BW.input.txt): BED-like file with one variant per line.
# Output: tab-separated file with one score column per position.
#
# Tools required:
#   bigWigAverageOverBed (UCSC): https://hgdownload.soe.ucsc.edu/admin/exe/
#
# Reference bigWig files (human hg38):
#   GERP:      dbNsfpGerpRs.bw          (dbNSFP, https://sites.google.com/site/jpopgen/dbNSFP)
#   PhastCons: hg38.phastCons100way.bw  (UCSC, https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/)
#   PhyloP:    hg38.phyloP100way.bw     (UCSC, https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/)
#
# Reference bigWig files (mouse mm39):
#   PhyloP:    mm39.phyloP35way.bw      (UCSC, https://hgdownload.soe.ucsc.edu/goldenPath/mm39/phyloP35way/)

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────

BIGWIG_BIN="path/to/bigWigAverageOverBed"   # UCSC tool binary

# bigWig reference files
GERP_BW="path/to/dbNsfpGerpRs.bw"
PHASTCONS_BW="path/to/hg38.phastCons100way.bw"
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

    echo "  GERP ..."
    "$BIGWIG_BIN" "$GERP_BW"       "$INPUT_POS" "${OUTPUT_DIR}/${DATASET}.GERP.precpos.tsv"
    "$BIGWIG_BIN" "$GERP_BW"       "$INPUT_FULL" "${OUTPUT_DIR}/${DATASET}.GERP.tsv"

    echo "  PhastCons ..."
    "$BIGWIG_BIN" "$PHASTCONS_BW"  "$INPUT_POS" "${OUTPUT_DIR}/${DATASET}.PHASTCONS.precpos.tsv"
    "$BIGWIG_BIN" "$PHASTCONS_BW"  "$INPUT_FULL" "${OUTPUT_DIR}/${DATASET}.PHASTCONS.tsv"

    echo "  PhyloP ..."
    "$BIGWIG_BIN" "$PHYLOP_HUMAN_BW" "$INPUT_POS" "${OUTPUT_DIR}/${DATASET}.PHYLOP.precpos.tsv"
    "$BIGWIG_BIN" "$PHYLOP_HUMAN_BW" "$INPUT_FULL" "${OUTPUT_DIR}/${DATASET}.PHYLOP.tsv"
done

# ── mouse dataset (mm39) ──────────────────────────────────────────────────────

MOUSE_DATASET="mouse_GRCm39_8categories_5994_noN.BW.input"
MOUSE_INPUT_FULL="${INPUT_DIR}/${MOUSE_DATASET}.txt"

echo "=== mouse GRCm39 ==="
echo "  PhyloP ..."
"$BIGWIG_BIN" "$PHYLOP_MOUSE_BW" "$MOUSE_INPUT_FULL" "${OUTPUT_DIR}/${MOUSE_DATASET}.PHYLOP.tsv"

echo "Done. Scores written to: ${OUTPUT_DIR}"
