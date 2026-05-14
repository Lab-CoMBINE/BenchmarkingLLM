#!/usr/bin/env bash
# Post-process VEP output: keep one row per variant (the transcript with the
# highest score for each metric).
#
# VEP output structure assumed:
#   Lines 1-43  : metadata comments (##)
#   Line  44    : column header (#Uploaded_variation ...)
#   Lines 45+   : data (one row per variant/transcript pair)
#
# Column indices (1-based, tab-separated):
#   1   : Uploaded_variation (variant ID, used as grouping key)
#   18  : SIFT score
#   19  : PolyPhen score
#   20  : CADD_RAW
#   21  : CADD_PHRED
#
# Output files per dataset:
#   <dataset>.VEP.SIFT.MAX.tsv      — columns 1-18  (max SIFT per variant)
#   <dataset>.VEP.POLYPHEN.MAX.tsv  — columns 1-17 + 19
#   <dataset>.VEP.CADD.MAX.tsv      — columns 1-17 + 20-21 (max CADD_RAW per variant)

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────

INPUT_DIR="path/to/vep_output"     # directory containing the raw VEP .tsv files
OUTPUT_DIR="path/to/vep_filtered"  # directory for filtered output files

HEADER_LINES=43    # number of ## metadata lines before the column header
DATA_START=45      # first data line (header is line 44)

mkdir -p "$OUTPUT_DIR"

# ── datasets ─────────────────────────────────────────────────────────────────

DATASETS=(
    "Benign.single.nucleotide.variant.clinvar.Filt"
    "Pathogenic.single.nucleotide.variant.clinvar.Filt"
    "Uncertain_significance.single.nucleotide.variant.clinvar.Filt.Sampled"
)

# ── filter function ───────────────────────────────────────────────────────────
# Usage: filter_max VEP_TSV OUT_TSV SORT_COL "col1,col2,..."
#   VEP_TSV   : full path to raw VEP output
#   OUT_TSV   : full path for filtered output
#   SORT_COL  : column number to sort descending (to pick max score)
#   AWK_COLS  : awk print expression for desired output columns

filter_max() {
    local vep_tsv="$1"
    local out_tsv="$2"
    local sort_col="$3"
    local awk_cols="$4"

    # copy metadata comments (lines 1 to HEADER_LINES)
    head -"$HEADER_LINES" "$vep_tsv" > "$out_tsv"

    # copy column header line with selected columns
    head -$((HEADER_LINES + 1)) "$vep_tsv" | tail -1 \
        | awk -v OFS="\t" "{print ${awk_cols}}" >> "$out_tsv"

    # data: sort by variant ID (col 1) then by score descending, deduplicate
    tail -n +"$DATA_START" "$vep_tsv" \
        | sort -k1,1 -k"${sort_col}","${sort_col}"rn \
        | sort -uk1,1 \
        | awk -v OFS="\t" "{print ${awk_cols}}" >> "$out_tsv"
}

# ── process each dataset ──────────────────────────────────────────────────────

SIFT_COLS='$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18'
POLYPHEN_COLS='$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$19'
CADD_COLS='$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$20,$21'

for DATASET in "${DATASETS[@]}"; do
    VEP_TSV="${INPUT_DIR}/${DATASET}.VEP.tsv"

    if [[ ! -f "$VEP_TSV" ]]; then
        echo "WARNING: input not found, skipping: ${VEP_TSV}"
        continue
    fi

    echo "=== ${DATASET} ==="

    filter_max "$VEP_TSV" "${OUTPUT_DIR}/${DATASET}.VEP.SIFT.MAX.tsv"    18 "$SIFT_COLS"
    echo "  SIFT done"

    filter_max "$VEP_TSV" "${OUTPUT_DIR}/${DATASET}.VEP.POLYPHEN.MAX.tsv" 19 "$POLYPHEN_COLS"
    echo "  PolyPhen done"

    filter_max "$VEP_TSV" "${OUTPUT_DIR}/${DATASET}.VEP.CADD.MAX.tsv"     20 "$CADD_COLS"
    echo "  CADD done"
done

echo "Done. Filtered files written to: ${OUTPUT_DIR}"
