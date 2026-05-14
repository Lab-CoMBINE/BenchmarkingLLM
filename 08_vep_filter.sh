#!/usr/bin/env bash
# Post-process VEP output: keep one row per variant (the transcript with the
# highest CADD_RAW score).
#
# VEP output structure assumed:
#   Lines 1-43  : metadata comments (##)
#   Line  44    : column header (#Uploaded_variation ...)
#   Lines 45+   : data (one row per variant/transcript pair)
#
# Column indices for CADD_RAW and CADD_PHRED are detected dynamically
# from the header line, so the script works regardless of VEP column order.
#
# Output file per dataset:
#   <dataset>.VEP.CADD.MAX.tsv  — one row per variant (max CADD_RAW transcript)

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
# Keeps one row per variant: the transcript with the highest CADD_RAW score.
# Column indices for CADD_RAW and CADD_PHRED are read from the header line.

filter_cadd_max() {
    local vep_tsv="$1"
    local out_tsv="$2"

    # find CADD_RAW and CADD_PHRED column indices from the header line
    local header_line
    header_line=$(head -$((HEADER_LINES + 1)) "$vep_tsv" | tail -1)

    local cadd_raw_col cadd_phred_col
    cadd_raw_col=$(echo "$header_line" | tr '\t' '\n' | grep -n "^CADD_RAW$"   | cut -d: -f1)
    cadd_phred_col=$(echo "$header_line" | tr '\t' '\n' | grep -n "^CADD_PHRED$" | cut -d: -f1)

    if [[ -z "$cadd_raw_col" || -z "$cadd_phred_col" ]]; then
        echo "ERROR: CADD_RAW or CADD_PHRED column not found in ${vep_tsv}" >&2
        return 1
    fi

    echo "  Detected CADD_RAW col=${cadd_raw_col}, CADD_PHRED col=${cadd_phred_col}"

    # copy metadata comments
    head -"$HEADER_LINES" "$vep_tsv" > "$out_tsv"

    # copy header line
    echo "$header_line" >> "$out_tsv"

    # data: sort by variant ID then by CADD_RAW descending, deduplicate by variant
    tail -n +"$DATA_START" "$vep_tsv" \
        | sort -k1,1 -k"${cadd_raw_col}","${cadd_raw_col}"rn \
        | sort -uk1,1 >> "$out_tsv"
}

# ── process each dataset ──────────────────────────────────────────────────────

for DATASET in "${DATASETS[@]}"; do
    VEP_TSV="${INPUT_DIR}/${DATASET}.VEP.tsv"

    if [[ ! -f "$VEP_TSV" ]]; then
        echo "WARNING: input not found, skipping: ${VEP_TSV}"
        continue
    fi

    echo "=== ${DATASET} ==="
    filter_cadd_max "$VEP_TSV" "${OUTPUT_DIR}/${DATASET}.VEP.CADD.MAX.tsv"
    echo "  CADD done"
done

echo "Done. Filtered files written to: ${OUTPUT_DIR}"
