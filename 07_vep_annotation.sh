#!/usr/bin/env bash
# Annotate ClinVar SNV datasets with Ensembl VEP.
#
# Plugins used:
#   CADD    — Combined Annotation Dependent Depletion (PHRED-scaled score)
#   SIFT    — sorting intolerant from tolerant (per-transcript)
#   PolyPhen — polymorphism phenotyping v2 (per-transcript)
#
# Output: tab-separated VEP output (--most_severe: one line per variant/transcript).
#
# Tools required:
#   vep (Ensembl VEP): https://www.ensembl.org/info/docs/tools/vep/script/index.html
#   Conda environment recommended — see https://www.ensembl.org/info/docs/tools/vep/script/vep_download.html
#
# Reference files required:
#   VEP cache:  download with `vep_install -a cf -s homo_sapiens -y GRCh38`
#   CADD db:    whole_genome_SNVs.tsv.gz (CADD v1.7, GRCh38)
#               https://krishna.gs.washington.edu/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz
#               (must be bgzipped and tabix-indexed)

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────

VEP_BIN="vep"                                   # or full path to the vep executable
VEP_CACHE_DIR="path/to/vep_cache"               # VEP cache directory
VEP_PLUGINS_DIR="path/to/vep_plugins"           # VEP plugins directory
CADD_DB="path/to/whole_genome_SNVs.tsv.gz"      # CADD v1.7 GRCh38 database

INPUT_DIR="path/to/vep_input_files"             # directory with .VEP.input.txt files
OUTPUT_DIR="path/to/vep_output"                 # output directory

FORKS=20   # parallel threads for VEP

mkdir -p "$OUTPUT_DIR"

# ── datasets ─────────────────────────────────────────────────────────────────

DATASETS=(
    "Benign.single.nucleotide.variant.clinvar.Filt"
    "Pathogenic.single.nucleotide.variant.clinvar.Filt"
    "Uncertain_significance.single.nucleotide.variant.clinvar.Filt.Sampled"
)

# ── run VEP ──────────────────────────────────────────────────────────────────

for DATASET in "${DATASETS[@]}"; do
    INPUT="${INPUT_DIR}/${DATASET}.VEP.input.txt"
    OUTPUT="${OUTPUT_DIR}/${DATASET}.VEP.tsv"

    echo "=== Annotating: ${DATASET} ==="

    "$VEP_BIN" \
        --cache \
        --dir_cache   "$VEP_CACHE_DIR" \
        --dir_plugins "$VEP_PLUGINS_DIR" \
        --plugin CADD,"$CADD_DB" \
        --offline \
        --fork "$FORKS" \
        --tab \
        --sift s \
        --polyphen s \
        --most_severe \
        --input_file  "$INPUT" \
        --output_file "$OUTPUT"

    echo "  -> ${OUTPUT}"
done

echo "Done."
