# DNA LLM Benchmark

Inference pipelines used to evaluate genomic language models on single-nucleotide variant (SNV) datasets.
Each notebook computes embedding-space distances (cosine, Euclidean, Manhattan, Jensen-Shannon, Hellinger, cross-entropy) between the reference and alternative sequence of each variant.

## Scripts

### Sequence preparation and LLM inference

| Script | Description |
|---|---|
| `00_prepare_sequences.py` | Build REF/ALT sequence pairs from a VCF + reference genome FASTA |
| `01_nt_inference.py` | Nucleotide Transformer v2 (100M / 250M / 500M / 2.5B) — InstaDeepAI |
| `02_dnabert_inference.py` | DNABERT-6 — zhihan1996 |
| `03_hyenadna_1k_inference.py` | HyenaDNA tiny-1k — LongSafari |
| `04_hyenadna_32k_inference.py` | HyenaDNA small-32k — LongSafari |
| `05_evo2_inference.py` | Evo 2 7B — ARC Institute |

## Input format

A plain-text file, one variant per line:

```
<REF_sequence> <ALT_sequence>
```

- Both sequences are the same length (5994 bp in the paper)
- They differ only at the central position (index 2997, 0-based)
- No header line

`data/example_human_clinvar_100.txt` contains 100 human ClinVar variants of uncertain significance in this format (source: ClinVar, GRCh38).

## Sequence preparation

To build your own dataset from a VCF + FASTA, see `00_PrepareSequences.ipynb`.
It requires:
- A VCF file with annotated consequence terms (e.g., `missense_variant`, `intron_variant`, …)
- The matching reference genome FASTA (indexed with `pysam.faidx`)
- The `pysam` library

The script selects up to N variants per functional category, extracts a window of ±2997 bp around each SNV, applies the alternative allele, and filters out sequences containing ambiguous bases (N).

## Usage

```bash
# Nucleotide Transformer (all sizes)
python 01_nt_inference.py --seq_file data/example_human_clinvar_100.txt --out_dir results/

# DNABERT-6
python 02_dnabert_inference.py --seq_file data/example_human_clinvar_100.txt --out_dir results/

# HyenaDNA 1k
python 03_hyenadna_1k_inference.py --seq_file data/example_human_clinvar_100.txt --out_dir results/

# HyenaDNA 32k
python 04_hyenadna_32k_inference.py --seq_file data/example_human_clinvar_100.txt --out_dir results/

# Evo 2 7B
python 05_evo2_inference.py --seq_file data/example_human_clinvar_100.txt --out_dir results/

# Prepare sequences from VCF + FASTA
python 00_prepare_sequences.py \
    --vcf path/to/variants.vcf.gz \
    --fasta path/to/genome.fna \
    --out_dir results/ \
    --n_per_cat 5000
```

All scripts accept `--help` for a full list of options.
Models are downloaded automatically from HuggingFace on first run (`--cache_dir cache/`).
For gated models set your token:

```bash
export HF_TOKEN="your_token_here"
```

## Requirements

**Python scripts (01–05):**
```
torch
transformers
numpy
pandas
tqdm
pysam   # only for 00_prepare_sequences.py
evo2    # only for 05_evo2_inference.py
```
GPU with CUDA is strongly recommended. Tested with Python 3.10.
