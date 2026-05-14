"""
Prepare REF/ALT sequence pairs from a VCF + reference genome FASTA.

For each SNV in the VCF the script extracts a window of 5994 bp
(2997 bp upstream + variant + 2996 bp downstream), applies the ALT allele,
and writes REF and ALT sequences space-separated on a single line.

Variant categories (from VEP CSQ field):
  Nonsense, Splicing, Intronic, Missense, Stop, Synonymous, 3UTR, 5UTR

Requirements: bcftools (in PATH), pysam
"""

import argparse
import os
import random
import re
import subprocess
from collections import Counter

import pandas as pd
import pysam


LEFT_FLANK = 2997
RIGHT_FLANK = 2996
TOTAL_LEN = LEFT_FLANK + 1 + RIGHT_FLANK  # 5994
VAR_POS_0 = LEFT_FLANK                    # 0-based index of the variant in the window

CATEGORY_MAP = {
    "Nonsense":   {"stop_gained"},
    "Splicing":   {"splice_donor_variant", "splice_acceptor_variant"},
    "Intronic":   {"intron_variant"},
    "Missense":   {"missense_variant"},
    "Stop":       {"stop_lost"},
    "Synonymous": {"synonymous_variant"},
    "3UTR":       {"3_prime_UTR_variant"},
    "5UTR":       {"5_prime_UTR_variant"},
}

RAW_TO_CATEGORY = {
    term: cat for cat, terms in CATEGORY_MAP.items() for term in terms
}
TARGET_TERMS = set(RAW_TO_CATEGORY)


# ── helpers ──────────────────────────────────────────────────────────────────

def build_chr_to_contig(fasta_path, fasta):
    """Map chromosome names (1, 2, …, X, Y, MT) to FASTA contig IDs."""
    mapping = {}
    with open(fasta_path) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            short_name = header.split()[0]
            m = re.search(r'chromosome\s+([0-9XYM]+)', header)
            if m:
                chrom = m.group(1)
                if chrom == "M":
                    chrom = "MT"
                mapping[chrom] = short_name
    for ref_name in fasta.references:
        if ref_name in [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]:
            mapping.setdefault(ref_name, ref_name)
    return mapping


def classify_variant(alt, csq):
    """Return (category, vep_annotation) or (None, None) if not eligible."""
    if not csq or csq in (".", ""):
        return None, None
    matched_cats = set()
    matched_terms = set()
    for entry in csq.split(","):
        parts = entry.split("|")
        if len(parts) < 2 or parts[0] != alt:
            continue
        for term in parts[1].split("&"):
            term = term.strip()
            if term in TARGET_TERMS:
                matched_terms.add(term)
                matched_cats.add(RAW_TO_CATEGORY[term])
    if len(matched_cats) != 1:
        return None, None
    return next(iter(matched_cats)), ";".join(sorted(matched_terms))


def has_valid_window(chrom, pos, chr_to_contig, fasta):
    if chrom not in chr_to_contig:
        return False
    contig_len = fasta.get_reference_length(chr_to_contig[chrom])
    return LEFT_FLANK < pos <= contig_len - RIGHT_FLANK


def bcftools_stream(vcf_path):
    cmd = [
        "bcftools", "query",
        "-i", 'TYPE="snp" && FILTER="PASS"',
        "-f", "%CHROM\t%POS\t%REF\t%ALT\t%INFO/CSQ\n",
        vcf_path,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)


def make_sequences(chrom, pos_1based, ref, alt, chr_to_contig, fasta):
    """Extract REF and ALT 5994-bp windows. Returns (seq_ref, seq_alt, status)."""
    if chrom not in chr_to_contig:
        return None, None, f"missing_chrom:{chrom}"
    contig = chr_to_contig[chrom]
    contig_len = fasta.get_reference_length(contig)
    start_1 = pos_1based - LEFT_FLANK
    end_1 = pos_1based + RIGHT_FLANK
    if start_1 < 1:
        return None, None, "left_boundary"
    if end_1 > contig_len:
        return None, None, "right_boundary"
    seq_ref = fasta.fetch(contig, start_1 - 1, end_1).upper()
    if len(seq_ref) != TOTAL_LEN:
        return None, None, f"bad_length:{len(seq_ref)}"
    if seq_ref[VAR_POS_0] != ref:
        return None, None, f"ref_mismatch:genome={seq_ref[VAR_POS_0]},vcf={ref}"
    seq_alt = seq_ref[:VAR_POS_0] + alt + seq_ref[VAR_POS_0 + 1:]
    return seq_ref, seq_alt, "ok"


# ── pipeline steps ────────────────────────────────────────────────────────────

def step1_select_variants(vcf_path, fasta, chr_to_contig, target_per_cat, seed):
    """Reservoir-sample up to target_per_cat variants per category from the VCF."""
    # Pass 1: count eligible variants
    eligible = Counter()
    proc = bcftools_stream(vcf_path)
    for i, line in enumerate(proc.stdout, 1):
        chrom, pos, ref, alt_field, csq = line.rstrip("\n").split("\t", 4)
        pos = int(pos)
        if not has_valid_window(chrom, pos, chr_to_contig, fasta):
            continue
        for alt in alt_field.split(","):
            if len(ref) != 1 or len(alt) != 1 or alt in {".", "*"} or alt.startswith("<"):
                continue
            cat, _ = classify_variant(alt, csq)
            if cat:
                eligible[cat] += 1
        if i % 1_000_000 == 0:
            print(f"  pass 1: {i:,} VCF records read")
    proc.communicate()

    print("Eligible variants per category:")
    for cat in CATEGORY_MAP:
        print(f"  {cat}: {eligible.get(cat, 0):,}")

    target_n = {cat: min(target_per_cat, eligible.get(cat, 0)) for cat in CATEGORY_MAP}

    # Pass 2: reservoir sampling
    rng = random.Random(seed)
    selected = {cat: [] for cat in CATEGORY_MAP}
    seen = Counter()

    proc = bcftools_stream(vcf_path)
    for i, line in enumerate(proc.stdout, 1):
        chrom, pos, ref, alt_field, csq = line.rstrip("\n").split("\t", 4)
        pos = int(pos)
        if not has_valid_window(chrom, pos, chr_to_contig, fasta):
            continue
        for alt in alt_field.split(","):
            if len(ref) != 1 or len(alt) != 1 or alt in {".", "*"} or alt.startswith("<"):
                continue
            cat, vep = classify_variant(alt, csq)
            if cat is None:
                continue
            row = {"chr": chrom, "pos": pos, "ref": ref, "alt": alt,
                   "vep_annotation": vep, "new_category": cat}
            n_target = target_n[cat]
            if n_target == 0:
                continue
            if eligible[cat] <= target_per_cat:
                selected[cat].append(row)
            else:
                seen[cat] += 1
                k = seen[cat]
                if len(selected[cat]) < n_target:
                    selected[cat].append(row)
                else:
                    j = rng.randint(1, k)
                    if j <= n_target:
                        selected[cat][j - 1] = row
        if i % 1_000_000 == 0:
            print(f"  pass 2: {i:,} VCF records read")
    proc.communicate()

    rows = [r for cat in CATEGORY_MAP for r in selected[cat]]
    df = pd.DataFrame(rows).sort_values(
        ["new_category", "chr", "pos", "ref", "alt"]
    ).reset_index(drop=True)
    return df


def step2_extract_sequences(df, fasta, chr_to_contig, out_txt, out_meta_csv):
    """Extract REF/ALT windows for each selected variant."""
    kept = []
    status_counts = Counter()

    with open(out_txt, "w") as fout:
        for _, row in df.iterrows():
            chrom, pos, ref, alt = str(row["chr"]), int(row["pos"]), row["ref"], row["alt"]
            seq_ref, seq_alt, status = make_sequences(chrom, pos, ref, alt,
                                                       chr_to_contig, fasta)
            status_counts[status] += 1
            if status != "ok":
                continue
            fout.write(seq_ref + " " + seq_alt + "\n")
            kept.append({**row.to_dict(), "seq_ref": seq_ref, "seq_alt": seq_alt})

    meta_df = pd.DataFrame(kept)
    meta_df.to_csv(out_meta_csv, index=False)

    print("\nSequence extraction summary:")
    for k, v in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}")
    print(f"\nSequences written: {len(meta_df):,}")
    return meta_df


def step3_filter_N(seq_txt, meta_csv, out_seq, out_meta):
    """Remove sequence pairs that contain ambiguous bases (N)."""
    seq_ref, seq_alt = [], []
    with open(seq_txt) as f:
        for line in f:
            a, b = line.strip().split()
            seq_ref.append(a)
            seq_alt.append(b)

    df = pd.read_csv(meta_csv)
    assert len(seq_ref) == len(df), "Sequences and metadata are not aligned"

    keep = [i for i, (r, a) in enumerate(zip(seq_ref, seq_alt))
            if "N" not in r and "N" not in a]

    with open(out_seq, "w") as f:
        for i in keep:
            f.write(seq_ref[i] + " " + seq_alt[i] + "\n")

    df.iloc[keep].reset_index(drop=True).to_csv(out_meta, index=False)
    print(f"Kept {len(keep):,} / {len(seq_ref):,} sequences after N-filtering")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vcf",        required=True,
                   help="VCF file with VEP CSQ annotations (bgzipped + tabixed)")
    p.add_argument("--fasta",      required=True,
                   help="Reference genome FASTA (will be indexed with pysam if needed)")
    p.add_argument("--out_dir",    default="results/",
                   help="Output directory (default: results/)")
    p.add_argument("--n_per_cat",  type=int, default=5000,
                   help="Max variants to sample per category (default: 5000)")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    variants_csv  = os.path.join(args.out_dir, "variants_selected.csv")
    seq_txt       = os.path.join(args.out_dir, "sequences_5994.txt")
    meta_csv      = os.path.join(args.out_dir, "sequences_5994.metadata.csv")
    seq_noN_txt   = os.path.join(args.out_dir, "sequences_5994_noN.txt")
    meta_noN_csv  = os.path.join(args.out_dir, "sequences_5994_noN.metadata.csv")

    if not os.path.exists(args.fasta + ".fai"):
        print("Indexing FASTA …")
        pysam.faidx(args.fasta)

    fasta = pysam.FastaFile(args.fasta)
    chr_to_contig = build_chr_to_contig(args.fasta, fasta)
    print(f"Chromosome → contig mappings found: {len(chr_to_contig)}")

    print("\n=== Step 1: selecting variants from VCF ===")
    df = step1_select_variants(args.vcf, fasta, chr_to_contig,
                                args.n_per_cat, args.seed)
    df.to_csv(variants_csv, index=False)
    print(f"Selected variants saved to: {variants_csv}")

    print("\n=== Step 2: extracting sequence windows ===")
    step2_extract_sequences(df, fasta, chr_to_contig, seq_txt, meta_csv)
    print(f"Sequences: {seq_txt}\nMetadata:  {meta_csv}")

    print("\n=== Step 3: filtering ambiguous bases (N) ===")
    step3_filter_N(seq_txt, meta_csv, seq_noN_txt, meta_noN_csv)
    print(f"Filtered sequences: {seq_noN_txt}\nFiltered metadata:  {meta_noN_csv}")


if __name__ == "__main__":
    main()
