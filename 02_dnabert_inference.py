"""
DNABERT-6 inference — embedding-space distance profiles for SNV pairs.

Sequences are trimmed to the central 515 bp (257 bp + variant + 257 bp)
and converted to 6-mer tokens before inference.

Output files:
  DNABERT_cosine_profile.csv
  DNABERT_euclidean_profile.csv
  DNABERT_manhattan_profile.csv
  DNABERT_js_profile.csv
  DNABERT_hellinger_profile.csv
  DNABERT_cross_entropy_profile.csv
"""

import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertForMaskedLM


MODEL_NAME = "zhihan1996/DNA_bert_6"
KMER = 6
WINDOW_BP = 515    # central window extracted from the 5994-bp input
VAR_IDX = 2997     # 0-based position of the variant in the 5994-bp sequence


# ── distance metrics ──────────────────────────────────────────────────────────

def hellinger(p, q):
    return torch.sqrt(0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=1))

def jensen_shannon(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (
        torch.sum(p * (torch.log(p) - torch.log(m)), dim=1) +
        torch.sum(q * (torch.log(q) - torch.log(m)), dim=1)
    )

def cross_entropy(p, q):
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=1)


# ── sequence helpers ──────────────────────────────────────────────────────────

def load_sequences(path):
    s_ref, s_alt = [], []
    with open(path) as f:
        for line in f:
            a, b = line.strip().split()
            s_ref.append(a)
            s_alt.append(b)
    return s_ref, s_alt


def extract_central_window(seq, total_len=5994, window=WINDOW_BP, var_idx=VAR_IDX):
    if len(seq) != total_len:
        raise ValueError(f"Expected {total_len} bp, got {len(seq)}")
    half = window // 2
    start = var_idx - half
    end = start + window
    sub = seq[start:end]
    if len(sub) != window:
        raise ValueError(f"Window extraction failed: got {len(sub)} bp")
    return sub


def seq_to_kmers(seq, k=KMER):
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_file",   default="data/example_human_clinvar_100.txt")
    p.add_argument("--out_dir",    default="results/")
    p.add_argument("--cache_dir",  default="cache/")
    p.add_argument("--batch_size", type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=args.cache_dir, trust_remote_code=True)
    model = BertForMaskedLM.from_pretrained(
        MODEL_NAME, cache_dir=args.cache_dir, trust_remote_code=True,
        output_hidden_states=True).to(device)
    model.eval()

    seq_ref_full, seq_alt_full = load_sequences(args.seq_file)
    print(f"Sequences loaded: {len(seq_ref_full)}")

    seq_ref_kmers = [seq_to_kmers(extract_central_window(s)) for s in seq_ref_full]
    seq_alt_kmers = [seq_to_kmers(extract_central_window(s)) for s in seq_alt_full]
    print(f"Window: {WINDOW_BP} bp → {len(seq_ref_kmers[0].split())} {KMER}-mers")

    cosine_rows, euclidean_rows, manhattan_rows = [], [], []
    js_rows, hellinger_rows, ce_rows = [], [], []

    for i in tqdm(range(0, len(seq_ref_kmers), args.batch_size)):
        batch_ref = seq_ref_kmers[i:i + args.batch_size]
        batch_alt = seq_alt_kmers[i:i + args.batch_size]

        tok_ref = tokenizer(batch_ref, padding=True, return_tensors="pt").to(device)
        tok_alt = tokenizer(batch_alt, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            out_ref = model(**tok_ref)
            out_alt = model(**tok_alt)

        # strip CLS and SEP tokens
        emb_ref = out_ref.hidden_states[-1][:, 1:-1, :]
        emb_alt = out_alt.hidden_states[-1][:, 1:-1, :]

        cos  = F.cosine_similarity(emb_ref, emb_alt, dim=2)
        eucl = torch.norm(emb_ref - emb_alt, dim=2)
        manh = torch.norm(emb_ref - emb_alt, p=1, dim=2)

        logits_ref = out_ref.logits[:, 1:-1, :].float()
        logits_alt = out_alt.logits[:, 1:-1, :].float()
        probs_ref  = F.softmax(logits_ref, dim=1)
        probs_alt  = F.softmax(logits_alt, dim=1)
        input_ids  = tok_ref["input_ids"][:, 1:-1]

        for b in range(len(batch_ref)):
            ids   = input_ids[b]
            prob1 = probs_ref[b, :, ids].clamp(min=1e-12).cpu()
            prob2 = probs_alt[b, :, ids].clamp(min=1e-12).cpu()

            cosine_rows.append(cos[b].cpu().numpy())
            euclidean_rows.append(eucl[b].cpu().numpy())
            manhattan_rows.append(manh[b].cpu().numpy())
            js_rows.append(jensen_shannon(prob1, prob2).numpy())
            hellinger_rows.append(hellinger(prob1, prob2).numpy())
            ce_rows.append(cross_entropy(prob1, prob2).numpy())

        del out_ref, out_alt
        gc.collect()
        torch.cuda.empty_cache()

    PREFIX = "DNABERT"
    for name, rows in [
        ("cosine",        cosine_rows),
        ("euclidean",     euclidean_rows),
        ("manhattan",     manhattan_rows),
        ("js",            js_rows),
        ("hellinger",     hellinger_rows),
        ("cross_entropy", ce_rows),
    ]:
        pd.DataFrame(np.vstack(rows)).to_csv(
            os.path.join(args.out_dir, f"{PREFIX}_{name}_profile.csv"), index=False)

    print(f"Results saved to: {args.out_dir}")
    print(f"Output shape: {np.vstack(cosine_rows).shape}")


if __name__ == "__main__":
    main()
