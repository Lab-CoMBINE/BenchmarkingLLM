"""
HyenaDNA small-32k inference — embedding-space distance profiles for SNV pairs.

The full 5994-bp input sequences are used directly (well within the 32k context).

Output files:
  HyenaDNA_32k_cosine_profile.csv
  HyenaDNA_32k_euclidean_profile.csv
  HyenaDNA_32k_manhattan_profile.csv
  HyenaDNA_32k_js_profile.csv
  HyenaDNA_32k_hellinger_profile.csv
  HyenaDNA_32k_cross_entropy_profile.csv
"""

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"
PREFIX     = "HyenaDNA_32k"


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


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_sequences(path):
    s_ref, s_alt = [], []
    with open(path) as f:
        for line in f:
            a, b = line.strip().split()
            s_ref.append(a)
            s_alt.append(b)
    return s_ref, s_alt


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_file",   default="data/example_human_clinvar_100.txt")
    p.add_argument("--out_dir",    default="results/")
    p.add_argument("--cache_dir",  default="cache/")
    p.add_argument("--batch_size", type=int, default=60)
    args = p.parse_args()

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, cache_dir=args.cache_dir,
        use_flash_attention=False, output_hidden_states=True).to(device)
    model.eval()

    seq_ref, seq_alt = load_sequences(args.seq_file)
    print(f"Sequences loaded: {len(seq_ref)}")

    cosine_rows, euclidean_rows, manhattan_rows = [], [], []
    js_rows, hellinger_rows, ce_rows = [], [], []

    total_batches = (len(seq_ref) + args.batch_size - 1) // args.batch_size
    batch_times   = []
    start_time    = time.time()

    for batch_idx, i in enumerate(tqdm(range(0, len(seq_ref), args.batch_size)), start=1):
        t0 = time.time()
        batch_ref = seq_ref[i:i + args.batch_size]
        batch_alt = seq_alt[i:i + args.batch_size]

        tok_ref = tokenizer(batch_ref, padding=True, return_tensors="pt").to(device)
        tok_alt = tokenizer(batch_alt, padding=True, return_tensors="pt").to(device)

        tok_ref["input_ids"] = tok_ref["input_ids"][:, 1:]
        tok_alt["input_ids"] = tok_alt["input_ids"][:, 1:]
        for key in ("attention_mask",):
            if key in tok_ref:
                tok_ref[key] = tok_ref[key][:, 1:]
                tok_alt[key] = tok_alt[key][:, 1:]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            out_ref = model(**tok_ref)
            out_alt = model(**tok_alt)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        emb_ref = out_ref.hidden_states[-1]
        emb_alt = out_alt.hidden_states[-1]

        cos  = F.cosine_similarity(emb_ref, emb_alt, dim=2)
        eucl = torch.norm(emb_ref - emb_alt, p=2, dim=2)
        manh = torch.norm(emb_ref - emb_alt, p=1, dim=2)

        logits_ref = out_ref.logits.float()[:, :, 4:]
        logits_alt = out_alt.logits.float()[:, :, 4:]
        probs_ref  = F.softmax(logits_ref, dim=1)
        probs_alt  = F.softmax(logits_alt, dim=1)

        for b in range(len(batch_ref)):
            ids   = tok_ref["input_ids"][b].flatten()
            prob1 = probs_ref[b, :, ids].clamp(min=1e-12).cpu()
            prob2 = probs_alt[b, :, ids].clamp(min=1e-12).cpu()

            cosine_rows.append(cos[b].detach().cpu().numpy())
            euclidean_rows.append(eucl[b].detach().cpu().numpy())
            manhattan_rows.append(manh[b].detach().cpu().numpy())
            js_rows.append(jensen_shannon(prob1, prob2).numpy())
            hellinger_rows.append(hellinger(prob1, prob2).numpy())
            ce_rows.append(cross_entropy(prob1, prob2).numpy())

        del tok_ref, tok_alt, out_ref, out_alt, emb_ref, emb_alt
        del logits_ref, logits_alt, probs_ref, probs_alt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_times.append(time.time() - t0)
        if batch_idx <= 5 or batch_idx % 50 == 0:
            eta = np.mean(batch_times) * (total_batches - batch_idx) / 60
            print(f"  [{batch_idx}/{total_batches}] ETA ~{eta:.1f} min")

    elapsed = (time.time() - start_time) / 60
    print(f"\nCompleted in {elapsed:.2f} min  |  mean {np.mean(batch_times):.3f} s/batch")

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
