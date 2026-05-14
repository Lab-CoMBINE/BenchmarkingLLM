"""
Evo 2 (7B) inference — embedding-space distance profiles for SNV pairs.

Embeddings are extracted from layer blocks.28.mlp.l3.
Distances are computed both on embeddings and on the full token-probability
distributions, and saved as tab-separated CSVs (one row per sequence pair,
one column per token position).

Output files:
  EVO2_7B_cosine_profile.csv
  EVO2_7B_euclidean_profile.csv
  EVO2_7B_manhattan_profile.csv
  EVO2_7B_hellinger_profile.csv
  EVO2_7B_js_profile.csv
  EVO2_7B_cross_entropy_profile.csv

Requirements: pip install evo2
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from evo2 import Evo2


LAYER_NAME = "blocks.28.mlp.l3"
PREFIX     = "EVO2_7B"


# ── distance metrics ──────────────────────────────────────────────────────────

def hellinger(p, q):
    return torch.sqrt(0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=1))

def kl(p, q, eps=1e-12):
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * torch.log(p / q), dim=1)

def jensen_shannon(p, q):
    m = 0.5 * (p + q)
    return torch.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m))

def cross_entropy(p, q, eps=1e-12):
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return -torch.sum(p * torch.log(q), dim=1)


# ── inference helper ──────────────────────────────────────────────────────────

def get_logits_and_embeddings(seq, model, tokenizer, device):
    ids = tokenizer.tokenize(seq)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs, embeddings = model(
            input_ids, return_embeddings=True, layer_names=[LAYER_NAME])
    logits = outputs[0].squeeze(0).float()              # [L, vocab]
    emb    = embeddings[LAYER_NAME].squeeze(0).float()  # [L, hidden]
    return logits, emb, ids


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_file", default="data/example_human_clinvar_100.txt")
    p.add_argument("--out_dir",  default="results/")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    print("Loading Evo 2 7B …")
    evo2_model = Evo2("evo2_7b_base")
    evo2_model.model.to(device)
    evo2_model.model.eval()
    tokenizer = evo2_model.tokenizer

    # open output files
    out_files = {
        key: open(os.path.join(args.out_dir, f"{PREFIX}_{name}_profile.csv"),
                  "w", newline="")
        for key, name in [
            ("cos", "cosine"), ("euc", "euclidean"), ("man", "manhattan"),
            ("hel", "hellinger"), ("js",  "js"), ("ce",  "cross_entropy"),
        ]
    }
    writers = {k: csv.writer(v, delimiter="\t") for k, v in out_files.items()}

    # read first pair to get sequence length and write header
    with open(args.seq_file) as f:
        first_ref, _ = f.readline().strip().split()
    L = len(first_ref)
    header = list(range(L))
    for w in writers.values():
        w.writerow(header)

    # count lines for progress bar
    with open(args.seq_file) as f:
        num_lines = sum(1 for _ in f)

    with open(args.seq_file) as f:
        for line in tqdm(f, total=num_lines, desc="Evo2 inference"):
            seq_ref, seq_alt = line.strip().split()

            log_ref, emb_ref, tok_ref = get_logits_and_embeddings(
                seq_ref, evo2_model, tokenizer, device)
            log_alt, emb_alt, tok_alt = get_logits_and_embeddings(
                seq_alt, evo2_model, tokenizer, device)

            p_ref = torch.softmax(log_ref, dim=1)
            p_alt = torch.softmax(log_alt, dim=1)

            tok_ref_t = torch.tensor(tok_ref, device=device, dtype=torch.long)
            tok_alt_t = torch.tensor(tok_alt, device=device, dtype=torch.long)

            P_ref = p_ref[:, tok_ref_t]
            P_alt = p_alt[:, tok_alt_t]

            writers["hel"].writerow(hellinger(P_ref, P_alt).cpu().tolist())
            writers["js"].writerow(jensen_shannon(P_ref, P_alt).cpu().tolist())
            writers["ce"].writerow(cross_entropy(P_ref, P_alt).cpu().tolist())

            writers["cos"].writerow(
                F.cosine_similarity(emb_ref, emb_alt, dim=1).cpu().tolist())
            writers["euc"].writerow(
                torch.norm(emb_ref - emb_alt, dim=1).cpu().tolist())
            writers["man"].writerow(
                torch.sum(torch.abs(emb_ref - emb_alt), dim=1).cpu().tolist())

            torch.cuda.empty_cache()

    for f in out_files.values():
        f.close()

    print(f"Done. Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
