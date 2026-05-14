"""
Nucleotide Transformer inference — embedding-space distance profiles for SNV pairs.

Runs all NT v2 model sizes (100M to 2.5B) sequentially and saves one CSV per
metric per model to OUT_DIR.

Output files (per model, e.g. NT_500M):
  NT_500M_cosine_profile.csv
  NT_500M_euclidean_profile.csv
  NT_500M_manhattan_profile.csv
  NT_500M_js_profile.csv
  NT_500M_hellinger_profile.csv
  NT_500M_cross_entropy_profile.csv
  run_status.csv

Each row in a profile CSV corresponds to one sequence pair;
each column corresponds to one token position.
"""

import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


NT_MODELS = {
    "NT_100M":    "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
    "NT_250M":    "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    "NT_500M":    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    "NT_HR_500M": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "NT_1G_500M": "InstaDeepAI/nucleotide-transformer-500m-1000g",
    "NT_1G_2_5B": "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    "NT_2_5B":    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
}

BATCH_SIZE_MAP = {
    "NT_100M": 20, "NT_250M": 20, "NT_500M": 20, "NT_HR_500M": 20,
    "NT_1G_500M": 16, "NT_1G_2_5B": 4, "NT_2_5B": 4,
}


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


def save_csv(mat, path):
    pd.DataFrame(mat).to_csv(path, index=False)


# ── inference ─────────────────────────────────────────────────────────────────

def run_model(short_name, model_name, seq_ref, seq_alt, out_dir, cache_dir, device):
    batch_size = BATCH_SIZE_MAP[short_name]
    print(f"\n=== {short_name} | {model_name} ===")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    model_kwargs = dict(cache_dir=cache_dir, trust_remote_code=True,
                        output_hidden_states=True)
    if "2_5B" in short_name:
        model_kwargs["device_map"] = "auto"
        model = AutoModelForMaskedLM.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, **model_kwargs).to(device)
    model.eval()

    cosine_rows, euclidean_rows, manhattan_rows = [], [], []
    js_rows, hellinger_rows, ce_rows = [], [], []

    for i in tqdm(range(0, len(seq_ref), batch_size), desc=short_name):
        batch_ref = seq_ref[i:i + batch_size]
        batch_alt = seq_alt[i:i + batch_size]

        tok_ref = tokenizer(batch_ref, padding=True, return_tensors="pt")
        tok_alt = tokenizer(batch_alt, padding=True, return_tensors="pt")
        tok_ref = {k: v.to(device) for k, v in tok_ref.items()}
        tok_alt = {k: v.to(device) for k, v in tok_alt.items()}

        # strip CLS token
        tok_ref["input_ids"]      = tok_ref["input_ids"][:, 1:]
        tok_ref["attention_mask"] = tok_ref["attention_mask"][:, 1:]
        tok_alt["input_ids"]      = tok_alt["input_ids"][:, 1:]
        tok_alt["attention_mask"] = tok_alt["attention_mask"][:, 1:]

        with torch.no_grad():
            out_ref = model(**tok_ref)
            out_alt = model(**tok_alt)

        emb_ref = out_ref.hidden_states[-1]
        emb_alt = out_alt.hidden_states[-1]

        cos  = F.cosine_similarity(emb_ref, emb_alt, dim=2)
        eucl = torch.norm(emb_ref - emb_alt, dim=2)
        manh = torch.norm(emb_ref - emb_alt, p=1, dim=2)

        probs_ref = F.softmax(out_ref.logits.float(), dim=1)
        probs_alt = F.softmax(out_alt.logits.float(), dim=1)

        for b in range(len(batch_ref)):
            ids   = tok_ref["input_ids"][b]
            prob1 = probs_ref[b, :, ids].clamp(min=1e-12).cpu()
            prob2 = probs_alt[b, :, ids].clamp(min=1e-12).cpu()

            cosine_rows.append(cos[b].detach().cpu().numpy())
            euclidean_rows.append(eucl[b].detach().cpu().numpy())
            manhattan_rows.append(manh[b].detach().cpu().numpy())
            js_rows.append(jensen_shannon(prob1, prob2).numpy())
            hellinger_rows.append(hellinger(prob1, prob2).numpy())
            ce_rows.append(cross_entropy(prob1, prob2).numpy())

        del tok_ref, tok_alt, out_ref, out_alt, emb_ref, emb_alt
        del probs_ref, probs_alt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cosine_mat    = np.vstack(cosine_rows)
    euclidean_mat = np.vstack(euclidean_rows)
    manhattan_mat = np.vstack(manhattan_rows)
    js_mat        = np.vstack(js_rows)
    hellinger_mat = np.vstack(hellinger_rows)
    ce_mat        = np.vstack(ce_rows)

    print(f"Output shape: {cosine_mat.shape}")

    for name, mat in [
        ("cosine",        cosine_mat),
        ("euclidean",     euclidean_mat),
        ("manhattan",     manhattan_mat),
        ("js",            js_mat),
        ("hellinger",     hellinger_mat),
        ("cross_entropy", ce_mat),
    ]:
        save_csv(mat, os.path.join(out_dir, f"{short_name}_{name}_profile.csv"))

    del model, tokenizer, cosine_mat, euclidean_mat, manhattan_mat
    del js_mat, hellinger_mat, ce_mat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"short_name": short_name, "model_name": model_name, "status": "ok",
            "n_rows": cosine_mat.shape[0] if False else len(cosine_rows),
            "n_cols": len(cosine_rows[0]) if cosine_rows else 0}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_file",  default="data/example_human_clinvar_100.txt",
                   help="Input sequence file (default: data/example_human_clinvar_100.txt)")
    p.add_argument("--out_dir",   default="results/",
                   help="Output directory (default: results/)")
    p.add_argument("--cache_dir", default="cache/",
                   help="HuggingFace model cache directory (default: cache/)")
    p.add_argument("--models",    nargs="+", default=list(NT_MODELS.keys()),
                   choices=list(NT_MODELS.keys()),
                   help="Which NT models to run (default: all)")
    args = p.parse_args()

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    seq_ref, seq_alt = load_sequences(args.seq_file)
    print(f"Sequences loaded: {len(seq_ref)}")

    run_status = []
    for short_name in args.models:
        model_name = NT_MODELS[short_name]
        try:
            status = run_model(short_name, model_name, seq_ref, seq_alt,
                               args.out_dir, args.cache_dir, device)
            run_status.append(status)
        except Exception as e:
            print(f"FAILED {short_name}: {e}")
            run_status.append({"short_name": short_name, "model_name": model_name,
                                "status": f"failed: {e}", "n_rows": None, "n_cols": None})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    status_df = pd.DataFrame(run_status)
    status_df.to_csv(os.path.join(args.out_dir, "nt_run_status.csv"), index=False)
    print("\nRun summary:")
    print(status_df.to_string(index=False))


if __name__ == "__main__":
    main()
