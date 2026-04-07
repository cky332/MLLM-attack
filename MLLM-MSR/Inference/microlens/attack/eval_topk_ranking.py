#!/usr/bin/env python3
"""eval_topk_ranking.py — Real top-K ranking evaluation under attack.

For each attack, this script:
  1. Builds two LLaVA scoring datasets (clean vs attacked) over the SAME
     (user, candidate-item) test pairs, differing only in the user-preference
     text injected into the prompt.
  2. Runs the LLaVA-Next scoring model on both, gets P(Yes) per (user, item).
  3. Reshapes to (n_users, n_candidates_per_user) and computes:
       - Recall@K, NDCG@K, MRR@K   (global)
       - mean_rank_delta            (positive items: rank_attacked - rank_clean)
       - topk_hit_rate_before/after (for positive items that are in the
         attacked-item set, fraction whose rank <= K)

Assumes the same scoring conventions as
MLLM-MSR/test/microlens/test_with_llava.py:
  - 21 candidates per user (1 positive + 20 negatives)
  - prompt template from multi_col_dataset.py
  - LLaVA-Next-Mistral-7B, Yes/No token logits

Usage:
    python eval_topk_ranking.py \
        --test_pairs_csv     /path/to/test_pairs.csv \
        --image_dir          /path/to/MicroLens-50k_covers \
        --title_csv          /path/to/MicroLens-50k_titles.csv \
        --clean_pref_csv     /path/to/user_preference.csv \
        --attacked_pref_csv  /path/to/user_preference_attacked_<atk>.csv \
        --attacked_summary_csv results/image_summary_<atk>.csv \
        --attack_name <atk> \
        --output_report results/topk_ranking_<atk>.json \
        --candidates_per_user 21 --batch_size 12 --num_proc 4
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROMPT_TEMPLATE = (
    "[INST]<image>\nBased on the previous interaction history, the user's "
    "preference can be summarized as: {}"
    "Please predict whether this user would interact with the video at the "
    "next opportunity. The video's title is'{}', and the given image is this "
    "video's cover? Please only response 'yes' or 'no' based on your "
    "judgement, do not include any other content including words, space, and "
    "punctuations in your response. [/INST]"
)


# ---------------------------------------------------------------------------
# Metrics (mirroring test_with_llava.py)
# ---------------------------------------------------------------------------
def recall_at_k(y_true, y_prob, k):
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_labels = np.take_along_axis(y_true, sorted_indices, axis=1)
    retrieved_positives = np.sum(sorted_labels[:, :k], axis=1)
    return float(np.mean(retrieved_positives))  # 1 positive per user => hit rate


def mrr_at_k(y_true, y_prob, k):
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_labels = np.take_along_axis(y_true, sorted_indices, axis=1)
    rr = np.zeros(y_true.shape[0])
    for i, labels in enumerate(sorted_labels[:, :k]):
        pos = np.where(labels == 1)[0]
        if pos.size > 0:
            rr[i] = 1.0 / (pos[0] + 1)
    return float(np.mean(rr))


def ndcg_at_k(y_true, y_prob, k):
    def dcg(scores, k):
        discounts = np.log2(np.arange(2, k + 2))
        return np.sum((2 ** scores - 1) / discounts, axis=1)
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_scores = np.take_along_axis(y_true, sorted_indices, axis=1)[:, :k]
    dcg_scores = dcg(sorted_scores, k)
    ideal = np.sort(y_true, axis=1)[:, ::-1][:, :k]
    idcg = dcg(ideal, k)
    return float(np.mean(dcg_scores / (idcg + 1e-10)))


def positive_ranks(y_true, y_prob):
    """For each row, return the 1-indexed rank of the positive item."""
    order = np.argsort(-y_prob, axis=1)
    ranks = np.zeros(y_true.shape[0], dtype=int)
    for i in range(y_true.shape[0]):
        pos_idx = np.where(y_true[i] == 1)[0]
        if pos_idx.size == 0:
            ranks[i] = -1
            continue
        pos_idx = pos_idx[0]
        ranks[i] = int(np.where(order[i] == pos_idx)[0][0]) + 1
    return ranks


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def _norm_id(x):
    return str(x)


def build_scored_df(test_pairs_csv, image_dir, title_csv, pref_csv):
    """Merge pairs + image paths + titles + user preferences -> df with prompt."""
    pairs = pd.read_csv(test_pairs_csv)
    pairs.columns = [c.strip().lower() for c in pairs.columns]
    pairs["item"] = pairs["item"].astype(str)
    pairs["user"] = pairs["user"].astype(str)

    # --- titles: detect whether file has a header row ---
    _peek = pd.read_csv(title_csv, nrows=1, header=None)
    has_header = any(isinstance(v, str) and not str(v).strip().isdigit()
                     for v in _peek.iloc[0].tolist()[:1])
    if has_header:
        titles = pd.read_csv(title_csv)
    else:
        titles = pd.read_csv(title_csv, header=None, names=["item", "title"])
    titles.columns = [c.strip().lower() for c in titles.columns]
    if "item_id" in titles.columns:
        titles.rename(columns={"item_id": "item"}, inplace=True)
    titles["item"] = titles["item"].astype(str)

    # --- preferences: handle (user_id,summary) | (user,preference) | header-less ---
    prefs = pd.read_csv(pref_csv)
    prefs.columns = [c.strip().lower() for c in prefs.columns]
    if "user_id" in prefs.columns:
        prefs.rename(columns={"user_id": "user"}, inplace=True)
    if "summary" in prefs.columns and "preference" not in prefs.columns:
        prefs.rename(columns={"summary": "preference"}, inplace=True)
    if "user" not in prefs.columns or "preference" not in prefs.columns:
        # fallback: assume header-less 2-col file
        prefs = pd.read_csv(pref_csv, header=None, names=["user", "preference"])
    prefs["user"] = prefs["user"].astype(str)

    img_dir = Path(image_dir)
    files = []
    for p in img_dir.glob("*"):
        if p.is_file():
            files.append({"item": p.stem, "image": str(p.absolute())})
    images = pd.DataFrame(files)
    images["item"] = images["item"].astype(str)

    df = pairs.merge(images, on="item").merge(titles, on="item").merge(prefs, on="user")
    df["prompt"] = df.apply(
        lambda r: PROMPT_TEMPLATE.format(r["preference"], r["title"]), axis=1
    )
    df = df[["user", "item", "label", "prompt", "image"]].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Scoring (LLaVA-Next, mirroring test_with_llava.py)
# ---------------------------------------------------------------------------
_MODEL = None
_PROCESSOR = None
_YES_ID = None
_NO_ID = None


def _lazy_load_model(model_id):
    global _MODEL, _PROCESSOR, _YES_ID, _NO_ID
    if _MODEL is not None:
        return
    import torch
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    _MODEL = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    ).eval()
    torch.set_default_dtype(default_dtype)
    _MODEL.tie_weights()
    _PROCESSOR = LlavaNextProcessor.from_pretrained(model_id)
    _PROCESSOR.tokenizer.pad_token = _PROCESSOR.tokenizer.eos_token
    _PROCESSOR.tokenizer.add_tokens(["<|image|>", "<pad>"], special_tokens=True)
    _YES_ID = _PROCESSOR.tokenizer.convert_tokens_to_ids("Yes")
    _NO_ID = _PROCESSOR.tokenizer.convert_tokens_to_ids("No")


def score_dataframe(df, model_id, batch_size=12, num_proc=1):
    """Run LLaVA scoring on df rows. Returns yes_prob[N], no_prob[N]."""
    import torch
    from datasets import Dataset, Image
    from PIL import ImageOps
    from torch.cuda.amp import autocast

    _lazy_load_model(model_id)
    ds = Dataset.from_pandas(df).cast_column("image", Image())

    def gpu_fn(batch, rank):
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        _MODEL.to(device)
        imgs = batch["image"]
        max_w = max(im.width for im in imgs)
        max_h = max(im.height for im in imgs)
        padded = []
        for im in imgs:
            if im.width == max_w and im.height == max_h:
                padded.append(im)
            else:
                dw, dh = max_w - im.width, max_h - im.height
                pad = (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2)
                padded.append(ImageOps.expand(im, border=pad, fill="black"))
        inputs = _PROCESSOR(text=batch["prompt"], images=padded,
                            return_tensors="pt", padding=True).to(device)
        with torch.no_grad(), autocast():
            outputs = _MODEL.generate(**inputs, max_new_tokens=1,
                                      return_dict_in_generate=True,
                                      output_scores=True)
        scores = outputs["scores"][0]
        return {
            "yes_logits": scores[:, _YES_ID].float().cpu().tolist(),
            "no_logits": scores[:, _NO_ID].float().cpu().tolist(),
        }

    out = ds.map(gpu_fn, batched=True, batch_size=batch_size,
                 with_rank=True, num_proc=num_proc)
    yl = np.array(out["yes_logits"], dtype=np.float64)
    nl = np.array(out["no_logits"], dtype=np.float64)
    # softmax over [no, yes]
    m = np.maximum(yl, nl)
    eyes = np.exp(yl - m)
    enos = np.exp(nl - m)
    yes_prob = eyes / (eyes + enos)
    return yes_prob


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------
def evaluate(args):
    print(f"[topk] Building CLEAN dataset...")
    df_clean = build_scored_df(args.test_pairs_csv, args.image_dir,
                               args.title_csv, args.clean_pref_csv)
    print(f"[topk] Building ATTACKED dataset...")
    df_atk = build_scored_df(args.test_pairs_csv, args.image_dir,
                             args.title_csv, args.attacked_pref_csv)

    # Align row order so reshape into (n_users, K) is consistent.
    df_clean = df_clean.sort_values(["user", "item"]).reset_index(drop=True)
    df_atk = df_atk.sort_values(["user", "item"]).reset_index(drop=True)
    assert (df_clean[["user", "item"]].values == df_atk[["user", "item"]].values).all(), \
        "Clean and attacked dataframes are not aligned"

    K_per = args.candidates_per_user
    n = len(df_clean)
    assert n % K_per == 0, f"{n} rows not divisible by {K_per} candidates/user"
    n_users = n // K_per

    print(f"[topk] Scoring CLEAN ({n} rows)...")
    yes_clean = score_dataframe(df_clean, args.model_id,
                                batch_size=args.batch_size, num_proc=args.num_proc)
    print(f"[topk] Scoring ATTACKED ({n} rows)...")
    yes_atk = score_dataframe(df_atk, args.model_id,
                              batch_size=args.batch_size, num_proc=args.num_proc)

    labels = df_clean["label"].values.astype(int).reshape(n_users, K_per)
    items_grid = df_clean["item"].values.reshape(n_users, K_per)
    yes_clean_g = yes_clean.reshape(n_users, K_per)
    yes_atk_g = yes_atk.reshape(n_users, K_per)

    # Identify attacked items
    atk_items = set()
    if args.attacked_summary_csv and os.path.exists(args.attacked_summary_csv):
        s = pd.read_csv(args.attacked_summary_csv)
        s.columns = [c.strip().lower() for c in s.columns]
        atk_items = set(s["item_id"].astype(str).values) if "item_id" in s.columns \
            else set(s["item"].astype(str).values)

    # Per-user positive rank
    rank_clean = positive_ranks(labels, yes_clean_g)
    rank_atk = positive_ranks(labels, yes_atk_g)
    valid = (rank_clean > 0) & (rank_atk > 0)
    delta = (rank_atk - rank_clean)[valid]

    # Mask: positive item belongs to attacked set
    pos_items_per_user = []
    for i in range(n_users):
        pos_idx = np.where(labels[i] == 1)[0]
        pos_items_per_user.append(items_grid[i, pos_idx[0]] if pos_idx.size else None)
    pos_items_per_user = np.array(pos_items_per_user)
    is_attacked_pos = np.array([str(it) in atk_items for it in pos_items_per_user])

    K = args.topk
    hit_clean_all = (rank_clean > 0) & (rank_clean <= K)
    hit_atk_all = (rank_atk > 0) & (rank_atk <= K)

    def _frac(mask):
        return float(np.mean(mask)) if mask.size else 0.0

    report = {
        "attack_name": args.attack_name,
        "n_users": int(n_users),
        "candidates_per_user": int(K_per),
        "n_attacked_items_in_test": int(is_attacked_pos.sum()),
        "global_metrics": {
            "clean": {
                "recall@10": recall_at_k(labels, yes_clean_g, 10),
                "ndcg@10": ndcg_at_k(labels, yes_clean_g, 10),
                "mrr@10": mrr_at_k(labels, yes_clean_g, 10),
                "recall@5": recall_at_k(labels, yes_clean_g, 5),
                "ndcg@5": ndcg_at_k(labels, yes_clean_g, 5),
            },
            "attacked": {
                "recall@10": recall_at_k(labels, yes_atk_g, 10),
                "ndcg@10": ndcg_at_k(labels, yes_atk_g, 10),
                "mrr@10": mrr_at_k(labels, yes_atk_g, 10),
                "recall@5": recall_at_k(labels, yes_atk_g, 5),
                "ndcg@5": ndcg_at_k(labels, yes_atk_g, 5),
            },
        },
        "rank_analysis_all_positives": {
            "mean_rank_clean": float(np.mean(rank_clean[valid])),
            "mean_rank_attacked": float(np.mean(rank_atk[valid])),
            "mean_rank_delta": float(np.mean(delta)),
            "median_rank_delta": float(np.median(delta)),
            "n_users_rank_improved": int((delta < 0).sum()),
            "n_users_rank_worsened": int((delta > 0).sum()),
            "n_users_rank_unchanged": int((delta == 0).sum()),
        },
        f"topk_hit_rate@{K}_all_positives": {
            "before": _frac(hit_clean_all),
            "after": _frac(hit_atk_all),
            "delta": _frac(hit_atk_all) - _frac(hit_clean_all),
        },
    }

    if is_attacked_pos.any():
        sel = is_attacked_pos & valid
        rc = rank_clean[sel]
        ra = rank_atk[sel]
        d = ra - rc
        hc = (rc > 0) & (rc <= K)
        ha = (ra > 0) & (ra <= K)
        report["rank_analysis_attacked_items_only"] = {
            "n": int(sel.sum()),
            "mean_rank_clean": float(np.mean(rc)),
            "mean_rank_attacked": float(np.mean(ra)),
            "mean_rank_delta": float(np.mean(d)),
            "n_rank_improved": int((d < 0).sum()),
            "n_rank_worsened": int((d > 0).sum()),
            f"topk_hit_rate@{K}_before": _frac(hc),
            f"topk_hit_rate@{K}_after": _frac(ha),
            f"topk_hit_rate@{K}_delta": _frac(ha) - _frac(hc),
        }

    if args.output_report:
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[topk] Report saved to {args.output_report}")

    print("\n" + "=" * 70)
    print(f"TOP-K RANKING REPORT  ({args.attack_name})")
    print("=" * 70)
    gm = report["global_metrics"]
    print(f"Recall@10:  clean={gm['clean']['recall@10']:.4f}  "
          f"attacked={gm['attacked']['recall@10']:.4f}  "
          f"delta={gm['attacked']['recall@10']-gm['clean']['recall@10']:+.4f}")
    print(f"NDCG@10:    clean={gm['clean']['ndcg@10']:.4f}  "
          f"attacked={gm['attacked']['ndcg@10']:.4f}  "
          f"delta={gm['attacked']['ndcg@10']-gm['clean']['ndcg@10']:+.4f}")
    ra = report["rank_analysis_all_positives"]
    print(f"Mean rank delta (all positives): {ra['mean_rank_delta']:+.3f}  "
          f"(improved={ra['n_users_rank_improved']}, "
          f"worsened={ra['n_users_rank_worsened']})")
    if "rank_analysis_attacked_items_only" in report:
        rai = report["rank_analysis_attacked_items_only"]
        print(f"\nFor positive items in attacked set (n={rai['n']}):")
        print(f"  mean rank: {rai['mean_rank_clean']:.2f} -> {rai['mean_rank_attacked']:.2f}  "
              f"(delta={rai['mean_rank_delta']:+.3f})")
        print(f"  top-{K} hit rate: {rai[f'topk_hit_rate@{K}_before']:.4f} -> "
              f"{rai[f'topk_hit_rate@{K}_after']:.4f}  "
              f"(delta={rai[f'topk_hit_rate@{K}_delta']:+.4f})")
    print()
    return report


def main():
    p = argparse.ArgumentParser(description="Real top-K ranking eval under attack")
    p.add_argument("--test_pairs_csv", required=True,
                   help="CSV with columns user,item,label (1 pos + N negs per user)")
    p.add_argument("--image_dir", required=True, help="Item cover images dir")
    p.add_argument("--title_csv", required=True, help="Item titles CSV (item,title)")
    p.add_argument("--clean_pref_csv", required=True,
                   help="Clean user preferences CSV (user,preference)")
    p.add_argument("--attacked_pref_csv", required=True,
                   help="Attacked user preferences CSV (from eval_recommendation_impact run_preference)")
    p.add_argument("--attacked_summary_csv", default=None,
                   help="Attacked image_summary CSV (to identify which items are attacked)")
    p.add_argument("--attack_name", default="attack")
    p.add_argument("--output_report", default="results/topk_ranking_report.json")
    p.add_argument("--candidates_per_user", type=int, default=21)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--num_proc", type=int, default=1)
    p.add_argument("--model_id", default="llava-hf/llava-v1.6-mistral-7b-hf")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    try:
        from multiprocess import set_start_method
        set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
