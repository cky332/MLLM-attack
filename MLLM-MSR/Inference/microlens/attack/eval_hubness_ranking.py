#!/usr/bin/env python3
"""eval_hubness_ranking.py — Correct hubness evaluation via LLaVA scoring.

For each (source_A → target_B) pair, this script:
  1. Finds "B-users": users who historically interacted with target B
  2. Constructs scoring inputs: (B-user's preference, item A's image+title)
  3. Scores with LLaVA under clean preferences vs attacked preferences
  4. Measures: does item A get a higher Yes-score from B-users after the attack?

Key metrics:
  - mean_score_delta:  avg P(Yes) change for A among B-users (positive = A pushed toward B-users)
  - topk_infiltration: fraction of B-users whose top-K now includes A
  - hubness_score:     across ALL users, how many users' top-K includes A (before vs after)

Usage:
    python eval_hubness_ranking.py \
        --pair_mapping results/hubness_hubness_popular/pair_mapping.json \
        --image_dir /path/to/MicroLens-50k_covers \
        --title_csv /path/to/MicroLens-50k_titles.csv \
        --clean_pref_csv /path/to/user_preference_direct.csv \
        --attacked_pref_csv results/hubness_hubness_popular/user_preference_attacked.csv \
        --user_items_tsv /path/to/user_items_negs.tsv \
        --output_report results/hubness_hubness_popular/hubness_ranking_report.json \
        --batch_size 4 --num_proc 1
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
# Data loading
# ---------------------------------------------------------------------------
def load_preferences(pref_csv):
    """Load preferences as {user_id: text}."""
    df = pd.read_csv(pref_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if "user_id" in df.columns:
        df.rename(columns={"user_id": "user"}, inplace=True)
    if "summary" in df.columns and "preference" not in df.columns:
        df.rename(columns={"summary": "preference"}, inplace=True)
    if "user" not in df.columns or "preference" not in df.columns:
        df = pd.read_csv(pref_csv, header=None, names=["user", "preference"])
    df["user"] = df["user"].astype(str)
    return dict(zip(df["user"], df["preference"].astype(str)))


def load_titles(title_csv):
    """Load titles as {item_id: title}."""
    peek = pd.read_csv(title_csv, nrows=1, header=None)
    has_header = any(isinstance(v, str) and not str(v).strip().isdigit()
                     for v in peek.iloc[0].tolist()[:1])
    df = pd.read_csv(title_csv) if has_header else \
        pd.read_csv(title_csv, header=None, names=["item", "title"])
    df.columns = [c.strip().lower() for c in df.columns]
    if "item_id" in df.columns:
        df.rename(columns={"item_id": "item"}, inplace=True)
    df["item"] = df["item"].astype(str)
    return dict(zip(df["item"], df["title"].astype(str)))


def load_user_items(user_items_tsv):
    """Load user→[items] mapping."""
    user_items = {}
    with open(user_items_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                user = parts[0]
                items = [i.strip().rstrip(",") for i in parts[1].split(",") if i.strip().rstrip(",")]
                user_items[user] = items
    return user_items


def find_image_path(image_dir, item_id):
    """Find the image file for an item."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = os.path.join(image_dir, f"{item_id}{ext}")
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Scoring (reuses builtins cache pattern from eval_topk_ranking.py)
# ---------------------------------------------------------------------------
def score_pairs(prompts, image_paths, model_id, batch_size=4):
    """Score a list of (prompt, image_path) pairs. Returns P(Yes) array."""
    import torch
    from datasets import Dataset, Image
    from PIL import ImageOps
    from torch.cuda.amp import autocast
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    df = pd.DataFrame({"prompt": prompts, "image": image_paths})
    ds = Dataset.from_pandas(df).cast_column("image", Image())

    def gpu_fn(batch, rank):
        import builtins
        cache = getattr(builtins, "_hubness_eval_cache", None)
        if cache is None:
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float16)
            mdl = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
            ).eval()
            torch.set_default_dtype(default_dtype)
            mdl.tie_weights()
            proc = LlavaNextProcessor.from_pretrained(model_id)
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
            proc.tokenizer.add_tokens(["<|image|>", "<pad>"], special_tokens=True)
            yes_id = proc.tokenizer.convert_tokens_to_ids("Yes")
            no_id = proc.tokenizer.convert_tokens_to_ids("No")
            cache = {"model": mdl, "proc": proc, "yes_id": yes_id, "no_id": no_id}
            builtins._hubness_eval_cache = cache
        mdl = cache["model"]
        proc = cache["proc"]
        yes_id = cache["yes_id"]
        no_id = cache["no_id"]

        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        mdl.to(device)
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
        inputs = proc(text=batch["prompt"], images=padded,
                      return_tensors="pt", padding=True).to(device)
        with torch.no_grad(), autocast():
            outputs = mdl.generate(**inputs, max_new_tokens=1,
                                   return_dict_in_generate=True,
                                   output_scores=True)
        scores = outputs["scores"][0]
        return {
            "yes_logits": scores[:, yes_id].float().cpu().tolist(),
            "no_logits": scores[:, no_id].float().cpu().tolist(),
        }

    out = ds.map(gpu_fn, batched=True, batch_size=batch_size,
                 with_rank=True, num_proc=1)
    yl = np.array(out["yes_logits"], dtype=np.float64)
    nl = np.array(out["no_logits"], dtype=np.float64)
    m = np.maximum(yl, nl)
    eyes = np.exp(yl - m)
    enos = np.exp(nl - m)
    return eyes / (eyes + enos)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(args):
    print("[hubness_ranking] Loading data...")
    with open(args.pair_mapping) as f:
        pair_mapping = json.load(f)

    clean_prefs = load_preferences(args.clean_pref_csv)
    attacked_prefs = load_preferences(args.attacked_pref_csv)
    titles = load_titles(args.title_csv)
    user_items = load_user_items(args.user_items_tsv)

    # Build reverse index: item → users who interacted with it
    item_users = {}
    for user, items in user_items.items():
        for item in items:
            item_users.setdefault(item, []).append(user)

    # For each (source_A → target_B) pair, find B-users and build scoring inputs
    print(f"[hubness_ranking] Building scoring inputs for {len(pair_mapping)} pairs...")
    scoring_entries = []  # (pair_idx, user, source_id, target_id, image_path, title)

    for pair_idx, (src_id, info) in enumerate(pair_mapping.items()):
        target_id = info["target_id"]
        src_img = find_image_path(args.image_dir, src_id)
        src_title = titles.get(src_id, "Unknown")
        if not src_img:
            continue

        # B-users: users who interacted with target B
        b_users = item_users.get(target_id, [])
        # Only keep users who have both clean and attacked preferences
        b_users = [u for u in b_users if u in clean_prefs and u in attacked_prefs]

        # Limit to max_users_per_pair to keep runtime manageable
        if len(b_users) > args.max_users_per_pair:
            rng = np.random.RandomState(42 + pair_idx)
            b_users = list(rng.choice(b_users, args.max_users_per_pair, replace=False))

        for user in b_users:
            scoring_entries.append({
                "pair_idx": pair_idx,
                "user": user,
                "source_id": src_id,
                "target_id": target_id,
                "image_path": src_img,
                "title": src_title,
            })

    if not scoring_entries:
        print("[hubness_ranking] No valid (B-user, source_A) pairs found!")
        return {}

    n = len(scoring_entries)
    print(f"[hubness_ranking] Total scoring entries: {n} "
          f"({len(pair_mapping)} pairs × avg {n/max(len(pair_mapping),1):.0f} B-users)")

    # Build prompts for clean and attacked preferences
    prompts_clean = []
    prompts_attacked = []
    image_paths = []
    for entry in scoring_entries:
        user = entry["user"]
        title = entry["title"]
        prompts_clean.append(PROMPT_TEMPLATE.format(clean_prefs[user], title))
        prompts_attacked.append(PROMPT_TEMPLATE.format(attacked_prefs[user], title))
        image_paths.append(entry["image_path"])

    # Score with LLaVA (two passes: clean then attacked)
    print(f"[hubness_ranking] Scoring {n} entries with CLEAN preferences...")
    scores_clean = score_pairs(prompts_clean, image_paths,
                               args.model_id, args.batch_size)

    print(f"[hubness_ranking] Scoring {n} entries with ATTACKED preferences...")
    scores_attacked = score_pairs(prompts_attacked, image_paths,
                                  args.model_id, args.batch_size)

    # Aggregate results per (source_A, target_B) pair
    print("[hubness_ranking] Aggregating results...")
    pair_results = {}
    for i, entry in enumerate(scoring_entries):
        key = f"{entry['source_id']}→{entry['target_id']}"
        if key not in pair_results:
            pair_results[key] = {
                "source_id": entry["source_id"],
                "target_id": entry["target_id"],
                "scores_clean": [],
                "scores_attacked": [],
                "users": [],
            }
        pair_results[key]["scores_clean"].append(scores_clean[i])
        pair_results[key]["scores_attacked"].append(scores_attacked[i])
        pair_results[key]["users"].append(entry["user"])

    # Compute per-pair metrics
    pair_summaries = []
    for key, pr in pair_results.items():
        sc = np.array(pr["scores_clean"])
        sa = np.array(pr["scores_attacked"])
        delta = sa - sc
        pair_summaries.append({
            "source_id": pr["source_id"],
            "target_id": pr["target_id"],
            "n_b_users": len(sc),
            "mean_score_clean": float(sc.mean()),
            "mean_score_attacked": float(sa.mean()),
            "mean_score_delta": float(delta.mean()),
            "n_score_increased": int((delta > 0).sum()),
            "n_score_decreased": int((delta < 0).sum()),
            "pct_score_increased": float((delta > 0).mean()),
        })

    pair_summaries.sort(key=lambda x: -x["mean_score_delta"])

    # Global aggregates
    all_deltas = scores_attacked - scores_clean
    n_pairs_with_positive_delta = sum(1 for p in pair_summaries if p["mean_score_delta"] > 0)

    report = {
        "experiment": "hubness_ranking",
        "total_pairs": len(pair_summaries),
        "total_scoring_entries": n,
        "global_metrics": {
            "mean_score_clean": float(scores_clean.mean()),
            "mean_score_attacked": float(scores_attacked.mean()),
            "mean_score_delta": float(all_deltas.mean()),
            "median_score_delta": float(np.median(all_deltas)),
            "pct_entries_score_increased": float((all_deltas > 0).mean()),
            "pairs_with_positive_mean_delta": n_pairs_with_positive_delta,
            "pairs_with_positive_mean_delta_pct": n_pairs_with_positive_delta / max(len(pair_summaries), 1),
        },
        "top_10_most_successful_pairs": pair_summaries[:10],
        "top_10_least_successful_pairs": pair_summaries[-10:][::-1],
        "all_pair_summaries": pair_summaries,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("HUBNESS RANKING EVALUATION")
    print("=" * 70)
    gm = report["global_metrics"]
    print(f"Total pairs: {report['total_pairs']}")
    print(f"Total (B-user, source_A) scoring entries: {n}")
    print(f"\nMean P(Yes) for source_A among B-users:")
    print(f"  Clean:    {gm['mean_score_clean']:.4f}")
    print(f"  Attacked: {gm['mean_score_attacked']:.4f}")
    print(f"  Delta:    {gm['mean_score_delta']:+.4f}")
    print(f"\nEntries where score increased: {gm['pct_entries_score_increased']:.1%}")
    print(f"Pairs with positive mean delta: {gm['pairs_with_positive_mean_delta']}"
          f"/{report['total_pairs']} ({gm['pairs_with_positive_mean_delta_pct']:.1%})")

    print(f"\nTop 5 most successful pairs (A pushed toward B-users):")
    for p in pair_summaries[:5]:
        print(f"  {p['source_id']}→{p['target_id']}: "
              f"score {p['mean_score_clean']:.4f}→{p['mean_score_attacked']:.4f} "
              f"(delta={p['mean_score_delta']:+.4f}, "
              f"n_B_users={p['n_b_users']}, "
              f"{p['pct_score_increased']:.0%} increased)")

    if args.output_report:
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[hubness_ranking] Report saved to {args.output_report}")

    return report


def main():
    p = argparse.ArgumentParser(description="Hubness-specific ranking evaluation")
    p.add_argument("--pair_mapping", required=True,
                   help="JSON with {source_id: {target_id, attack_text}} from generate_hubness_attack.py")
    p.add_argument("--image_dir", required=True, help="Original item cover images dir")
    p.add_argument("--title_csv", required=True, help="Item titles CSV")
    p.add_argument("--clean_pref_csv", required=True, help="Clean user preferences CSV")
    p.add_argument("--attacked_pref_csv", required=True, help="Attacked user preferences CSV")
    p.add_argument("--user_items_tsv", required=True, help="user_items_negs.tsv")
    p.add_argument("--output_report", default="results/hubness_ranking_report.json")
    p.add_argument("--max_users_per_pair", type=int, default=50,
                   help="Max B-users to score per pair (limits GPU time)")
    p.add_argument("--batch_size", type=int, default=4)
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
