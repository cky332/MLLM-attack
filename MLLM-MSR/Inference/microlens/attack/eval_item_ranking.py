#!/usr/bin/env python3
"""eval_item_ranking.py — Test how attacked images affect item scores
with CLEAN (unpolluted) user preferences.

Unlike eval_topk_ranking.py (which changes user preferences), this script
keeps user preferences fixed and instead swaps the candidate item's IMAGE
from clean to attacked.

This answers: "If a seller adds negative/positive text to their product
image, does LLaVA score that item higher or lower for users?"

For each user in the test set:
  - Score all 21 candidates with clean images → baseline ranking
  - Replace attacked items' images with attacked versions → new ranking
  - Compare: did the attacked items' scores go up or down?

Usage:
    python eval_item_ranking.py \
        --test_pairs_csv /path/to/test_pairs.csv \
        --clean_image_dir /path/to/MicroLens-50k_covers \
        --attacked_image_dir /path/to/attacked_images/negative_en \
        --title_csv /path/to/MicroLens-50k_titles.csv \
        --pref_csv /path/to/user_preference_direct.csv \
        --attack_name negative_en \
        --output_report results/item_ranking_negative_en.json \
        --batch_size 4
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


def load_prefs(pref_csv):
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


def find_image(image_dir, item_id):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = os.path.join(image_dir, f"{item_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def score_batch(prompts, image_paths, model_id, batch_size=4):
    """Score (prompt, image) pairs with LLaVA. Returns P(Yes) array."""
    import torch
    from datasets import Dataset, Image
    from PIL import ImageOps
    from torch.cuda.amp import autocast
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    df = pd.DataFrame({"prompt": prompts, "image": image_paths})
    ds = Dataset.from_pandas(df).cast_column("image", Image())

    def gpu_fn(batch, rank):
        import builtins
        cache = getattr(builtins, "_item_ranking_cache", None)
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
            builtins._item_ranking_cache = cache
        mdl, proc = cache["model"], cache["proc"]
        yes_id, no_id = cache["yes_id"], cache["no_id"]

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


def evaluate(args):
    print("[item_ranking] Loading data...")
    prefs = load_prefs(args.pref_csv)
    titles = load_titles(args.title_csv)

    pairs = pd.read_csv(args.test_pairs_csv)
    pairs.columns = [c.strip().lower() for c in pairs.columns]
    pairs["item"] = pairs["item"].astype(str)
    pairs["user"] = pairs["user"].astype(str)

    # Find which items have attacked images
    attacked_dir = Path(args.attacked_image_dir)
    attacked_items = set()
    for p in attacked_dir.glob("*"):
        if p.is_file():
            attacked_items.add(p.stem)

    print(f"[item_ranking] Attacked images found: {len(attacked_items)}")
    print(f"[item_ranking] Test pairs: {len(pairs)}")

    # Build two sets of (prompt, image_path):
    # - clean: all items use clean images
    # - attacked: items with attacked images use attacked versions, rest use clean
    prompts = []
    images_clean = []
    images_attacked = []
    is_attacked_item = []
    valid_mask = []

    for _, row in pairs.iterrows():
        user, item = row["user"], row["item"]
        if user not in prefs:
            valid_mask.append(False)
            prompts.append("")
            images_clean.append("")
            images_attacked.append("")
            is_attacked_item.append(False)
            continue

        title = titles.get(item, "Unknown")
        prompt = PROMPT_TEMPLATE.format(prefs[user], title)

        clean_img = find_image(args.clean_image_dir, item)
        if not clean_img:
            valid_mask.append(False)
            prompts.append("")
            images_clean.append("")
            images_attacked.append("")
            is_attacked_item.append(False)
            continue

        atk_img = find_image(args.attacked_image_dir, item) if item in attacked_items else None

        valid_mask.append(True)
        prompts.append(prompt)
        images_clean.append(clean_img)
        images_attacked.append(atk_img if atk_img else clean_img)
        is_attacked_item.append(atk_img is not None)

    valid_mask = np.array(valid_mask)
    is_attacked_item = np.array(is_attacked_item)
    labels = pairs["label"].values

    # Filter to valid rows
    v_prompts = [p for p, v in zip(prompts, valid_mask) if v]
    v_clean = [p for p, v in zip(images_clean, valid_mask) if v]
    v_attacked = [p for p, v in zip(images_attacked, valid_mask) if v]
    v_labels = labels[valid_mask]
    v_is_atk = is_attacked_item[valid_mask]
    n = len(v_prompts)

    print(f"[item_ranking] Valid scoring entries: {n}")
    print(f"[item_ranking] Entries with attacked images: {v_is_atk.sum()}")

    # Score twice: same prompts (same clean preference), different images
    print(f"[item_ranking] Scoring with CLEAN images ({n} entries)...")
    scores_clean = score_batch(v_prompts, v_clean, args.model_id, args.batch_size)

    print(f"[item_ranking] Scoring with ATTACKED images ({n} entries)...")
    scores_attacked = score_batch(v_prompts, v_attacked, args.model_id, args.batch_size)

    # Analyze score changes for attacked items specifically
    delta = scores_attacked - scores_clean

    # Per-entry analysis for attacked items only
    atk_delta = delta[v_is_atk]
    atk_scores_clean = scores_clean[v_is_atk]
    atk_scores_attacked = scores_attacked[v_is_atk]

    # Reshape for Recall@K (21 candidates per user)
    K_per = args.candidates_per_user
    assert n % K_per == 0, f"{n} not divisible by {K_per}"
    n_users = n // K_per

    labels_g = v_labels.reshape(n_users, K_per)
    clean_g = scores_clean.reshape(n_users, K_per)
    atk_g = scores_attacked.reshape(n_users, K_per)

    def recall_at_k(y_true, y_prob, k):
        order = np.argsort(-y_prob, axis=1)
        sorted_labels = np.take_along_axis(y_true, order, axis=1)
        return float(np.mean(np.sum(sorted_labels[:, :k], axis=1)))

    def positive_ranks(y_true, y_prob):
        order = np.argsort(-y_prob, axis=1)
        ranks = np.zeros(y_true.shape[0], dtype=int)
        for i in range(y_true.shape[0]):
            pos = np.where(y_true[i] == 1)[0]
            if pos.size == 0:
                ranks[i] = -1
                continue
            ranks[i] = int(np.where(order[i] == pos[0])[0][0]) + 1
        return ranks

    rank_clean = positive_ranks(labels_g, clean_g)
    rank_atk = positive_ranks(labels_g, atk_g)
    valid_r = (rank_clean > 0) & (rank_atk > 0)
    rank_delta = (rank_atk - rank_clean)[valid_r]

    report = {
        "attack_name": args.attack_name,
        "experiment": "item_ranking_with_clean_preferences",
        "description": "User preferences are FIXED (clean). Only item images change.",
        "n_users": int(n_users),
        "n_entries": int(n),
        "n_attacked_entries": int(v_is_atk.sum()),
        "attacked_items_score_analysis": {
            "n": int(v_is_atk.sum()),
            "mean_score_clean": float(atk_scores_clean.mean()) if atk_delta.size else None,
            "mean_score_attacked": float(atk_scores_attacked.mean()) if atk_delta.size else None,
            "mean_score_delta": float(atk_delta.mean()) if atk_delta.size else None,
            "pct_score_increased": float((atk_delta > 0).mean()) if atk_delta.size else None,
            "pct_score_decreased": float((atk_delta < 0).mean()) if atk_delta.size else None,
        },
        "global_ranking": {
            "recall@10_clean": recall_at_k(labels_g, clean_g, 10),
            "recall@10_attacked": recall_at_k(labels_g, atk_g, 10),
            "recall@10_delta": recall_at_k(labels_g, atk_g, 10) - recall_at_k(labels_g, clean_g, 10),
            "mean_rank_clean": float(np.mean(rank_clean[valid_r])),
            "mean_rank_attacked": float(np.mean(rank_atk[valid_r])),
            "mean_rank_delta": float(np.mean(rank_delta)),
        },
    }

    # Print
    print("\n" + "=" * 70)
    print(f"ITEM RANKING REPORT ({args.attack_name})")
    print(f"(User preferences FIXED, item images changed)")
    print("=" * 70)

    if atk_delta.size:
        ai = report["attacked_items_score_analysis"]
        print(f"\nAttacked items P(Yes) change (n={ai['n']}):")
        print(f"  Clean image:    {ai['mean_score_clean']:.4f}")
        print(f"  Attacked image: {ai['mean_score_attacked']:.4f}")
        print(f"  Delta:          {ai['mean_score_delta']:+.4f}")
        print(f"  Score increased: {ai['pct_score_increased']:.1%}")
        print(f"  Score decreased: {ai['pct_score_decreased']:.1%}")

    gr = report["global_ranking"]
    print(f"\nGlobal ranking (all {n_users} users):")
    print(f"  Recall@10: {gr['recall@10_clean']:.4f} → {gr['recall@10_attacked']:.4f} "
          f"(delta={gr['recall@10_delta']:+.4f})")
    print(f"  Mean rank: {gr['mean_rank_clean']:.2f} → {gr['mean_rank_attacked']:.2f} "
          f"(delta={gr['mean_rank_delta']:+.3f})")

    if args.output_report:
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[item_ranking] Report saved to {args.output_report}")

    return report


def main():
    p = argparse.ArgumentParser(
        description="Evaluate attacked item images with CLEAN user preferences")
    p.add_argument("--test_pairs_csv", required=True)
    p.add_argument("--clean_image_dir", required=True, help="Original clean images dir")
    p.add_argument("--attacked_image_dir", required=True, help="Attacked images dir")
    p.add_argument("--title_csv", required=True)
    p.add_argument("--pref_csv", required=True, help="Clean user preferences (FIXED)")
    p.add_argument("--attack_name", default="attack")
    p.add_argument("--output_report", default="results/item_ranking_report.json")
    p.add_argument("--candidates_per_user", type=int, default=21)
    p.add_argument("--batch_size", type=int, default=4)
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
