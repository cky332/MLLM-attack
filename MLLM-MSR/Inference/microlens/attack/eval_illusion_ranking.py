#!/usr/bin/env python3
"""eval_illusion_ranking.py — Recommendation-level ASR for the adversarial-
illusion attack (illusion_attack.py) against MLLM-MSR.

The illusion attack perturbs a candidate item's COVER IMAGE so its CLIP
embedding aligns with popular text. This script measures whether that actually
changes MLLM-MSR's interaction decision.

Method (preferences are held FIXED — only the candidate image changes):
  For every (user, candidate-item) test pair, build the LLaVA scoring prompt
  with the user's CLEAN preference + the item title, then score it twice:
    - clean image   -> P(Yes)_clean
    - adversarial image (if one exists for that item) -> P(Yes)_adv
  and report:
    - decision-flip ASR : among attacked pairs the model scored No (P(Yes)<0.5)
                          on the clean image, the fraction flipped to Yes;
    - mean P(Yes) lift  on attacked pairs;
    - rank-promotion ASR: for users whose POSITIVE item was attacked, whether it
                          is pushed into top-K (illusion_metrics.rank_promotion_asr);
    - global Recall@K / NDCG@K, clean vs attacked.

Scoring (LLaVA-Next Yes/No logits) is reused verbatim from eval_item_ranking.py
so numbers are directly comparable to the rest of the repo.

Usage:
    python eval_illusion_ranking.py \
        --test_pairs_csv    /path/to/test_pairs.csv \
        --clean_image_dir   results/illusion/clean_resized \
        --attacked_image_dir results/illusion/images \
        --title_csv         ../../data/microlens/MicroLens-50k_titles.csv \
        --pref_csv          /path/to/user_preference_recurrent.csv \
        --attack_name       illusion_popular_eps16 \
        --output_report     results/illusion/recsys_asr.json \
        --candidates_per_user 21 --topk 10 --batch_size 12
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the exact scoring + IO helpers from the existing item-ranking eval.
from eval_item_ranking import (
    PROMPT_TEMPLATE,
    find_image,
    load_prefs,
    load_titles,
    score_batch,
)
from illusion_metrics import (
    decision_flip_asr,
    ndcg_at_k,
    rank_promotion_asr,
    recall_at_k,
)


def evaluate(args):
    print("[illusion-eval] loading data ...")
    prefs = load_prefs(args.pref_csv)
    titles = load_titles(args.title_csv)

    pairs = pd.read_csv(args.test_pairs_csv)
    pairs.columns = [c.strip().lower() for c in pairs.columns]
    # test_pairs.csv has stray spaces in some rows (e.g. "40139, 1837,0").
    pairs["item"] = pairs["item"].astype(str).str.strip()
    pairs["user"] = pairs["user"].astype(str).str.strip()
    # Stable grid order: group candidates by user.
    pairs = pairs.sort_values(["user", "item"]).reset_index(drop=True)

    attacked_dir = Path(args.attacked_image_dir)
    attacked_items = {p.stem for p in attacked_dir.glob("*") if p.is_file()}
    print(f"[illusion-eval] adversarial images available for {len(attacked_items)} items")

    prompts, clean_imgs, adv_imgs, is_atk, valid = [], [], [], [], []
    for _, row in pairs.iterrows():
        user, item = row["user"], row["item"]
        clean_img = find_image(args.clean_image_dir, item)
        if user not in prefs or clean_img is None:
            prompts.append(""); clean_imgs.append(""); adv_imgs.append("")
            is_atk.append(False); valid.append(False)
            continue
        prompts.append(PROMPT_TEMPLATE.format(prefs[user], titles.get(item, "Unknown")))
        clean_imgs.append(clean_img)
        atk_img = find_image(args.attacked_image_dir, item) if item in attacked_items else None
        adv_imgs.append(atk_img if atk_img else clean_img)
        is_atk.append(atk_img is not None)
        valid.append(True)

    valid = np.array(valid)
    is_atk = np.array(is_atk)
    labels = pairs["label"].values.astype(int)
    all_valid = bool(valid.all())

    if not all_valid:
        n_bad = int((~valid).sum())
        print(f"[illusion-eval] WARNING: {n_bad} pairs missing image/preference; "
              f"excluded from scoring (and ranking is skipped — grid incomplete).")

    # Score only valid rows (empty paths must never reach the image loader).
    v_idx = np.where(valid)[0]
    v_prompts = [prompts[i] for i in v_idx]
    v_clean = [clean_imgs[i] for i in v_idx]
    v_adv = [adv_imgs[i] for i in v_idx]

    print(f"[illusion-eval] scoring CLEAN images ({len(v_prompts)}) ...")
    pyes_clean = score_batch(v_prompts, v_clean, args.model_id, args.batch_size)
    print(f"[illusion-eval] scoring ADVERSARIAL images ({len(v_prompts)}) ...")
    pyes_adv = score_batch(v_prompts, v_adv, args.model_id, args.batch_size)

    # ---- Per-pair decision-flip ASR (over attacked, valid pairs) ----
    v_is_atk = is_atk[v_idx]
    flip = decision_flip_asr(pyes_clean[v_is_atk], pyes_adv[v_is_atk],
                             threshold=args.decision_threshold, direction="promote")

    report = {
        "attack_name": args.attack_name,
        "experiment": "adversarial_illusion_on_candidate_image",
        "description": "User preferences FIXED; candidate cover image perturbed "
                       "to align with popular text (Zhang et al. 2025).",
        "n_pairs_total": int(len(prompts)),
        "n_pairs_attacked": int(v_is_atk.sum()),
        "decision_threshold": args.decision_threshold,
        "decision_flip_asr": flip,
    }

    # ---- Ranking impact (needs a complete K-per-user grid) ----
    # Safe only when no rows were dropped: then pyes_* line up with the full,
    # (user,item)-sorted `pairs`, so reshape into (n_users, K) is consistent.
    K_per = args.candidates_per_user
    if all_valid and len(prompts) % K_per == 0:
        n_users = len(prompts) // K_per
        labels_g = labels.reshape(n_users, K_per)
        clean_g = pyes_clean.reshape(n_users, K_per)
        adv_g = pyes_adv.reshape(n_users, K_per)
        is_atk_g = is_atk.reshape(n_users, K_per)

        # Was each user's positive item attacked?
        pos_attacked = np.array([
            bool(is_atk_g[i][np.where(labels_g[i] == 1)[0][0]])
            if (labels_g[i] == 1).any() else False
            for i in range(n_users)
        ])

        report["global_ranking"] = {
            "recall@%d_clean" % args.topk: recall_at_k(labels_g, clean_g, args.topk),
            "recall@%d_attacked" % args.topk: recall_at_k(labels_g, adv_g, args.topk),
            "ndcg@%d_clean" % args.topk: ndcg_at_k(labels_g, clean_g, args.topk),
            "ndcg@%d_attacked" % args.topk: ndcg_at_k(labels_g, adv_g, args.topk),
        }
        gr = report["global_ranking"]
        gr["recall@%d_delta" % args.topk] = (
            gr["recall@%d_attacked" % args.topk] - gr["recall@%d_clean" % args.topk]
        )
        gr["ndcg@%d_delta" % args.topk] = (
            gr["ndcg@%d_attacked" % args.topk] - gr["ndcg@%d_clean" % args.topk]
        )

        if pos_attacked.any():
            promo = rank_promotion_asr(
                labels_g[pos_attacked], clean_g[pos_attacked], adv_g[pos_attacked],
                k=args.topk,
            )
            report["positive_item_promotion"] = promo
    else:
        report["global_ranking"] = (
            "skipped (incomplete %d-per-user grid)" % K_per
        )

    # ---- Print ----
    print("\n" + "=" * 70)
    print(f"ADVERSARIAL ILLUSION — RECOMMENDATION ASR  ({args.attack_name})")
    print("=" * 70)
    f = report["decision_flip_asr"]
    print(f"Attacked (user,item) pairs:          {report['n_pairs_attacked']}")
    print(f"  clean No -> adv Yes (flippable={f['n_flippable']}):")
    print(f"  DECISION-FLIP ASR:                 {f['asr']:.1%}")
    print(f"  mean P(Yes): {f['mean_pyes_clean']:.4f} -> {f['mean_pyes_attacked']:.4f} "
          f"(lift {f['mean_pyes_lift']:+.4f}, increased on {f['pct_pyes_increased']:.1%})")
    if isinstance(report["global_ranking"], dict):
        gr = report["global_ranking"]
        rk = args.topk
        print(f"Recall@{rk}: {gr['recall@%d_clean'%rk]:.4f} -> {gr['recall@%d_attacked'%rk]:.4f} "
              f"(delta {gr['recall@%d_delta'%rk]:+.4f})")
        print(f"NDCG@{rk}:   {gr['ndcg@%d_clean'%rk]:.4f} -> {gr['ndcg@%d_attacked'%rk]:.4f} "
              f"(delta {gr['ndcg@%d_delta'%rk]:+.4f})")
        if "positive_item_promotion" in report:
            p = report["positive_item_promotion"]
            print(f"Positive-item promotion (n={p['n_users']}): "
                  f"mean rank {p['mean_rank_clean']:.2f} -> {p['mean_rank_attacked']:.2f} "
                  f"(delta {p['mean_rank_delta']:+.3f})")
            print(f"  top-{rk} hit: {p['topk_hit_rate_clean']:.1%} -> "
                  f"{p['topk_hit_rate_attacked']:.1%}   PROMOTION ASR: {p['promotion_asr']:.1%}")

    if args.output_report:
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        print(f"\n[illusion-eval] report -> {args.output_report}")
    return report


def main():
    p = argparse.ArgumentParser(description="Recommendation-level ASR for the illusion attack")
    p.add_argument("--test_pairs_csv", required=True, help="user,item,label (K per user)")
    p.add_argument("--clean_image_dir", required=True,
                   help="clean baseline images (ideally the resized-clean dir from "
                        "illusion_attack.py generate, for a fair comparison)")
    p.add_argument("--attacked_image_dir", required=True, help="adversarial images dir")
    p.add_argument("--title_csv", required=True)
    p.add_argument("--pref_csv", required=True, help="clean user preferences (FIXED)")
    p.add_argument("--attack_name", default="illusion")
    p.add_argument("--output_report", default="results/illusion/recsys_asr.json")
    p.add_argument("--candidates_per_user", type=int, default=21)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--decision_threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=12)
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
