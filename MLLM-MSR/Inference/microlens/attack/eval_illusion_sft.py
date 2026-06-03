#!/usr/bin/env python3
"""eval_illusion_sft.py — Re-evaluate ONLY the final MLLM-MSR recommendation
ranking with clean vs adversarially-perturbed candidate images, using your
ALREADY FINE-TUNED LoRA recommender.

This is for the case where the full MLLM-MSR pipeline has already been run:
the LoRA-SFT recommender is trained, user preferences are generated, and the
test pairs exist. Nothing here retrains the model or regenerates preferences —
it only swaps each candidate's cover IMAGE (clean -> adversarial illusion) and
re-scores the final Yes/No judgment, exactly mirroring test_with_llava_sft.py:

    base = llava-v1.6-mistral-7b-hf
    model = PeftModel.from_pretrained(base, <your LoRA>)        # your recommender
    P(Yes) = softmax([logit("No"), logit("Yes")])[1]

It reports the same metric suite as test_with_llava_sft.py (AUC, Recall@K,
MRR@K, NDCG@K for K in {3,5,10}) for BOTH the clean and the attacked images,
side by side, plus attack-success metrics (decision-flip ASR, P(Yes) lift,
positive-item rank promotion).

Inputs are the SAME files your test pipeline already uses
(MLLM-MSR/test/microlens/multi_col_dataset.py builds the test set from these):
  - test_pairs.csv               (user,item,label; 21 candidates/user)
  - MicroLens-50k_titles.csv     (item,title)
  - user_preference_recurrent.csv(user,preference)   <- your generated prefs
  - clean cover images dir
  - adversarial images dir        (from illusion_attack.py generate)

Usage:
    python eval_illusion_sft.py \
        --peft_model_id /home/.../llava-v1.6-mistral-7b-hf-lora-recurrent-e4-r16 \
        --test_pairs_csv /path/to/Split/test_pairs.csv \
        --clean_image_dir    results/illusion/clean_resized \
        --attacked_image_dir results/illusion/images \
        --title_csv ../../data/microlens/MicroLens-50k_titles.csv \
        --pref_csv  /path/to/user_preference_recurrent.csv \
        --output_report results/illusion/recsys_asr_sft.json \
        --candidates_per_user 21 --batch_size 4 --num_proc 4
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse IO helpers + the exact prompt template used by multi_col_dataset.py.
from eval_item_ranking import PROMPT_TEMPLATE, find_image, load_prefs, load_titles
from illusion_metrics import (
    decision_flip_asr,
    mrr_at_k,
    ndcg_at_k,
    rank_promotion_asr,
    recall_at_k,
    yesno_softmax,
)


# ---------------------------------------------------------------------------
# Scoring with the FINE-TUNED LoRA recommender (mirrors test_with_llava_sft.py)
# ---------------------------------------------------------------------------
def score_with_lora(df, base_model_id, peft_model_id, batch_size=4, num_proc=1):
    """Score df rows (columns: prompt, image[path]) with base+LoRA. -> P(Yes)[N]."""
    import torch
    from datasets import Dataset, Image
    from PIL import ImageOps
    from torch.cuda.amp import autocast

    ds = Dataset.from_pandas(df[["prompt", "image"]]).cast_column("image", Image())

    def gpu_fn(batch, rank):
        import builtins
        cache = getattr(builtins, "_illusion_sft_cache", None)
        if cache is None:
            from transformers import (
                LlavaNextForConditionalGeneration,
                LlavaNextProcessor,
            )
            mdl = LlavaNextForConditionalGeneration.from_pretrained(
                base_model_id,
                cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
            )
            proc = LlavaNextProcessor.from_pretrained(base_model_id)
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
            if peft_model_id:
                from peft import PeftModel
                mdl = PeftModel.from_pretrained(mdl, peft_model_id)
            mdl = mdl.eval()
            mdl.tie_weights()
            proc.tokenizer.add_tokens(["<|image|>", "<pad>"], special_tokens=True)
            yes_id = proc.tokenizer.convert_tokens_to_ids("Yes")
            no_id = proc.tokenizer.convert_tokens_to_ids("No")
            cache = {"model": mdl, "proc": proc, "yes_id": yes_id, "no_id": no_id}
            builtins._illusion_sft_cache = cache
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
                                   return_dict_in_generate=True, output_scores=True)
        scores = outputs["scores"][0]
        result = {
            "yes_logits": scores[:, yes_id].float().cpu().tolist(),
            "no_logits": scores[:, no_id].float().cpu().tolist(),
        }
        # Free per-batch activations/KV-cache so 24GB cards don't creep into OOM.
        del inputs, outputs, scores
        torch.cuda.empty_cache()
        return result

    out = ds.map(gpu_fn, batched=True, batch_size=batch_size,
                 with_rank=True, num_proc=num_proc)
    return yesno_softmax(np.array(out["yes_logits"]), np.array(out["no_logits"]))


# ---------------------------------------------------------------------------
# Full metric suite (clean & attacked), identical to test_with_llava_sft.py
# ---------------------------------------------------------------------------
def metric_suite(labels_g, scores_g, flat_labels=None, flat_scores=None):
    out = {}
    if flat_labels is not None:
        try:
            from sklearn.metrics import roc_auc_score
            out["auc"] = float(roc_auc_score(flat_labels, flat_scores))
        except Exception as e:
            out["auc"] = None
            out["auc_error"] = str(e)
    for k in (3, 5, 10):
        out[f"recall@{k}"] = recall_at_k(labels_g, scores_g, k)
        out[f"mrr@{k}"] = mrr_at_k(labels_g, scores_g, k)
        out[f"ndcg@{k}"] = ndcg_at_k(labels_g, scores_g, k)
    return out


def evaluate(args):
    print("[illusion-sft] loading data ...")
    prefs = load_prefs(args.pref_csv)
    titles = load_titles(args.title_csv)

    pairs = pd.read_csv(args.test_pairs_csv)
    pairs.columns = [c.strip().lower() for c in pairs.columns]
    # test_pairs.csv has stray spaces in some rows (e.g. "40139, 1837,0").
    pairs["item"] = pairs["item"].astype(str).str.strip()
    pairs["user"] = pairs["user"].astype(str).str.strip()
    pairs = pairs.sort_values(["user", "item"]).reset_index(drop=True)

    if args.max_users > 0:
        keep = pd.unique(pairs["user"])[: args.max_users]
        pairs = pairs[pairs["user"].isin(keep)].reset_index(drop=True)
        print(f"[illusion-sft] PILOT: limited to first {len(keep)} users "
              f"({len(pairs)} rows)")

    attacked_items = {p.stem for p in Path(args.attacked_image_dir).glob("*") if p.is_file()}
    print(f"[illusion-sft] adversarial images for {len(attacked_items)} items")

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
        atk = find_image(args.attacked_image_dir, item) if item in attacked_items else None
        adv_imgs.append(atk if atk else clean_img)
        is_atk.append(atk is not None)
        valid.append(True)

    valid = np.array(valid)
    is_atk = np.array(is_atk)
    labels = pairs["label"].values.astype(int)
    all_valid = bool(valid.all())
    if not all_valid:
        print(f"[illusion-sft] WARNING: {int((~valid).sum())} pairs missing "
              f"image/preference; excluded (ranking metrics need a full grid).")

    v = np.where(valid)[0]
    df_clean = pd.DataFrame({"prompt": [prompts[i] for i in v],
                             "image": [clean_imgs[i] for i in v]})
    df_adv = pd.DataFrame({"prompt": [prompts[i] for i in v],
                           "image": [adv_imgs[i] for i in v]})

    print(f"[illusion-sft] scoring CLEAN images with LoRA ({len(df_clean)}) ...")
    pyes_clean = score_with_lora(df_clean, args.base_model_id, args.peft_model_id,
                                 args.batch_size, args.num_proc)
    print(f"[illusion-sft] scoring ADVERSARIAL images with LoRA ({len(df_adv)}) ...")
    pyes_adv = score_with_lora(df_adv, args.base_model_id, args.peft_model_id,
                               args.batch_size, args.num_proc)

    v_is_atk = is_atk[v]
    flip = decision_flip_asr(pyes_clean[v_is_atk], pyes_adv[v_is_atk],
                             threshold=args.decision_threshold, direction="promote")
    # Backfire / collateral check: of attacked pairs the clean model said YES to,
    # how many did our (promotion) attack accidentally flip to NO?  A clean attack
    # should keep this low.  Its "asr" field = Yes->No rate among clean-Yes pairs.
    backfire = decision_flip_asr(pyes_clean[v_is_atk], pyes_adv[v_is_atk],
                                 threshold=args.decision_threshold, direction="demote")

    report = {
        "attack_name": args.attack_name,
        "experiment": "illusion_final_reranking_with_finetuned_lora",
        "base_model_id": args.base_model_id,
        "peft_model_id": args.peft_model_id,
        "description": "Fine-tuned LoRA recommender re-scored on clean vs "
                       "adversarial candidate images; preferences/training reused.",
        "n_pairs_total": int(len(prompts)),
        "n_pairs_attacked": int(v_is_atk.sum()),
        "decision_threshold": args.decision_threshold,
        "decision_flip_asr": flip,
        "backfire_yes_to_no": backfire,
    }

    K_per = args.candidates_per_user
    if all_valid and len(prompts) % K_per == 0:
        n_users = len(prompts) // K_per
        labels_g = labels.reshape(n_users, K_per)
        clean_g = pyes_clean.reshape(n_users, K_per)
        adv_g = pyes_adv.reshape(n_users, K_per)
        is_atk_g = is_atk.reshape(n_users, K_per)

        report["metrics_clean"] = metric_suite(labels_g, clean_g, labels, pyes_clean)
        report["metrics_attacked"] = metric_suite(labels_g, adv_g, labels, pyes_adv)
        report["metrics_delta"] = {
            k: (report["metrics_attacked"][k] - report["metrics_clean"][k])
            for k in report["metrics_clean"]
            if isinstance(report["metrics_clean"].get(k), (int, float))
            and isinstance(report["metrics_attacked"].get(k), (int, float))
        }

        pos_attacked = np.array([
            bool(is_atk_g[i][np.where(labels_g[i] == 1)[0][0]])
            if (labels_g[i] == 1).any() else False
            for i in range(n_users)
        ])
        if pos_attacked.any():
            report["positive_item_promotion"] = rank_promotion_asr(
                labels_g[pos_attacked], clean_g[pos_attacked], adv_g[pos_attacked],
                k=args.topk,
            )
    else:
        report["metrics_clean"] = "skipped (incomplete %d-per-user grid)" % K_per

    # ---- Print ----
    print("\n" + "=" * 72)
    print(f"ILLUSION FINAL RE-RANKING (fine-tuned LoRA)  —  {args.attack_name}")
    print("=" * 72)
    f = report["decision_flip_asr"]
    print(f"Attacked (user,item) pairs: {report['n_pairs_attacked']}  "
          f"| clean No->adv Yes flippable={f['n_flippable']}")
    print(f"  DECISION-FLIP ASR: {f['asr']:.1%}   "
          f"P(Yes) {f['mean_pyes_clean']:.4f} -> {f['mean_pyes_attacked']:.4f} "
          f"(lift {f['mean_pyes_lift']:+.4f})")
    bf = report["backfire_yes_to_no"]
    print(f"  BACKFIRE (Yes->No): {bf['asr']:.1%}  "
          f"({bf['n_flipped']}/{bf['n_flippable']} clean-Yes pairs demoted)   "
          f"P(Yes) up on {f['pct_pyes_increased']:.1%} of attacked pairs")
    if isinstance(report.get("metrics_clean"), dict):
        mc, ma = report["metrics_clean"], report["metrics_attacked"]
        print(f"\n{'metric':<12}{'clean':>10}{'attacked':>12}{'delta':>10}")
        for k in ("auc", "recall@3", "recall@5", "recall@10",
                  "mrr@5", "mrr@10", "ndcg@5", "ndcg@10"):
            if isinstance(mc.get(k), (int, float)) and isinstance(ma.get(k), (int, float)):
                print(f"{k:<12}{mc[k]:>10.4f}{ma[k]:>12.4f}{ma[k]-mc[k]:>+10.4f}")
        if "positive_item_promotion" in report:
            p = report["positive_item_promotion"]
            print(f"\nPositive-item promotion (n={p['n_users']}): "
                  f"rank {p['mean_rank_clean']:.2f} -> {p['mean_rank_attacked']:.2f} "
                  f"({p['mean_rank_delta']:+.3f}); "
                  f"top-{args.topk} hit {p['topk_hit_rate_clean']:.1%} -> "
                  f"{p['topk_hit_rate_attacked']:.1%}; PROMOTION ASR {p['promotion_asr']:.1%}")

    if args.output_report:
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        print(f"\n[illusion-sft] report -> {args.output_report}")
    return report


def main():
    p = argparse.ArgumentParser(
        description="Final-stage illusion re-evaluation with a fine-tuned LoRA recommender")
    p.add_argument("--test_pairs_csv", required=True)
    p.add_argument("--clean_image_dir", required=True,
                   help="clean baseline (ideally the resized-clean dir from generate)")
    p.add_argument("--attacked_image_dir", required=True)
    p.add_argument("--title_csv", required=True)
    p.add_argument("--pref_csv", required=True, help="your generated user preferences (FIXED)")
    p.add_argument("--base_model_id", default="llava-hf/llava-v1.6-mistral-7b-hf")
    p.add_argument("--peft_model_id", default=None,
                   help="your fine-tuned LoRA adapter dir (e.g. "
                        "llava-v1.6-mistral-7b-hf-lora-recurrent-e4-r16). "
                        "Omit to score the base model.")
    p.add_argument("--attack_name", default="illusion")
    p.add_argument("--output_report", default="results/illusion/recsys_asr_sft.json")
    p.add_argument("--candidates_per_user", type=int, default=21)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--decision_threshold", type=float, default=0.5)
    p.add_argument("--max_users", type=int, default=0,
                   help="evaluate only the first N users (0 = all). Use for a "
                        "cheap pilot, mirroring test_with_llava_sft.py's select().")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_proc", type=int, default=1, help="set to #GPUs for multi-GPU")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    try:
        from multiprocess import set_start_method
        set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
