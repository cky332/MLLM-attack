#!/usr/bin/env python3
"""eval_baseline_noise.py — Measure the natural randomness of preference generation.

Runs Llama-3 preference inference TWICE on the exact same clean summaries,
then compares the two outputs. This establishes the "baseline noise" level:
how much Jaccard difference comes purely from do_sample=True randomness,
NOT from any attack.

If baseline Jaccard > 0.8: randomness is small, attack results are reliable.
If baseline Jaccard 0.5-0.8: moderate noise, attack effects should be discounted.
If baseline Jaccard < 0.5: large noise, attack conclusions need re-evaluation.

Usage:
    python eval_baseline_noise.py \
        --clean_summary_csv /path/to/image_summary.csv \
        --original_pref_csv /path/to/user_preference_direct.csv \
        --title_csv /path/to/MicroLens-50k_titles.csv \
        --user_items_tsv /path/to/user_items_negs.tsv \
        --output_dir results/baseline_noise/ \
        --batch_size 12
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def load_prefs(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "user_id" in df.columns:
        df.rename(columns={"user_id": "user"}, inplace=True)
    if "summary" in df.columns and "preference" not in df.columns:
        df.rename(columns={"summary": "preference"}, inplace=True)
    df["user"] = df["user"].astype(str)
    return df


def compute_jaccard_stats(df_a, df_b, col_a="preference", col_b="preference"):
    merged = pd.merge(df_a[["user", col_a]], df_b[["user", col_b]],
                      on="user", suffixes=("_a", "_b"))
    jaccards = []
    for _, row in merged.iterrows():
        w1 = set(str(row[f"{col_a}_a"]).lower().split())
        w2 = set(str(row[f"{col_b}_b"]).lower().split())
        jac = len(w1 & w2) / max(len(w1 | w2), 1)
        jaccards.append(jac)
    jaccards = np.array(jaccards)
    return {
        "n_users": len(jaccards),
        "mean_jaccard": float(jaccards.mean()),
        "median_jaccard": float(np.median(jaccards)),
        "std_jaccard": float(jaccards.std()),
        "highly_changed_lt_0.5": int((jaccards < 0.5).sum()),
        "highly_changed_pct": float((jaccards < 0.5).mean() * 100),
        "moderately_changed_0.5_0.8": int(((jaccards >= 0.5) & (jaccards < 0.8)).sum()),
        "barely_changed_gte_0.8": int((jaccards >= 0.8).sum()),
        "barely_changed_pct": float((jaccards >= 0.8).mean() * 100),
    }


def main():
    p = argparse.ArgumentParser(description="Measure baseline noise of preference generation")
    p.add_argument("--clean_summary_csv", required=True, help="Clean image_summary.csv")
    p.add_argument("--original_pref_csv", required=True, help="Original user_preference_direct.csv")
    p.add_argument("--title_csv", required=True, help="Item titles CSV")
    p.add_argument("--user_items_tsv", required=True, help="user_items_negs.tsv")
    p.add_argument("--output_dir", default="results/baseline_noise/")
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--skip_inference", action="store_true",
                   help="Skip GPU inference, only analyze existing rerun CSV")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rerun_csv = os.path.join(args.output_dir, "user_preference_rerun.csv")
    report_path = os.path.join(args.output_dir, "baseline_noise_report.json")

    # Step 1: Re-run preference inference on clean summaries (same input, different random seed)
    if not args.skip_inference:
        if os.path.exists(rerun_csv):
            print(f"[baseline] Rerun CSV exists: {rerun_csv}, skipping inference (use --skip_inference=False and delete file to force)")
        else:
            print("[baseline] Step 1: Re-running preference inference on clean summaries...")
            from eval_recommendation_impact import run_preference_inference
            run_preference_inference(
                visual_csv=args.clean_summary_csv,
                title_csv=args.title_csv,
                user_items_tsv=args.user_items_tsv,
                output_csv=rerun_csv,
                num_proc=1,
                batch_size=args.batch_size,
            )
    else:
        print("[baseline] Step 1: Skipped (--skip_inference)")

    # Step 2: Compare original vs rerun
    if not os.path.exists(rerun_csv):
        print(f"[baseline] ERROR: {rerun_csv} not found. Run without --skip_inference first.")
        return

    print("[baseline] Step 2: Comparing original vs rerun preferences...")
    original_df = load_prefs(args.original_pref_csv)
    rerun_df = load_prefs(rerun_csv)

    stats = compute_jaccard_stats(original_df, rerun_df)

    report = {
        "experiment": "baseline_noise",
        "description": "Same clean summaries, re-run preference inference. "
                       "Measures natural randomness from do_sample=True, temperature=0.6.",
        "original_csv": args.original_pref_csv,
        "rerun_csv": rerun_csv,
        "comparison": stats,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print results
    print("\n" + "=" * 70)
    print("BASELINE NOISE ANALYSIS")
    print("(Same input, different random generation → how much changes?)")
    print("=" * 70)
    print(f"  Users compared:         {stats['n_users']}")
    print(f"  Mean Jaccard:           {stats['mean_jaccard']:.4f}")
    print(f"  Median Jaccard:         {stats['median_jaccard']:.4f}")
    print(f"  Std Jaccard:            {stats['std_jaccard']:.4f}")
    print(f"  Highly changed (<0.5):  {stats['highly_changed_lt_0.5']} ({stats['highly_changed_pct']:.1f}%)")
    print(f"  Moderately (0.5-0.8):   {stats['moderately_changed_0.5_0.8']}")
    print(f"  Barely changed (≥0.8):  {stats['barely_changed_gte_0.8']} ({stats['barely_changed_pct']:.1f}%)")

    print(f"\n--- 对比攻击实验 ---")
    print(f"  Baseline rerun Jaccard:       {stats['mean_jaccard']:.4f}  ← 纯随机波动")
    print(f"  inject_en N=1 attack Jaccard: 0.3470  ← 攻击1个item")
    print(f"  inject_en N=6 attack Jaccard: 0.3394  ← 攻击6个item")

    if stats["mean_jaccard"] > 0.8:
        verdict = "基线波动很小 → 攻击效果完全可信"
    elif stats["mean_jaccard"] > 0.5:
        verdict = "基线波动中等 → 攻击有效但实际效果需扣除基线噪声"
    else:
        verdict = "基线波动很大 → 攻击效果可能大部分来自随机性，需要重新评估"
    print(f"\n  结论: {verdict}")
    print(f"\n[baseline] Report saved to {report_path}")

    return report


if __name__ == "__main__":
    main()
