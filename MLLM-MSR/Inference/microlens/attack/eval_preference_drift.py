"""
Evaluate user preference drift when users interact with attacked items.

Experiment: For each user, replace N of their top-6 historical items' summaries
with attacked versions, then measure how much their preference text changes.

This tests the "dose-response" relationship: does more exposure to attacked items
cause greater preference drift?

Usage:
    # Run preference drift analysis with varying N (1..6)
    python eval_preference_drift.py \
        --clean_summary_csv ../image_summary.csv \
        --attacked_summary_csv results/attacked_inject_en_summary.csv \
        --title_csv ../../data/MicroLens-50k/MicroLens-50k_titles.csv \
        --user_items_tsv ../../data/MicroLens-50k/Split/user_items_negs.tsv \
        --attack_name inject_en \
        --output_dir results/preference_drift/ \
        --max_n 6

    # Analyze pre-computed drift results (no GPU needed)
    python eval_preference_drift.py --analyze_only \
        --clean_preference_csv ../user_preference_direct.csv \
        --drift_dir results/preference_drift/ \
        --output_report results/preference_drift_report.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def prepare_drift_summaries(clean_summary_csv, attacked_summary_csv,
                            user_items_tsv, n_replace, output_csv, seed=42):
    """Create a visual summary file where N of each user's historical items
    have their summaries replaced with attacked versions.

    Args:
        clean_summary_csv: Path to clean image_summary.csv
        attacked_summary_csv: Path to attacked summary CSV
        user_items_tsv: Path to user_items_negs.tsv
        n_replace: Number of items per user to replace (1..6)
        output_csv: Output path for the modified summary CSV
        seed: Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)

    clean_df = pd.read_csv(clean_summary_csv)
    attacked_df = pd.read_csv(attacked_summary_csv)

    clean_df.columns = [c.strip().lower() for c in clean_df.columns]
    attacked_df.columns = [c.strip().lower() for c in attacked_df.columns]
    clean_df["item_id"] = clean_df["item_id"].astype(str)
    attacked_df["item_id"] = attacked_df["item_id"].astype(str)

    attacked_ids = set(attacked_df["item_id"].values)
    attacked_lookup = attacked_df.set_index("item_id")["summary"].to_dict()

    # Read user histories
    user_items = {}
    with open(user_items_tsv, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                user = parts[0]
                items = parts[1].split(", ")[:-1] if parts[1].endswith(", ") else parts[1].split(", ")
                items = [i.strip() for i in items if i.strip()]
                user_items[user] = items

    # Determine which items to replace per user
    items_to_replace = set()
    replacement_stats = {"users_with_replaceable": 0, "total_replacements": 0}

    for user, items in user_items.items():
        replaceable = [i for i in items if i in attacked_ids]
        if not replaceable:
            continue
        replacement_stats["users_with_replaceable"] += 1
        n = min(n_replace, len(replaceable))
        selected = rng.choice(replaceable, size=n, replace=False)
        items_to_replace.update(selected)
        replacement_stats["total_replacements"] += n

    # Build the modified summary file
    modified_df = clean_df.copy()
    replace_mask = modified_df["item_id"].isin(items_to_replace)
    for idx in modified_df[replace_mask].index:
        item_id = modified_df.loc[idx, "item_id"]
        if item_id in attacked_lookup:
            modified_df.loc[idx, "summary"] = attacked_lookup[item_id]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    modified_df.to_csv(output_csv, index=False)

    print(f"[drift N={n_replace}] Replaced {len(items_to_replace)} unique items "
          f"across {replacement_stats['users_with_replaceable']} users -> {output_csv}")
    return replacement_stats


def run_preference_for_drift(visual_csv, title_csv, user_items_tsv, output_csv,
                             model_id="/home/chenkuiyun/.cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct",
                             num_proc=4, batch_size=12):
    """Run preference inference using the drift-modified summaries.

    Delegates to eval_recommendation_impact.run_preference_inference.
    """
    from eval_recommendation_impact import run_preference_inference
    return run_preference_inference(
        visual_csv, title_csv, user_items_tsv, output_csv,
        model_id=model_id, num_proc=num_proc, batch_size=batch_size,
    )


def analyze_drift(clean_preference_csv, drift_dir, output_report=None):
    """Analyze preference drift across different N values.

    Reads preference files named user_preference_drift_N{n}.csv from drift_dir.
    """
    clean_df = pd.read_csv(clean_preference_csv)
    clean_df.columns = [c.strip().lower() for c in clean_df.columns]
    if "user_id" not in clean_df.columns and "user" in clean_df.columns:
        clean_df.rename(columns={"user": "user_id"}, inplace=True)
    if "summary" not in clean_df.columns and "preference" in clean_df.columns:
        clean_df.rename(columns={"preference": "summary"}, inplace=True)
    clean_df["user_id"] = clean_df["user_id"].astype(str)

    results = {}
    for n in range(1, 7):
        drift_csv = os.path.join(drift_dir, f"user_preference_drift_N{n}.csv")
        if not os.path.exists(drift_csv):
            continue

        drift_df = pd.read_csv(drift_csv)
        drift_df.columns = [c.strip().lower() for c in drift_df.columns]
        if "user_id" not in drift_df.columns and "user" in drift_df.columns:
            drift_df.rename(columns={"user": "user_id"}, inplace=True)
        if "summary" not in drift_df.columns and "preference" in drift_df.columns:
            drift_df.rename(columns={"preference": "summary"}, inplace=True)
        drift_df["user_id"] = drift_df["user_id"].astype(str)

        merged = pd.merge(clean_df, drift_df, on="user_id", suffixes=("_clean", "_drift"))

        # Compute Jaccard similarity
        similarities = []
        for _, row in merged.iterrows():
            clean_words = set(str(row["summary_clean"]).lower().split())
            drift_words = set(str(row["summary_drift"]).lower().split())
            union = clean_words | drift_words
            jaccard = len(clean_words & drift_words) / max(len(union), 1)
            similarities.append(jaccard)

        similarities = np.array(similarities)

        # Keyword analysis
        from attack_config import POSITIVE_KEYWORDS
        keyword_changes = {}
        for kw in POSITIVE_KEYWORDS:
            kw_lower = kw.lower()
            clean_count = sum(1 for s in merged["summary_clean"].astype(str) if kw_lower in s.lower())
            drift_count = sum(1 for s in merged["summary_drift"].astype(str) if kw_lower in s.lower())
            if clean_count > 0 or drift_count > 0:
                keyword_changes[kw] = {"clean": clean_count, "drift": drift_count, "diff": drift_count - clean_count}

        results[f"N={n}"] = {
            "n_replace": n,
            "n_users": len(merged),
            "mean_jaccard": float(similarities.mean()),
            "std_jaccard": float(similarities.std()),
            "median_jaccard": float(np.median(similarities)),
            "highly_changed_lt_0.5": int((similarities < 0.5).sum()),
            "moderately_changed_0.5_0.8": int(((similarities >= 0.5) & (similarities < 0.8)).sum()),
            "barely_changed_gte_0.8": int((similarities >= 0.8).sum()),
            "keyword_changes": keyword_changes,
        }

    report = {
        "experiment": "preference_drift_dose_response",
        "description": "Measures how N attacked items in user history affect preference text",
        "results": results,
    }

    # Extract dose-response curve
    curve = []
    for key, val in results.items():
        curve.append({"n": val["n_replace"], "mean_jaccard": val["mean_jaccard"]})
    report["dose_response_curve"] = sorted(curve, key=lambda x: x["n"])

    if output_report:
        os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[analyze_drift] Report saved to {output_report}")

    # Print summary
    print("\n" + "=" * 60)
    print("PREFERENCE DRIFT ANALYSIS (Dose-Response)")
    print("=" * 60)
    print(f"{'N':>3} | {'Mean Jaccard':>12} | {'Highly Changed':>14} | {'Moderately':>10} | {'Barely':>6}")
    print("-" * 60)
    for key in sorted(results.keys()):
        r = results[key]
        print(f"{r['n_replace']:>3} | {r['mean_jaccard']:>12.4f} | "
              f"{r['highly_changed_lt_0.5']:>14} | "
              f"{r['moderately_changed_0.5_0.8']:>10} | "
              f"{r['barely_changed_gte_0.8']:>6}")

    if report["dose_response_curve"]:
        print("\nDose-Response Curve (N → Mean Jaccard):")
        for pt in report["dose_response_curve"]:
            bar = "#" * int((1 - pt["mean_jaccard"]) * 40)
            print(f"  N={pt['n']:1d}: {pt['mean_jaccard']:.4f} |{bar}")

    return report


def run_full_drift_experiment(args):
    """Run the complete drift experiment: prepare summaries for N=1..max_n,
    run preference inference, then analyze."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    max_n = args.max_n

    # Step 1: Prepare modified summary files for each N
    print("\n>>> Step 1: Preparing drift summary files...")
    for n in range(1, max_n + 1):
        summary_csv = os.path.join(output_dir, f"image_summary_drift_N{n}.csv")
        if os.path.exists(summary_csv) and not args.force:
            print(f"  N={n}: {summary_csv} exists, skipping (use --force to regenerate)")
            continue
        prepare_drift_summaries(
            args.clean_summary_csv, args.attacked_summary_csv,
            args.user_items_tsv, n, summary_csv,
        )

    # Step 2: Run preference inference for each N
    if not args.skip_inference:
        print("\n>>> Step 2: Running preference inference for each N...")
        for n in range(1, max_n + 1):
            summary_csv = os.path.join(output_dir, f"image_summary_drift_N{n}.csv")
            pref_csv = os.path.join(output_dir, f"user_preference_drift_N{n}.csv")
            if os.path.exists(pref_csv) and not args.force:
                print(f"  N={n}: {pref_csv} exists, skipping")
                continue
            if not os.path.exists(summary_csv):
                print(f"  N={n}: {summary_csv} not found, skipping")
                continue
            print(f"\n  N={n}: Running preference inference...")
            run_preference_for_drift(
                summary_csv, args.title_csv, args.user_items_tsv, pref_csv,
                num_proc=args.num_proc, batch_size=args.batch_size,
            )
    else:
        print("\n>>> Step 2: Skipped (--skip_inference)")

    # Step 3: Analyze drift
    print("\n>>> Step 3: Analyzing preference drift...")
    report_path = os.path.join(output_dir, f"preference_drift_report_{args.attack_name}.json")
    analyze_drift(args.clean_preference_csv, output_dir, report_path)

    print(f"\n>>> Drift experiment complete. Results in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate user preference drift from attacked items")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze existing results, no GPU needed")

    # Full experiment args
    parser.add_argument("--clean_summary_csv", help="Clean image_summary.csv")
    parser.add_argument("--attacked_summary_csv", help="Attacked summary CSV")
    parser.add_argument("--title_csv", help="Item titles CSV")
    parser.add_argument("--user_items_tsv", help="user_items_negs.tsv")
    parser.add_argument("--clean_preference_csv", help="Clean user_preference.csv")
    parser.add_argument("--attack_name", default="inject_en", help="Attack name for labeling")
    parser.add_argument("--output_dir", default="results/preference_drift/")
    parser.add_argument("--max_n", type=int, default=6, help="Max N items to replace (1..N)")
    parser.add_argument("--skip_inference", action="store_true", help="Skip GPU inference step")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=12)

    # Analyze-only args
    parser.add_argument("--drift_dir", help="Directory with drift preference CSVs")
    parser.add_argument("--output_report", help="Output report JSON path")

    args = parser.parse_args()

    if args.analyze_only:
        if not args.clean_preference_csv or not args.drift_dir:
            parser.error("--analyze_only requires --clean_preference_csv and --drift_dir")
        analyze_drift(args.clean_preference_csv, args.drift_dir, args.output_report)
    else:
        if not all([args.clean_summary_csv, args.attacked_summary_csv,
                     args.title_csv, args.user_items_tsv]):
            parser.error("Full experiment requires --clean_summary_csv, --attacked_summary_csv, "
                         "--title_csv, --user_items_tsv")
        if not args.clean_preference_csv:
            parser.error("Full experiment requires --clean_preference_csv for analysis step")
        run_full_drift_experiment(args)


if __name__ == "__main__":
    main()
