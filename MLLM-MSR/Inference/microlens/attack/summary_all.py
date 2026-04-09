#!/usr/bin/env python3
"""Unified summary across all attack evaluations.

Aggregates outputs from:
  - evaluate.py                    -> results/report_*.json                (Part 1)
  - eval_recommendation_impact.py  -> results/recommendation_impact_*.json (Part 2 + Exposure)
  - generate_hubness_attack.py     -> results/hubness_report*.json         (Part 3)
  - targeted_recommendation.py     -> results/targeted_attack_report*.json (Part 4)
  - eval_preference_drift.py       -> results/preference_drift/*.json      (Part 5)
  - eval_topk_ranking.py           -> results/topk_ranking_*.json          (Part 6)

Usage:
    python summary_all.py [--results_dir results]
"""
import argparse
import glob
import json
import os


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}


def part1_summary(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "report_*.json")))
    if not files:
        print("(no report_*.json found)")
        return
    print(f'{"Attack":<30} {"Verbatim%":>10} {"MeanJaccard":>12} {"HighlyChanged":>14} {"LengthDiff":>10}')
    print("-" * 80)
    for f in files:
        name = os.path.basename(f).replace("report_", "").replace(".json", "")
        r = _load(f)
        if "_error" in r:
            continue
        vd = r.get("verbatim_detection", {})
        ts = r.get("text_similarity", {})
        la = r.get("length_analysis", {})
        print(f'{name:<30} '
              f'{vd.get("full_match_rate_attacked", 0)*100:>9.1f}% '
              f'{ts.get("mean_jaccard", 0):>12.4f} '
              f'{ts.get("num_highly_changed", 0):>14} '
              f'{la.get("mean_length_diff", 0):>+10.0f}')


def part2_summary(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "recommendation_impact_*.json")))
    if not files:
        print("(no recommendation_impact_*.json found)")
        return
    print(f'{"Attack":<30} {"MeanJaccard":>12} {"Affected":>10} {"HighChg":>9} {"BarelyChg":>10} {"ExpDelta":>10} {"Dir":>9}')
    print("-" * 95)
    for f in files:
        name = os.path.basename(f).replace("recommendation_impact_", "").replace(".json", "")
        r = _load(f)
        if "_error" in r:
            continue
        ps = r.get("preference_similarity", {}).get("all_users", {})
        er = r.get("exposure_direction", {})
        delta = er.get("total_delta", "-")
        direction = er.get("direction", "-")
        delta_str = f"{delta:+d}" if isinstance(delta, int) else str(delta)
        print(f'{name:<30} '
              f'{ps.get("mean_jaccard", 0):>12.4f} '
              f'{r.get("affected_users", 0):>10} '
              f'{ps.get("highly_changed_lt_0.5", 0):>9} '
              f'{ps.get("barely_changed_gte_0.8", 0):>10} '
              f'{delta_str:>10} '
              f'{direction:>9}')


def part3_hubness(results_dir):
    files = sorted(
        glob.glob(os.path.join(results_dir, "hubness_report*.json")) +
        glob.glob(os.path.join(results_dir, "hubness", "*.json"))
    )
    if not files:
        print("(no hubness reports found — run generate_hubness_attack.py eval)")
        return
    for f in files:
        r = _load(f)
        name = os.path.basename(f).replace(".json", "")
        if "_error" in r:
            print(f"  {name}: load error")
            continue
        print(f"  {name}:")
        # generic dump of top-level scalar metrics
        for k, v in r.items():
            if isinstance(v, (int, float, str)):
                print(f"    {k}: {v}")


def part4_targeted(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "targeted_attack_report*.json")))
    if not files:
        print("(no targeted_attack_report*.json — run targeted_recommendation.py eval)")
        return
    for f in files:
        r = _load(f)
        name = os.path.basename(f).replace(".json", "")
        print(f"  {name}:")
        for k, v in r.items():
            if isinstance(v, (int, float, str)):
                print(f"    {k}: {v}")


def part5_drift(results_dir):
    drift_dir = os.path.join(results_dir, "preference_drift")
    files = sorted(glob.glob(os.path.join(drift_dir, "*.json"))) if os.path.isdir(drift_dir) else []
    files += sorted(glob.glob(os.path.join(results_dir, "preference_drift_*.json")))
    if not files:
        print("(no preference drift reports — run eval_preference_drift.py)")
        return
    for f in files:
        r = _load(f)
        name = os.path.basename(f).replace(".json", "")
        print(f"  {name}:")
        if "drift_curve" in r:
            for pt in r["drift_curve"]:
                print(f"    N={pt.get('n_replaced')}: mean_jaccard={pt.get('mean_jaccard'):.4f}")
        else:
            for k, v in r.items():
                if isinstance(v, (int, float, str)):
                    print(f"    {k}: {v}")


def part6_topk_ranking(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "topk_ranking_*.json")))
    if not files:
        print("(no topk_ranking_*.json found — run eval_topk_ranking.py)")
        return

    # Print clean baseline once (same across all attacks)
    first = _load(files[0])
    if "_error" not in first:
        gc = first.get("global_metrics", {}).get("clean", {})
        print(f"  Clean baseline: Recall@10={gc.get('recall@10', 0):.4f}  "
              f"NDCG@10={gc.get('ndcg@10', 0):.4f}  "
              f"MRR@10={gc.get('mrr@10', 0):.4f}")
        print()

    print(f'{"Attack":<30} '
          f'{"R@10":>7} {"dR@10":>7} '
          f'{"N@10":>7} {"dN@10":>7} '
          f'{"MRR@10":>7} {"dMRR":>7} '
          f'{"MeanRk":>7} {"dRank":>7} '
          f'{"Improv":>7} {"Worse":>7}')
    print("-" * 112)

    for f in files:
        name = os.path.basename(f).replace("topk_ranking_", "").replace(".json", "")
        r = _load(f)
        if "_error" in r:
            continue
        gm = r.get("global_metrics", {})
        gc = gm.get("clean", {})
        ga = gm.get("attacked", {})
        ra = r.get("rank_analysis_all_positives", {})

        r10_c = gc.get("recall@10", 0)
        r10_a = ga.get("recall@10", 0)
        n10_c = gc.get("ndcg@10", 0)
        n10_a = ga.get("ndcg@10", 0)
        mrr_c = gc.get("mrr@10", 0)
        mrr_a = ga.get("mrr@10", 0)
        mr_a = ra.get("mean_rank_attacked", 0)
        mr_d = ra.get("mean_rank_delta", 0)
        n_imp = ra.get("n_users_rank_improved", 0)
        n_wor = ra.get("n_users_rank_worsened", 0)

        print(f'{name:<30} '
              f'{r10_a:>7.4f} {r10_a - r10_c:>+7.4f} '
              f'{n10_a:>7.4f} {n10_a - n10_c:>+7.4f} '
              f'{mrr_a:>7.4f} {mrr_a - mrr_c:>+7.4f} '
              f'{mr_a:>7.2f} {mr_d:>+7.3f} '
              f'{n_imp:>7} {n_wor:>7}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    print("=" * 100)
    print("UNIFIED ATTACK EVALUATION SUMMARY")
    print("=" * 100)

    print("\n--- Part 1: Image Summary Attack Effectiveness ---")
    part1_summary(args.results_dir)

    print("\n--- Part 2: Recommendation System Impact (+ Exposure Direction) ---")
    part2_summary(args.results_dir)

    print("\n--- Part 3: Adversarial Hubness ---")
    part3_hubness(args.results_dir)

    print("\n--- Part 4: Targeted User-Group Recommendation ---")
    part4_targeted(args.results_dir)

    print("\n--- Part 5: User Preference Drift (multi-interaction) ---")
    part5_drift(args.results_dir)

    print("\n--- Part 6: Real Top-K Ranking Impact ---")
    part6_topk_ranking(args.results_dir)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
