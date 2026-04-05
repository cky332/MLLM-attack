"""
Evaluate targeted recommendation impact of attacks (Direction 3).

Answers two key questions using EXISTING data (no GPU needed):
  Q1: Do attacks boost/reduce an item's recommendation presence?
      - Check if attack keywords leak from item summaries into user preferences
      - Compare keyword leakage for users who interacted vs didn't interact with attacked items
  Q2: Can attacks misdirect recommendations to unrelated users?
      - Check if users who NEVER interacted with attacked items still got "contaminated"
      - Measure cross-user keyword contamination

Required files (all already generated):
  - clean user preferences:    user_preference_direct.csv
  - attacked user preferences: results/user_preference_attacked_{attack}.csv
  - clean image summaries:     image_summary.csv
  - attacked image summaries:  results/attacked_{attack}_summary.csv
  - user-item interactions:    user_items_negs.tsv

Usage:
    # Analyze a single attack
    python eval_targeted_impact.py \
        --clean_preference_csv /path/to/user_preference_direct.csv \
        --attacked_preference_csv results/user_preference_attacked_inject_en.csv \
        --clean_summary_csv /path/to/image_summary.csv \
        --attacked_summary_csv results/attacked_inject_en_summary.csv \
        --user_items_tsv /path/to/user_items_negs.tsv \
        --attack_name inject_en \
        --output_dir results/

    # Analyze all attacks at once
    python eval_targeted_impact.py --run_all \
        --clean_preference_csv /path/to/user_preference_direct.csv \
        --clean_summary_csv /path/to/image_summary.csv \
        --user_items_tsv /path/to/user_items_negs.tsv \
        --results_dir results/ \
        --output_dir results/
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from attack_config import (
    ATTACK_CONFIGS,
    ATTACK_TEXTS,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_user_items(user_items_tsv):
    """Load user-item interactions. Returns {user_id: [item_ids]}."""
    user_items = {}
    with open(user_items_tsv, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                user = parts[0]
                items = parts[1].split(", ")
                items = [i.strip().rstrip(",") for i in items if i.strip().rstrip(",")]
                user_items[user] = items
    return user_items


def load_preferences(csv_path):
    """Load user preferences as {user_id: summary_text}."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "user" in df.columns:
        df.rename(columns={"user": "user_id"}, inplace=True)
    df["user_id"] = df["user_id"].astype(str)
    return dict(zip(df["user_id"], df["summary"].astype(str)))


def load_summaries(csv_path):
    """Load item summaries as {item_id: summary_text}."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["item_id"] = df["item_id"].astype(str)
    return dict(zip(df["item_id"], df["summary"].astype(str)))


def get_attack_keywords(attack_name):
    """Get relevant keywords to track for a given attack type."""
    config = ATTACK_CONFIGS.get(attack_name, {})
    text_key = config.get("text_key", attack_name)
    attack_text = ATTACK_TEXTS.get(text_key, "")

    # Determine if positive or negative attack
    is_negative = "negative" in attack_name
    base_keywords = NEGATIVE_KEYWORDS if is_negative else POSITIVE_KEYWORDS

    # Also extract key phrases from the attack text itself
    attack_phrases = [
        f.strip().lower()
        for f in re.split(r"[,，.。!！?？:：]+", attack_text)
        if len(f.strip()) > 3
    ]

    return base_keywords, attack_phrases, attack_text


# ---------------------------------------------------------------------------
# Analysis 1: Item-level keyword leakage (item summary → user preference)
# ---------------------------------------------------------------------------
def analyze_keyword_leakage(clean_prefs, attacked_prefs, user_items,
                            attacked_item_ids, keywords):
    """Measure if attack keywords leak from item summaries into user preferences.

    Compares keyword appearance in preferences BEFORE vs AFTER attack,
    separated by whether the user interacted with attacked items.
    """
    # Classify users
    interacted_users = set()
    non_interacted_users = set()
    for user, items in user_items.items():
        if user not in clean_prefs or user not in attacked_prefs:
            continue
        if attacked_item_ids & set(items):
            interacted_users.add(user)
        else:
            non_interacted_users.add(user)

    results = {
        "total_users_with_prefs": len(set(clean_prefs) & set(attacked_prefs)),
        "interacted_users": len(interacted_users),
        "non_interacted_users": len(non_interacted_users),
        "keywords": {},
    }

    for kw in keywords:
        kw_lower = kw.lower()

        # Count keyword in clean vs attacked preferences for each group
        stats = {
            "interacted": {"clean": 0, "attacked": 0, "new_appearances": 0},
            "non_interacted": {"clean": 0, "attacked": 0, "new_appearances": 0},
        }

        for user in interacted_users:
            clean_has = kw_lower in clean_prefs[user].lower()
            attacked_has = kw_lower in attacked_prefs[user].lower()
            if clean_has:
                stats["interacted"]["clean"] += 1
            if attacked_has:
                stats["interacted"]["attacked"] += 1
            if attacked_has and not clean_has:
                stats["interacted"]["new_appearances"] += 1

        for user in non_interacted_users:
            clean_has = kw_lower in clean_prefs[user].lower()
            attacked_has = kw_lower in attacked_prefs[user].lower()
            if clean_has:
                stats["non_interacted"]["clean"] += 1
            if attacked_has:
                stats["non_interacted"]["attacked"] += 1
            if attacked_has and not clean_has:
                stats["non_interacted"]["new_appearances"] += 1

        # Only report keywords with some change
        total_diff = (stats["interacted"]["attacked"] - stats["interacted"]["clean"] +
                      stats["non_interacted"]["attacked"] - stats["non_interacted"]["clean"])
        if total_diff != 0 or stats["interacted"]["new_appearances"] > 0:
            results["keywords"][kw] = stats

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Per-item influence score
# ---------------------------------------------------------------------------
def analyze_per_item_influence(clean_prefs, attacked_prefs, user_items,
                               attacked_item_ids, keywords):
    """For each attacked item, measure how many users' preferences were
    influenced by the attack keywords after interacting with that item.

    Returns per-item influence scores.
    """
    # Build reverse index: item_id -> [users who interacted with it]
    item_users = defaultdict(list)
    for user, items in user_items.items():
        if user not in clean_prefs or user not in attacked_prefs:
            continue
        for item in items:
            if item in attacked_item_ids:
                item_users[item].append(user)

    kw_set = [kw.lower() for kw in keywords]

    item_scores = []
    for item_id, users in item_users.items():
        if not users:
            continue

        keyword_gained = 0  # users who gained attack keywords
        total_users = len(users)

        for user in users:
            clean_text = clean_prefs[user].lower()
            attacked_text = attacked_prefs[user].lower()

            # Count how many keywords newly appeared
            new_kw_count = sum(
                1 for kw in kw_set
                if kw in attacked_text and kw not in clean_text
            )
            if new_kw_count > 0:
                keyword_gained += 1

        item_scores.append({
            "item_id": item_id,
            "users_interacted": total_users,
            "users_keyword_gained": keyword_gained,
            "influence_rate": keyword_gained / total_users if total_users > 0 else 0,
        })

    item_scores.sort(key=lambda x: -x["influence_rate"])
    return item_scores


# ---------------------------------------------------------------------------
# Analysis 3: Cross-contamination (misdirected push)
# ---------------------------------------------------------------------------
def analyze_cross_contamination(clean_prefs, attacked_prefs, user_items,
                                 attacked_item_ids, keywords):
    """Detect if attack keywords appear in preferences of users who
    NEVER interacted with any attacked item — evidence of indirect
    contamination through the recommendation pipeline.

    Also identifies users whose preferences shifted TOWARD attack content
    despite having no direct exposure.
    """
    kw_set = [kw.lower() for kw in keywords]

    # Find users with NO interaction with attacked items
    non_interacted = []
    for user in set(clean_prefs) & set(attacked_prefs):
        user_item_set = set(user_items.get(user, []))
        if not (user_item_set & attacked_item_ids):
            non_interacted.append(user)

    contaminated_users = []
    total_new_keywords = 0

    for user in non_interacted:
        clean_text = clean_prefs[user].lower()
        attacked_text = attacked_prefs[user].lower()

        new_kws = [kw for kw in kw_set if kw in attacked_text and kw not in clean_text]
        lost_kws = [kw for kw in kw_set if kw in clean_text and kw not in attacked_text]

        if new_kws:
            contaminated_users.append({
                "user_id": user,
                "new_keywords": new_kws,
                "lost_keywords": lost_kws,
                "pref_clean_preview": clean_prefs[user][:300],
                "pref_attacked_preview": attacked_prefs[user][:300],
            })
            total_new_keywords += len(new_kws)

    result = {
        "non_interacted_users": len(non_interacted),
        "contaminated_users": len(contaminated_users),
        "contamination_rate": len(contaminated_users) / max(len(non_interacted), 1),
        "total_new_keyword_appearances": total_new_keywords,
        "sample_contaminated": contaminated_users[:10],
    }

    return result


# ---------------------------------------------------------------------------
# Analysis 4: Interaction depth vs preference shift
# ---------------------------------------------------------------------------
def analyze_interaction_depth(clean_prefs, attacked_prefs, user_items,
                               attacked_item_ids):
    """Group users by how many attacked items they interacted with,
    then measure preference Jaccard for each group.

    This answers: "Do users who interact with MORE attacked items
    experience greater preference drift?"
    """
    depth_groups = defaultdict(list)  # n_attacked -> [jaccard values]

    for user in set(clean_prefs) & set(attacked_prefs):
        items = user_items.get(user, [])
        n_attacked = len(set(items) & attacked_item_ids)

        clean_words = set(clean_prefs[user].lower().split())
        attacked_words = set(attacked_prefs[user].lower().split())
        union = clean_words | attacked_words
        jaccard = len(clean_words & attacked_words) / max(len(union), 1)

        depth_groups[n_attacked].append(jaccard)

    results = {}
    for n in sorted(depth_groups.keys()):
        jaccards = depth_groups[n]
        results[n] = {
            "n_attacked_items": n,
            "n_users": len(jaccards),
            "mean_jaccard": sum(jaccards) / len(jaccards),
            "min_jaccard": min(jaccards),
            "max_jaccard": max(jaccards),
            "highly_changed_pct": sum(1 for j in jaccards if j < 0.5) / len(jaccards) * 100,
        }

    return results


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------
def run_analysis(clean_preference_csv, attacked_preference_csv,
                 clean_summary_csv, attacked_summary_csv,
                 user_items_tsv, attack_name, output_dir):
    """Run all targeted impact analyses for a single attack."""

    print(f"\n{'=' * 80}")
    print(f"TARGETED IMPACT ANALYSIS: {attack_name}")
    print(f"{'=' * 80}")

    # Load data
    print("Loading data...")
    clean_prefs = load_preferences(clean_preference_csv)
    attacked_prefs = load_preferences(attacked_preference_csv)
    user_items = load_user_items(user_items_tsv)
    attacked_summaries = load_summaries(attacked_summary_csv)
    attacked_item_ids = set(attacked_summaries.keys())

    base_keywords, attack_phrases, attack_text = get_attack_keywords(attack_name)

    print(f"  Users with preferences: {len(set(clean_prefs) & set(attacked_prefs))}")
    print(f"  Attacked items: {len(attacked_item_ids)}")
    print(f"  Attack text: {attack_text}")
    print(f"  Tracking {len(base_keywords)} base keywords + {len(attack_phrases)} attack phrases")

    all_keywords = base_keywords + [p for p in attack_phrases if p not in [k.lower() for k in base_keywords]]

    # --- Analysis 1: Keyword leakage ---
    print("\n--- Analysis 1: Keyword Leakage (item summary → user preference) ---")
    leakage = analyze_keyword_leakage(
        clean_prefs, attacked_prefs, user_items, attacked_item_ids, all_keywords
    )
    print(f"  Interacted users: {leakage['interacted_users']}")
    print(f"  Non-interacted users: {leakage['non_interacted_users']}")
    print(f"\n  {'Keyword':<20} {'Interacted':>40}  {'Non-interacted':>40}")
    print(f"  {'':20} {'clean→attacked (new)':>40}  {'clean→attacked (new)':>40}")
    print("  " + "-" * 105)
    for kw, stats in sorted(leakage["keywords"].items(),
                             key=lambda x: -(x[1]["interacted"]["new_appearances"])):
        ia = stats["interacted"]
        ni = stats["non_interacted"]
        print(f"  {kw:<20} {ia['clean']:>6}→{ia['attacked']:<6} (+{ia['new_appearances']:<5})  "
              f"{ni['clean']:>6}→{ni['attacked']:<6} (+{ni['new_appearances']:<5})")

    # --- Analysis 2: Per-item influence ---
    print("\n--- Analysis 2: Per-Item Influence Score (top 20) ---")
    item_scores = analyze_per_item_influence(
        clean_prefs, attacked_prefs, user_items, attacked_item_ids, all_keywords
    )
    print(f"  {'Item ID':<12} {'Users':>8} {'KW Gained':>10} {'Influence%':>12}")
    print("  " + "-" * 45)
    for item in item_scores[:20]:
        print(f"  {item['item_id']:<12} {item['users_interacted']:>8} "
              f"{item['users_keyword_gained']:>10} {item['influence_rate']*100:>11.1f}%")

    avg_influence = (sum(i["influence_rate"] for i in item_scores) / len(item_scores)
                     if item_scores else 0)
    print(f"\n  Average influence rate across {len(item_scores)} attacked items: {avg_influence*100:.1f}%")

    # --- Analysis 3: Cross-contamination ---
    print("\n--- Analysis 3: Cross-Contamination (misdirected push) ---")
    contamination = analyze_cross_contamination(
        clean_prefs, attacked_prefs, user_items, attacked_item_ids, all_keywords
    )
    print(f"  Users with NO attacked-item interaction: {contamination['non_interacted_users']}")
    print(f"  Contaminated (gained attack keywords):   {contamination['contaminated_users']} "
          f"({contamination['contamination_rate']:.1%})")
    print(f"  Total new keyword appearances:           {contamination['total_new_keyword_appearances']}")

    if contamination["sample_contaminated"]:
        print(f"\n  Sample contaminated users (no direct exposure):")
        for i, u in enumerate(contamination["sample_contaminated"][:5], 1):
            print(f"    [{i}] user={u['user_id']}, new keywords: {u['new_keywords']}")
            print(f"        clean:    {u['pref_clean_preview'][:150]}")
            print(f"        attacked: {u['pref_attacked_preview'][:150]}")

    # --- Analysis 4: Interaction depth ---
    print("\n--- Analysis 4: Interaction Depth vs Preference Shift ---")
    depth = analyze_interaction_depth(
        clean_prefs, attacked_prefs, user_items, attacked_item_ids
    )
    print(f"  {'N_attacked':>10} {'Users':>8} {'MeanJaccard':>12} {'HighlyChg%':>12}")
    print("  " + "-" * 45)
    for n in sorted(depth.keys()):
        d = depth[n]
        print(f"  {d['n_attacked_items']:>10} {d['n_users']:>8} "
              f"{d['mean_jaccard']:>12.4f} {d['highly_changed_pct']:>11.1f}%")

    # --- Save report ---
    report = {
        "attack_name": attack_name,
        "attack_text": attack_text,
        "keyword_leakage": {
            "interacted_users": leakage["interacted_users"],
            "non_interacted_users": leakage["non_interacted_users"],
            "keywords": leakage["keywords"],
        },
        "per_item_influence": {
            "avg_influence_rate": avg_influence,
            "total_items": len(item_scores),
            "top_20": item_scores[:20],
        },
        "cross_contamination": contamination,
        "interaction_depth": {str(k): v for k, v in depth.items()},
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"targeted_impact_{attack_name}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {report_path}")

    return report


def run_all_attacks(clean_preference_csv, clean_summary_csv, user_items_tsv,
                    results_dir, output_dir):
    """Run analysis for all attacks that have both attacked preferences and summaries."""
    # Find all available attacked preference files
    pref_files = glob.glob(os.path.join(results_dir, "user_preference_attacked_*.csv"))

    all_reports = {}
    for pref_path in sorted(pref_files):
        # Extract attack name
        basename = os.path.basename(pref_path)
        attack_name = basename.replace("user_preference_attacked_", "").replace(".csv", "")

        # Find matching attacked summary
        summary_path = os.path.join(results_dir, f"attacked_{attack_name}_summary.csv")
        if not os.path.exists(summary_path):
            print(f"SKIP {attack_name}: no matching {summary_path}")
            continue

        report = run_analysis(
            clean_preference_csv, pref_path,
            clean_summary_csv, summary_path,
            user_items_tsv, attack_name, output_dir,
        )
        all_reports[attack_name] = report

    # Print cross-attack summary
    if all_reports:
        print("\n" + "=" * 100)
        print("CROSS-ATTACK TARGETED IMPACT SUMMARY")
        print("=" * 100)

        print(f"\n{'Attack':<28} {'AvgInfluence%':>14} {'Contaminated':>13} {'ContamRate':>11} "
              f"{'Depth0_Jacc':>12} {'DepthMax_Jacc':>14}")
        print("-" * 95)

        for name in sorted(all_reports):
            r = all_reports[name]
            inf_rate = r["per_item_influence"]["avg_influence_rate"] * 100
            contam = r["cross_contamination"]["contaminated_users"]
            contam_rate = r["cross_contamination"]["contamination_rate"] * 100
            depths = r["interaction_depth"]
            d0 = depths.get("0", {}).get("mean_jaccard", 0)
            max_n = max((int(k) for k in depths if k != "0"), default=0)
            d_max = depths.get(str(max_n), {}).get("mean_jaccard", 0)
            print(f"{name:<28} {inf_rate:>13.1f}% {contam:>13} {contam_rate:>10.1f}% "
                  f"{d0:>12.4f} {d_max:>14.4f}")

        # Save combined summary
        summary_path = os.path.join(output_dir, "targeted_impact_summary.json")
        summary = {}
        for name, r in all_reports.items():
            summary[name] = {
                "avg_influence_rate": r["per_item_influence"]["avg_influence_rate"],
                "contaminated_users": r["cross_contamination"]["contaminated_users"],
                "contamination_rate": r["cross_contamination"]["contamination_rate"],
                "interaction_depth": r["interaction_depth"],
            }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nCombined summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate targeted recommendation impact (Direction 3)"
    )
    parser.add_argument("--run_all", action="store_true",
                        help="Analyze all attacks in results_dir")

    # Single attack args
    parser.add_argument("--clean_preference_csv", required=True,
                        help="Clean user_preference_direct.csv")
    parser.add_argument("--clean_summary_csv", required=True,
                        help="Clean image_summary.csv")
    parser.add_argument("--user_items_tsv", required=True,
                        help="user_items_negs.tsv")

    # Single attack mode
    parser.add_argument("--attacked_preference_csv",
                        help="Attacked user preference CSV")
    parser.add_argument("--attacked_summary_csv",
                        help="Attacked image summary CSV")
    parser.add_argument("--attack_name", default="inject_en",
                        help="Attack name for labeling")

    # Batch mode
    parser.add_argument("--results_dir", default="results/",
                        help="Directory with attacked CSVs (for --run_all)")
    parser.add_argument("--output_dir", default="results/",
                        help="Output directory for reports")

    args = parser.parse_args()

    if args.run_all:
        run_all_attacks(
            args.clean_preference_csv, args.clean_summary_csv,
            args.user_items_tsv, args.results_dir, args.output_dir,
        )
    else:
        if not args.attacked_preference_csv or not args.attacked_summary_csv:
            parser.error("Single attack mode requires --attacked_preference_csv "
                         "and --attacked_summary_csv")
        run_analysis(
            args.clean_preference_csv, args.attacked_preference_csv,
            args.clean_summary_csv, args.attacked_summary_csv,
            args.user_items_tsv, args.attack_name, args.output_dir,
        )


if __name__ == "__main__":
    main()
