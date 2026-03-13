"""
Targeted recommendation attack: inject specific item descriptions to push items
to users with particular preferences.

Experiment design:
1. Select target items and target user groups (users who prefer a certain category)
2. Inject target users' preference keywords into the target item images
3. Measure whether the target item's recommendation score increases for those users

Usage:
    # Analyze which user preference clusters exist
    python targeted_recommendation.py analyze_users \
        --user_preference_csv ../user_preference_direct.csv \
        --output_report results/user_clusters.json

    # Generate targeted attack texts based on user preference clusters
    python targeted_recommendation.py generate_attacks \
        --user_preference_csv ../user_preference_direct.csv \
        --title_csv ../../data/MicroLens-50k/MicroLens-50k_titles.csv \
        --target_items 1234,5678 \
        --target_cluster 0 \
        --output_config results/targeted_attack_config.json

    # Evaluate targeted attack effectiveness
    python targeted_recommendation.py evaluate \
        --clean_preference_csv ../user_preference_direct.csv \
        --attacked_preference_csv results/user_preference_attacked_targeted.csv \
        --user_items_tsv ../../data/MicroLens-50k/Split/user_items_negs.tsv \
        --target_items 1234,5678 \
        --output_report results/targeted_attack_report.json
"""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd


def extract_keywords(text, top_k=20):
    """Extract top-k most frequent meaningful words from a text."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "nor", "not", "no", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "only", "own", "same", "than", "too", "very", "just", "about", "up",
        "out", "also", "that", "this", "these", "those", "it", "its", "they",
        "their", "them", "he", "she", "his", "her", "him", "we", "our", "us",
        "you", "your", "who", "which", "what", "when", "where", "how", "there",
        "then", "if", "user", "appears", "preference", "prefers", "content",
        "videos", "video", "features", "style", "visual", "particularly",
    }
    words = re.findall(r"[a-z]+", text.lower())
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return [w for w, _ in Counter(words).most_common(top_k)]


def analyze_user_preferences(preference_csv, output_report=None, n_clusters=5):
    """Analyze user preferences to find clusters of similar users.

    Uses keyword-based clustering (no external ML dependencies required).
    """
    df = pd.read_csv(preference_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    if "user_id" not in df.columns and "user" in df.columns:
        df.rename(columns={"user": "user_id"}, inplace=True)
    if "summary" not in df.columns and "preference" in df.columns:
        df.rename(columns={"preference": "summary"}, inplace=True)

    df["user_id"] = df["user_id"].astype(str)

    # Extract keywords per user
    all_keywords = Counter()
    user_keywords = {}
    for _, row in df.iterrows():
        text = str(row["summary"])
        kws = extract_keywords(text, top_k=15)
        user_keywords[str(row["user_id"])] = kws
        all_keywords.update(kws)

    # Find top global keywords (these represent content categories)
    top_global = [kw for kw, _ in all_keywords.most_common(50)]

    # Simple keyword-based clustering: assign each user to dominant keyword group
    # Group users by their most distinctive keyword
    cluster_keywords = top_global[:n_clusters]
    clusters = {i: [] for i in range(n_clusters)}
    cluster_other = []

    for user_id, kws in user_keywords.items():
        assigned = False
        for i, ck in enumerate(cluster_keywords):
            if ck in kws[:5]:  # if cluster keyword is in user's top-5
                clusters[i].append(user_id)
                assigned = True
                break
        if not assigned:
            cluster_other.append(user_id)

    report = {
        "total_users": len(df),
        "top_global_keywords": top_global[:30],
        "clusters": {},
    }

    for i, ck in enumerate(cluster_keywords):
        sample_users = clusters[i][:5]
        sample_prefs = []
        for uid in sample_users:
            row = df[df["user_id"] == uid]
            if len(row) > 0:
                sample_prefs.append(str(row.iloc[0]["summary"])[:200])

        report["clusters"][f"cluster_{i}"] = {
            "keyword": ck,
            "size": len(clusters[i]),
            "user_ids_sample": sample_users,
            "preference_samples": sample_prefs,
        }

    report["unassigned_users"] = len(cluster_other)

    if output_report:
        os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[analyze_users] Report saved to {output_report}")

    # Print summary
    print("\n" + "=" * 60)
    print("USER PREFERENCE ANALYSIS")
    print("=" * 60)
    print(f"Total users: {report['total_users']}")
    print(f"Top keywords: {', '.join(top_global[:15])}")
    for cid, cinfo in report["clusters"].items():
        print(f"\n  {cid} (keyword='{cinfo['keyword']}', size={cinfo['size']}):")
        for sp in cinfo["preference_samples"][:2]:
            print(f"    - {sp[:100]}...")
    print(f"\n  Unassigned: {report['unassigned_users']}")

    return report


def generate_targeted_attack_text(preference_csv, title_csv, target_items,
                                  target_cluster_keyword=None, output_config=None):
    """Generate attack text designed to push items to specific user groups.

    The attack text is constructed from the target cluster's preference keywords,
    making the item appear relevant to that user group.
    """
    pref_df = pd.read_csv(preference_csv)
    pref_df.columns = [c.strip().lower() for c in pref_df.columns]
    if "summary" not in pref_df.columns and "preference" in pref_df.columns:
        pref_df.rename(columns={"preference": "summary"}, inplace=True)

    title_df = pd.read_csv(title_csv)
    title_df.columns = [c.strip().lower() for c in title_df.columns]
    title_df["item"] = title_df["item"].astype(str)

    # Extract cluster-specific keywords
    if target_cluster_keyword:
        # Find users whose preferences mention this keyword
        cluster_users = pref_df[
            pref_df["summary"].astype(str).str.lower().str.contains(target_cluster_keyword.lower(), na=False)
        ]
        cluster_keywords = extract_keywords(
            " ".join(cluster_users["summary"].astype(str).values), top_k=20
        )
    else:
        cluster_keywords = extract_keywords(
            " ".join(pref_df["summary"].astype(str).values), top_k=20
        )

    # Generate attack text for each target item
    attacks = {}
    for item_id in target_items:
        item_title = "Unknown"
        if item_id in title_df["item"].values:
            item_title = title_df[title_df["item"] == item_id].iloc[0]["title"]

        # Construct attack text using cluster keywords
        keyword_str = ", ".join(cluster_keywords[:10])
        attack_text = (
            f"This is a highly popular video featuring {keyword_str}. "
            f"Perfect content for fans of {cluster_keywords[0]} and {cluster_keywords[1]}. "
            f"Must watch recommendation."
        )

        attacks[item_id] = {
            "original_title": item_title,
            "attack_text": attack_text,
            "target_keywords": cluster_keywords[:10],
        }

    config = {
        "target_cluster_keyword": target_cluster_keyword,
        "cluster_keywords": cluster_keywords[:20],
        "target_items": attacks,
    }

    if output_config:
        os.makedirs(os.path.dirname(output_config) or ".", exist_ok=True)
        with open(output_config, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"[generate_attacks] Config saved to {output_config}")

    print("\n" + "=" * 60)
    print("TARGETED ATTACK TEXT GENERATION")
    print("=" * 60)
    print(f"Target cluster keyword: {target_cluster_keyword}")
    print(f"Cluster keywords: {', '.join(cluster_keywords[:10])}")
    for item_id, info in attacks.items():
        print(f"\n  Item {item_id} ('{info['original_title'][:50]}'):")
        print(f"    Attack: {info['attack_text'][:100]}...")

    return config


def evaluate_targeted_attack(clean_pref_csv, attacked_pref_csv, user_items_tsv,
                             target_items, target_cluster_keyword=None,
                             output_report=None):
    """Evaluate whether targeted attack successfully pushes items to target users.

    Compares how preference text similarity to target items changes
    between clean and attacked conditions.
    """
    clean_df = pd.read_csv(clean_pref_csv)
    attacked_df = pd.read_csv(attacked_pref_csv)

    for df in [clean_df, attacked_df]:
        df.columns = [c.strip().lower() for c in df.columns]
        if "user_id" not in df.columns and "user" in df.columns:
            df.rename(columns={"user": "user_id"}, inplace=True)
        if "summary" not in df.columns and "preference" in df.columns:
            df.rename(columns={"preference": "summary"}, inplace=True)
        df["user_id"] = df["user_id"].astype(str)

    merged = pd.merge(clean_df, attacked_df, on="user_id", suffixes=("_clean", "_attacked"))

    # Identify target users (those who have target items in their negatives)
    target_set = set(target_items)
    target_users = set()
    non_target_users = set()

    with open(user_items_tsv, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                user = parts[0]
                neg_items = set(parts[2].replace(" ", "").split(","))
                if target_set & neg_items:
                    target_users.add(user)
                else:
                    non_target_users.add(user)

    # If target_cluster_keyword provided, filter target users to those matching the cluster
    if target_cluster_keyword:
        cluster_target_users = set()
        for _, row in merged.iterrows():
            uid = str(row["user_id"])
            if uid in target_users:
                pref = str(row["summary_clean"]).lower()
                if target_cluster_keyword.lower() in pref:
                    cluster_target_users.add(uid)
        target_users = cluster_target_users

    # Analyze preference changes for target vs non-target users
    def compute_keyword_overlap(text, keywords):
        text_lower = text.lower()
        return sum(1 for kw in keywords if kw.lower() in text_lower) / max(len(keywords), 1)

    target_keywords = target_items  # simplified; in practice use item descriptions

    report = {
        "target_items": target_items,
        "target_cluster_keyword": target_cluster_keyword,
        "target_users_count": len(target_users),
        "total_users": len(merged),
    }

    # Preference similarity analysis per group
    for group_name, user_set in [("target_users", target_users), ("all_users", set(merged["user_id"]))]:
        group_df = merged[merged["user_id"].isin(user_set)] if user_set else merged

        similarities = []
        for _, row in group_df.iterrows():
            clean_words = set(str(row["summary_clean"]).lower().split())
            attacked_words = set(str(row["summary_attacked"]).lower().split())
            union = clean_words | attacked_words
            jaccard = len(clean_words & attacked_words) / max(len(union), 1)
            similarities.append(jaccard)

        report[f"{group_name}_preference_change"] = {
            "count": len(group_df),
            "mean_jaccard": float(np.mean(similarities)) if similarities else None,
            "std_jaccard": float(np.std(similarities)) if similarities else None,
        }

    if output_report:
        os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[evaluate_targeted] Report saved to {output_report}")

    print("\n" + "=" * 60)
    print("TARGETED ATTACK EVALUATION")
    print("=" * 60)
    print(f"Target items: {target_items}")
    print(f"Target users: {report['target_users_count']}")
    for group in ["target_users", "all_users"]:
        info = report[f"{group}_preference_change"]
        print(f"\n  {group} (n={info['count']}):")
        if info["mean_jaccard"] is not None:
            print(f"    Mean preference Jaccard: {info['mean_jaccard']:.4f}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Targeted recommendation attack experiments")
    subparsers = parser.add_subparsers(dest="command")

    # analyze_users
    p_analyze = subparsers.add_parser("analyze_users")
    p_analyze.add_argument("--user_preference_csv", required=True)
    p_analyze.add_argument("--output_report", default="results/user_clusters.json")
    p_analyze.add_argument("--n_clusters", type=int, default=5)

    # generate_attacks
    p_gen = subparsers.add_parser("generate_attacks")
    p_gen.add_argument("--user_preference_csv", required=True)
    p_gen.add_argument("--title_csv", required=True)
    p_gen.add_argument("--target_items", required=True, help="Comma-separated item IDs")
    p_gen.add_argument("--target_cluster_keyword", default=None)
    p_gen.add_argument("--output_config", default="results/targeted_attack_config.json")

    # evaluate
    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--clean_preference_csv", required=True)
    p_eval.add_argument("--attacked_preference_csv", required=True)
    p_eval.add_argument("--user_items_tsv", required=True)
    p_eval.add_argument("--target_items", required=True, help="Comma-separated item IDs")
    p_eval.add_argument("--target_cluster_keyword", default=None)
    p_eval.add_argument("--output_report", default="results/targeted_attack_report.json")

    args = parser.parse_args()

    if args.command == "analyze_users":
        analyze_user_preferences(args.user_preference_csv, args.output_report, args.n_clusters)
    elif args.command == "generate_attacks":
        items = [i.strip() for i in args.target_items.split(",")]
        generate_targeted_attack_text(
            args.user_preference_csv, args.title_csv, items,
            args.target_cluster_keyword, args.output_config,
        )
    elif args.command == "evaluate":
        items = [i.strip() for i in args.target_items.split(",")]
        evaluate_targeted_attack(
            args.clean_preference_csv, args.attacked_preference_csv,
            args.user_items_tsv, items, args.target_cluster_keyword,
            args.output_report,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
