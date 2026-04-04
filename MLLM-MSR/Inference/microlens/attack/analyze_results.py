"""
Comprehensive analysis of all attack evaluation results.

Reads report_*.json and recommendation_impact_*.json to produce:
  1. Cross-attack comparison tables
  2. Summary-level vs recommendation-level impact correlation
  3. Attack category analysis (EN vs CN, injection vs non-injection, stealth variants)
  4. Keyword penetration analysis
  5. Stealth-effectiveness tradeoff ranking

Usage:
    python analyze_results.py --results_dir results/

    # Save analysis to file
    python analyze_results.py --results_dir results/ --output results/full_analysis.json
"""

import argparse
import glob
import json
import os


# ---------------------------------------------------------------------------
# Attack metadata: category, language, stealth level (opacity 0-255)
# ---------------------------------------------------------------------------
ATTACK_META = {
    "inject_en":               {"category": "injection",     "lang": "en", "opacity": 255, "font_ratio": 0.06},
    "inject_cn":               {"category": "injection",     "lang": "cn", "opacity": 255, "font_ratio": 0.06},
    "inject_cn_watermark":     {"category": "injection",     "lang": "cn", "opacity": 128, "font_ratio": 0.05},
    "inject_en_corner":        {"category": "injection",     "lang": "en", "opacity": 255, "font_ratio": 0.06},
    "inject_en_small":         {"category": "injection",     "lang": "en", "opacity":  96, "font_ratio": 0.04},
    "inject_en_stealth_low":   {"category": "injection",     "lang": "en", "opacity":  64, "font_ratio": 0.03},
    "inject_en_stealth_mid":   {"category": "injection",     "lang": "en", "opacity":  96, "font_ratio": 0.04},
    "inject_en_stealth_blend": {"category": "injection",     "lang": "en", "opacity":  48, "font_ratio": 0.03},
    "negative_en":             {"category": "negative",      "lang": "en", "opacity": 255, "font_ratio": 0.06},
    "negative_cn":             {"category": "negative",      "lang": "cn", "opacity": 255, "font_ratio": 0.06},
    "negative_inject_en":      {"category": "neg_injection", "lang": "en", "opacity": 255, "font_ratio": 0.06},
    "five_star":               {"category": "promo",         "lang": "cn", "opacity": 255, "font_ratio": 0.06},
    "promo_cn":                {"category": "promo",         "lang": "cn", "opacity": 255, "font_ratio": 0.06},
}


def load_all_results(results_dir):
    """Load all report and recommendation_impact JSON files."""
    reports = {}
    impacts = {}

    for f in sorted(glob.glob(os.path.join(results_dir, "report_*.json"))):
        name = os.path.basename(f).replace("report_", "").replace(".json", "")
        with open(f) as fp:
            reports[name] = json.load(fp)

    for f in sorted(glob.glob(os.path.join(results_dir, "recommendation_impact_*.json"))):
        name = os.path.basename(f).replace("recommendation_impact_", "").replace(".json", "")
        with open(f) as fp:
            impacts[name] = json.load(fp)

    return reports, impacts


# ===== Analysis 1: Cross-Attack Summary Table =====
def analysis_summary_table(reports, impacts):
    """Print a combined table ranking attacks by effectiveness."""
    print("\n" + "=" * 120)
    print("ANALYSIS 1: CROSS-ATTACK EFFECTIVENESS RANKING (sorted by Summary Jaccard, ascending = more effective)")
    print("=" * 120)

    rows = []
    for name in reports:
        r = reports[name]
        ts = r["text_similarity"]
        vd = r["verbatim_detection"]
        la = r["length_analysis"]
        meta = ATTACK_META.get(name, {})

        row = {
            "name": name,
            "category": meta.get("category", "?"),
            "lang": meta.get("lang", "?"),
            "opacity": meta.get("opacity", "?"),
            "summary_jaccard": ts["mean_jaccard"],
            "highly_changed": ts["num_highly_changed"],
            "total": ts["total"],
            "verbatim_pct": vd["full_match_rate_attacked"] * 100,
            "length_diff": la["mean_length_diff"],
        }

        if name in impacts:
            imp = impacts[name]
            ps = imp["preference_similarity"]["all_users"]
            row["pref_jaccard"] = ps["mean_jaccard"]
            row["pref_highly_changed"] = ps["highly_changed_lt_0.5"]
            row["pref_barely_changed"] = ps["barely_changed_gte_0.8"]
        else:
            row["pref_jaccard"] = None
            row["pref_highly_changed"] = None
            row["pref_barely_changed"] = None

        rows.append(row)

    rows.sort(key=lambda x: x["summary_jaccard"])

    header = (f"{'Rank':<5} {'Attack':<28} {'Cat':<13} {'Lang':<5} {'Opacity':<8} "
              f"{'SumJacc':>8} {'Verbatim%':>10} {'HighChg':>8}/{rows[0]['total']:<6} "
              f"{'LenDiff':>8} {'PrefJacc':>9} {'PrefHigh':>9} {'PrefBarely':>11}")
    print(header)
    print("-" * len(header))

    for i, row in enumerate(rows, 1):
        pj = f"{row['pref_jaccard']:.4f}" if row["pref_jaccard"] is not None else "N/A"
        ph = str(row["pref_highly_changed"]) if row["pref_highly_changed"] is not None else "N/A"
        pb = str(row["pref_barely_changed"]) if row["pref_barely_changed"] is not None else "N/A"
        print(f"{i:<5} {row['name']:<28} {row['category']:<13} {row['lang']:<5} {row['opacity']:<8} "
              f"{row['summary_jaccard']:>8.4f} {row['verbatim_pct']:>9.1f}% {row['highly_changed']:>8}/{row['total']:<6} "
              f"{row['length_diff']:>+8.0f} {pj:>9} {ph:>9} {pb:>11}")

    return rows


# ===== Analysis 2: Category-Level Comparison =====
def analysis_by_category(reports, impacts):
    """Compare attack categories: injection vs negative vs promo."""
    print("\n" + "=" * 120)
    print("ANALYSIS 2: ATTACK CATEGORY COMPARISON (averaged within each category)")
    print("=" * 120)

    categories = {}
    for name, r in reports.items():
        meta = ATTACK_META.get(name, {})
        cat = meta.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []

        entry = {
            "name": name,
            "summary_jaccard": r["text_similarity"]["mean_jaccard"],
            "verbatim_pct": r["verbatim_detection"]["full_match_rate_attacked"] * 100,
        }
        if name in impacts:
            entry["pref_jaccard"] = impacts[name]["preference_similarity"]["all_users"]["mean_jaccard"]
        categories[cat].append(entry)

    print(f"\n{'Category':<15} {'Count':>6} {'AvgSumJacc':>11} {'AvgVerbatim%':>13} {'AvgPrefJacc':>12}")
    print("-" * 60)

    for cat in sorted(categories):
        entries = categories[cat]
        n = len(entries)
        avg_sj = sum(e["summary_jaccard"] for e in entries) / n
        avg_vb = sum(e["verbatim_pct"] for e in entries) / n
        pref_entries = [e for e in entries if "pref_jaccard" in e]
        avg_pj = sum(e["pref_jaccard"] for e in pref_entries) / len(pref_entries) if pref_entries else None
        pj_str = f"{avg_pj:.4f}" if avg_pj is not None else "N/A"
        print(f"{cat:<15} {n:>6} {avg_sj:>11.4f} {avg_vb:>12.1f}% {pj_str:>12}")
        for e in sorted(entries, key=lambda x: x["summary_jaccard"]):
            print(f"  - {e['name']:<26} SumJacc={e['summary_jaccard']:.4f}  Verbatim={e['verbatim_pct']:.1f}%")


# ===== Analysis 3: Language Comparison (EN vs CN) =====
def analysis_by_language(reports, impacts):
    """Compare English vs Chinese attacks."""
    print("\n" + "=" * 120)
    print("ANALYSIS 3: LANGUAGE COMPARISON (EN vs CN)")
    print("=" * 120)

    langs = {"en": [], "cn": []}
    for name, r in reports.items():
        meta = ATTACK_META.get(name, {})
        lang = meta.get("lang", "?")
        if lang not in langs:
            continue
        entry = {
            "name": name,
            "summary_jaccard": r["text_similarity"]["mean_jaccard"],
            "highly_changed_pct": r["text_similarity"]["num_highly_changed"] / r["text_similarity"]["total"] * 100,
        }
        if name in impacts:
            entry["pref_jaccard"] = impacts[name]["preference_similarity"]["all_users"]["mean_jaccard"]
        langs[lang].append(entry)

    for lang in ["en", "cn"]:
        entries = langs[lang]
        n = len(entries)
        avg_sj = sum(e["summary_jaccard"] for e in entries) / n
        pref_entries = [e for e in entries if "pref_jaccard" in e]
        avg_pj = sum(e["pref_jaccard"] for e in pref_entries) / len(pref_entries) if pref_entries else None
        avg_hc = sum(e["highly_changed_pct"] for e in entries) / n
        pj_str = f"{avg_pj:.4f}" if avg_pj is not None else "N/A"

        print(f"\n  {lang.upper()} attacks ({n} total):")
        print(f"    Avg Summary Jaccard:      {avg_sj:.4f}")
        print(f"    Avg Preference Jaccard:   {pj_str}")
        print(f"    Avg Highly Changed:       {avg_hc:.1f}%")

    # Direct comparison for same-type attacks
    print("\n  Head-to-head (same attack type, different language):")
    pairs = [
        ("inject_en", "inject_cn"),
        ("negative_en", "negative_cn"),
    ]
    for en_name, cn_name in pairs:
        if en_name in reports and cn_name in reports:
            en_j = reports[en_name]["text_similarity"]["mean_jaccard"]
            cn_j = reports[cn_name]["text_similarity"]["mean_jaccard"]
            winner = "EN" if en_j < cn_j else "CN"
            print(f"    {en_name} ({en_j:.4f}) vs {cn_name} ({cn_j:.4f}) → {winner} more effective (lower=better)")


# ===== Analysis 4: Stealth-Effectiveness Tradeoff =====
def analysis_stealth_tradeoff(reports):
    """Analyze opacity/font-size vs effectiveness for stealth variants."""
    print("\n" + "=" * 120)
    print("ANALYSIS 4: STEALTH-EFFECTIVENESS TRADEOFF (English injection variants only)")
    print("=" * 120)

    stealth_attacks = [
        "inject_en", "inject_en_corner", "inject_en_small",
        "inject_en_stealth_mid", "inject_en_stealth_low", "inject_en_stealth_blend",
    ]

    rows = []
    for name in stealth_attacks:
        if name not in reports:
            continue
        meta = ATTACK_META.get(name, {})
        r = reports[name]
        rows.append({
            "name": name,
            "opacity": meta.get("opacity", "?"),
            "font_ratio": meta.get("font_ratio", "?"),
            "summary_jaccard": r["text_similarity"]["mean_jaccard"],
            "verbatim_pct": r["verbatim_detection"]["full_match_rate_attacked"] * 100,
            "highly_changed": r["text_similarity"]["num_highly_changed"],
            "total": r["text_similarity"]["total"],
        })

    print(f"\n{'Attack':<28} {'Opacity':>8} {'FontRatio':>10} {'SumJaccard':>11} {'Verbatim%':>10} {'HighChg%':>9}")
    print("-" * 80)
    for row in rows:
        hc_pct = row["highly_changed"] / row["total"] * 100
        print(f"{row['name']:<28} {row['opacity']:>8} {row['font_ratio']:>10} "
              f"{row['summary_jaccard']:>11.4f} {row['verbatim_pct']:>9.1f}% {hc_pct:>8.1f}%")

    if len(rows) >= 2:
        best = min(rows, key=lambda x: x["summary_jaccard"])
        worst = max(rows, key=lambda x: x["summary_jaccard"])
        print(f"\n  Most effective:  {best['name']} (Jaccard={best['summary_jaccard']:.4f}, opacity={best['opacity']})")
        print(f"  Least effective: {worst['name']} (Jaccard={worst['summary_jaccard']:.4f}, opacity={worst['opacity']})")
        drop = (worst["summary_jaccard"] - best["summary_jaccard"]) / best["summary_jaccard"] * 100
        print(f"  Effectiveness drop from full to stealthiest: {drop:+.1f}% Jaccard increase")


# ===== Analysis 5: Keyword Penetration Across Attacks =====
def analysis_keyword_penetration(reports):
    """Compare how different attacks shift keyword frequencies."""
    print("\n" + "=" * 120)
    print("ANALYSIS 5: KEYWORD PENETRATION (top keywords with largest diff across attacks)")
    print("=" * 120)

    # Collect all keyword diffs across all attacks
    all_kw_diffs = {}  # keyword -> {attack_name: diff}

    for name, r in reports.items():
        kf = r.get("keyword_frequency", {})
        for category in ["positive", "negative"]:
            cat_data = kf.get(category, {})
            for kw, counts in cat_data.items():
                if kw not in all_kw_diffs:
                    all_kw_diffs[kw] = {"type": category}
                all_kw_diffs[kw][name] = counts.get("diff", 0)

    # Find keywords with largest total impact
    kw_impact = []
    for kw, diffs in all_kw_diffs.items():
        attack_diffs = {k: v for k, v in diffs.items() if k != "type"}
        total_abs_diff = sum(abs(v) for v in attack_diffs.values())
        kw_impact.append((kw, diffs["type"], total_abs_diff, attack_diffs))

    kw_impact.sort(key=lambda x: -x[2])

    attack_names = sorted(reports.keys())

    # Top 10 most impacted keywords
    print(f"\n{'Keyword':<15} {'Type':<9} ", end="")
    for name in attack_names:
        short = name.replace("inject_en_", "").replace("inject_", "").replace("negative_", "neg_")[:10]
        print(f"{short:>10}", end="")
    print()
    print("-" * (26 + 10 * len(attack_names)))

    for kw, ktype, total, diffs in kw_impact[:15]:
        print(f"{kw:<15} {ktype:<9} ", end="")
        for name in attack_names:
            d = diffs.get(name, 0)
            if d > 0:
                print(f"{'+' + str(d):>10}", end="")
            elif d < 0:
                print(f"{d:>10}", end="")
            else:
                print(f"{'·':>10}", end="")
        print()


# ===== Analysis 6: Summary vs Recommendation Correlation =====
def analysis_correlation(reports, impacts):
    """Analyze correlation between summary-level and recommendation-level impact."""
    print("\n" + "=" * 120)
    print("ANALYSIS 6: SUMMARY IMPACT vs RECOMMENDATION IMPACT CORRELATION")
    print("=" * 120)

    pairs = []
    for name in reports:
        if name not in impacts:
            continue
        sj = reports[name]["text_similarity"]["mean_jaccard"]
        pj = impacts[name]["preference_similarity"]["all_users"]["mean_jaccard"]
        pairs.append((name, sj, pj))

    if len(pairs) < 2:
        print("  Not enough data for correlation analysis.")
        return

    pairs.sort(key=lambda x: x[1])

    print(f"\n{'Attack':<28} {'SummaryJacc':>12} {'PrefJacc':>12} {'Gap':>8} {'Amplified?':>11}")
    print("-" * 75)
    for name, sj, pj in pairs:
        gap = pj - sj
        amplified = "YES" if pj < sj else "no"
        print(f"{name:<28} {sj:>12.4f} {pj:>12.4f} {gap:>+8.4f} {amplified:>11}")

    # Summary statistics
    summary_jaccards = [p[1] for p in pairs]
    pref_jaccards = [p[2] for p in pairs]
    mean_sj = sum(summary_jaccards) / len(summary_jaccards)
    mean_pj = sum(pref_jaccards) / len(pref_jaccards)

    # Pearson correlation (manual, no numpy dependency)
    n = len(pairs)
    sum_xy = sum(s * p for _, s, p in pairs)
    sum_x = sum(s for _, s, _ in pairs)
    sum_y = sum(p for _, _, p in pairs)
    sum_x2 = sum(s ** 2 for _, s, _ in pairs)
    sum_y2 = sum(p ** 2 for _, _, p in pairs)
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
    pearson_r = numerator / denominator if denominator > 0 else 0

    print(f"\n  Mean Summary Jaccard:     {mean_sj:.4f}")
    print(f"  Mean Preference Jaccard:  {mean_pj:.4f}")
    print(f"  Pearson correlation:      {pearson_r:.4f}")

    amplified_count = sum(1 for _, sj, pj in pairs if pj < sj)
    print(f"  Attacks where preference impact > summary impact: {amplified_count}/{len(pairs)}")

    if pearson_r > 0.7:
        print("  → Strong positive correlation: stronger summary attacks lead to stronger recommendation impact")
    elif pearson_r > 0.4:
        print("  → Moderate correlation: summary impact partially predicts recommendation impact")
    else:
        print("  → Weak correlation: recommendation impact is not simply proportional to summary changes")


# ===== Analysis 7: Per-Attack Detailed Card =====
def analysis_detail_cards(reports, impacts):
    """Print a detailed card for each attack."""
    print("\n" + "=" * 120)
    print("ANALYSIS 7: DETAILED ATTACK CARDS")
    print("=" * 120)

    for name in sorted(reports.keys()):
        r = reports[name]
        meta = ATTACK_META.get(name, {})
        ts = r["text_similarity"]
        vd = r["verbatim_detection"]
        la = r["length_analysis"]
        kf = r.get("keyword_frequency", {})

        print(f"\n┌─ {name} ─{'─' * (80 - len(name))}┐")
        print(f"│ Category: {meta.get('category', '?'):<15} Language: {meta.get('lang', '?'):<5} "
              f"Opacity: {meta.get('opacity', '?'):<5} FontRatio: {meta.get('font_ratio', '?')}")

        print(f"│ [Summary Level]")
        print(f"│   Jaccard: {ts['mean_jaccard']:.4f}   "
              f"Verbatim: {vd['full_match_rate_attacked']*100:.1f}%   "
              f"LenDiff: {la['mean_length_diff']:+.0f}")
        hc_pct = ts["num_highly_changed"] / ts["total"] * 100
        mc_pct = ts["num_moderately_changed"] / ts["total"] * 100
        bc_pct = ts["num_barely_changed"] / ts["total"] * 100
        print(f"│   HighlyChanged: {hc_pct:.1f}%   Moderate: {mc_pct:.1f}%   Barely: {bc_pct:.1f}%")

        # Top keyword changes
        top_pos = []
        for kw, counts in kf.get("positive", {}).items():
            if counts["diff"] > 10:
                top_pos.append((kw, counts["diff"]))
        top_neg = []
        for kw, counts in kf.get("negative", {}).items():
            if counts["diff"] > 10:
                top_neg.append((kw, counts["diff"]))
        top_pos.sort(key=lambda x: -x[1])
        top_neg.sort(key=lambda x: -x[1])

        if top_pos:
            kw_str = ", ".join(f"{kw}(+{d})" for kw, d in top_pos[:5])
            print(f"│   Top positive keywords: {kw_str}")
        if top_neg:
            kw_str = ", ".join(f"{kw}(+{d})" for kw, d in top_neg[:5])
            print(f"│   Top negative keywords: {kw_str}")

        if name in impacts:
            imp = impacts[name]
            ps = imp["preference_similarity"]["all_users"]
            print(f"│ [Recommendation Level]")
            print(f"│   PrefJaccard: {ps['mean_jaccard']:.4f}   "
                  f"HighlyChanged: {ps['highly_changed_lt_0.5']}   "
                  f"BarelyChanged: {ps['barely_changed_gte_0.8']}")
            if imp.get("keyword_in_preferences"):
                pref_kw = [(kw, c["diff"]) for kw, c in imp["keyword_in_preferences"].items() if abs(c["diff"]) > 5]
                pref_kw.sort(key=lambda x: -abs(x[1]))
                if pref_kw:
                    kw_str = ", ".join(f"{kw}({d:+d})" for kw, d in pref_kw[:5])
                    print(f"│   Preference keyword shifts: {kw_str}")

        print(f"└{'─' * 85}┘")


def save_analysis_json(reports, impacts, output_path):
    """Save structured analysis results to JSON."""
    result = {"attacks": {}}

    for name in sorted(reports.keys()):
        r = reports[name]
        meta = ATTACK_META.get(name, {})
        entry = {
            "metadata": meta,
            "summary_level": {
                "jaccard": r["text_similarity"]["mean_jaccard"],
                "verbatim_rate": r["verbatim_detection"]["full_match_rate_attacked"],
                "highly_changed_pct": r["text_similarity"]["num_highly_changed"] / r["text_similarity"]["total"],
                "length_diff": r["length_analysis"]["mean_length_diff"],
            },
        }
        if name in impacts:
            ps = impacts[name]["preference_similarity"]["all_users"]
            entry["recommendation_level"] = {
                "pref_jaccard": ps["mean_jaccard"],
                "highly_changed": ps["highly_changed_lt_0.5"],
                "barely_changed": ps["barely_changed_gte_0.8"],
            }
        result["attacks"][name] = entry

    # Rankings
    sorted_by_summary = sorted(result["attacks"].items(), key=lambda x: x[1]["summary_level"]["jaccard"])
    result["rankings"] = {
        "by_summary_effectiveness": [name for name, _ in sorted_by_summary],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nStructured analysis saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze all attack evaluation results")
    parser.add_argument("--results_dir", type=str, default="results/", help="Directory with report/impact JSONs")
    parser.add_argument("--output", type=str, default=None, help="Optional: save structured analysis to JSON")
    args = parser.parse_args()

    reports, impacts = load_all_results(args.results_dir)

    if not reports:
        print(f"No report_*.json found in {args.results_dir}")
        return

    print(f"Loaded {len(reports)} report(s) and {len(impacts)} recommendation impact(s)")
    print(f"Attacks: {', '.join(sorted(reports.keys()))}")

    analysis_summary_table(reports, impacts)
    analysis_by_category(reports, impacts)
    analysis_by_language(reports, impacts)
    analysis_stealth_tradeoff(reports)
    analysis_keyword_penetration(reports)
    analysis_correlation(reports, impacts)
    analysis_detail_cards(reports, impacts)

    if args.output:
        save_analysis_json(reports, impacts, args.output)


if __name__ == "__main__":
    main()
