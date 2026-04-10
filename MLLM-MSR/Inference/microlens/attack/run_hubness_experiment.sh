#!/bin/bash
# ===========================================================================
# run_hubness_experiment.sh  —  End-to-end Adversarial Hubness pipeline
#
# Task 1: Inject item B's description into item A's image
#         → make A become top-1 for users who liked B
#
# Steps:
#   1. Generate (source A, target B) pairs + attacked images     [CPU]
#   2. Run LLaVA image summary on attacked images                [GPU]
#   3. Replace summaries + run Llama-3 preference inference      [GPU]
#   4. Run LLaVA top-K ranking on test pairs                     [GPU]
#   5. Evaluate hubness-specific metrics                         [CPU]
# ===========================================================================
set -euo pipefail

# ---- Configurable paths (EDIT THESE) ----
SRC_IMG_DIR="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_covers"
TITLE_CSV="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_titles.csv"
PAIRS_CSV="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/microlens/MicroLens-50k_pairs.csv"
USER_ITEMS_TSV="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/MicroLens-50k/Split/user_items_negs.tsv"
TEST_PAIRS_CSV="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/MicroLens-50k/Split/test_pairs.csv"
CLEAN_SUMMARY="/home/chenkuiyun/MLLM-attack/image_summary.csv"
CLEAN_PREF="/home/chenkuiyun/MLLM-attack/user_preference_direct.csv"

# ---- Experiment parameters ----
STRATEGY="${1:-hubness_popular}"   # hubness_popular | hubness_random | hubness_cross
N_TARGETS="${2:-100}"              # Number of source items to attack
STYLE="${3:-bold_white}"
BATCH_SIZE="${4:-4}"

# ---- Derived paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES="$SCRIPT_DIR/results"
HUBNESS_DIR="$RES/hubness_${STRATEGY}"
ATTACKED_IMG_DIR="$HUBNESS_DIR/images"
PAIR_MAPPING="$HUBNESS_DIR/pair_mapping.json"
ATTACKED_SUMMARY="$HUBNESS_DIR/attacked_summary.csv"
MERGED_SUMMARY="$HUBNESS_DIR/image_summary_merged.csv"
ATTACKED_PREF="$HUBNESS_DIR/user_preference_attacked.csv"
TOPK_REPORT="$HUBNESS_DIR/topk_ranking_report.json"
HUBNESS_REPORT="$HUBNESS_DIR/hubness_report.json"
HUBNESS_RANKING_REPORT="$HUBNESS_DIR/hubness_ranking_report.json"

mkdir -p "$HUBNESS_DIR"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "HUBNESS EXPERIMENT: strategy=$STRATEGY, n_targets=$N_TARGETS"
echo "============================================================"

# ── Step 1: Generate attack pairs + images [CPU] ──
if [[ -f "$PAIR_MAPPING" ]]; then
    echo "[Step 1] SKIP (pair_mapping exists: $PAIR_MAPPING)"
else
    echo "[Step 1] Generating hubness attack images..."
    python generate_hubness_attack.py generate \
        --src_dir    "$SRC_IMG_DIR" \
        --output_dir "$ATTACKED_IMG_DIR" \
        --title_csv  "$TITLE_CSV" \
        --pairs_csv  "$PAIRS_CSV" \
        --strategy   "$STRATEGY" \
        --n_targets  "$N_TARGETS" \
        --style      "$STYLE"

    # generate_hubness_attack.py saves pair_mapping one level up from output_dir
    # Move it to our canonical location
    GENERATED_MAPPING="$(dirname "$ATTACKED_IMG_DIR")/${STRATEGY}_pairs.json"
    if [[ -f "$GENERATED_MAPPING" && "$GENERATED_MAPPING" != "$PAIR_MAPPING" ]]; then
        mv "$GENERATED_MAPPING" "$PAIR_MAPPING"
    fi
fi

# ── Step 2: Run LLaVA image summary on attacked images [GPU] ──
if [[ -f "$ATTACKED_SUMMARY" ]]; then
    echo "[Step 2] SKIP (attacked summary exists)"
else
    echo "[Step 2] Running LLaVA image summary on attacked images..."
    python run_inference.py \
        --img_dir    "$ATTACKED_IMG_DIR" \
        --output_csv "$ATTACKED_SUMMARY" \
        --batch_size "$BATCH_SIZE"
fi

# ── Step 3: Merge summaries + run preference inference [GPU] ──
if [[ -f "$ATTACKED_PREF" ]]; then
    echo "[Step 3] SKIP (attacked preferences exist)"
else
    echo "[Step 3a] Merging attacked summaries into clean summary..."
    python eval_recommendation_impact.py replace_summaries \
        --clean_summary_csv    "$CLEAN_SUMMARY" \
        --attacked_summary_csv "$ATTACKED_SUMMARY" \
        --output_csv           "$MERGED_SUMMARY"

    echo "[Step 3b] Running Llama-3 preference inference..."
    python eval_recommendation_impact.py run_preference \
        --visual_csv    "$MERGED_SUMMARY" \
        --title_csv     "$TITLE_CSV" \
        --user_items_tsv "$USER_ITEMS_TSV" \
        --output_csv    "$ATTACKED_PREF" \
        --batch_size    12
fi

# ── Step 4: Run top-K ranking evaluation [GPU] ──
if [[ -f "$TOPK_REPORT" ]]; then
    echo "[Step 4] SKIP (topk report exists)"
else
    echo "[Step 4] Running top-K ranking evaluation..."
    python eval_topk_ranking.py \
        --test_pairs_csv       "$TEST_PAIRS_CSV" \
        --image_dir            "$SRC_IMG_DIR" \
        --title_csv            "$TITLE_CSV" \
        --clean_pref_csv       "$CLEAN_PREF" \
        --attacked_pref_csv    "$ATTACKED_PREF" \
        --attacked_summary_csv "$ATTACKED_SUMMARY" \
        --attack_name          "hubness_${STRATEGY}" \
        --output_report        "$TOPK_REPORT" \
        --candidates_per_user 21 --topk 10 \
        --batch_size "$BATCH_SIZE" --num_proc 1
fi

# ── Step 5: Evaluate hubness text-level metrics [CPU] ──
if [[ -f "$HUBNESS_REPORT" ]]; then
    echo "[Step 5] SKIP (hubness report exists)"
else
    echo "[Step 5] Evaluating hubness attack effectiveness..."
    python generate_hubness_attack.py evaluate \
        --clean_summary_csv    "$CLEAN_SUMMARY" \
        --attacked_summary_csv "$ATTACKED_SUMMARY" \
        --pair_mapping         "$PAIR_MAPPING" \
        --output_report        "$HUBNESS_REPORT"
fi

# ── Step 6: Hubness-specific ranking evaluation [GPU] ──
# Scores (B-user, source_A) pairs to measure if A is pushed toward B's users
if [[ -f "$HUBNESS_RANKING_REPORT" ]]; then
    echo "[Step 6] SKIP (hubness ranking report exists)"
else
    echo "[Step 6] Evaluating hubness via LLaVA scoring (B-user, source_A)..."
    python eval_hubness_ranking.py \
        --pair_mapping         "$PAIR_MAPPING" \
        --image_dir            "$SRC_IMG_DIR" \
        --title_csv            "$TITLE_CSV" \
        --clean_pref_csv       "$CLEAN_PREF" \
        --attacked_pref_csv    "$ATTACKED_PREF" \
        --user_items_tsv       "$USER_ITEMS_TSV" \
        --output_report        "$HUBNESS_RANKING_REPORT" \
        --max_users_per_pair 50 \
        --batch_size "$BATCH_SIZE"
fi

echo ""
echo "============================================================"
echo "HUBNESS EXPERIMENT COMPLETE"
echo "Results in: $HUBNESS_DIR/"
echo "  pair_mapping.json              — source→target pairs"
echo "  hubness_report.json            — text-level hubness gain"
echo "  topk_ranking_report.json       — global top-K ranking impact"
echo "  hubness_ranking_report.json    — hubness-specific: A's score among B-users"
echo "============================================================"
