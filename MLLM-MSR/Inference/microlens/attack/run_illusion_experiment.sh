#!/bin/bash
# ===========================================================================
# run_illusion_experiment.sh — End-to-end adversarial-illusion attack on MLLM-MSR
#
# Idea (per Zhang et al., "Adversarial Illusions in Multi-Modal Embeddings",
# USENIX Security 2025): perturb a candidate item's COVER IMAGE with an
# imperceptible L_inf perturbation so its CLIP embedding aligns with POPULAR
# text. LLaVA (MLLM-MSR's recommender) then "sees" popular content and is more
# likely to predict the user will interact -> the item gets recommended.
#
# Steps:
#   1. Build the popular-text target embedding (popularity x titles)   [GPU*]
#   2. Generate adversarial cover images + resized-clean baseline      [GPU]
#      (also reports embedding-level ASR, paper Table-1 style)
#   3. Recommendation-level ASR: decision-flip / P(Yes) lift / rank    [GPU]
#      promotion / Recall@K / NDCG@K on the real LLaVA Yes/No judgment
#
#   *Step 1 only needs CLIP's text encoder; runs on a single GPU quickly.
# ===========================================================================
set -euo pipefail

# ---- Configurable paths (EDIT THESE) ----
SRC_IMG_DIR="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_covers"
TITLE_CSV="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/microlens/MicroLens-50k_titles.csv"
PAIRS_CSV="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/microlens/MicroLens-50k_pairs.csv"
TEST_PAIRS_CSV="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/MicroLens-50k/Split/test_pairs.csv"
CLEAN_PREF="/home/chenkuiyun/MLLM-attack/user_preference_recurrent.csv"

# ---- Experiment parameters ----
EPS="${1:-16}"        # L_inf budget /255 (paper standard = 16; try 8 / 32 too)
ITERS="${2:-300}"     # PGD iterations
TOP_N="${3:-20}"      # # most-popular titles forming the target centroid
N_ITEMS="${4:-0}"     # limit # attacked items (0 = every item in TEST_PAIRS_CSV)
BATCH_SIZE="${5:-12}"
DEVICE="${6:-cuda:0}"

# ---- Derived paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN="$SCRIPT_DIR/results/illusion_eps${EPS}_it${ITERS}_top${TOP_N}"
TARGET_NPZ="$RUN/popular_target.npz"
ADV_DIR="$RUN/images"
CLEAN_RESIZED_DIR="$RUN/clean_resized"
RECSYS_REPORT="$RUN/recsys_asr.json"

mkdir -p "$RUN"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "ILLUSION EXPERIMENT  eps=${EPS}/255  iters=${ITERS}  top_n=${TOP_N}"
echo "  results -> $RUN"
echo "============================================================"

# ── Step 1: Build popular-text target [GPU: CLIP text encoder] ──
if [[ -f "$TARGET_NPZ" ]]; then
    echo "[Step 1] SKIP (target exists: $TARGET_NPZ)"
else
    echo "[Step 1] Building popular-text target embedding..."
    python illusion_attack.py build_target \
        --pairs_csv  "$PAIRS_CSV" \
        --title_csv  "$TITLE_CSV" \
        --top_n      "$TOP_N" \
        --out_target "$TARGET_NPZ" \
        --device     "$DEVICE"
    # Alternative: a single hand-written popular target instead of popularity:
    #   --target_text "A trending viral video everyone is watching and loves"
fi

# ── Step 2: Generate adversarial cover images [GPU: CLIP image encoder] ──
if [[ -f "$ADV_DIR/manifest.csv" ]]; then
    echo "[Step 2] SKIP (adversarial images exist: $ADV_DIR)"
else
    echo "[Step 2] Generating adversarial cover images (PGD on CLIP)..."
    python illusion_attack.py generate \
        --src_dir           "$SRC_IMG_DIR" \
        --out_dir           "$ADV_DIR" \
        --clean_resized_dir "$CLEAN_RESIZED_DIR" \
        --target            "$TARGET_NPZ" \
        --items_csv         "$TEST_PAIRS_CSV" \
        --max_items         "$N_ITEMS" \
        --eps "$EPS" --alpha 1 --iters "$ITERS" \
        --batch_size "$BATCH_SIZE" --device "$DEVICE"
fi

# ── Step 3: Recommendation-level ASR [GPU: LLaVA Yes/No judgment] ──
if [[ -f "$RECSYS_REPORT" ]]; then
    echo "[Step 3] SKIP (recsys ASR exists: $RECSYS_REPORT)"
else
    echo "[Step 3] Measuring recommendation-level ASR (clean vs adversarial image)..."
    python eval_illusion_ranking.py \
        --test_pairs_csv     "$TEST_PAIRS_CSV" \
        --clean_image_dir    "$CLEAN_RESIZED_DIR" \
        --attacked_image_dir "$ADV_DIR" \
        --title_csv          "$TITLE_CSV" \
        --pref_csv           "$CLEAN_PREF" \
        --attack_name        "illusion_eps${EPS}_it${ITERS}" \
        --output_report      "$RECSYS_REPORT" \
        --candidates_per_user 21 --topk 10 \
        --batch_size "$BATCH_SIZE"
fi

echo ""
echo "============================================================"
echo "ILLUSION EXPERIMENT COMPLETE — results in: $RUN/"
echo "  popular_target.npz   — popular-text target embedding"
echo "  images/manifest.csv  — per-image clean/adv cosine alignment + L_inf"
echo "  images/summary.json  — embedding-level ASR (paper Table-1 style)"
echo "  recsys_asr.json      — decision-flip / rank-promotion ASR on LLaVA"
echo "============================================================"
