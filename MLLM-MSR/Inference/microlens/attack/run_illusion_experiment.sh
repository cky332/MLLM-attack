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

# ---- Configurable paths (defaults match the real /home/chenkuiyun/MLLM-attack
#      layout confirmed by preflight_illusion.py; override if yours differs) ----
ROOT_DIR="/home/chenkuiyun/MLLM-attack"
SRC_IMG_DIR="$ROOT_DIR/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_covers"
TITLE_CSV="$ROOT_DIR/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_titles.csv"
PAIRS_CSV="$ROOT_DIR/MLLM-MSR/data/microlens/MicroLens-50k_pairs.csv"
TEST_PAIRS_CSV="$ROOT_DIR/MLLM-MSR/data/MicroLens-50k/Split/test_pairs.csv"
CLEAN_PREF="$ROOT_DIR/user_preference_recurrent.csv"
# Your ALREADY fine-tuned LoRA recommender (the one test_with_llava_sft.py loads).
# Leave empty ("") to score the base model instead.
PEFT_MODEL_ID="$ROOT_DIR/output/llava-v1.6-mistral-7b-hf-lora-recurrent-e4-r16"

# ---- Experiment parameters ----
EPS="${1:-16}"        # L_inf budget /255 (paper standard = 16; try 8 / 32 too)
ITERS="${2:-300}"     # PGD iterations
TOP_N="${3:-20}"      # # most-popular titles forming the target centroid
N_ITEMS="${4:-0}"     # limit # attacked items (0 = every item in TEST_PAIRS_CSV)
BATCH_SIZE="${5:-4}"
DEVICE="${6:-cuda:0}"
NUM_PROC="${7:-1}"    # set to #GPUs for multi-GPU LoRA scoring

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

# ── Step 0: Validate paths + environment before spending GPU time ──
echo "[Step 0] Preflight path/env check..."
python preflight_illusion.py --root "$ROOT_DIR" \
    --covers_dir "$SRC_IMG_DIR" --test_pairs_csv "$TEST_PAIRS_CSV" \
    --title_csv "$TITLE_CSV" --pairs_csv "$PAIRS_CSV" \
    --pref_csv "$CLEAN_PREF" --peft_model_id "$PEFT_MODEL_ID" \
    | tail -40 || { echo "Preflight failed — fix the paths above."; exit 1; }

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
# Set GPUS="0 1 2 3 4 5 6 7" to shard the attack across GPUs (one process each).
GPUS="${GPUS:-}"
if compgen -G "$ADV_DIR/manifest*.csv" > /dev/null; then
    echo "[Step 2] SKIP (adversarial images exist in $ADV_DIR)"
else
    echo "[Step 2] Generating adversarial cover images (PGD on CLIP)..."
    if [[ -n "$GPUS" ]]; then
        NSH=$(echo $GPUS | wc -w); i=0
        for g in $GPUS; do
            CUDA_VISIBLE_DEVICES=$g python illusion_attack.py generate \
                --src_dir "$SRC_IMG_DIR" --out_dir "$ADV_DIR" \
                --clean_resized_dir "$CLEAN_RESIZED_DIR" --target "$TARGET_NPZ" \
                --items_csv "$TEST_PAIRS_CSV" --max_items "$N_ITEMS" \
                --eps "$EPS" --alpha 1 --iters "$ITERS" --batch_size "$BATCH_SIZE" \
                --num_shards "$NSH" --shard_id "$i" --device cuda:0 &
            i=$((i + 1))
        done
        wait
    else
        python illusion_attack.py generate \
            --src_dir "$SRC_IMG_DIR" --out_dir "$ADV_DIR" \
            --clean_resized_dir "$CLEAN_RESIZED_DIR" --target "$TARGET_NPZ" \
            --items_csv "$TEST_PAIRS_CSV" --max_items "$N_ITEMS" \
            --eps "$EPS" --alpha 1 --iters "$ITERS" --batch_size "$BATCH_SIZE" \
            --device "$DEVICE"
    fi
    echo "[Step 2] Merged embedding-level ASR:"
    python illusion_attack.py embed_asr --manifest "$ADV_DIR" || true
fi

# ── Step 3: Final re-ranking with your FINE-TUNED LoRA recommender [GPU] ──
# Reuses your trained adapter + generated preferences; only re-scores the final
# Yes/No judgment on clean vs adversarial images. No retraining, no pref regen.
if [[ -f "$RECSYS_REPORT" ]]; then
    echo "[Step 3] SKIP (recsys ASR exists: $RECSYS_REPORT)"
else
    echo "[Step 3] Re-scoring final ranking with fine-tuned LoRA (clean vs adversarial)..."
    python eval_illusion_sft.py \
        --peft_model_id      "$PEFT_MODEL_ID" \
        --test_pairs_csv     "$TEST_PAIRS_CSV" \
        --clean_image_dir    "$CLEAN_RESIZED_DIR" \
        --attacked_image_dir "$ADV_DIR" \
        --title_csv          "$TITLE_CSV" \
        --pref_csv           "$CLEAN_PREF" \
        --attack_name        "illusion_eps${EPS}_it${ITERS}" \
        --output_report      "$RECSYS_REPORT" \
        --candidates_per_user 21 --topk 10 \
        --batch_size "$BATCH_SIZE" --num_proc "$NUM_PROC"
fi

echo ""
echo "============================================================"
echo "ILLUSION EXPERIMENT COMPLETE — results in: $RUN/"
echo "  popular_target.npz   — popular-text target embedding"
echo "  images/manifest.csv  — per-image clean/adv cosine alignment + L_inf"
echo "  images/summary.json  — embedding-level ASR (paper Table-1 style)"
echo "  recsys_asr.json      — decision-flip / rank-promotion ASR on LLaVA"
echo "============================================================"
