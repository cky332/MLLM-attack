#!/bin/bash
# ===========================================================================
# run_targeted_experiment.sh  —  End-to-end Targeted User-Group Push pipeline
#
# Task 2: Inject cluster-specific keywords into target items' images
#         → push those items to users with matching preference profiles
#
# Steps:
#   1. Analyze user preference clusters                          [CPU]
#   2. Generate targeted attack text using cluster keywords      [CPU]
#   3. Generate attacked images for target items                 [CPU]
#   4. Run LLaVA image summary on attacked images                [GPU]
#   5. Replace summaries + run Llama-3 preference inference      [GPU]
#   6. Run LLaVA top-K ranking with user-group breakdown         [GPU]
#   7. Evaluate targeted push effectiveness                      [CPU]
# ===========================================================================
set -euo pipefail

# ---- Configurable paths (EDIT THESE) ----
SRC_IMG_DIR="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_covers"
TITLE_CSV="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_titles.csv"
USER_ITEMS_TSV="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/MicroLens-50k/Split/user_items_negs.tsv"
TEST_PAIRS_CSV="/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/MicroLens-50k/Split/test_pairs.csv"
CLEAN_SUMMARY="/home/chenkuiyun/MLLM-attack/image_summary.csv"
CLEAN_PREF="/home/chenkuiyun/MLLM-attack/user_preference_direct.csv"

# ---- Experiment parameters ----
TARGET_KEYWORD="${1:-gaming}"           # Target user cluster keyword
TARGET_ITEMS="${2:-}"                   # Comma-separated item IDs (auto-selected if empty)
N_CLUSTERS="${3:-5}"
BATCH_SIZE="${4:-4}"

# ---- Derived paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES="$SCRIPT_DIR/results"
TARGETED_DIR="$RES/targeted_${TARGET_KEYWORD}"
CLUSTERS_REPORT="$TARGETED_DIR/user_clusters.json"
ATTACK_CONFIG="$TARGETED_DIR/attack_config.json"
ATTACKED_IMG_DIR="$TARGETED_DIR/images"
ATTACKED_SUMMARY="$TARGETED_DIR/attacked_summary.csv"
MERGED_SUMMARY="$TARGETED_DIR/image_summary_merged.csv"
ATTACKED_PREF="$TARGETED_DIR/user_preference_attacked.csv"
TOPK_REPORT="$TARGETED_DIR/topk_ranking_report.json"
TARGETED_EVAL_REPORT="$TARGETED_DIR/targeted_eval_report.json"

mkdir -p "$TARGETED_DIR"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "TARGETED PUSH EXPERIMENT: keyword=$TARGET_KEYWORD"
echo "============================================================"

# ── Step 1: Analyze user preference clusters [CPU] ──
if [[ -f "$CLUSTERS_REPORT" ]]; then
    echo "[Step 1] SKIP (user clusters exist)"
else
    echo "[Step 1] Analyzing user preference clusters..."
    python targeted_recommendation.py analyze_users \
        --user_preference_csv "$CLEAN_PREF" \
        --output_report       "$CLUSTERS_REPORT" \
        --n_clusters          "$N_CLUSTERS"
fi

# ── Step 2: Auto-select target items if not provided ──
if [[ -z "$TARGET_ITEMS" ]]; then
    echo "[Step 2a] Auto-selecting target items unrelated to '$TARGET_KEYWORD'..."
    TARGET_ITEMS=$(python3 -c "
import pandas as pd, random
random.seed(42)
df = pd.read_csv('$TITLE_CSV')
df.columns = [c.strip().lower() for c in df.columns]
# Pick items whose titles do NOT contain the target keyword
unrelated = df[~df['title'].str.lower().str.contains('$TARGET_KEYWORD', na=False)]
if len(unrelated) >= 20:
    sample = unrelated.sample(20, random_state=42)
else:
    sample = unrelated
print(','.join(sample['item'].astype(str).tolist()))
")
    echo "  Selected items: ${TARGET_ITEMS:0:80}..."
fi

# ── Step 3: Generate targeted attack text [CPU] ──
if [[ -f "$ATTACK_CONFIG" ]]; then
    echo "[Step 3] SKIP (attack config exists)"
else
    echo "[Step 3] Generating targeted attack text..."
    python targeted_recommendation.py generate_attacks \
        --user_preference_csv    "$CLEAN_PREF" \
        --title_csv              "$TITLE_CSV" \
        --target_items           "$TARGET_ITEMS" \
        --target_cluster_keyword "$TARGET_KEYWORD" \
        --output_config          "$ATTACK_CONFIG"
fi

# ── Step 4: Generate attacked images [CPU] ──
if [[ -d "$ATTACKED_IMG_DIR" && $(ls "$ATTACKED_IMG_DIR" 2>/dev/null | head -1) ]]; then
    echo "[Step 4] SKIP (attacked images exist)"
else
    echo "[Step 4] Generating attacked images for target items..."
    mkdir -p "$ATTACKED_IMG_DIR"
    python3 -c "
import json, os, sys
sys.path.insert(0, '.')
from generate_attacked_images import overlay_text
from attack_config import TEXT_STYLES
from PIL import Image

with open('$ATTACK_CONFIG') as f:
    config = json.load(f)

style = TEXT_STYLES['bold_white']
src_dir = '$SRC_IMG_DIR'
dst_dir = '$ATTACKED_IMG_DIR'
success, fail = 0, 0

for item_id, info in config['target_items'].items():
    text = info['attack_text']
    src_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        p = os.path.join(src_dir, f'{item_id}{ext}')
        if os.path.exists(p):
            src_path = p
            break
    if not src_path:
        fail += 1
        continue
    try:
        img = Image.open(src_path).convert('RGB')
        attacked = overlay_text(img, text, 'center', style)
        attacked.save(os.path.join(dst_dir, os.path.basename(src_path)), quality=95)
        success += 1
    except Exception as e:
        print(f'  ERROR {item_id}: {e}')
        fail += 1

print(f'Generated {success} attacked images, {fail} failed')
"
fi

# ── Step 5: Run LLaVA image summary on attacked images [GPU] ──
if [[ -f "$ATTACKED_SUMMARY" ]]; then
    echo "[Step 5] SKIP (attacked summary exists)"
else
    echo "[Step 5] Running LLaVA image summary..."
    python run_inference.py \
        --img_dir    "$ATTACKED_IMG_DIR" \
        --output_csv "$ATTACKED_SUMMARY" \
        --batch_size "$BATCH_SIZE"
fi

# ── Step 6: Merge summaries + preference inference [GPU] ──
if [[ -f "$ATTACKED_PREF" ]]; then
    echo "[Step 6] SKIP (attacked preferences exist)"
else
    echo "[Step 6a] Merging attacked summaries into clean summary..."
    python eval_recommendation_impact.py replace_summaries \
        --clean_summary_csv    "$CLEAN_SUMMARY" \
        --attacked_summary_csv "$ATTACKED_SUMMARY" \
        --output_csv           "$MERGED_SUMMARY"

    echo "[Step 6b] Running Llama-3 preference inference..."
    python eval_recommendation_impact.py run_preference \
        --visual_csv    "$MERGED_SUMMARY" \
        --title_csv     "$TITLE_CSV" \
        --user_items_tsv "$USER_ITEMS_TSV" \
        --output_csv    "$ATTACKED_PREF" \
        --batch_size    12
fi

# ── Step 7: Run top-K ranking evaluation [GPU] ──
if [[ -f "$TOPK_REPORT" ]]; then
    echo "[Step 7] SKIP (topk report exists)"
else
    echo "[Step 7] Running top-K ranking evaluation..."
    python eval_topk_ranking.py \
        --test_pairs_csv       "$TEST_PAIRS_CSV" \
        --image_dir            "$SRC_IMG_DIR" \
        --title_csv            "$TITLE_CSV" \
        --clean_pref_csv       "$CLEAN_PREF" \
        --attacked_pref_csv    "$ATTACKED_PREF" \
        --attacked_summary_csv "$ATTACKED_SUMMARY" \
        --attack_name          "targeted_${TARGET_KEYWORD}" \
        --output_report        "$TOPK_REPORT" \
        --target_cluster_keyword "$TARGET_KEYWORD" \
        --candidates_per_user 21 --topk 10 \
        --batch_size "$BATCH_SIZE" --num_proc 1
fi

# ── Step 8: Evaluate targeted push effectiveness [CPU] ──
if [[ -f "$TARGETED_EVAL_REPORT" ]]; then
    echo "[Step 8] SKIP (targeted eval report exists)"
else
    echo "[Step 8] Evaluating targeted push effectiveness..."
    python targeted_recommendation.py evaluate \
        --clean_preference_csv    "$CLEAN_PREF" \
        --attacked_preference_csv "$ATTACKED_PREF" \
        --user_items_tsv          "$USER_ITEMS_TSV" \
        --target_items            "$TARGET_ITEMS" \
        --target_cluster_keyword  "$TARGET_KEYWORD" \
        --output_report           "$TARGETED_EVAL_REPORT"
fi

echo ""
echo "============================================================"
echo "TARGETED PUSH EXPERIMENT COMPLETE"
echo "Results in: $TARGETED_DIR/"
echo "  user_clusters.json        — user preference clusters"
echo "  attack_config.json        — generated attack text"
echo "  topk_ranking_report.json  — real top-K ranking impact"
echo "  targeted_eval_report.json — targeted push evaluation"
echo "============================================================"
