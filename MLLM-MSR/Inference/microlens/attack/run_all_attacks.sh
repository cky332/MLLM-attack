#!/bin/bash
# =============================================================================
# Run all attack inference efficiently.
#
# Mode 1 (Recommended): Single-process, model loaded once, all attacks sequential
#   ./run_all_attacks.sh single
#
# Mode 2: Dual-GPU parallel — split attacks across 2 GPUs
#   ./run_all_attacks.sh parallel
#
# Mode 3: Legacy — one python process per attack (slow, reloads model each time)
#   ./run_all_attacks.sh legacy
# =============================================================================

set -e

# ---- Configuration (edit these) ----
ATTACK_BASE_DIR="${ATTACK_BASE_DIR:-/home/chenkuiyun/MLLM-attack/MLLM-MSR/data/attacked_covers}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_A="${GPU_A:-0}"          # First GPU id (for parallel mode)
GPU_B="${GPU_B:-1}"          # Second GPU id (for parallel mode)

ALL_ATTACKS="promo_cn,inject_cn,inject_en,five_star,inject_cn_watermark"
# Split for parallel mode: group A on GPU_A, group B on GPU_B
ATTACKS_A="promo_cn,inject_cn,inject_en"
ATTACKS_B="five_star,inject_cn_watermark"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-single}"

case "$MODE" in
  single)
    echo "=== Mode: single process, model loaded once ==="
    echo "=== Batch size: $BATCH_SIZE, Workers: $NUM_WORKERS ==="
    CUDA_VISIBLE_DEVICES=$GPU_A python run_inference_batch.py \
        --attack_base_dir "$ATTACK_BASE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --attacks "$ALL_ATTACKS" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --device cuda:0
    ;;

  parallel)
    echo "=== Mode: dual-GPU parallel (GPU $GPU_A + GPU $GPU_B) ==="
    echo "=== Batch size: $BATCH_SIZE, Workers: $NUM_WORKERS ==="

    # Launch group A on GPU_A in background
    CUDA_VISIBLE_DEVICES=$GPU_A python run_inference_batch.py \
        --attack_base_dir "$ATTACK_BASE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --attacks "$ATTACKS_A" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --device cuda:0 &
    PID_A=$!

    # Launch group B on GPU_B
    CUDA_VISIBLE_DEVICES=$GPU_B python run_inference_batch.py \
        --attack_base_dir "$ATTACK_BASE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --attacks "$ATTACKS_B" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --device cuda:0 &
    PID_B=$!

    echo "GPU $GPU_A (PID $PID_A): $ATTACKS_A"
    echo "GPU $GPU_B (PID $PID_B): $ATTACKS_B"
    echo "Waiting for both to finish..."

    wait $PID_A
    echo "GPU $GPU_A done."
    wait $PID_B
    echo "GPU $GPU_B done."
    ;;

  legacy)
    echo "=== Mode: legacy (one process per attack, slow) ==="
    for attack in promo_cn inject_cn inject_en five_star inject_cn_watermark; do
        CUDA_VISIBLE_DEVICES=$GPU_A python run_inference.py \
            --img_dir "$ATTACK_BASE_DIR/${attack}" \
            --output_csv "$OUTPUT_DIR/attacked_${attack}_summary.csv" \
            --device cuda:0 \
            --batch_size "$BATCH_SIZE"
    done
    ;;

  *)
    echo "Usage: $0 {single|parallel|legacy}"
    echo "  single   - (Recommended) Load model once, process all attacks"
    echo "  parallel - Dual-GPU: split attacks across 2 GPUs"
    echo "  legacy   - One process per attack (reloads model each time)"
    exit 1
    ;;
esac

echo ""
echo "=== All done. Results in: $OUTPUT_DIR ==="
