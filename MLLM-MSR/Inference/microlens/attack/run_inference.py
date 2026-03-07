"""
Run LLaVA image summarization inference on a given image directory.
Reuses the same logic as image_summary.py but with configurable paths via CLI args.

Usage:
    python run_inference.py \
        --img_dir /path/to/images \
        --output_csv results/clean_summary.csv

    python run_inference.py \
        --img_dir /path/to/attacked_images/inject_cn \
        --output_csv results/attacked_inject_cn_summary.csv
"""

import argparse
import os
import glob

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.amp import autocast
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# Prompt must match the original image_summary.py exactly for comparable results
PROMPT = (
    "[INST] <image>\n"
    "Please describe this image, which is a cover of a video."
    " Provide a detailed description in one continuous paragraph,"
    " including content information and visual features such as colors, objects, text,"
    " and any notable elements present in the image.[/INST]"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaVA image summarization inference")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf",
                        help="HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use, e.g. cuda:0")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs (sets CUDA_VISIBLE_DEVICES)")
    return parser.parse_args()


def collect_images(img_dir):
    """Collect all image file paths and their item_ids."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(img_dir, ext)))
        paths.extend(glob.glob(os.path.join(img_dir, ext.upper())))
    paths = sorted(set(paths))
    item_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return paths, item_ids


def pad_batch_images(images):
    """Pad images to the same size within a batch (same logic as image_summary.py)."""
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    padded = []
    for img in images:
        if img.width == max_width and img.height == max_height:
            padded.append(img)
        else:
            delta_w = max_width - img.width
            delta_h = max_height - img.height
            padding = (
                delta_w // 2,
                delta_h // 2,
                delta_w - (delta_w // 2),
                delta_h - (delta_h // 2),
            )
            padded.append(ImageOps.expand(img, border=padding, fill="black"))
    return padded


def main():
    args = parse_args()

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ["CURL_CA_BUNDLE"] = ""

    # Load model
    print(f"Loading model: {args.model_id}")

    # Detect number of visible GPUs for automatic multi-GPU support
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    load_kwargs = dict(
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )

    if num_gpus > 1:
        # Spread model across multiple GPUs automatically
        load_kwargs["device_map"] = "auto"
        print("Using device_map='auto' for multi-GPU inference")

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id, **load_kwargs
    ).eval()

    # Ensure vision tower returns hidden_states (required by newer transformers).
    # model.config.vision_config and model.vision_tower.config are separate objects.
    for cfg in [model.config]:
        cfg.output_hidden_states = True
        if hasattr(cfg, "vision_config"):
            cfg.vision_config.output_hidden_states = True
        if hasattr(cfg, "text_config"):
            cfg.text_config.output_hidden_states = True
    if hasattr(model, "vision_tower"):
        model.vision_tower.config.output_hidden_states = True
        if hasattr(model.vision_tower, "vision_model"):
            model.vision_tower.vision_model.config.output_hidden_states = True

    if num_gpus <= 1:
        device = args.device
        model.to(device)
    else:
        # With device_map="auto", model is already distributed; use first device for inputs
        device = args.device
    print(f"Model loaded, input device: {device}")

    processor = AutoProcessor.from_pretrained(args.model_id, return_tensors="pt")

    # Collect images
    print(f"Loading images from: {args.img_dir}")
    image_paths, item_ids = collect_images(args.img_dir)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found. Exiting.")
        return

    # Process in batches
    all_summaries = []
    all_item_ids = []
    total = len(image_paths)

    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch_paths = image_paths[batch_start:batch_end]
        batch_ids = item_ids[batch_start:batch_end]

        # Load and pad images
        batch_images = []
        valid_ids = []
        for path, iid in zip(batch_paths, batch_ids):
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                valid_ids.append(iid)
            except Exception as e:
                print(f"  SKIP {path}: {e}")

        if not batch_images:
            continue

        padded_images = pad_batch_images(batch_images)

        model_inputs = processor(
            text=[PROMPT] * len(padded_images),
            images=padded_images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad(), autocast("cuda"):
            outputs = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens)

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        summaries = [d.split("[/INST]")[-1].strip() for d in decoded]

        all_summaries.extend(summaries)
        all_item_ids.extend(valid_ids)

        # Free GPU memory between batches
        del model_inputs, outputs, decoded
        torch.cuda.empty_cache()

        if (batch_start // args.batch_size + 1) % 50 == 0 or batch_end == total:
            print(f"  Progress: {batch_end}/{total}")

    # Save results
    df = pd.DataFrame({"item_id": all_item_ids, "summary": all_summaries})
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} summaries to {args.output_csv}")


if __name__ == "__main__":
    main()
