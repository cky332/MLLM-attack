"""
Optimized batch inference: load the model ONCE, then process multiple attack directories.
Uses DataLoader with multi-worker prefetch so GPU never waits for CPU image loading.

Usage:
    # Process all attack dirs under a base directory
    python run_inference_batch.py \
        --attack_base_dir /path/to/attacked_covers \
        --output_dir results/ \
        --attacks promo_cn,inject_cn,inject_en,five_star,inject_cn_watermark

    # Dual-GPU: run on specific GPU(s)
    CUDA_VISIBLE_DEVICES=1 python run_inference_batch.py \
        --attack_base_dir /path/to/attacked_covers \
        --output_dir results/ \
        --attacks promo_cn,inject_cn

    # Also supports single-directory mode (drop-in replacement for run_inference.py)
    python run_inference_batch.py \
        --img_dir /path/to/images \
        --output_csv results/summary.csv
"""

import argparse
import os
import glob

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from torch.amp import autocast
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

PROMPT = (
    "[INST] <image>\n"
    "Please describe this image, which is a cover of a video."
    " Provide a detailed description in one continuous paragraph,"
    " including content information and visual features such as colors, objects, text,"
    " and any notable elements present in the image.[/INST]"
)


# ---------------------------------------------------------------------------
# Dataset & collate – enables DataLoader prefetch with num_workers
# ---------------------------------------------------------------------------

class ImageDataset(Dataset):
    """Loads images from disk. Used with DataLoader for parallel prefetch."""

    def __init__(self, image_paths, item_ids):
        self.image_paths = image_paths
        self.item_ids = item_ids

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        item_id = self.item_ids[idx]
        try:
            img = Image.open(path).convert("RGB")
            return img, item_id, True
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            # Return a tiny placeholder; will be filtered out
            return Image.new("RGB", (1, 1)), item_id, False


def collate_fn(batch):
    """Custom collate that keeps PIL images (not tensors)."""
    images, item_ids, valids = zip(*batch)
    return list(images), list(item_ids), list(valids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_images(img_dir):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(img_dir, ext)))
        paths.extend(glob.glob(os.path.join(img_dir, ext.upper())))
    paths = sorted(set(paths))
    item_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return paths, item_ids


def pad_batch_images(images):
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
                delta_w // 2, delta_h // 2,
                delta_w - (delta_w // 2), delta_h - (delta_h // 2),
            )
            padded.append(ImageOps.expand(img, border=padding, fill="black"))
    return padded


# ---------------------------------------------------------------------------
# Model loading (once)
# ---------------------------------------------------------------------------

def load_model(model_id, device):
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    load_kwargs = dict(
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )

    if num_gpus > 1:
        load_kwargs["device_map"] = "auto"
        print("Using device_map='auto' for multi-GPU inference")

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, **load_kwargs
    ).eval()

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
        model.to(device)

    processor = AutoProcessor.from_pretrained(model_id, return_tensors="pt")
    print(f"Model loaded, input device: {device}")
    return model, processor


# ---------------------------------------------------------------------------
# Inference on one directory
# ---------------------------------------------------------------------------

def run_inference_on_dir(img_dir, output_csv, model, processor, device,
                         batch_size=4, max_new_tokens=200, num_workers=4):
    """Run inference on a single image directory using the already-loaded model."""
    print(f"\n{'='*60}")
    print(f"Processing: {img_dir}")
    print(f"Output:     {output_csv}")

    image_paths, item_ids = collect_images(img_dir)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found. Skipping.")
        return

    dataset = ImageDataset(image_paths, item_ids)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
    )

    all_summaries = []
    all_item_ids = []
    processed = 0

    for batch_images, batch_ids, batch_valids in loader:
        # Filter out failed loads
        valid_images = []
        valid_ids = []
        for img, iid, ok in zip(batch_images, batch_ids, batch_valids):
            if ok:
                valid_images.append(img)
                valid_ids.append(iid)

        if not valid_images:
            continue

        padded_images = pad_batch_images(valid_images)

        model_inputs = processor(
            text=[PROMPT] * len(padded_images),
            images=padded_images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad(), autocast("cuda"):
            outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        summaries = [d.split("[/INST]")[-1].strip() for d in decoded]

        all_summaries.extend(summaries)
        all_item_ids.extend(valid_ids)

        del model_inputs, outputs, decoded
        torch.cuda.empty_cache()

        processed += len(valid_ids)
        if processed % 200 == 0 or processed == len(image_paths):
            print(f"  Progress: {processed}/{len(image_paths)}")

    df = pd.DataFrame({"item_id": all_item_ids, "summary": all_summaries})
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} summaries to {output_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimized batch inference: load model once, process multiple attack dirs")

    # Multi-attack mode
    parser.add_argument("--attack_base_dir", type=str, default=None,
                        help="Base dir containing attack subdirectories")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for CSVs")
    parser.add_argument("--attacks", type=str, default=None,
                        help="Comma-separated attack names (subdirs of attack_base_dir)")

    # Single-dir mode (backward compatible with run_inference.py)
    parser.add_argument("--img_dir", type=str, default=None,
                        help="Single image directory (backward compat mode)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Single output CSV (backward compat mode)")

    # Model & performance
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers for image prefetch")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CURL_CA_BUNDLE"] = ""

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id, args.device)

    # Single-dir backward-compatible mode
    if args.img_dir and args.output_csv:
        run_inference_on_dir(
            args.img_dir, args.output_csv, model, processor, args.device,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            num_workers=args.num_workers,
        )
        return

    # Multi-attack mode
    if not args.attack_base_dir:
        print("ERROR: Provide either --img_dir/--output_csv or --attack_base_dir/--attacks")
        return

    if args.attacks:
        attack_names = [a.strip() for a in args.attacks.split(",")]
    else:
        # Auto-detect subdirectories
        attack_names = sorted([
            d for d in os.listdir(args.attack_base_dir)
            if os.path.isdir(os.path.join(args.attack_base_dir, d))
        ])
        print(f"Auto-detected attack dirs: {attack_names}")

    for attack_name in attack_names:
        img_dir = os.path.join(args.attack_base_dir, attack_name)
        output_csv = os.path.join(args.output_dir, f"attacked_{attack_name}_summary.csv")

        if not os.path.isdir(img_dir):
            print(f"WARNING: {img_dir} not found, skipping '{attack_name}'")
            continue

        run_inference_on_dir(
            img_dir, output_csv, model, processor, args.device,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            num_workers=args.num_workers,
        )

    print(f"\n{'='*60}")
    print("All attacks processed.")


if __name__ == "__main__":
    main()
