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

import pandas as pd
import torch
from datasets import load_dataset
from multiprocess import set_start_method
from PIL import ImageOps
from torch.cuda.amp import autocast
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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3",
                        help="Comma-separated GPU IDs to use")
    return parser.parse_args()


def add_image_file_path(example):
    """Extract item_id from image filename."""
    file_path = example["image"].filename
    filename = os.path.splitext(os.path.basename(file_path))[0]
    example["item_id"] = filename
    return example


def main():
    args = parse_args()

    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    print(f"Loading model: {args.model_id}")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    ).eval()

    processor = AutoProcessor.from_pretrained(args.model_id, return_tensors="pt")

    print(f"Loading images from: {args.img_dir}")
    dataset = load_dataset("imagefolder", data_dir=args.img_dir)
    dataset = dataset.map(lambda x: add_image_file_path(x))
    print(dataset)

    def gpu_computation(batch, rank):
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        model.to(device)

        batch_images = batch["image"]

        # Pad images to same size within batch (same logic as image_summary.py)
        max_width = max(img.width for img in batch_images)
        max_height = max(img.height for img in batch_images)

        padded_images = []
        for img in batch_images:
            if img.width == max_width and img.height == max_height:
                padded_images.append(img)
            else:
                delta_width = max_width - img.width
                delta_height = max_height - img.height
                padding = (
                    delta_width // 2,
                    delta_height // 2,
                    delta_width - (delta_width // 2),
                    delta_height - (delta_height // 2),
                )
                new_img = ImageOps.expand(img, border=padding, fill="black")
                padded_images.append(new_img)

        batch["image"] = padded_images

        model_inputs = processor(
            text=[PROMPT] * len(batch["image"]),
            images=batch["image"],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad() and autocast():
            outputs = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens)

        ans = processor.batch_decode(outputs, skip_special_tokens=True)
        ans = [a.split("[/INST]")[1] for a in ans]
        return {"summary": ans}

    set_start_method("spawn", force=True)
    updated_dataset = dataset.map(
        gpu_computation,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True,
        num_proc=args.num_proc,
    )

    train_dataset = updated_dataset["train"]
    df = pd.DataFrame({
        "item_id": train_dataset["item_id"],
        "summary": train_dataset["summary"],
    })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} summaries to {args.output_csv}")


if __name__ == "__main__":
    main()
