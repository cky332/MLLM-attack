"""
Generate attacked images by overlaying prompt injection text onto product/video cover images.

Usage:
    python generate_attacked_images.py \
        --src_dir /path/to/original_covers \
        --output_base_dir /path/to/attacked_images \
        --attacks inject_cn,inject_en,promo_cn

    # Use --attacks all to generate all configured attacks
    python generate_attacked_images.py \
        --src_dir /path/to/original_covers \
        --output_base_dir /path/to/attacked_images \
        --attacks all
"""

import argparse
import os
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont

from attack_config import (
    ATTACK_CONFIGS,
    ATTACK_TEXTS,
    CJK_FONT_PATH,
    TEXT_POSITIONS,
    TEXT_STYLES,
)


def load_font(font_path, size_pixels):
    """Load a TrueType font, falling back to default if not found."""
    try:
        return ImageFont.truetype(font_path, size=size_pixels)
    except (OSError, IOError):
        print(f"Warning: Cannot load font {font_path}, using default font.")
        return ImageFont.load_default()


def wrap_text_to_width(draw, text, font, max_width):
    """Wrap text so each line fits within max_width pixels."""
    # Try to fit the whole text first
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] - bbox[0] <= max_width:
        return text

    # Wrap character by character for CJK compatibility
    lines = []
    current_line = ""
    for char in text:
        test_line = current_line + char
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > max_width:
            if current_line:
                lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def overlay_text(image, text, position_name, style_config):
    """
    Overlay text onto an image using RGBA compositing.

    Args:
        image: PIL Image (any mode)
        text: str, the text to overlay
        position_name: str, key from TEXT_POSITIONS
        style_config: dict with font_size_ratio, color, opacity, stroke_width, stroke_fill

    Returns:
        PIL Image in RGB mode
    """
    # Convert to RGBA for alpha compositing
    base = image.convert("RGBA")
    img_w, img_h = base.size

    # Create transparent overlay
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Load font
    font_size_px = max(12, int(style_config["font_size_ratio"] * img_h))
    font = load_font(CJK_FONT_PATH, font_size_px)

    # Wrap text to fit within 90% of image width
    max_text_width = int(img_w * 0.9)
    wrapped_text = wrap_text_to_width(draw, text, font, max_text_width)

    # Compute text bounding box
    text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    txt_w = text_bbox[2] - text_bbox[0]
    txt_h = text_bbox[3] - text_bbox[1]

    # Get position
    pos_func = TEXT_POSITIONS.get(position_name, TEXT_POSITIONS["center"])
    x, y = pos_func(img_w, img_h, txt_w, txt_h)

    # Draw text with opacity
    fill_color = (*style_config["color"], style_config["opacity"])

    draw_kwargs = {
        "xy": (x, y),
        "text": wrapped_text,
        "font": font,
        "fill": fill_color,
    }
    if style_config.get("stroke_width", 0) > 0 and style_config.get("stroke_fill"):
        draw_kwargs["stroke_width"] = style_config["stroke_width"]
        draw_kwargs["stroke_fill"] = (*style_config["stroke_fill"], style_config["opacity"])

    draw.multiline_text(**draw_kwargs)

    # Composite and convert back to RGB
    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


def process_single_image(args):
    """Process a single image. Used for parallel execution."""
    src_path, dst_path, text, position_name, style_config = args
    try:
        image = Image.open(src_path)
        attacked = overlay_text(image, text, position_name, style_config)
        attacked.save(dst_path, quality=95)
        return True, src_path
    except Exception as e:
        return False, f"{src_path}: {e}"


def process_images(src_dir, dst_dir, attack_name, max_workers=8):
    """
    Generate attacked images for a given attack config.

    Args:
        src_dir: source image directory
        dst_dir: destination directory for attacked images
        attack_name: key from ATTACK_CONFIGS
        max_workers: number of parallel workers
    """
    config = ATTACK_CONFIGS[attack_name]
    text = ATTACK_TEXTS[config["text_key"]]
    position = config["position"]
    style = TEXT_STYLES[config["style"]]

    os.makedirs(dst_dir, exist_ok=True)

    # Collect all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    tasks = []
    for fname in os.listdir(src_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in image_extensions:
            continue
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        tasks.append((src_path, dst_path, text, position, style))

    print(f"[{attack_name}] Processing {len(tasks)} images -> {dst_dir}")

    success_count = 0
    fail_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, t): t for t in tasks}
        for future in as_completed(futures):
            ok, info = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                print(f"  FAILED: {info}")
            if (success_count + fail_count) % 500 == 0:
                print(f"  Progress: {success_count + fail_count}/{len(tasks)}")

    print(f"[{attack_name}] Done. Success: {success_count}, Failed: {fail_count}")


def main():
    parser = argparse.ArgumentParser(description="Generate attacked images with text overlays")
    parser.add_argument("--src_dir", type=str, required=True, help="Source image directory")
    parser.add_argument("--output_base_dir", type=str, required=True, help="Base output directory")
    parser.add_argument("--attacks", type=str, default="all",
                        help="Comma-separated attack names, or 'all' for all configs")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    if args.attacks == "all":
        attack_names = list(ATTACK_CONFIGS.keys())
    else:
        attack_names = [a.strip() for a in args.attacks.split(",")]

    for name in attack_names:
        if name not in ATTACK_CONFIGS:
            print(f"Warning: Unknown attack '{name}', skipping. Available: {list(ATTACK_CONFIGS.keys())}")
            continue
        dst_dir = os.path.join(args.output_base_dir, name)
        process_images(args.src_dir, dst_dir, name, max_workers=args.workers)


if __name__ == "__main__":
    main()
