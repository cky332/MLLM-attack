#!/usr/bin/env python3
"""compare_overlay_variants.py — one row = [clean | variant1 | variant2 | ...] for an item.

Builds a side-by-side montage so you can eyeball the SAME cover under the clean
condition and several attack variants (e.g. visible vs stealth text overlays).

Usage:
    python compare_overlay_variants.py \
        --clean_dir results/illusion_v1_image_pilot/clean_resized \
        --variant_dirs results/overlay_pilot/rank_first_en,results/overlay_pilot/rank_first_en_stealth,results/overlay_pilot/rank_first_en_stealth_low \
        --labels clean,visible_a255,stealth_a96,stealth_a64 \
        --items 16981 --out overlay_variants.png
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def find(d, item):
    for ext in IMG_EXTS:
        p = Path(d) / f"{item}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Side-by-side clean + attack-variant montage")
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--variant_dirs", required=True, help="comma-separated attack dirs")
    ap.add_argument("--labels", default=None,
                    help="comma-separated column labels (incl clean); len = 1 + #variant_dirs")
    ap.add_argument("--items", default=None, help="comma-separated item ids; default: auto-pick --n")
    ap.add_argument("--n", type=int, default=1, help="how many items (rows) if --items omitted")
    ap.add_argument("--out", default="overlay_variants.png")
    args = ap.parse_args()

    variant_dirs = [d.strip() for d in args.variant_dirs.split(",") if d.strip()]
    cols = [args.clean_dir] + variant_dirs
    labels = ([s.strip() for s in args.labels.split(",")] if args.labels
              else ["clean"] + [Path(d).name for d in variant_dirs])
    if len(labels) != len(cols):
        raise SystemExit(f"--labels has {len(labels)} entries but there are {len(cols)} columns")

    if args.items:
        items = [s.strip() for s in args.items.split(",") if s.strip()]
    else:
        common = {p.stem for p in Path(args.clean_dir).glob("*") if p.is_file()}
        for d in variant_dirs:
            common &= {p.stem for p in Path(d).glob("*") if p.is_file()}
        items = sorted(common)[: args.n]

    rows = []
    for it in items:
        imgs = []
        for d in cols:
            f = find(d, it)
            if f is None:
                imgs = None
                break
            imgs.append(Image.open(f).convert("RGB"))
        if imgs is None:
            print(f"  skip {it}: missing in one of the dirs")
            continue
        W, H = imgs[0].size
        imgs = [im.resize((W, H)) for im in imgs]
        rows.append((it, imgs))

    if not rows:
        raise SystemExit("No items rendered — check --items / dirs.")

    W, H = rows[0][1][0].size
    ncol, head, pad = len(cols), 18, 4
    canvas = Image.new("RGB",
                       (ncol * W + (ncol + 1) * pad, len(rows) * (H + head + pad) + pad),
                       (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for r, (it, imgs) in enumerate(rows):
        y = pad + r * (H + head + pad)
        for c, im in enumerate(imgs):
            x = pad + c * (W + pad)
            draw.text((x + 2, y + 3), f"{it} | {labels[c]}", fill=(0, 0, 0))
            canvas.paste(im, (x, y + head))
    canvas.save(args.out)
    print(f"saved -> {args.out}   ({len(rows)} item(s) x {ncol} cols: {labels})")


if __name__ == "__main__":
    main()
