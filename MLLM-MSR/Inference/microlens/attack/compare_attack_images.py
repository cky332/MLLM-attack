#!/usr/bin/env python3
"""compare_attack_images.py — eyeball the perturbation: clean vs adversarial vs diff.

The L_inf budget here is 16/255 (~6% per pixel), so clean and adversarial covers
look almost identical to the human eye. To actually SEE what the attack changed,
this builds a montage where each row is:

    [ clean cover | adversarial cover | amplified difference ]

The third panel is the per-pixel difference centred at grey and multiplied by
`--amp` (default 10x) so the structured adversarial noise becomes visible.

It also prints the real per-image max|delta| (should be ~16/255).

Usage (run on the machine that has the results/):
    python compare_attack_images.py \
        --clean_dir results/illusion_v1_image_pilot/clean_resized \
        --adv_dir   results/illusion_v1_image_pilot/images \
        --manifest  results/illusion_v1_image_pilot/images/manifest.csv \
        --n 6 --amp 10 --out attack_compare.png

    # or pick specific items
    python compare_attack_images.py --clean_dir ... --adv_dir ... \
        --items 16981,6539,15775 --out attack_compare.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _pick_items(args, adv_dir):
    if args.items:
        return [s.strip() for s in args.items.split(",") if s.strip()]
    if args.manifest and Path(args.manifest).exists():
        import pandas as pd
        m = pd.read_csv(args.manifest)
        m.columns = [c.strip().lower() for c in m.columns]
        key = "item_id" if "item_id" in m.columns else m.columns[0]
        if {"cos_adv", "cos_clean"}.issubset(m.columns):  # show best-aligned first
            m = m.assign(_gain=m["cos_adv"] - m["cos_clean"]).sort_values("_gain", ascending=False)
        return m[key].astype(str).tolist()[: args.n]
    return [p.stem for p in sorted(adv_dir.glob("*.png"))][: args.n]


def main():
    ap = argparse.ArgumentParser(description="Clean vs adversarial cover comparison montage")
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--adv_dir", required=True)
    ap.add_argument("--manifest", default=None, help="optional manifest.csv to pick/sort items")
    ap.add_argument("--items", default=None, help="comma-separated item ids (overrides manifest)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--amp", type=float, default=10.0, help="amplify the diff panel for visibility")
    ap.add_argument("--out", default="attack_compare.png")
    args = ap.parse_args()

    clean_dir, adv_dir = Path(args.clean_dir), Path(args.adv_dir)
    items = _pick_items(args, adv_dir)

    rows = []
    for it in items:
        cp, av = clean_dir / f"{it}.png", adv_dir / f"{it}.png"
        if not (cp.exists() and av.exists()):
            print(f"  skip {it}: missing clean or adv png")
            continue
        c = _load_rgb(cp)
        a = _load_rgb(av)
        if c.shape != a.shape:
            c = np.asarray(Image.open(cp).convert("RGB").resize((a.shape[1], a.shape[0])),
                           dtype=np.float32)
        d = a - c
        linf = float(np.abs(d).max())
        diff_panel = np.clip(128.0 + d * args.amp, 0, 255)
        panel = np.concatenate([c, a, diff_panel], axis=1).astype(np.uint8)
        rows.append((str(it), linf, panel))
        print(f"  item {it}: max|delta|={linf:.1f}/255  (~{linf/255*100:.1f}% per pixel)")

    if not rows:
        raise SystemExit("No items rendered — check --clean_dir / --adv_dir / --items.")

    H, W = rows[0][2].shape[:2]
    label_h = 22
    canvas = Image.fromarray(np.full((len(rows) * (H + label_h), W, 3), 255, np.uint8))
    draw = ImageDraw.Draw(canvas)
    for i, (it, linf, panel) in enumerate(rows):
        y = i * (H + label_h)
        draw.text((4, y + 5),
                  f"item {it}   max|d|={linf:.0f}/255     [ clean | adversarial | diff x{args.amp:g} ]",
                  fill=(0, 0, 0))
        canvas.paste(Image.fromarray(panel), (0, y + label_h))
    canvas.save(args.out)
    print(f"\nsaved -> {args.out}   ({len(rows)} rows; each = clean | adversarial | amplified diff)")
    print("open/download this PNG to see the attack. Left vs middle ~ identical; "
          "right shows the (amplified) adversarial noise.")


if __name__ == "__main__":
    main()
