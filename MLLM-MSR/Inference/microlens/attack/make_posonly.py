#!/usr/bin/env python3
"""make_posonly.py — build a *_posonly attacked-image dir (positives only).

Realistic threat model: the attacker perturbs ONLY their own item's cover, while
the competing (negative) candidates keep clean covers. eval_illusion_sft.py
applies an adversarial image to any item that has a file in --attacked_image_dir,
so to attack only the positives we copy just the positive items' images into a
separate dir and point the eval at it.

Usage:
    python make_posonly.py --test_pairs /tmp/test_pairs_pilot.csv \
        --adv_dir results/overlay_pilot/rank_first_en
    # -> writes results/overlay_pilot/rank_first_en_posonly/<item>.<ext>
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser(description="Copy positive items' attacked images into a *_posonly dir")
    ap.add_argument("--test_pairs", required=True, help="test_pairs.csv with user,item,label")
    ap.add_argument("--adv_dir", required=True, help="dir of attacked images (keyed <item>.<ext>)")
    ap.add_argument("--out", default=None, help="output dir (default: <adv_dir>_posonly)")
    args = ap.parse_args()

    p = pd.read_csv(args.test_pairs)
    p.columns = [c.strip().lower() for c in p.columns]
    p["item"] = p["item"].astype(str).str.strip()
    pos = pd.unique(p[p["label"] == 1]["item"])

    src = Path(args.adv_dir)
    dst = Path(args.out) if args.out else src.parent / (src.name + "_posonly")
    dst.mkdir(parents=True, exist_ok=True)

    copied = missing = 0
    for it in pos:
        found = next((src / f"{it}{e}" for e in IMG_EXTS if (src / f"{it}{e}").exists()), None)
        if found is not None:
            shutil.copy(found, dst / found.name)
            copied += 1
        else:
            missing += 1
    print(f"posonly -> {dst}")
    print(f"  positives={len(pos)}  copied={copied}  missing={missing}")
    if missing:
        print(f"  (missing = positive items with no attacked image in {src}; "
              f"check the attack covered all test_pairs items)")


if __name__ == "__main__":
    main()
