#!/usr/bin/env python3
r"""illusion_attack.py — Adversarial-illusion attack on MLLM-MSR candidate images.

Implements the white-box attack of:

    Tingwei Zhang, Rishi Jha, Eugene Bagdasaryan, Vitaly Shmatikov,
    "Adversarial Illusions in Multi-Modal Embeddings", USENIX Security 2025.
    https://github.com/ebagdasa/adversarial_illusions

Threat model applied to MLLM-MSR
--------------------------------
MLLM-MSR decides whether a user will interact with a candidate item by feeding
(user-preference text + candidate cover IMAGE + candidate title) to LLaVA and
reading P(Yes). LLaVA-1.6-Mistral-7B's vision tower IS the CLIP ViT-L/14-336
image encoder. So a white-box adversary who perturbs the candidate cover image
can drive LLaVA's visual features wherever they want in CLIP space.

The attack (the user's idea): perturb a candidate cover image x so its CLIP
image embedding aligns with the text embedding of "popular / trending" content
(``热门文本``). LLaVA then "sees" a popular video and is more likely to answer
Yes -> the item is recommended.

Objective (paper Eq. 3), optimised with L_inf PGD (paper Eq. 2):

    min_delta  L_WB(x+delta, y_t) = 1 - cos( theta_img(x+delta), theta_txt(y_t) )
    s.t.       ||delta||_inf <= eps,   x+delta in [0, 1]

theta_img, theta_txt are CLIP's image / text encoders. y_t is the popular-text
target; when several popular titles are used we align to their (normalised)
centroid embedding.

This module is intentionally encoder-agnostic at its core: ``pgd_illusion`` takes
an ``encode_image_01`` callable, so the exact same optimisation loop can be unit
tested with a tiny synthetic encoder (see test_illusion_attack.py) and run for
real against CLIP on a GPU.

Subcommands
-----------
    build_target   Build the popular-text target embedding from interaction
                   popularity + titles (or a custom --target_text).
    generate       Perturb candidate cover images and save them (+ a manifest
                   with per-image clean/adv cosine alignment), and report the
                   embedding-level ASR.
    embed_asr      Aggregate embedding-level ASR from a manifest.

Usage
-----
    # 1) Build the popular-text target (CPU-cheap, needs CLIP text encoder)
    python illusion_attack.py build_target \
        --pairs_csv ../../data/microlens/MicroLens-50k_pairs.csv \
        --title_csv ../../data/microlens/MicroLens-50k_titles.csv \
        --top_n 20 \
        --out_target results/illusion/popular_target.npz

    # 2) Generate adversarial cover images aligned to that target  [GPU]
    python illusion_attack.py generate \
        --src_dir   /path/to/MicroLens-50k_covers \
        --out_dir   results/illusion/images \
        --clean_resized_dir results/illusion/clean_resized \
        --target    results/illusion/popular_target.npz \
        --items_csv ../../data/MicroLens-50k/Split/test_pairs.csv \
        --eps 16 --alpha 1 --iters 300 --batch_size 16

    # 3) Embedding-level ASR (paper Table-1 style)
    python illusion_attack.py embed_asr \
        --manifest results/illusion/images/manifest.csv \
        --cos_threshold 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from illusion_metrics import embedding_alignment_asr

# CLIP ViT-L/14-336 is the vision tower inside llava-v1.6-mistral-7b-hf.
DEFAULT_CLIP_ID = "openai/clip-vit-large-patch14-336"
CLIP_INPUT_SIZE = 336
# CLIP image normalisation constants (OpenAI CLIP).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# CLIP encoder wrapper (lazy torch import so the module loads without torch)
# ---------------------------------------------------------------------------
class CLIPIllusionEncoder:
    """Wraps CLIP so images can be encoded directly from [0,1] pixel tensors
    (keeping the whole resize->normalize->encode path differentiable)."""

    def __init__(self, model_id=DEFAULT_CLIP_ID, device="cuda:0", dtype="fp32"):
        import torch
        from transformers import CLIPModel, CLIPTokenizer

        self.torch = torch
        self.device = device
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
        self.dtype = torch_dtype
        self.model = (
            CLIPModel.from_pretrained(
                model_id, cache_dir=os.path.expanduser("~/.cache/huggingface/hub")
            )
            .to(device)
            .to(torch_dtype)
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.input_size = self.model.config.vision_config.image_size  # 336
        mean = torch.tensor(CLIP_MEAN, device=device, dtype=torch_dtype).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=device, dtype=torch_dtype).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def encode_image_01(self, x01):
        """x01: (B,3,H,W) float in [0,1] at CLIP input size. -> (B,D) L2-normalised."""
        x = (x01.to(self.dtype) - self._mean) / self._std
        feats = self.model.get_image_features(pixel_values=x)
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @property
    def encode_image_fn(self):
        return self.encode_image_01

    def encode_text(self, texts, batch_size=64):
        torch = self.torch
        embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                tok = self.tokenizer(
                    chunk, padding=True, truncation=True, max_length=77,
                    return_tensors="pt",
                ).to(self.device)
                f = self.model.get_text_features(**tok)
                f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                embs.append(f.float().cpu())
        return torch.cat(embs, dim=0).numpy()


# ---------------------------------------------------------------------------
# Image IO
# ---------------------------------------------------------------------------
def load_image_as_01(path, size=CLIP_INPUT_SIZE):
    """Open an image, resize to size x size (bicubic), return CHW float in [0,1]."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    return np.transpose(arr, (2, 0, 1))  # CHW


def save_01_image(chw01, path):
    """Save a CHW float[0,1] array losslessly (PNG) so the perturbation is exact."""
    from PIL import Image

    arr = np.clip(np.transpose(chw01, (1, 2, 0)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)  # extension (.png) decides format


# ---------------------------------------------------------------------------
# PGD core — encoder-agnostic, faithful to paper Eq. 2 & 3
# ---------------------------------------------------------------------------
def pgd_illusion(x01, target_emb, encode_image_01, eps, alpha, iters,
                 torch_mod, random_init=False, progress=None):
    """Run L_inf PGD to align encode_image_01(x+delta) with target_emb.

    Args:
        x01:        (B,3,H,W) torch tensor in [0,1] on the target device.
        target_emb: (D,) or (T,D) torch tensor; if (T,D) we align to the
                    normalised centroid (mean target embedding).
        encode_image_01: callable (B,3,H,W in [0,1]) -> (B,D) L2-normalised.
        eps, alpha: L_inf budget and step size, in [0,1] pixel units.
        iters:      number of PGD steps.
        torch_mod:  the torch module (injected to avoid a hard import here).
    Returns:
        x_adv:     (B,3,H,W) detached tensor in [0,1].
        final_cos: (B,) numpy array, cos(image_emb, target centroid) at the end.
    """
    torch = torch_mod
    if target_emb.dim() == 2:
        t = target_emb.mean(dim=0)
    else:
        t = target_emb
    t = (t / t.norm().clamp_min(1e-12)).detach()

    x01 = x01.detach()
    if random_init:
        delta = (torch.rand_like(x01) * 2 - 1) * eps
        delta = (torch.clamp(x01 + delta, 0, 1) - x01).detach()
    else:
        delta = torch.zeros_like(x01)
    delta.requires_grad_(True)

    last_cos = None
    for it in range(iters):
        emb = encode_image_01(torch.clamp(x01 + delta, 0, 1))  # (B,D), normalised
        cos = emb @ t.to(emb.dtype)                            # (B,) cosine sim
        loss = (1.0 - cos).mean()                              # paper Eq. 3
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            # Minimise L_WB -> descend the loss -> step against sign(grad).
            delta -= alpha * grad.sign()
            delta.clamp_(-eps, eps)                            # L_inf projection
            delta.copy_(torch.clamp(x01 + delta, 0, 1) - x01)  # keep pixels valid
        last_cos = cos.detach()
        if progress is not None and (it % max(1, iters // 5) == 0 or it == iters - 1):
            progress(it + 1, iters, float(loss.item()), float(last_cos.mean().item()))

    x_adv = torch.clamp(x01 + delta, 0, 1).detach()
    return x_adv, last_cos.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Popularity / target construction
# ---------------------------------------------------------------------------
def load_item_popularity(pairs_path):
    """Item interaction counts from either pairs.csv (user,item,timestamp) or
    pairs.tsv (user \\t item item item ...)."""
    p = str(pairs_path)
    counts = {}
    if p.endswith(".tsv"):
        with open(p) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                for it in parts[1].split():
                    counts[it] = counts.get(it, 0) + 1
    else:
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        col = "item" if "item" in df.columns else df.columns[1]
        vc = df[col].astype(str).value_counts()
        counts = {str(k): int(v) for k, v in vc.items()}
    return counts


def load_titles(title_csv):
    peek = pd.read_csv(title_csv, nrows=1, header=None)
    has_header = any(
        isinstance(v, str) and not str(v).strip().isdigit()
        for v in peek.iloc[0].tolist()[:1]
    )
    df = pd.read_csv(title_csv) if has_header else pd.read_csv(
        title_csv, header=None, names=["item", "title"]
    )
    df.columns = [c.strip().lower() for c in df.columns]
    if "item_id" in df.columns:
        df.rename(columns={"item_id": "item"}, inplace=True)
    df["item"] = df["item"].astype(str)
    return dict(zip(df["item"], df["title"].astype(str)))


def build_target(args):
    """Build and save the popular-text target embedding (npz)."""
    enc = CLIPIllusionEncoder(args.clip_id, device=args.device, dtype="fp32")

    if args.target_text:
        texts = [t.strip() for t in args.target_text.split("||") if t.strip()]
        source = "custom"
        top_items = []
    else:
        counts = load_item_popularity(args.pairs_csv)
        titles = load_titles(args.title_csv)
        ranked = sorted(counts.keys(), key=lambda k: -counts[k])
        top_items = [it for it in ranked if it in titles][: args.top_n]
        texts = [titles[it] for it in top_items]
        if args.popular_prefix:
            texts = [f"{args.popular_prefix} {t}" for t in texts]
        source = "popularity"

    if not texts:
        raise SystemExit("No target texts — check --pairs_csv/--title_csv/--target_text")

    embs = enc.encode_text(texts)                     # (T, D), normalised
    centroid = embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    out = Path(args.out_target)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        centroid=centroid.astype(np.float32),
        embs=embs.astype(np.float32),
        texts=np.array(texts, dtype=object),
        items=np.array(top_items, dtype=object),
        clip_id=args.clip_id,
        source=source,
    )
    print(f"[build_target] source={source}  n_texts={len(texts)}  dim={centroid.shape[0]}")
    for t in texts[: min(5, len(texts))]:
        print(f"    target text: {t[:90]}")
    print(f"[build_target] Saved target -> {out}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _resolve_items(args):
    """Return ordered list of item ids to attack and a map item->src image path."""
    src = Path(args.src_dir)
    path_by_item = {}
    for p in src.glob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            path_by_item.setdefault(p.stem, str(p))

    if args.items_csv:
        df = pd.read_csv(args.items_csv)
        df.columns = [c.strip().lower() for c in df.columns]
        col = "item" if "item" in df.columns else df.columns[0]
        items = list(dict.fromkeys(df[col].astype(str).tolist()))  # de-dup, keep order
        items = [it for it in items if it in path_by_item]
    else:
        items = sorted(path_by_item.keys())

    if args.max_items > 0:
        items = items[: args.max_items]
    # Multi-GPU: launch one process per GPU, each taking an interleaved slice.
    if getattr(args, "num_shards", 1) > 1:
        items = items[args.shard_id :: args.num_shards]
        print(f"[generate] shard {args.shard_id}/{args.num_shards}: {len(items)} items")
    return items, path_by_item


def generate(args):
    import torch

    enc = CLIPIllusionEncoder(args.clip_id, device=args.device, dtype=args.encoder_dtype)
    target = np.load(args.target, allow_pickle=True)
    centroid = torch.tensor(target["centroid"], device=args.device)

    eps = args.eps / 255.0
    alpha = args.alpha / 255.0
    items, path_by_item = _resolve_items(args)
    print(f"[generate] attacking {len(items)} items | eps={args.eps}/255 "
          f"alpha={args.alpha}/255 iters={args.iters} batch={args.batch_size}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = Path(args.clean_resized_dir) if args.clean_resized_dir else None
    if clean_dir:
        clean_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    size = enc.input_size
    for b in range(0, len(items), args.batch_size):
        chunk = items[b : b + args.batch_size]
        batch_np, ok_items = [], []
        for it in chunk:
            try:
                batch_np.append(load_image_as_01(path_by_item[it], size))
                ok_items.append(it)
            except Exception as e:  # unreadable image
                print(f"  SKIP {it}: {e}")
        if not ok_items:
            continue
        x01 = torch.tensor(np.stack(batch_np), device=args.device)

        with torch.no_grad():
            cos_clean = (enc.encode_image_01(x01) @ centroid.to(enc.dtype)).float().cpu().numpy()

        x_adv, cos_adv = pgd_illusion(
            x01, centroid, enc.encode_image_01, eps, alpha, args.iters,
            torch_mod=torch, random_init=args.random_init,
        )
        x_adv_np = x_adv.float().cpu().numpy()
        x01_np = x01.float().cpu().numpy()

        for j, it in enumerate(ok_items):
            adv_path = out_dir / f"{it}.png"
            save_01_image(x_adv_np[j], adv_path)
            if clean_dir:
                save_01_image(x01_np[j], clean_dir / f"{it}.png")
            linf = float(np.max(np.abs(x_adv_np[j] - x01_np[j])))
            l2 = float(np.sqrt(np.sum((x_adv_np[j] - x01_np[j]) ** 2)))
            rows.append({
                "item_id": it,
                "cos_clean": float(cos_clean[j]),
                "cos_adv": float(cos_adv[j]),
                "linf": linf,
                "linf_255": linf * 255.0,
                "l2": l2,
                "path": str(adv_path),
            })
        done = b + len(chunk)
        if (done // args.batch_size) % 10 == 0 or done >= len(items):
            mc = np.mean([r["cos_clean"] for r in rows])
            ma = np.mean([r["cos_adv"] for r in rows])
            print(f"  {done}/{len(items)}  cos clean={mc:.3f} -> adv={ma:.3f}")

    # When sharded across GPUs, each process writes its own manifest; merge with
    # `embed_asr --manifest <out_dir>` afterwards.
    suffix = f"_shard{args.shard_id}" if getattr(args, "num_shards", 1) > 1 else ""
    manifest = pd.DataFrame(rows)
    man_path = out_dir / f"manifest{suffix}.csv"
    manifest.to_csv(man_path, index=False)

    asr = embedding_alignment_asr(
        manifest["cos_clean"].values, manifest["cos_adv"].values, args.cos_threshold
    )
    summary = {
        "attack": "adversarial_illusion",
        "paper": "Zhang et al., Adversarial Illusions in Multi-Modal Embeddings (USENIX Sec 2025)",
        "clip_id": args.clip_id,
        "eps_255": args.eps, "alpha_255": args.alpha, "iters": args.iters,
        "target_source": str(target["source"]),
        "target_texts_preview": [str(t) for t in list(target["texts"])[:5]],
        "n_images": len(manifest),
        "mean_linf_255": float(manifest["linf_255"].mean()) if len(manifest) else 0.0,
        "embedding_asr": asr,
    }
    with open(out_dir / f"summary{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print("ADVERSARIAL ILLUSION — EMBEDDING-LEVEL ASR")
    print("=" * 64)
    print(f"images:            {asr['n']}")
    print(f"mean cos (clean):  {asr['mean_cos_clean']:.4f}")
    print(f"mean cos (adv):    {asr['mean_cos_attacked']:.4f}  "
          f"(gain {asr['mean_cos_gain']:+.4f})")
    print(f"ASR (improved & cos>= {asr['cos_threshold']}): {asr['asr']:.1%}")
    print(f"  - improved-only: {asr['asr_improved']:.1%}   "
          f"threshold-only: {asr['asr_threshold']:.1%}")
    print(f"mean ||delta||_inf: {summary['mean_linf_255']:.2f}/255  (budget {args.eps}/255)")
    print(f"\nManifest -> {man_path}\nSummary  -> {out_dir/'summary.json'}")
    if clean_dir:
        print(f"Resized-clean baseline images -> {clean_dir} "
              f"(use as --clean_image_dir in eval_illusion_ranking.py)")


def embed_asr(args):
    p = Path(args.manifest)
    files = sorted(p.glob("manifest*.csv")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"No manifest*.csv found at {p}")
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    asr = embedding_alignment_asr(m["cos_clean"].values, m["cos_adv"].values,
                                  args.cos_threshold)
    print(f"[embed_asr] merged {len(files)} manifest(s), {len(m)} images")
    print(json.dumps(asr, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Adversarial-illusion attack on MLLM-MSR images")
    sub = ap.add_subparsers(dest="cmd")

    bt = sub.add_parser("build_target", help="Build popular-text target embedding")
    bt.add_argument("--pairs_csv", help="interaction pairs (.csv user,item,ts | .tsv)")
    bt.add_argument("--title_csv", help="item titles csv")
    bt.add_argument("--top_n", type=int, default=20, help="# most-popular titles to use")
    bt.add_argument("--popular_prefix", default="",
                    help="optional phrase prepended to each title, e.g. "
                         "'A trending viral popular video:'")
    bt.add_argument("--target_text", default=None,
                    help="custom target text(s), '||'-separated; overrides popularity")
    bt.add_argument("--out_target", required=True, help="output .npz path")
    bt.add_argument("--clip_id", default=DEFAULT_CLIP_ID)
    bt.add_argument("--device", default="cuda:0")
    bt.set_defaults(func=build_target)

    gn = sub.add_parser("generate", help="Generate adversarial cover images")
    gn.add_argument("--src_dir", required=True, help="clean cover images dir")
    gn.add_argument("--out_dir", required=True, help="output dir for adversarial images")
    gn.add_argument("--clean_resized_dir", default=None,
                    help="also save resized (un-perturbed) covers here for a fair "
                         "clean baseline in the recommendation eval")
    gn.add_argument("--target", required=True, help="target .npz from build_target")
    gn.add_argument("--items_csv", default=None,
                    help="restrict to items in this csv (e.g. test_pairs.csv); "
                         "default = all images in --src_dir")
    gn.add_argument("--max_items", type=int, default=0, help="0 = no limit")
    gn.add_argument("--eps", type=float, default=16.0, help="L_inf budget /255 (paper std=16)")
    gn.add_argument("--alpha", type=float, default=1.0, help="PGD step /255")
    gn.add_argument("--iters", type=int, default=300,
                    help="PGD iterations (paper uses up to 7500 for ImageBind; "
                         "CLIP converges far sooner)")
    gn.add_argument("--batch_size", type=int, default=16)
    gn.add_argument("--random_init", action="store_true", help="random PGD start")
    gn.add_argument("--shard_id", type=int, default=0,
                    help="this process's shard index (multi-GPU)")
    gn.add_argument("--num_shards", type=int, default=1,
                    help="split items across N processes (launch one per GPU)")
    gn.add_argument("--cos_threshold", type=float, default=0.5)
    gn.add_argument("--encoder_dtype", default="fp32", choices=["fp32", "fp16"])
    gn.add_argument("--clip_id", default=DEFAULT_CLIP_ID)
    gn.add_argument("--device", default="cuda:0")
    gn.set_defaults(func=generate)

    ea = sub.add_parser("embed_asr", help="Aggregate embedding ASR from a manifest")
    ea.add_argument("--manifest", required=True)
    ea.add_argument("--cos_threshold", type=float, default=0.5)
    ea.set_defaults(func=embed_asr)

    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
