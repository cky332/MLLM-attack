#!/usr/bin/env python3
r"""illusion_attack_llava.py — Feature-space "popular-cover" illusion against
MLLM-MSR, in the representation LLaVA actually consumes.

Why this exists
---------------
The pure CLIP image<->text illusion (illusion_attack.py) aligns CLIP's *pooled
contrastive embedding* (get_image_features) to popular text. A pilot showed that
drives embedding ASR ~99% but barely moves LLaVA's P(Yes): LLaVA-Next does NOT
read that embedding. It feeds the LLM the **patch hidden states of the vision
tower's `vision_feature_layer` (=-2), CLS dropped, passed through
`multi_modal_projector`** — a different representation.

So this attack perturbs the candidate cover so its *LLaVA visual tokens* align
with those of POPULAR item covers ("make it look like a popular video to LLaVA"
in LLaVA's own feature space). It backprops only through the vision tower +
projector (cheap, no 7B LLM), and stays in the space that determines P(Yes).

Objective (L_inf PGD, same threat model as the paper):

    min_delta  1 - mean_t cos( g_t(x+delta),  g_t(target) )      [impersonate]
        or     1 - cos( pool(g(x+delta)),  popular_centroid )    [centroid]
    s.t.       ||delta||_inf <= eps,   x+delta in [0,1]

where g(x) = projector( vision_tower(x).hidden_states[-2][:, 1:] ) are the visual
tokens LLaVA feeds to the LLM, and the target is the same features computed on
the top-N most-interacted item covers.

Subcommands
-----------
    build_target   Compute LLaVA visual-token targets from the top-N popular
                   item COVERS (saves per-cover token grids + pooled centroid).
    generate       Perturb candidate covers to match those features; save images
                   + manifest (per-image feature cos, clean vs adv) + ASR.
    embed_asr      Aggregate feature-alignment ASR from manifest(s).

The output images plug straight into eval_illusion_sft.py (same as the CLIP
attack) for the recommendation-level ASR on your fine-tuned LoRA.

Usage
-----
    python illusion_attack_llava.py build_target \
        --src_dir   /path/to/MicroLens-50k_covers \
        --pairs_csv ../../data/microlens/MicroLens-50k_pairs.csv \
        --title_csv ../../data/MicroLens-50k/MicroLens-50k_titles.csv \
        --top_n 10 --out_target results/illusion_llava/popular_target.pt

    python illusion_attack_llava.py generate \
        --src_dir /path/to/MicroLens-50k_covers \
        --out_dir results/illusion_llava/images --clean_resized_dir results/illusion_llava/clean_resized \
        --target  results/illusion_llava/popular_target.pt \
        --items_csv /path/to/Split/test_pairs.csv \
        --target_mode impersonate --eps 16 --iters 300 --batch_size 8
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from illusion_attack import (
    CLIP_INPUT_SIZE,
    CLIP_MEAN,
    CLIP_STD,
    IMG_EXTS,
    load_image_as_01,
    load_item_popularity,
    load_titles,
    save_01_image,
)
from illusion_metrics import embedding_alignment_asr

DEFAULT_LLAVA_ID = "llava-hf/llava-v1.6-mistral-7b-hf"


# ---------------------------------------------------------------------------
# LLaVA visual encoder: vision tower + multimodal projector ONLY (no 7B LLM)
# ---------------------------------------------------------------------------
class LlavaVisualEncoder:
    """Produces the exact visual tokens LLaVA feeds to its LLM, from [0,1] pixels.

    Loads the full LlavaNext checkpoint (to obtain the trained projector), keeps
    only the vision tower + projector on GPU in fp32, and frees the language
    model so this fits comfortably on a single 24GB card.
    """

    def __init__(self, model_id=DEFAULT_LLAVA_ID, device="cuda:0"):
        import torch
        from transformers import LlavaNextForConditionalGeneration

        self.torch = torch
        self.device = device
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.vision_tower = model.vision_tower.to(device=device, dtype=torch.float32).eval()
        self.projector = model.multi_modal_projector.to(device=device, dtype=torch.float32).eval()
        self.vfl = getattr(model.config, "vision_feature_layer", -2)
        self.vfs = getattr(model.config, "vision_feature_select_strategy", "default")
        self.input_size = model.config.vision_config.image_size  # 336
        for p in list(self.vision_tower.parameters()) + list(self.projector.parameters()):
            p.requires_grad_(False)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def tokens(self, x01):
        """x01: (B,3,336,336) in [0,1] -> (B, T, D) LLaVA visual tokens."""
        torch = self.torch
        x = (x01 - self._mean) / self._std
        out = self.vision_tower(x, output_hidden_states=True)
        feat = out.hidden_states[self.vfl]
        if self.vfs == "default":
            feat = feat[:, 1:]  # drop CLS, exactly like LlavaNext
        return self.projector(feat)  # (B, T, D=4096)


# ---------------------------------------------------------------------------
# Image-set resolution
# ---------------------------------------------------------------------------
def _path_by_item(src_dir):
    src = Path(src_dir)
    m = {}
    for p in src.glob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            m.setdefault(p.stem, str(p))
    return m


def _resolve_items(args, path_by_item):
    if args.items_csv:
        df = pd.read_csv(args.items_csv)
        df.columns = [c.strip().lower() for c in df.columns]
        col = "item" if "item" in df.columns else df.columns[0]
        items = list(dict.fromkeys(df[col].astype(str).str.strip().tolist()))
        items = [it for it in items if it in path_by_item]
    else:
        items = sorted(path_by_item.keys())
    if args.max_items > 0:
        items = items[: args.max_items]
    if getattr(args, "num_shards", 1) > 1:
        items = items[args.shard_id :: args.num_shards]
        print(f"[generate] shard {args.shard_id}/{args.num_shards}: {len(items)} items")
    return items


# ---------------------------------------------------------------------------
# build_target — popular covers' LLaVA features
# ---------------------------------------------------------------------------
def build_target(args):
    import torch

    enc = LlavaVisualEncoder(args.llava_id, device=args.device)
    path_by_item = _path_by_item(args.src_dir)
    counts = load_item_popularity(args.pairs_csv)
    titles = load_titles(args.title_csv) if args.title_csv else {}
    ranked = [it for it in sorted(counts, key=lambda k: -counts[k]) if it in path_by_item]
    top = ranked[: args.top_n]
    if not top:
        raise SystemExit("No popular covers found (check --src_dir / --pairs_csv)")

    grids, pooled, used = [], [], []
    size = enc.input_size
    with torch.no_grad():
        for it in top:
            try:
                x = torch.tensor(load_image_as_01(path_by_item[it], size)[None], device=args.device)
            except Exception as e:
                print(f"  skip {it}: {e}")
                continue
            g = enc.tokens(x)[0]               # (T, D)
            grids.append(g.cpu())
            pooled.append(g.mean(0).cpu())     # (D,)
            used.append(it)
    grids = torch.stack(grids)                 # (K, T, D)
    centroid = torch.stack(pooled).mean(0)     # (D,)

    out = Path(args.out_target)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "grids": grids, "centroid": centroid,
        "items": used, "titles": [titles.get(i, "") for i in used],
        "llava_id": args.llava_id,
    }, out)
    print(f"[build_target] {len(used)} popular covers -> grids {tuple(grids.shape)}, "
          f"centroid {tuple(centroid.shape)}")
    for it in used[:5]:
        print(f"    popular item {it} (count={counts.get(it)}): {titles.get(it,'')[:60]}")
    print(f"[build_target] saved -> {out}")


# ---------------------------------------------------------------------------
# PGD in LLaVA feature space
# ---------------------------------------------------------------------------
def _pgd_llava(enc, x01, target_grids, mode, centroid, eps, alpha, iters,
               random_init=False):
    """target_grids: (B,T,D) per-sample target tokens (used when mode=impersonate)."""
    torch = enc.torch
    x01 = x01.detach()
    if random_init:
        delta = ((torch.rand_like(x01) * 2 - 1) * eps)
        delta = (torch.clamp(x01 + delta, 0, 1) - x01).detach()
    else:
        delta = torch.zeros_like(x01)
    delta.requires_grad_(True)

    if mode == "centroid":
        tgt = (centroid / centroid.norm().clamp_min(1e-12)).to(x01.device)
    else:  # impersonate: per-sample target grid, L2-normalized per token
        tgt = target_grids / target_grids.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    last = None
    for _ in range(iters):
        g = enc.tokens(torch.clamp(x01 + delta, 0, 1))            # (B, T, D)
        if mode == "centroid":
            pooled = g.mean(1)                                    # (B, D)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cos = pooled @ tgt                                    # (B,)
        else:
            gn = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cos = (gn * tgt).sum(-1).mean(1)                      # (B,) mean over tokens
        loss = (1.0 - cos).mean()
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta -= alpha * grad.sign()
            delta.clamp_(-eps, eps)
            delta.copy_(torch.clamp(x01 + delta, 0, 1) - x01)
        last = cos.detach()
    x_adv = torch.clamp(x01 + delta, 0, 1).detach()
    return x_adv, last.float().cpu().numpy()


def generate(args):
    import torch

    enc = LlavaVisualEncoder(args.llava_id, device=args.device)
    tgt = torch.load(args.target, map_location=args.device)
    grids = tgt["grids"].to(args.device).float()      # (K, T, D)
    centroid = tgt["centroid"].to(args.device).float()
    K = grids.shape[0]
    rng = np.random.default_rng(args.seed)

    eps, alpha = args.eps / 255.0, args.alpha / 255.0
    path_by_item = _path_by_item(args.src_dir)
    items = _resolve_items(args, path_by_item)
    print(f"[generate] mode={args.target_mode} items={len(items)} eps={args.eps}/255 "
          f"iters={args.iters} batch={args.batch_size} (targets K={K})")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = Path(args.clean_resized_dir) if args.clean_resized_dir else None
    if clean_dir:
        clean_dir.mkdir(parents=True, exist_ok=True)

    size = enc.input_size
    rows = []
    for b in range(0, len(items), args.batch_size):
        chunk = items[b: b + args.batch_size]
        arrs, ok = [], []
        for it in chunk:
            try:
                arrs.append(load_image_as_01(path_by_item[it], size)); ok.append(it)
            except Exception as e:
                print(f"  skip {it}: {e}")
        if not ok:
            continue
        x01 = torch.tensor(np.stack(arrs), device=args.device)
        # target grid: per-source random cover (impersonate), or the per-position
        # AVERAGED popular grid (token_centroid / centroid use a single shared grid).
        if args.target_mode == "impersonate":
            tidx = torch.tensor(rng.integers(0, K, size=len(ok)), device=args.device)
            tgrids = grids[tidx]                              # (B, T, D)
            tlabels = [tgt["items"][int(i)] for i in tidx.tolist()]
        else:  # token_centroid (per-token cos to avg grid) or centroid (pooled cos)
            tgrids = grids.mean(0, keepdim=True).expand(len(ok), -1, -1).contiguous()
            lbl = f"avg_top{K}" if args.target_mode == "token_centroid" else "pooled_centroid"
            tlabels = [lbl] * len(ok)

        with torch.no_grad():
            if args.target_mode == "centroid":
                p = enc.tokens(x01).mean(1)
                p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                cos_clean = (p @ (centroid / centroid.norm().clamp_min(1e-12))).float().cpu().numpy()
            else:
                g = enc.tokens(x01); g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                tn = tgrids / tgrids.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                cos_clean = (g * tn).sum(-1).mean(1).float().cpu().numpy()

        x_adv, cos_adv = _pgd_llava(enc, x01, tgrids, args.target_mode, centroid,
                                    eps, alpha, args.iters,
                                    random_init=args.random_init)
        x_adv_np = x_adv.float().cpu().numpy(); x01_np = x01.float().cpu().numpy()
        for j, it in enumerate(ok):
            save_01_image(x_adv_np[j], out_dir / f"{it}.png")
            if clean_dir:
                save_01_image(x01_np[j], clean_dir / f"{it}.png")
            linf = float(np.max(np.abs(x_adv_np[j] - x01_np[j])))
            rows.append({"item_id": it, "cos_clean": float(cos_clean[j]),
                         "cos_adv": float(cos_adv[j]), "linf_255": linf * 255.0,
                         "target_item": tlabels[j]})
        done = b + len(chunk)
        if (done // max(args.batch_size, 1)) % 10 == 0 or done >= len(items):
            mc = np.mean([r["cos_clean"] for r in rows]); ma = np.mean([r["cos_adv"] for r in rows])
            print(f"  {done}/{len(items)}  feature cos clean={mc:.3f} -> adv={ma:.3f}")

    suffix = f"_shard{args.shard_id}" if getattr(args, "num_shards", 1) > 1 else ""
    man = pd.DataFrame(rows)
    man.to_csv(out_dir / f"manifest{suffix}.csv", index=False)
    asr = embedding_alignment_asr(man["cos_clean"].values, man["cos_adv"].values, args.cos_threshold)
    with open(out_dir / f"summary{suffix}.json", "w", encoding="utf-8") as f:
        json.dump({"attack": "llava_feature_illusion", "mode": args.target_mode,
                   "eps_255": args.eps, "iters": args.iters, "feature_asr": asr,
                   "mean_linf_255": float(man["linf_255"].mean()) if len(man) else 0.0}, f,
                  ensure_ascii=False, indent=2)
    print("\n" + "=" * 64)
    print("LLaVA FEATURE-SPACE ILLUSION — feature alignment")
    print("=" * 64)
    print(f"images: {asr['n']}  mean cos clean={asr['mean_cos_clean']:.4f} -> "
          f"adv={asr['mean_cos_attacked']:.4f} (gain {asr['mean_cos_gain']:+.4f})")
    print(f"improved: {asr['asr_improved']:.1%}   mean ||delta||_inf="
          f"{man['linf_255'].mean():.2f}/255  (budget {args.eps}/255)")
    print(f"manifest -> {out_dir / ('manifest'+suffix+'.csv')}")


def embed_asr(args):
    p = Path(args.manifest)
    files = sorted(p.glob("manifest*.csv")) if p.is_dir() else [p]
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(json.dumps(embedding_alignment_asr(m["cos_clean"].values, m["cos_adv"].values,
                                             args.cos_threshold), indent=2))


def main():
    ap = argparse.ArgumentParser(description="LLaVA feature-space popular-cover illusion")
    sub = ap.add_subparsers(dest="cmd")

    bt = sub.add_parser("build_target")
    bt.add_argument("--src_dir", required=True)
    bt.add_argument("--pairs_csv", required=True)
    bt.add_argument("--title_csv", default=None)
    bt.add_argument("--top_n", type=int, default=10)
    bt.add_argument("--out_target", required=True)
    bt.add_argument("--llava_id", default=DEFAULT_LLAVA_ID)
    bt.add_argument("--device", default="cuda:0")
    bt.set_defaults(func=build_target)

    gn = sub.add_parser("generate")
    gn.add_argument("--src_dir", required=True)
    gn.add_argument("--out_dir", required=True)
    gn.add_argument("--clean_resized_dir", default=None)
    gn.add_argument("--target", required=True)
    gn.add_argument("--items_csv", default=None)
    gn.add_argument("--target_mode", default="impersonate",
                    choices=["impersonate", "centroid", "token_centroid"],
                    help="impersonate=per-token cos to 1 random popular cover; "
                         "token_centroid=per-token cos to the per-position AVG popular grid "
                         "(discriminative centroid); centroid=pooled cos to centroid (degenerate)")
    gn.add_argument("--max_items", type=int, default=0)
    gn.add_argument("--eps", type=float, default=16.0)
    gn.add_argument("--alpha", type=float, default=1.0)
    gn.add_argument("--iters", type=int, default=300)
    gn.add_argument("--batch_size", type=int, default=8)
    gn.add_argument("--random_init", action="store_true")
    gn.add_argument("--seed", type=int, default=0)
    gn.add_argument("--shard_id", type=int, default=0)
    gn.add_argument("--num_shards", type=int, default=1)
    gn.add_argument("--cos_threshold", type=float, default=0.5)
    gn.add_argument("--llava_id", default=DEFAULT_LLAVA_ID)
    gn.add_argument("--device", default="cuda:0")
    gn.set_defaults(func=generate)

    ea = sub.add_parser("embed_asr")
    ea.add_argument("--manifest", required=True)
    ea.add_argument("--cos_threshold", type=float, default=0.5)
    ea.set_defaults(func=embed_asr)

    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help(); return
    args.func(args)


if __name__ == "__main__":
    main()
