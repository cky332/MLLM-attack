#!/usr/bin/env python3
r"""xtransfer_attack.py — black-box TRANSFER attack on MLLM-MSR's CLIP, adapting
X-Transfer (Huang et al., ICML 2025) surrogate-scaling + UCB bandit selection.

Threat model (TRUE black-box w.r.t. the victim encoder):
  * We do NOT use MLLM-MSR's CLIP (openai/clip-vit-large-patch14-336) at all.
  * We craft the perturbation on a DIVERSE POOL of OTHER public CLIPs
    (LAION-2B/400M, DataComp-L, CommonPool-L, YFCC; arch B-32/B-16/L-14/RN50),
    and rely on transfer to the unseen victim CLIP.
  * No query to the recommender during crafting (zero-query transfer).

Method (per CANDIDATE image; targeted = impersonate the most-popular cover):
  Build per-surrogate target c_i = f_i(popular_cover)  (top-1, or centroid of top-N).
  For T steps (X-Transfer Algorithm 1, per-image targeted variant):
     UCB score per surrogate:  mu_i = R_i + sqrt(2 ln n / n_i)
     select top-k "hardest" surrogates (highest residual loss),
     push f_i(x0+delta) -> c_i:  L = mean_i (1 - cos),
     delta <- delta - alpha*sign(grad),  project ||delta||_inf <= eps, clamp [0,1],
     update reward R_i (moving avg of the 1-cos loss) so under-fooled surrogates
     get picked more -> perturbation becomes more transferable.
  Save x_adv = clip(x0+delta) keyed <item>.png  ->  feed to eval_illusion_sft.py.

Usage:
  pip install open_clip_torch
  python xtransfer_attack.py generate \
      --src_dir $COVERS --pairs_csv $PAIRS --title_csv $TITLE \
      --out_dir results/xtransfer/images --clean_resized_dir results/xtransfer/clean_resized \
      --items_csv $PILOT --top_n 1 --k 4 --eps 16 --iters 300 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from illusion_attack import (
    IMG_EXTS,
    load_image_as_01,
    load_item_popularity,
    load_titles,
    save_01_image,
)
from illusion_metrics import embedding_alignment_asr

# Diverse surrogate pool (arch:pretrained-tag), excluding the victim's openai CLIP.
# Spans LAION-2B / LAION-400M / DataComp-L / CommonPool-L / YFCC and B-32/B-16/L-14/RN50.
DEFAULT_SURROGATES = [
    "ViT-B-32:laion2b_s34b_b79k",
    "ViT-B-16:laion2b_s34b_b88k",
    "ViT-L-14:laion2b_s32b_b82k",
    "ViT-B-32:laion400m_e32",
    "ViT-B-16:datacomp_l_s1b_b8k",
    "ViT-B-16:commonpool_l_s1b_b8k",
    "RN50:yfcc15m",
]
ATTACK_INPUT = 336  # base resolution we optimise delta at (matches clean_resized)


class SurrogatePool:
    """A pool of frozen open_clip image encoders with differentiable forward."""

    def __init__(self, specs, device="cuda:0", dtype="fp32"):
        import torch
        try:
            import open_clip  # noqa: F401
        except ImportError as e:
            raise SystemExit("open_clip not installed. Run: pip install open_clip_torch") from e
        from torchvision.transforms import Normalize

        self.torch = torch
        self.device = device
        self.tdtype = torch.float16 if dtype == "fp16" else torch.float32
        self.models, self.sizes, self.means, self.stds, self.names = [], [], [], [], []
        for spec in specs:
            arch, tag = spec.split(":", 1)
            model, _, prep = open_clip.create_model_and_transforms(arch, pretrained=tag)
            model = model.to(device=device, dtype=self.tdtype).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            s = model.visual.image_size
            s = int(s[0]) if isinstance(s, (tuple, list)) else int(s)
            mean = std = None
            for t in getattr(prep, "transforms", []):
                if isinstance(t, Normalize):
                    mean, std = list(t.mean), list(t.std)
            if mean is None:  # CLIP defaults
                mean = [0.48145466, 0.4578275, 0.40821073]
                std = [0.26862954, 0.26130258, 0.27577711]
            self.models.append(model)
            self.sizes.append(s)
            self.means.append(torch.tensor(mean, device=device).view(1, 3, 1, 1))
            self.stds.append(torch.tensor(std, device=device).view(1, 3, 1, 1))
            self.names.append(spec)
            print(f"  [surrogate] {spec}  input={s}")
        self.N = len(self.models)
        print(f"[pool] loaded {self.N} surrogate encoders on {device} ({dtype})")

    def feat(self, i, x01):
        """Differentiable: x01 (1,3,H,W) in [0,1] -> L2-normalised embedding of surrogate i."""
        import torch.nn.functional as F
        s = self.sizes[i]
        x = F.interpolate(x01, size=(s, s), mode="bicubic", align_corners=False)
        x = (x - self.means[i]) / self.stds[i]
        z = self.models[i].encode_image(x.to(self.tdtype))
        return (z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)).float()

    def build_targets(self, cover_paths):
        """Per surrogate: c_i = normalised mean of f_i over the popular covers."""
        torch = self.torch
        targets = []
        with torch.no_grad():
            for i in range(self.N):
                embs = []
                for p in cover_paths:
                    x = torch.tensor(load_image_as_01(p, ATTACK_INPUT)[None], device=self.device)
                    embs.append(self.feat(i, x))
                c = torch.cat(embs, 0).mean(0)
                targets.append((c / c.norm().clamp_min(1e-12)))
        return targets

    def mean_cos(self, x01, targets):
        torch = self.torch
        with torch.no_grad():
            cs = [float((self.feat(i, x01) * targets[i]).sum().item()) for i in range(self.N)]
        return float(np.mean(cs))


def ucb_pgd(pool, x01, targets, k, iters, eps, alpha, m):
    """Per-image targeted UCB-PGD (X-Transfer Algorithm 1, sample-specific variant)."""
    torch = pool.torch
    N = pool.N
    R = np.zeros(N)
    T = np.zeros(N, dtype=np.int64)
    n = 0
    delta = torch.zeros_like(x01, requires_grad=True)
    for _ in range(iters):
        n += 1
        ucb = R + np.sqrt(2.0 * math.log(n + 1) / (T + 1e-6))
        sel = np.argsort(-ucb)[:k]
        x_adv = torch.clamp(x01 + delta, 0, 1)
        losses = [1.0 - (pool.feat(int(i), x_adv) * targets[int(i)]).sum() for i in sel]
        L = torch.stack(losses).mean()
        grad = torch.autograd.grad(L, delta)[0]
        with torch.no_grad():
            delta -= alpha * grad.sign()
            delta.clamp_(-eps, eps)
            delta.copy_(torch.clamp(x01 + delta, 0, 1) - x01)
            for idx, i in enumerate(sel):
                li = float(losses[idx].item())
                R[int(i)] = (1 - m) * R[int(i)] + m * li
                T[int(i)] += 1
    return torch.clamp(x01 + delta, 0, 1).detach(), T


def _path_by_item(src_dir):
    m = {}
    for p in Path(src_dir).glob("*"):
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


def generate(args):
    import torch

    specs = (args.surrogates.split(",") if args.surrogates else DEFAULT_SURROGATES)
    pool = SurrogatePool([s.strip() for s in specs], device=args.device, dtype=args.dtype)

    path_by_item = _path_by_item(args.src_dir)
    counts = load_item_popularity(args.pairs_csv)
    titles = load_titles(args.title_csv) if args.title_csv else {}
    ranked = [it for it in sorted(counts, key=lambda kk: -counts[kk]) if it in path_by_item]
    top = ranked[: args.top_n]
    if not top:
        raise SystemExit("No popular covers found (check --src_dir/--pairs_csv)")
    print(f"[target] {len(top)} popular cover(s): " +
          ", ".join(f"{it}(cnt={counts.get(it)})" for it in top[:5]))
    targets = pool.build_targets([path_by_item[it] for it in top])

    items = _resolve_items(args, path_by_item)
    eps, alpha = args.eps / 255.0, args.alpha / 255.0
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = Path(args.clean_resized_dir) if args.clean_resized_dir else None
    if clean_dir:
        clean_dir.mkdir(parents=True, exist_ok=True)
    print(f"[generate] items={len(items)} N_surrogates={pool.N} k={args.k} "
          f"eps={args.eps}/255 iters={args.iters} top_n={args.top_n}")

    rows = []
    for idx, it in enumerate(items):
        try:
            x01 = torch.tensor(load_image_as_01(path_by_item[it], ATTACK_INPUT)[None], device=args.device)
        except Exception as e:
            print(f"  skip {it}: {e}"); continue
        cos_clean = pool.mean_cos(x01, targets)
        x_adv, T = ucb_pgd(pool, x01, targets, args.k, args.iters, eps, alpha, args.m)
        cos_adv = pool.mean_cos(x_adv, targets)
        a = x_adv[0].float().cpu().numpy(); c = x01[0].float().cpu().numpy()
        save_01_image(a, out_dir / f"{it}.png")
        if clean_dir:
            save_01_image(c, clean_dir / f"{it}.png")
        rows.append({"item_id": it, "cos_clean": cos_clean, "cos_adv": cos_adv,
                     "linf_255": float(np.max(np.abs(a - c)) * 255.0)})
        if (idx + 1) % 20 == 0 or idx + 1 == len(items):
            mc = np.mean([r["cos_clean"] for r in rows]); ma = np.mean([r["cos_adv"] for r in rows])
            print(f"  {idx+1}/{len(items)}  surrogate cos {mc:.3f} -> {ma:.3f}")

    suffix = f"_shard{args.shard_id}" if getattr(args, "num_shards", 1) > 1 else ""
    man = pd.DataFrame(rows)
    man.to_csv(out_dir / f"manifest{suffix}.csv", index=False)
    asr = embedding_alignment_asr(man["cos_clean"].values, man["cos_adv"].values, args.cos_threshold)
    with open(out_dir / f"summary{suffix}.json", "w") as f:
        json.dump({"attack": "xtransfer_blackbox", "surrogates": pool.names,
                   "top_n": args.top_n, "k": args.k, "eps_255": args.eps,
                   "iters": args.iters, "surrogate_asr": asr}, f, indent=2)
    print("\n" + "=" * 64)
    print("X-TRANSFER BLACK-BOX ATTACK — surrogate-space alignment")
    print("=" * 64)
    print(f"images: {asr['n']}  mean surrogate cos {asr['mean_cos_clean']:.3f} -> "
          f"{asr['mean_cos_attacked']:.3f} (gain {asr['mean_cos_gain']:+.3f})")
    print(f"mean ||delta||_inf = {man['linf_255'].mean():.2f}/255 (budget {args.eps})")
    print("NOTE: surrogate cos is on the POOL, not the victim. Real test = eval_illusion_sft.py "
          "on MLLM-MSR's CLIP (transfer).")


def main():
    ap = argparse.ArgumentParser(description="X-Transfer-style black-box transfer attack on MLLM-MSR")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("generate")
    g.add_argument("--src_dir", required=True)
    g.add_argument("--pairs_csv", required=True)
    g.add_argument("--title_csv", default=None)
    g.add_argument("--out_dir", required=True)
    g.add_argument("--clean_resized_dir", default=None)
    g.add_argument("--items_csv", default=None)
    g.add_argument("--top_n", type=int, default=1, help="1=single hottest cover; >1=centroid of top-N")
    g.add_argument("--k", type=int, default=4, help="surrogates selected per step (UCB top-k)")
    g.add_argument("--m", type=float, default=0.5, help="reward moving-average factor")
    g.add_argument("--eps", type=float, default=16.0)
    g.add_argument("--alpha", type=float, default=1.0)
    g.add_argument("--iters", type=int, default=300)
    g.add_argument("--surrogates", default=None,
                   help="comma-sep arch:tag list; default = diverse pool (no openai CLIP)")
    g.add_argument("--max_items", type=int, default=0)
    g.add_argument("--num_shards", type=int, default=1)
    g.add_argument("--shard_id", type=int, default=0)
    g.add_argument("--cos_threshold", type=float, default=0.5)
    g.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    g.add_argument("--device", default="cuda:0")
    g.set_defaults(func=generate)

    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help(); return
    args.func(args)


if __name__ == "__main__":
    main()
