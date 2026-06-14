#!/usr/bin/env python3
r"""diffusion_illusion_attack.py — diffusion latent-space adversarial illusion.

Instead of pixel PGD (high-frequency noise), optimise a perturbation delta in the
LATENT space of a pretrained Stable Diffusion model, so the regenerated cover stays
on the natural-image manifold (looks like the original product) yet its CLIP
embedding is pulled toward the most-popular item's embedding (adversarial illusion).

Adapted to MLLM-MSR: the encoder Phi is MLLM-MSR's own visual backbone
openai/clip-vit-large-patch14-336 (336 px, pooled get_image_features), the same
representation as the v1 image<->image attack -> the produced cover plugs straight
into eval_illusion_sft.py.

Pipeline per cover x0 (steps B+C of the spec):
  z0   = VAE.encode(2x0-1).mean * 0.18215                 # latent (1,4,42,42) at 336px
  z_t  = DDIM.add_noise(z0, eps_fixed, t=timesteps[inject])
  for Adam step in 1..iters (only delta is trained):
     z   = z_t + eta * delta
     for t in timesteps[inject:]: z = DDIM.step(UNet(z,t,h_null), t, z)   # denoise
     x   = VAE.decode(z/0.18215) -> [0,1]
     x   = clip(x0 + clip(x-x0, +-eps_pix), 0, 1)          # pixel budget (stealth)
     L   = lam_align(1-cos(Phi(x),c_target)) + lam_clip||Phi(x)-Phi(x0)||^2
            + lam_ssim(1-SSIM(x,x0)) + lam_reg mean(delta^2)
     L.backward(); opt.step()                              # grad through decode/UNet x N/inject

Deps:  pip install diffusers
       a Stable Diffusion checkpoint (default runwayml/stable-diffusion-v1-5, ~4GB).
NOTE:  back-prop runs through the (inject..0) UNet chain -> memory heavy; uses UNet
       gradient checkpointing. Reduce --ddim_steps / --iters or use --dtype fp16 if OOM.

Known limitation (per the spec, observed on VIP5): under the same budget this is
usually WEAKER than free-pixel PGD, because diffusion keeps the perturbation on the
natural-image manifold (no high-frequency energy, which is where CLIP is most
attackable). It trades effectiveness for naturalness/stealth.

Usage:
    python diffusion_illusion_attack.py generate \
        --src_dir $COVERS --pairs_csv $PAIRS --title_csv $TITLE \
        --items_csv /tmp/pos_items.csv \
        --out_dir results/diffusion/images --clean_resized_dir results/diffusion/clean_resized \
        --target_mode top1 --iters 30 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from illusion_attack import (
    CLIPIllusionEncoder,
    IMG_EXTS,
    load_image_as_01,
    load_item_popularity,
    load_titles,
    save_01_image,
)

DEFAULT_SD = "runwayml/stable-diffusion-v1-5"


# ---------------------------------------------------------------------------
# Differentiable SSIM (Gaussian-window, [0,1] images)
# ---------------------------------------------------------------------------
def _gauss_window(torch, size, sigma, channels, device, dtype):
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win2d = g[:, None] @ g[None, :]
    return win2d.expand(channels, 1, size, size).contiguous()


def _ssim(torch, F, x, y, win):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad, ch = win.shape[-1] // 2, x.shape[1]
    mu_x = F.conv2d(x, win, padding=pad, groups=ch)
    mu_y = F.conv2d(y, win, padding=pad, groups=ch)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = F.conv2d(x * x, win, padding=pad, groups=ch) - mu_x2
    sy = F.conv2d(y * y, win, padding=pad, groups=ch) - mu_y2
    sxy = F.conv2d(x * y, win, padding=pad, groups=ch) - mu_xy
    s_map = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx + sy + C2))
    return s_map.mean()


# ---------------------------------------------------------------------------
# Diffusion + CLIP attack engine
# ---------------------------------------------------------------------------
class DiffusionIllusion:
    def __init__(self, sd_id, clip_id, device="cuda:0", dtype="fp32",
                 ddim_steps=20, inject_step=10, eta=1.0):
        import torch
        try:
            from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        except ImportError as e:
            raise SystemExit("diffusers not installed. Run: pip install diffusers") from e
        import torch.nn.functional as F

        self.torch = torch
        self.F = F
        self.device = device
        self.tdtype = torch.float16 if dtype == "fp16" else torch.float32
        self.eta = eta

        self.vae = AutoencoderKL.from_pretrained(sd_id, subfolder="vae").to(device, self.tdtype).eval()
        self.unet = UNet2DConditionModel.from_pretrained(sd_id, subfolder="unet").to(device, self.tdtype).eval()
        self.sched = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        self.sched.set_timesteps(ddim_steps, device=device)
        self.timesteps = self.sched.timesteps
        self.inject = inject_step
        for m in (self.vae, self.unet):
            for p in m.parameters():
                p.requires_grad_(False)
        self.unet.enable_gradient_checkpointing()
        # null (empty) text embedding
        cad = self.unet.config.cross_attention_dim
        self.h_null = torch.zeros(1, 77, cad, device=device, dtype=self.tdtype)

        # Phi = MLLM-MSR's visual backbone (CLIP ViT-L/14-336), pooled embedding
        self.phi = CLIPIllusionEncoder(clip_id, device=device, dtype="fp32")
        self.size = self.phi.input_size  # 336
        self.win = _gauss_window(torch, 11, 1.5, 3, device, torch.float32)

    # ---- Phi ----
    def embed(self, x01):
        return self.phi.encode_image_01(x01.float())  # L2-normalised (B,D)

    # ---- diffusion regen with current delta ----
    def _regen(self, z_t, delta):
        torch = self.torch
        z = z_t + self.eta * delta
        for t in self.timesteps[self.inject:]:
            eps_hat = self.unet(z, t, encoder_hidden_states=self.h_null).sample
            z = self.sched.step(eps_hat, t, z).prev_sample
        x = self.vae.decode(z / 0.18215).sample          # [-1,1]
        return (x / 2 + 0.5).clamp(0, 1)                  # [0,1]

    def encode_latent(self, x01):
        torch = self.torch
        with torch.no_grad():
            posterior = self.vae.encode((2 * x01 - 1).to(self.tdtype)).latent_dist
            return posterior.mean * 0.18215

    def attack(self, x01, c_target, iters, lr, lam_align, lam_clip, lam_ssim,
               lam_reg, eps_pix):
        torch, F = self.torch, self.F
        x0 = x01.detach()
        z0 = self.encode_latent(x0)
        eps_fixed = torch.randn_like(z0)
        t_inj = self.timesteps[self.inject]
        z_t = self.sched.add_noise(z0, eps_fixed, t_inj).detach()
        f0 = self.embed(x0).detach()

        delta = torch.zeros_like(z0, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        last = None
        for _ in range(iters):
            x = self._regen(z_t, delta)                          # (1,3,336,336)
            if eps_pix is not None:
                x = (x0 + (x - x0).clamp(-eps_pix, eps_pix)).clamp(0, 1)
            f = self.embed(x)
            cos = (f * c_target).sum()
            L = (lam_align * (1.0 - cos)
                 + lam_clip * F.mse_loss(f, f0)
                 + lam_ssim * (1.0 - _ssim(torch, F, x.float(), x0.float(), self.win))
                 + lam_reg * delta.float().pow(2).mean())
            opt.zero_grad()
            L.backward()
            opt.step()
            last = (float(cos.item()), float(L.item()))
        with torch.no_grad():
            x = self._regen(z_t, delta)
            if eps_pix is not None:
                x = (x0 + (x - x0).clamp(-eps_pix, eps_pix)).clamp(0, 1)
            cos_adv = float((self.embed(x) * c_target).sum().item())
        return x.detach(), float((f0 * c_target).sum().item()), cos_adv, last


# ---------------------------------------------------------------------------
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
        items = [it for it in dict.fromkeys(df[col].astype(str).str.strip()) if it in path_by_item]
    else:
        items = sorted(path_by_item)
    if args.max_items > 0:
        items = items[: args.max_items]
    if args.num_shards > 1:
        items = items[args.shard_id :: args.num_shards]
    return items


def build_target(eng, args, path_by_item):
    """c_target = (normalised) mean Phi over the top-N popular covers."""
    torch = eng.torch
    counts = load_item_popularity(args.pairs_csv)
    titles = load_titles(args.title_csv) if args.title_csv else {}
    ranked = [it for it in sorted(counts, key=lambda k: -counts[k]) if it in path_by_item]
    n = 1 if args.target_mode == "top1" else args.top_n
    top = ranked[:n]
    if not top:
        raise SystemExit("No popular covers found (check --src_dir/--pairs_csv)")
    print(f"[target] mode={args.target_mode} using {len(top)} cover(s): " +
          ", ".join(f"{it}(cnt={counts.get(it)})" for it in top[:5]))
    embs = []
    with torch.no_grad():
        for it in top:
            x = torch.tensor(load_image_as_01(path_by_item[it], eng.size)[None], device=eng.device)
            embs.append(eng.embed(x))
    c = torch.cat(embs, 0).mean(0)
    return c / c.norm().clamp_min(1e-12)


def generate(args):
    import torch

    eng = DiffusionIllusion(args.sd_id, args.clip_id, device=args.device, dtype=args.dtype,
                            ddim_steps=args.ddim_steps, inject_step=args.inject_step, eta=args.eta)
    path_by_item = _path_by_item(args.src_dir)
    c_target = build_target(eng, args, path_by_item)
    items = _resolve_items(args, path_by_item)
    eps_pix = None if args.eps_pix < 0 else args.eps_pix
    print(f"[diffusion] items={len(items)} ddim={args.ddim_steps} inject={args.inject_step} "
          f"iters={args.iters} eps_pix={eps_pix}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = Path(args.clean_resized_dir) if args.clean_resized_dir else None
    if clean_dir:
        clean_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, it in enumerate(items):
        try:
            x0 = torch.tensor(load_image_as_01(path_by_item[it], eng.size)[None], device=args.device)
        except Exception as e:
            print(f"  skip {it}: {e}"); continue
        x_adv, cos_clean, cos_adv, last = eng.attack(
            x0, c_target, args.iters, args.lr, args.lam_align, args.lam_clip,
            args.lam_ssim, args.lam_reg, eps_pix)
        a = x_adv[0].float().cpu().numpy(); c = x0[0].float().cpu().numpy()
        save_01_image(a, out_dir / f"{it}.png")
        if clean_dir:
            save_01_image(c, clean_dir / f"{it}.png")
        rows.append({"item_id": it, "cos_clean": cos_clean, "cos_adv": cos_adv,
                     "linf_255": float(np.max(np.abs(a - c)) * 255.0)})
        print(f"  {i+1}/{len(items)} {it}: cos {cos_clean:.3f} -> {cos_adv:.3f}")

    suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    man = pd.DataFrame(rows)
    man.to_csv(out_dir / f"manifest{suffix}.csv", index=False)
    if len(man):
        print("\n" + "=" * 60)
        print(f"DIFFUSION ILLUSION — cos {man['cos_clean'].mean():.3f} -> {man['cos_adv'].mean():.3f} "
              f"| mean ||delta||_inf={man['linf_255'].mean():.1f}/255 | n={len(man)}")
        with open(out_dir / f"summary{suffix}.json", "w") as f:
            json.dump({"attack": "diffusion_illusion", "target_mode": args.target_mode,
                       "ddim_steps": args.ddim_steps, "inject_step": args.inject_step,
                       "iters": args.iters, "eps_pix": eps_pix,
                       "mean_cos_clean": float(man["cos_clean"].mean()),
                       "mean_cos_adv": float(man["cos_adv"].mean())}, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Diffusion latent-space adversarial illusion on MLLM-MSR")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("generate")
    g.add_argument("--src_dir", required=True)
    g.add_argument("--pairs_csv", required=True)
    g.add_argument("--title_csv", default=None)
    g.add_argument("--items_csv", default=None)
    g.add_argument("--out_dir", required=True)
    g.add_argument("--clean_resized_dir", default=None)
    g.add_argument("--target_mode", default="top1", choices=["top1", "mean"])
    g.add_argument("--top_n", type=int, default=10, help="#covers for --target_mode mean")
    g.add_argument("--ddim_steps", type=int, default=20)
    g.add_argument("--inject_step", type=int, default=10)
    g.add_argument("--eta", type=float, default=1.0, help="latent perturbation scale")
    g.add_argument("--iters", type=int, default=30)
    g.add_argument("--lr", type=float, default=0.05)
    g.add_argument("--lam_align", type=float, default=1.0)
    g.add_argument("--lam_clip", type=float, default=0.5)
    g.add_argument("--lam_ssim", type=float, default=0.3)
    g.add_argument("--lam_reg", type=float, default=0.01)
    g.add_argument("--eps_pix", type=float, default=0.12, help="pixel L_inf budget; <0 = unbounded")
    g.add_argument("--max_items", type=int, default=0)
    g.add_argument("--num_shards", type=int, default=1)
    g.add_argument("--shard_id", type=int, default=0)
    g.add_argument("--sd_id", default=DEFAULT_SD)
    g.add_argument("--clip_id", default="openai/clip-vit-large-patch14-336")
    g.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    g.add_argument("--device", default="cuda:0")
    g.set_defaults(func=generate)
    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help(); return
    args.func(args)


if __name__ == "__main__":
    main()
