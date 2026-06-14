#!/usr/bin/env python3
r"""agenttypo_attack.py — ATPI-lite: AgentTypo's typographic AUTO-OPTIMISATION
core, transferred to MLLM-MSR.

What this is (and what it is NOT)
---------------------------------
AgentTypo (Li et al., "Adaptive Typographic Prompt Injection Attacks against
Black-box Multimodal Agents", arXiv 2510.04257) is black-box typographic prompt
injection against multimodal *web agents*. Its core algorithm, **ATPI**, does
NOT hand-pick the overlay's style; it SEARCHES the typographic parameters
(insert position, font size, colour, contrast, transparency, line count —
their Table I) with a Tree-structured Parzen Estimator (TPE / Bayesian
optimisation via `optuna`), jointly maximising machine-readability and
minimising human-visibility:

    AgentTypo (Eq.4):  min_x  L_prompt_rebuilt(x)  +  lambda * LPIPS(I_orig, I_alt)
                       L_prompt_rebuilt = -mean_i cos( E_text(P), E_text(caption_i(x)) )

Only the **ATPI core** transfers to MLLM-MSR. The AgentTypo-pro machinery
(attacker / scorer / summariser LLMs, RAG strategy library, multi-step action
hijacking) is agent-specific; MLLM-MSR is a single-turn Yes/No scorer, so that
part is intentionally NOT reproduced.

Adaptation (faithful structure, direct objective)
--------------------------------------------------
AgentTypo uses the prompt-reconstruction *proxy* (L_prompt_rebuilt) only because
it is black-box on the agent. We have white-box *query* access to MLLM-MSR, so
we replace the proxy with the quantity we actually want to move — the
recommender's P(Yes) — while keeping AgentTypo's exact optimiser (optuna TPE),
its Table-I search space, and its additive stealth trade-off:

    maximise   mean_u P(Yes | prompt_u, overlay(x; theta))  -  lambda * LPIPS(x, overlay)
    over theta = (x_frac, y_frac, font_px, colour|bg_avg, opacity, stroke, wrap_frac)

This is a **black-box QUERY attack** (forward passes only, no gradients) — it
slots between the white-box feature illusion (illusion_attack_llava.py) and the
hand-tuned overlay (generate_attacked_images.py / ipi_attack.py) in the threat
model. The injected TEXT is fixed (given, exactly like ATPI's prompt P); only
the TYPOGRAPHY is searched. Output images + clean_resized plug straight into
eval_illusion_sft.py, so the metrics are directly comparable to the other
attacks (decision-flip ASR, P(Yes) lift, rank promotion, backfire).

Stealthiness uses LPIPS when the `lpips` package is available (faithful to
AgentTypo Eq.3); otherwise it falls back to 1-SSIM, then pixel RMSE.

Usage
-----
    python agenttypo_attack.py generate \
        --src_dir /path/to/MicroLens-50k_covers \
        --out_dir results/agenttypo/images \
        --clean_resized_dir results/agenttypo/clean_resized \
        --items_csv  /path/to/Split/test_pairs.csv \
        --pref_csv   /path/to/user_preference_recurrent.csv \
        --title_csv  ../../data/microlens/MicroLens-50k_titles.csv \
        --peft_model_id /path/to/...llava-v1.6-mistral-7b-hf-lora-recurrent-e4-r16 \
        --text_key ipi_yes_en --posonly --n_trials 40 --n_prompts 8 --lam 0.3

Then evaluate exactly like the other attacks (posonly subset = realistic):
    python eval_illusion_sft.py --peft_model_id "$LORA" \
        --test_pairs_csv "$PILOT" --clean_image_dir results/agenttypo/clean_resized \
        --attacked_image_dir results/agenttypo/images \
        --title_csv "$TITLE" --pref_csv "$PREF" --attack_name agenttypo_atpi \
        --output_report results/agenttypo/recsys_atpi_posonly.json \
        --candidates_per_user 21 --batch_size 1 --num_proc 3
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Exact prompt template + IO the eval uses -> objective matches final scoring.
from eval_item_ranking import PROMPT_TEMPLATE, load_prefs, load_titles
from illusion_metrics import yesno_softmax
# Reuse the project's font loading + CJK-safe wrapping so rendering matches the
# hand-tuned overlay attack exactly.
from generate_attacked_images import load_font, wrap_text_to_width
from attack_config import ATTACK_TEXTS, CJK_FONT_PATH

DEFAULT_BASE = "llava-hf/llava-v1.6-mistral-7b-hf"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
COVER_SIZE = 336  # same 336 bicubic resize as illusion_attack.load_image_as_01


# ---------------------------------------------------------------------------
# MLLM-MSR P(Yes) scorer — in-process, mirrors eval_illusion_sft.score_with_lora
# ---------------------------------------------------------------------------
class LoraYesScorer:
    """Loads base + LoRA once; scores (prompt, PIL image) -> P(Yes), exactly like
    test_with_llava_sft.py / eval_illusion_sft.py (generate 1 token, read the
    Yes/No logits, softmax)."""

    def __init__(self, base_model_id, peft_model_id, device="cuda:0"):
        import torch
        from transformers import (
            LlavaNextForConditionalGeneration,
            LlavaNextProcessor,
        )

        self.torch = torch
        self.device = device
        kw = dict(cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
                  torch_dtype=torch.float16)
        try:
            mdl = LlavaNextForConditionalGeneration.from_pretrained(
                base_model_id, attn_implementation="flash_attention_2", **kw)
        except Exception as e:  # flash-attn not built in this env
            print(f"[agenttypo] flash_attention_2 unavailable ({e}); using sdpa")
            mdl = LlavaNextForConditionalGeneration.from_pretrained(
                base_model_id, attn_implementation="sdpa", **kw)
        proc = LlavaNextProcessor.from_pretrained(base_model_id)
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
        if peft_model_id:
            from peft import PeftModel
            mdl = PeftModel.from_pretrained(mdl, peft_model_id)
        mdl = mdl.eval().to(device)
        mdl.tie_weights()
        proc.tokenizer.add_tokens(["<|image|>", "<pad>"], special_tokens=True)
        self.model = mdl
        self.proc = proc
        self.yes_id = proc.tokenizer.convert_tokens_to_ids("Yes")
        self.no_id = proc.tokenizer.convert_tokens_to_ids("No")
        self.n_calls = 0

    def pyes(self, prompts, images):
        """prompts: list[str]; images: list[PIL]. -> np.array P(Yes) per pair."""
        import torch
        from PIL import ImageOps
        from torch.cuda.amp import autocast

        max_w = max(im.width for im in images)
        max_h = max(im.height for im in images)
        padded = []
        for im in images:
            if im.width == max_w and im.height == max_h:
                padded.append(im)
            else:
                dw, dh = max_w - im.width, max_h - im.height
                padded.append(ImageOps.expand(
                    im, border=(dw // 2, dh // 2, dw - dw // 2, dh - dh // 2),
                    fill="black"))
        inputs = self.proc(text=prompts, images=padded,
                           return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad(), autocast():
            out = self.model.generate(**inputs, max_new_tokens=1,
                                       return_dict_in_generate=True,
                                       output_scores=True)
        s = out["scores"][0]
        yes = s[:, self.yes_id].float().cpu().numpy()
        no = s[:, self.no_id].float().cpu().numpy()
        del inputs, out, s
        torch.cuda.empty_cache()
        self.n_calls += 1
        return yesno_softmax(yes, no)


# ---------------------------------------------------------------------------
# Stealthiness — LPIPS (faithful to AgentTypo Eq.3), else 1-SSIM, else RMSE
# ---------------------------------------------------------------------------
def _gray(a):  # a: HWC float[0,1]
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _ssim(a, b):
    """Global SSIM on grayscale, [0,1] arrays."""
    x, y = _gray(a), _gray(b)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mx * my + c1) * (2 * cov + c2)) /
                 ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2) + 1e-12))


class Stealth:
    def __init__(self, device="cuda:0", metric="auto"):
        self.metric = metric
        self.lpips = None
        if metric in ("auto", "lpips"):
            try:
                import lpips
                import torch
                self.torch = torch
                self.device = device
                self.lpips = lpips.LPIPS(net="alex").to(device).eval()
                self.metric = "lpips"
                print("[agenttypo] stealth metric: LPIPS(alex) (faithful to AgentTypo Eq.3)")
                return
            except Exception as e:
                if metric == "lpips":
                    print(f"[agenttypo] lpips unavailable ({e}); falling back to 1-SSIM")
                self.metric = "ssim"
        print(f"[agenttypo] stealth metric: {self.metric}")

    def distance(self, clean_pil, adv_pil):
        """Perceptual distance; higher = more visible (worse stealth)."""
        a = np.asarray(clean_pil.convert("RGB"), dtype=np.float32) / 255.0
        b = np.asarray(adv_pil.convert("RGB"), dtype=np.float32) / 255.0
        if self.metric == "lpips":
            torch = self.torch
            ta = torch.tensor(a.transpose(2, 0, 1)[None] * 2 - 1,
                              device=self.device, dtype=torch.float32)
            tb = torch.tensor(b.transpose(2, 0, 1)[None] * 2 - 1,
                              device=self.device, dtype=torch.float32)
            with torch.no_grad():
                return float(self.lpips(ta, tb).item())
        if self.metric == "ssim":
            return float(1.0 - _ssim(a, b))
        return float(np.sqrt(((a - b) ** 2).mean()))  # RMSE


# ---------------------------------------------------------------------------
# Typographic rendering with explicit (searched) parameters
# ---------------------------------------------------------------------------
def render_overlay(clean_pil, text, theta, font_path):
    """Draw `text` onto clean_pil using the typographic params in `theta`."""
    base = clean_pil.convert("RGBA")
    W, H = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = load_font(font_path, max(8, int(theta["font_px"])))
    wrapped = wrap_text_to_width(draw, text, font, max(8, int(theta["wrap_frac"] * W)))
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = int(theta["x_frac"] * max(1, W - tw))
    y = int(theta["y_frac"] * max(1, H - th))

    if theta["color_mode"] == "bg_avg":
        arr = np.asarray(base.convert("RGB"), dtype=np.float32)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + tw), min(H, y + th)
        region = (arr[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else arr).reshape(-1, 3).mean(0)
        off = theta.get("bright_off", 20)
        color = tuple(int(np.clip(region[c] + off, 0, 255)) for c in range(3))
    else:
        color = (theta["r"], theta["g"], theta["b"])

    opacity = int(theta["opacity"])
    kw = dict(xy=(x, y), text=wrapped, font=font, fill=(*color, opacity))
    if int(theta["stroke"]) > 0:
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        sfill = (0, 0, 0) if lum > 128 else (255, 255, 255)
        kw["stroke_width"] = int(theta["stroke"])
        kw["stroke_fill"] = (*sfill, opacity)
    draw.multiline_text(**kw)
    return Image.alpha_composite(base, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# ATPI search space (AgentTypo Table I) — one suggest() shared by TPE + fallback
# ---------------------------------------------------------------------------
def suggest_theta(trial, size):
    W, H = size
    color_mode = trial.suggest_categorical("color_mode", ["bg_avg", "rgb"])
    theta = {
        "color_mode": color_mode,
        "x_frac": trial.suggest_float("x_frac", 0.0, 1.0),       # Insert Position
        "y_frac": trial.suggest_float("y_frac", 0.0, 1.0),       # Insert Position
        "font_px": trial.suggest_int("font_px", 10, max(16, int(H * 0.18))),  # Font Size
        "opacity": trial.suggest_int("opacity", 40, 255),        # Transparency
        "stroke": trial.suggest_int("stroke", 0, 3),             # Contrast (via stroke)
        "wrap_frac": trial.suggest_float("wrap_frac", 0.30, 0.95),  # Line Numbers (via wrap)
    }
    if color_mode == "bg_avg":
        theta["bright_off"] = trial.suggest_int("bright_off", -40, 40)  # Colour (bg-avg)
    else:
        theta["r"] = trial.suggest_int("r", 0, 255)              # Colour
        theta["g"] = trial.suggest_int("g", 0, 255)
        theta["b"] = trial.suggest_int("b", 0, 255)
    return theta


class _RandTrial:
    """Minimal optuna-trial stand-in so the search still runs without optuna."""

    def __init__(self, rng):
        self.rng = rng

    def suggest_float(self, name, lo, hi):
        return float(self.rng.uniform(lo, hi))

    def suggest_int(self, name, lo, hi):
        return int(self.rng.integers(lo, hi + 1))

    def suggest_categorical(self, name, choices):
        return choices[int(self.rng.integers(0, len(choices)))]


def _make_study(sampler_name, seed):
    """Return (study, label) or (None, label) to signal the random fallback."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = (optuna.samplers.RandomSampler(seed=seed) if sampler_name == "random"
                   else optuna.samplers.TPESampler(seed=seed))
        return optuna.create_study(direction="maximize", sampler=sampler), f"optuna-{sampler_name}"
    except Exception as e:
        return None, f"builtin-random ({e})"


# ---------------------------------------------------------------------------
# Per-item ATPI optimisation
# ---------------------------------------------------------------------------
def optimize_item(clean_pil, text, scorer, stealth, prompts, font_path,
                  n_trials, lam, sampler_name, seed):
    size = clean_pil.size
    imgs_clean = [clean_pil] * len(prompts)
    pyes_clean = float(scorer.pyes(prompts, imgs_clean).mean())

    best = {"obj": -1e9, "pyes": pyes_clean, "stealth": 0.0, "theta": None, "adv": clean_pil}
    traj = []

    def evaluate(theta):
        adv = render_overlay(clean_pil, text, theta, font_path)
        py = float(scorer.pyes(prompts, [adv] * len(prompts)).mean())
        st = stealth.distance(clean_pil, adv)
        return adv, py, st, (py - lam * st)

    study, label = _make_study(sampler_name, seed)
    if study is not None:
        def objective(trial):
            theta = suggest_theta(trial, size)
            adv, py, st, obj = evaluate(theta)
            traj.append({"obj": obj, "pyes": py, "stealth": st})
            if obj > best["obj"]:
                best.update(obj=obj, pyes=py, stealth=st, theta=theta, adv=adv)
            return obj
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_trials):
            theta = suggest_theta(_RandTrial(rng), size)
            adv, py, st, obj = evaluate(theta)
            traj.append({"obj": obj, "pyes": py, "stealth": st})
            if obj > best["obj"]:
                best.update(obj=obj, pyes=py, stealth=st, theta=theta, adv=adv)

    return best, pyes_clean, label, traj


# ---------------------------------------------------------------------------
# Item / prompt resolution
# ---------------------------------------------------------------------------
def _path_by_item(src_dir):
    m = {}
    for p in Path(src_dir).glob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            m.setdefault(p.stem, str(p))
    return m


def _resolve_items(args, path_by_item):
    df = pd.read_csv(args.items_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    item_col = "item" if "item" in df.columns else df.columns[0]
    df[item_col] = df[item_col].astype(str).str.strip()
    if args.items:
        want = [s.strip() for s in args.items.split(",") if s.strip()]
    elif args.posonly and "label" in df.columns:
        want = df[df["label"].astype(int) == 1][item_col].tolist()
    else:
        want = df[item_col].tolist()
    items = [it for it in dict.fromkeys(want) if it in path_by_item]
    if args.max_items > 0:
        items = items[: args.max_items]
    return items, df, item_col


def build_prompts(item, df, item_col, prefs, titles, n_prompts, rng):
    """n_prompts user prompts for this item's title: prefer users who actually see
    this item (matches eval), then top up with random users for cross-user
    robustness (the 'push into more users' goal)."""
    title = titles.get(item, "Unknown")
    own = [u for u in df[df[item_col] == item].get("user", pd.Series([], dtype=str))
           .astype(str).str.strip().tolist() if u in prefs]
    own = list(dict.fromkeys(own))
    users = list(own)
    if len(users) < n_prompts:
        pool = [u for u in prefs.keys() if u not in set(users)]
        if pool:
            extra = rng.choice(pool, size=min(n_prompts - len(users), len(pool)),
                               replace=False).tolist()
            users += extra
    elif len(users) > n_prompts:
        users = rng.choice(users, size=n_prompts, replace=False).tolist()
    return [PROMPT_TEMPLATE.format(prefs[u], title) for u in users]


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def generate(args):
    if not CJK_FONT_PATH:
        print("[agenttypo] WARNING: no TTF font found; text size search will be a no-op.")
    text = args.text if args.text else ATTACK_TEXTS[args.text_key]
    print(f"[agenttypo] injected text: {text!r}")

    path_by_item = _path_by_item(args.src_dir)
    items, df, item_col = _resolve_items(args, path_by_item)
    prefs = load_prefs(args.pref_csv)
    titles = load_titles(args.title_csv)
    print(f"[agenttypo] items={len(items)}  n_trials={args.n_trials}  "
          f"n_prompts={args.n_prompts}  lam={args.lam}  sampler={args.sampler}")

    scorer = LoraYesScorer(args.base_model_id, args.peft_model_id, device=args.device)
    stealth = Stealth(device=args.device, metric=args.stealth_metric)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = Path(args.clean_resized_dir) if args.clean_resized_dir else None
    if clean_dir:
        clean_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, item in enumerate(items):
        clean_pil = Image.open(path_by_item[item]).convert("RGB").resize(
            (COVER_SIZE, COVER_SIZE), Image.BICUBIC)
        rng = np.random.default_rng(args.seed + idx)
        prompts = build_prompts(item, df, item_col, prefs, titles, args.n_prompts, rng)
        if not prompts:
            print(f"  skip {item}: no usable user prompts")
            continue

        best, pyes_clean, label, traj = optimize_item(
            clean_pil, text, scorer, stealth, prompts, CJK_FONT_PATH,
            args.n_trials, args.lam, args.sampler, args.seed + idx)

        best["adv"].save(out_dir / f"{item}.png")
        if clean_dir:
            clean_pil.save(clean_dir / f"{item}.png")

        rows.append({
            "item_id": item, "n_prompts": len(prompts),
            "pyes_clean": pyes_clean, "pyes_best": best["pyes"],
            "pyes_lift": best["pyes"] - pyes_clean,
            "stealth": best["stealth"], "objective": best["obj"],
            "theta": json.dumps(best["theta"], ensure_ascii=False),
        })
        if idx == 0:
            print(f"  [sampler] {label}")
        print(f"  [{idx + 1}/{len(items)}] {item}: P(Yes) {pyes_clean:.4f} -> "
              f"{best['pyes']:.4f} ({best['pyes'] - pyes_clean:+.4f})  "
              f"stealth={best['stealth']:.4f}  mode={best['theta']['color_mode']}")

    man = pd.DataFrame(rows)
    man.to_csv(out_dir / "manifest.csv", index=False)
    summary = {
        "attack": "agenttypo_atpi_lite",
        "paper": "AgentTypo (arXiv 2510.04257) — ATPI core only",
        "injected_text": text,
        "n_items": int(len(man)),
        "n_trials": args.n_trials, "n_prompts": args.n_prompts, "lam": args.lam,
        "sampler": args.sampler, "stealth_metric": stealth.metric,
        "mean_pyes_clean": float(man["pyes_clean"].mean()) if len(man) else 0.0,
        "mean_pyes_best": float(man["pyes_best"].mean()) if len(man) else 0.0,
        "mean_pyes_lift": float(man["pyes_lift"].mean()) if len(man) else 0.0,
        "mean_stealth": float(man["stealth"].mean()) if len(man) else 0.0,
        "scorer_forward_calls": scorer.n_calls,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 68)
    print("AGENTTYPO ATPI-LITE — typographic params auto-optimised for P(Yes)")
    print("=" * 68)
    if len(man):
        print(f"items: {len(man)}   mean P(Yes) clean={summary['mean_pyes_clean']:.4f} "
              f"-> best={summary['mean_pyes_best']:.4f} "
              f"(lift {summary['mean_pyes_lift']:+.4f})")
        print(f"mean stealth ({stealth.metric})={summary['mean_stealth']:.4f}   "
              f"improved on {int((man['pyes_lift'] > 1e-4).sum())}/{len(man)} items")
    print(f"manifest -> {out_dir / 'manifest.csv'}")
    print("Next: eval_illusion_sft.py on these images (posonly) to get the "
          "recommendation-level ASR / rank / backfire, comparable to your other attacks.")


def main():
    ap = argparse.ArgumentParser(description="ATPI-lite typographic injection (AgentTypo core) for MLLM-MSR")
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("generate")
    g.add_argument("--src_dir", required=True)
    g.add_argument("--out_dir", required=True)
    g.add_argument("--clean_resized_dir", default=None)
    g.add_argument("--items_csv", required=True, help="test_pairs.csv (user,item,label)")
    g.add_argument("--pref_csv", required=True)
    g.add_argument("--title_csv", required=True)
    g.add_argument("--peft_model_id", default=None, help="your fine-tuned LoRA adapter dir")
    g.add_argument("--base_model_id", default=DEFAULT_BASE)
    g.add_argument("--text_key", default="ipi_yes_en",
                   help=f"key in attack_config.ATTACK_TEXTS (default ipi_yes_en). "
                        f"Available: {list(ATTACK_TEXTS.keys())}")
    g.add_argument("--text", default=None, help="literal injected text (overrides --text_key)")
    g.add_argument("--items", default=None, help="comma-separated item ids (overrides --posonly)")
    g.add_argument("--posonly", action="store_true",
                   help="optimise only label==1 items (realistic promotion threat)")
    g.add_argument("--max_items", type=int, default=0)
    g.add_argument("--n_trials", type=int, default=40, help="ATPI TPE iterations (T)")
    g.add_argument("--n_prompts", type=int, default=8, help="user prompts averaged per objective")
    g.add_argument("--lam", type=float, default=0.3, help="stealth weight (AgentTypo lambda)")
    g.add_argument("--sampler", default="tpe", choices=["tpe", "random"],
                   help="tpe = faithful AgentTypo (optuna TPE); random = ablation")
    g.add_argument("--stealth_metric", default="auto", choices=["auto", "lpips", "ssim", "l2"])
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cuda:0")
    g.set_defaults(func=generate)

    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help(); return
    args.func(args)


if __name__ == "__main__":
    main()
