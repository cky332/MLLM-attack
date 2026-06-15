#!/usr/bin/env python3
r"""metainstruction_attack.py — Invisible, white-box "meta-instruction" attack
on MLLM-MSR (per "Self-interpreting Adversarial Images", Zhang et al.,
arXiv 2407.08970).

What this is
------------
The paper crafts an Lp-bounded INVISIBLE image perturbation that acts as a
cross-modal soft prompt: it backpropagates through the WHOLE VLM (vision tower
-> projector -> language decoder) to make the model's OUTPUT satisfy a
meta-objective, while staying imperceptible. Here the meta-objective is the only
thing MLLM-MSR outputs: the Yes/No interaction decision. So we directly maximise
the recommender's P(Yes) for the candidate item, by gradient descent on its
cover pixels, averaged over many user prompts (so one perturbation pushes the
item up for MANY users, not one).

This is the missing quadrant in the attack suite:
    visible  + output-steering   = ipi/overlay/agenttypo (hard prompt)
    invisible + feature-align    = illusion_attack_llava (stops at encoder)
    invisible + OUTPUT-steering  = THIS  (gradient through the decoder to P(Yes))

Threat model: WHITE-BOX (needs gradients of the full model). Contrast with
agenttypo_attack.py (black-box query) and illusion_attack_llava.py (white-box on
the encoder only).

Objective (L_inf PGD, faithful to the paper's Eq. 2 specialised to a Yes/No head):

    min_delta  mean_u  -log P(Yes | prompt_u, x+delta)
        s.t.   ||delta||_inf <= eps,  x+delta in [0,1]

where P(Yes) = softmax([logit("No"), logit("Yes")])[1] read off the first
generated token's logits — the SAME definition eval_illusion_sft.py scores with.
An optional --reg adds an L2 penalty on delta for extra stealth.

Implementation note (single-crop approximation)
-----------------------------------------------
For a differentiable, gradient-to-pixel forward that yields a SAVABLE single
cover PNG, this perturbs one 336x336 image and splices the vision tower's 576
patch tokens (projector output, CLS dropped, vision_feature_layer=-2) into the
LLM via inputs_embeds — the same single-crop path illusion_attack_llava.py uses
and which is proven to transfer to the eval's anyres forward. The final eval
(eval_illusion_sft.py) still uses the full LlavaNext anyres pipeline, so the
reported P(Yes) here (single-crop) is a proxy; trust the eval for the
comparable, anyres metrics. Sanity-check: the printed clean P(Yes) should be in
the same ballpark as the eval's clean P(Yes).

Usage
-----
    python metainstruction_attack.py generate \
        --src_dir /path/to/MicroLens-50k_covers \
        --out_dir results/metainstruction/images \
        --clean_resized_dir results/metainstruction/clean_resized \
        --items_csv  /path/to/Split/test_pairs.csv \
        --pref_csv   /path/to/user_preference_recurrent.csv \
        --title_csv  ../../data/microlens/MicroLens-50k_titles.csv \
        --peft_model_id /path/to/...llava-v1.6-mistral-7b-hf-lora-recurrent-e4-r16 \
        --posonly --eps 16 --iters 100 --n_prompts 4

Then evaluate exactly like the other attacks (posonly = realistic):
    python eval_illusion_sft.py --peft_model_id "$LORA" \
        --test_pairs_csv "$PILOT" --clean_image_dir results/metainstruction/clean_resized \
        --attacked_image_dir results/metainstruction/images \
        --title_csv "$TITLE" --pref_csv "$PREF" --attack_name metainstruction \
        --output_report results/metainstruction/recsys_meta_posonly.json \
        --candidates_per_user 21 --batch_size 1 --num_proc 3
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from eval_item_ranking import PROMPT_TEMPLATE, load_prefs, load_titles
from illusion_attack import (
    CLIP_MEAN,
    CLIP_STD,
    IMG_EXTS,
    load_image_as_01,
    save_01_image,
)

DEFAULT_BASE = "llava-hf/llava-v1.6-mistral-7b-hf"
COVER_SIZE = 336


# ---------------------------------------------------------------------------
# Full LLaVA+LoRA with a differentiable image->P(Yes) forward
# ---------------------------------------------------------------------------
class MetaInstructionModel:
    """Loads base + LoRA once; exposes a gradient-to-pixel forward to the Yes/No
    logits, plus PGD on the cover image to maximise P(Yes)."""

    def __init__(self, base_model_id, peft_model_id, device="cuda:0"):
        import torch
        from transformers import (
            LlavaNextForConditionalGeneration,
            LlavaNextProcessor,
        )

        self.torch = torch
        self.device = device
        kw = dict(cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
                  torch_dtype=torch.float16, low_cpu_mem_usage=True)
        try:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                base_model_id, attn_implementation="flash_attention_2", **kw)
        except Exception as e:
            print(f"[meta] flash_attention_2 unavailable ({e}); using sdpa")
            model = LlavaNextForConditionalGeneration.from_pretrained(
                base_model_id, attn_implementation="sdpa", **kw)
        proc = LlavaNextProcessor.from_pretrained(base_model_id)
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
        if peft_model_id:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, peft_model_id)
        model = model.eval().to(device)
        model.tie_weights()
        proc.tokenizer.add_tokens(["<|image|>", "<pad>"], special_tokens=True)
        for p in model.parameters():
            p.requires_grad_(False)

        # Resolve the LlavaNext core whether or not it is wrapped by PEFT.
        core = model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            core = model.base_model.model
        self.model = model
        self.core = core
        self.vision_tower = core.vision_tower
        self.projector = core.multi_modal_projector
        self.lm = core.language_model
        self.embed = core.get_input_embeddings()
        self.vfl = getattr(core.config, "vision_feature_layer", -2)
        self.vfs = getattr(core.config, "vision_feature_select_strategy", "default")
        core.config.use_cache = False
        try:
            self.lm.gradient_checkpointing_enable()
        except Exception:
            pass

        self.tok = proc.tokenizer
        self.yes_id = proc.tokenizer.convert_tokens_to_ids("Yes")
        self.no_id = proc.tokenizer.convert_tokens_to_ids("No")
        self.mean = torch.tensor(CLIP_MEAN, device=device, dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device, dtype=torch.float16).view(1, 3, 1, 1)

    # --- vision: differentiable 336 image -> (1, 576, 4096) LLaVA visual tokens
    def _img_tokens(self, x01):
        x = (x01 - self.mean) / self.std
        out = self.vision_tower(x, output_hidden_states=True)
        feat = out.hidden_states[self.vfl]
        if self.vfs == "default":
            feat = feat[:, 1:]  # drop CLS, exactly like LlavaNext
        return self.projector(feat)

    # --- text: prompt (with one <image>) -> (left_emb, right_emb), no grad
    def prep_prompt(self, prompt):
        torch = self.torch
        left, right = prompt.split("<image>")
        li = self.tok(left, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
        ri = self.tok(right, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        with torch.no_grad():
            return self.embed(li).detach(), self.embed(ri).detach()

    def _yesno(self, x01, le, re):
        """-> (logit_yes, logit_no) for one prompt, differentiable in x01."""
        img_tok = self._img_tokens(x01)
        embeds = self.torch.cat([le, img_tok, re], dim=1)
        out = self.lm(inputs_embeds=embeds, use_cache=False)
        logits = out.logits[0, -1].float()
        return logits[self.yes_id], logits[self.no_id]

    def pyes(self, x01, prompt_embs):
        """mean P(Yes) over prompts (no grad), for reporting."""
        torch = self.torch
        vals = []
        with torch.no_grad():
            for le, re in prompt_embs:
                ly, ln = self._yesno(x01, le, re)
                vals.append(torch.sigmoid(ly - ln).item())
        return float(np.mean(vals)) if vals else 0.0

    def attack(self, x01_0, prompt_embs, eps, alpha, iters, reg=0.0,
               random_init=False, log_every=20):
        """L_inf PGD on the cover to maximise mean P(Yes). Returns best image."""
        torch = self.torch
        x0 = x01_0.detach()
        if random_init:
            delta = ((torch.rand_like(x0) * 2 - 1) * eps)
            delta = (torch.clamp(x0 + delta, 0, 1) - x0).detach()
        else:
            delta = torch.zeros_like(x0)
        delta.requires_grad_(True)

        pyes_clean = self.pyes(x0, prompt_embs)
        best = {"pyes": pyes_clean, "delta": torch.zeros_like(x0)}

        for it in range(iters):
            if delta.grad is not None:
                delta.grad.zero_()
            pys = []
            for le, re in prompt_embs:
                # Recompute x_adv per prompt so each has its OWN graph; backward()
                # then frees only that graph and grads accumulate into delta.grad.
                # (Sharing one x_adv across prompts -> "backward a second time".)
                x_adv = torch.clamp(x0 + delta, 0, 1)
                ly, ln = self._yesno(x_adv, le, re)
                loss = torch.nn.functional.softplus(ln - ly)  # -log P(Yes), 2-way
                loss.backward()
                pys.append(torch.sigmoid(ly - ln).item())
            if reg > 0:  # L2 stealth penalty: one extra backward, accumulates into grad
                (reg * delta.float().pow(2).mean()).backward()
            mp = float(np.mean(pys))
            if mp > best["pyes"]:
                best = {"pyes": mp, "delta": delta.detach().clone()}
            with torch.no_grad():
                g = torch.nan_to_num(delta.grad)
                delta -= alpha * g.sign()
                delta.clamp_(-eps, eps)
                delta.copy_(torch.clamp(x0 + delta, 0, 1) - x0)
            if log_every and (it + 1) % log_every == 0:
                print(f"      iter {it + 1}/{iters}  P(Yes)={mp:.4f} "
                      f"(clean {pyes_clean:.4f}, best {best['pyes']:.4f})")

        x_best = torch.clamp(x0 + best["delta"], 0, 1).detach()
        linf = float((x_best - x0).abs().max().item()) * 255.0
        return x_best, pyes_clean, best["pyes"], linf


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
    if "user" in df.columns:
        df["user"] = df["user"].astype(str).str.strip()
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
    title = titles.get(item, "Unknown")
    own = []
    if "user" in df.columns:
        own = [u for u in df[df[item_col] == item]["user"].tolist() if u in prefs]
    own = list(dict.fromkeys(own))
    users = list(own)
    if len(users) < n_prompts:
        pool = [u for u in prefs.keys() if u not in set(users)]
        if pool:
            users += rng.choice(pool, size=min(n_prompts - len(users), len(pool)),
                                replace=False).tolist()
    elif len(users) > n_prompts:
        users = rng.choice(users, size=n_prompts, replace=False).tolist()
    return [PROMPT_TEMPLATE.format(prefs[u], title) for u in users]


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def generate(args):
    import torch

    path_by_item = _path_by_item(args.src_dir)
    items, df, item_col = _resolve_items(args, path_by_item)
    prefs = load_prefs(args.pref_csv)
    titles = load_titles(args.title_csv)
    print(f"[meta] items={len(items)} eps={args.eps}/255 iters={args.iters} "
          f"n_prompts={args.n_prompts} reg={args.reg}")

    M = MetaInstructionModel(args.base_model_id, args.peft_model_id, device=args.device)
    eps, alpha = args.eps / 255.0, args.alpha / 255.0

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = Path(args.clean_resized_dir) if args.clean_resized_dir else None
    if clean_dir:
        clean_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, item in enumerate(items):
        arr = load_image_as_01(path_by_item[item], COVER_SIZE)  # CHW [0,1]
        x0 = torch.tensor(arr[None], device=args.device, dtype=torch.float16)
        rng = np.random.default_rng(args.seed + idx)
        prompts = build_prompts(item, df, item_col, prefs, titles, args.n_prompts, rng)
        if not prompts:
            print(f"  skip {item}: no usable user prompts")
            continue
        prompt_embs = [M.prep_prompt(p) for p in prompts]

        x_best, pyes_clean, pyes_best, linf = M.attack(
            x0, prompt_embs, eps, alpha, args.iters, reg=args.reg,
            random_init=args.random_init, log_every=args.log_every)

        x_np = x_best[0].float().cpu().numpy()
        save_01_image(x_np, out_dir / f"{item}.png")
        if clean_dir:
            save_01_image(arr, clean_dir / f"{item}.png")
        rows.append({"item_id": item, "n_prompts": len(prompts),
                     "pyes_clean": pyes_clean, "pyes_best": pyes_best,
                     "pyes_lift": pyes_best - pyes_clean, "linf_255": linf})
        print(f"  [{idx + 1}/{len(items)}] {item}: P(Yes) {pyes_clean:.4f} -> "
              f"{pyes_best:.4f} ({pyes_best - pyes_clean:+.4f})  "
              f"||delta||inf={linf:.1f}/255")
        torch.cuda.empty_cache()

    man = pd.DataFrame(rows)
    man.to_csv(out_dir / "manifest.csv", index=False)
    summary = {
        "attack": "metainstruction_white_box",
        "paper": "Self-interpreting Adversarial Images (arXiv 2407.08970)",
        "threat_model": "white-box (gradient through full VLM to pixels)",
        "n_items": int(len(man)), "eps_255": args.eps, "iters": args.iters,
        "n_prompts": args.n_prompts, "reg": args.reg,
        "mean_pyes_clean": float(man["pyes_clean"].mean()) if len(man) else 0.0,
        "mean_pyes_best": float(man["pyes_best"].mean()) if len(man) else 0.0,
        "mean_pyes_lift": float(man["pyes_lift"].mean()) if len(man) else 0.0,
        "mean_linf_255": float(man["linf_255"].mean()) if len(man) else 0.0,
        "note": "single-crop proxy P(Yes); trust eval_illusion_sft.py (anyres) for comparable metrics",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 68)
    print("META-INSTRUCTION (white-box, invisible) — proxy P(Yes) (single-crop)")
    print("=" * 68)
    if len(man):
        print(f"items: {len(man)}  mean P(Yes) clean={summary['mean_pyes_clean']:.4f} "
              f"-> best={summary['mean_pyes_best']:.4f} (lift {summary['mean_pyes_lift']:+.4f})")
        print(f"mean ||delta||inf={summary['mean_linf_255']:.2f}/255 (budget {args.eps}/255)  "
              f"improved on {int((man['pyes_lift'] > 1e-4).sum())}/{len(man)} items")
    print(f"manifest -> {out_dir / 'manifest.csv'}")
    print("Next: eval_illusion_sft.py on these images (posonly) for the comparable "
          "anyres ASR / rank / backfire.")


def main():
    ap = argparse.ArgumentParser(description="White-box invisible meta-instruction attack for MLLM-MSR")
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
    g.add_argument("--items", default=None, help="comma-separated item ids (overrides --posonly)")
    g.add_argument("--posonly", action="store_true",
                   help="attack only label==1 items (realistic promotion threat)")
    g.add_argument("--max_items", type=int, default=0)
    g.add_argument("--eps", type=float, default=16.0, help="L_inf budget /255")
    g.add_argument("--alpha", type=float, default=1.0, help="PGD step /255")
    g.add_argument("--iters", type=int, default=100)
    g.add_argument("--n_prompts", type=int, default=4, help="user prompts averaged per item")
    g.add_argument("--reg", type=float, default=0.0, help="optional L2 stealth penalty on delta")
    g.add_argument("--random_init", action="store_true")
    g.add_argument("--log_every", type=int, default=20)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cuda:0")
    g.set_defaults(func=generate)

    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help(); return
    args.func(args)


if __name__ == "__main__":
    main()
