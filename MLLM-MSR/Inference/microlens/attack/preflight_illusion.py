#!/usr/bin/env python3
"""preflight_illusion.py — validate paths and print ready-to-run commands for the
adversarial-illusion experiment, tailored to an existing MLLM-MSR checkout.

It auto-detects (or takes overrides for) the clean covers dir, test_pairs.csv,
titles.csv, popularity pairs.csv, the user-preference CSV, and the fine-tuned
LoRA adapter; sanity-checks them (schemas, candidates-per-user, image coverage,
LoRA base model); and prints a copy-paste command block with the resolved paths.

Run it in your real MLLM conda env (the one with transformers/peft), e.g.:

    python preflight_illusion.py --root /home/chenkuiyun/MLLM-attack
"""
import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"


def first_existing(cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def find_image(image_dir, item):
    for ext in IMG_EXTS:
        p = os.path.join(image_dir, f"{item}{ext}")
        if os.path.exists(p):
            return p
    return None


def detect(args):
    root = args.root.rstrip("/")
    alt = "/home/chenkuiyun/MLLM"
    found, problems = {}, []

    # --- covers dir ---
    covers = args.covers_dir or first_existing([
        f"{root}/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_covers",
        f"{alt}/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_covers",
    ]) or next(iter(sorted(
        d for d in glob.glob(f"{root}/**/*covers", recursive=True)
        if os.path.isdir(d) and "attacked" not in d.lower())), None)
    found["covers_dir"] = covers

    # --- test pairs ---
    test_pairs = args.test_pairs_csv or first_existing([
        f"{root}/MLLM-MSR/data/MicroLens-50k/Split/test_pairs.csv",
        f"{alt}/MLLM-MSR/data/MicroLens-50k/Split/test_pairs.csv",
    ])
    found["test_pairs_csv"] = test_pairs

    # --- titles ---
    titles = args.title_csv or first_existing([
        f"{root}/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_titles.csv",
        f"{root}/MLLM-MSR/data/microlens/MicroLens-50k_titles.csv",
        f"{alt}/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_titles.csv",
    ])
    found["title_csv"] = titles

    # --- popularity pairs ---
    pairs = args.pairs_csv or first_existing([
        f"{root}/MLLM-MSR/data/microlens/MicroLens-50k_pairs.csv",
        f"{root}/MLLM-MSR/data/MicroLens-50k/MicroLens-50k_pairs.csv",
        f"{alt}/MLLM-MSR/data/microlens/MicroLens-50k_pairs.csv",
    ])
    found["pairs_csv"] = pairs

    # --- user preference CSV (pick the one covering the most test users) ---
    pref_cands = [args.pref_csv] if args.pref_csv else []
    pref_cands += [
        f"{root}/user_preference_recurrent.csv",
        f"{alt}/user_preference_recurrent.csv",
        f"{root}/MLLM-MSR/Inference/microlens/user_preference_recurrent.csv",
    ]
    pref_cands += sorted(glob.glob(f"{root}/**/user_preference*recurrent*.csv", recursive=True))
    pref_cands += sorted(glob.glob(f"{root}/**/user_preference*.csv", recursive=True))
    pref_cands = [p for p in dict.fromkeys(pref_cands) if p and os.path.exists(p)
                  and "attacked" not in os.path.basename(p).lower()]

    # --- LoRA adapter ---
    lora = args.peft_model_id or first_existing([
        f"{root}/output/llava-v1.6-mistral-7b-hf-lora-recurrent-e4-r16",
        f"{alt}/output/llava-v1.6-mistral-7b-hf-lora-recurrent-e4-r16",
    ]) or next(iter(sorted(
        os.path.dirname(p) for p in glob.glob(f"{root}/**/adapter_config.json", recursive=True))),
        None)
    found["peft_model_id"] = lora

    print("=" * 72)
    print("PREFLIGHT — resolved paths")
    print("=" * 72)

    # ---- validate covers ----
    if covers and os.path.isdir(covers):
        files = [f for f in os.listdir(covers)
                 if os.path.splitext(f)[1].lower() in IMG_EXTS]
        exts = sorted({os.path.splitext(f)[1].lower() for f in files})
        print(f"{OK} covers_dir      {covers}")
        print(f"        {len(files)} images, ext={exts}, e.g. {files[:3]}")
    else:
        print(f"{BAD} covers_dir      NOT FOUND (pass --covers_dir)")
        problems.append("covers_dir")

    # ---- validate test pairs ----
    test_users = set()
    cpu = None
    if test_pairs and os.path.exists(test_pairs):
        tp = pd.read_csv(test_pairs)
        tp.columns = [c.strip().lower() for c in tp.columns]
        tp["item"] = tp["item"].astype(str).str.strip()
        tp["user"] = tp["user"].astype(str).str.strip()
        test_users = set(tp["user"])
        per_user = tp.groupby("user").size()
        cpu = int(per_user.iloc[0]) if len(per_user) else None
        uniform = bool((per_user == cpu).all()) if cpu else False
        print(f"{OK if uniform else WARN} test_pairs_csv  {test_pairs}")
        print(f"        {len(tp)} rows, {len(test_users)} users, "
              f"candidates/user={cpu} (uniform={uniform}), "
              f"positives={int((tp['label']==1).sum())}")
        if not uniform:
            problems.append("candidates_per_user_not_uniform")
    else:
        print(f"{BAD} test_pairs_csv  NOT FOUND (pass --test_pairs_csv)")
        problems.append("test_pairs_csv")

    # ---- covers x test items coverage ----
    if covers and test_users and test_pairs:
        sample_items = tp["item"].drop_duplicates().head(50).tolist()
        hits = sum(find_image(covers, it) is not None for it in sample_items)
        rate = hits / max(len(sample_items), 1)
        tag = OK if rate > 0.95 else (WARN if rate > 0.5 else BAD)
        print(f"{tag} cover coverage  {hits}/{len(sample_items)} sampled test items "
              f"have a cover ({rate:.0%})")
        if rate <= 0.5:
            problems.append("cover_coverage")

    # ---- titles / pairs ----
    for key, path, cols in [("title_csv", titles, "item,title"),
                            ("pairs_csv", pairs, "user,item,timestamp")]:
        if path and os.path.exists(path):
            print(f"{OK} {key:<15} {path}   (expect: {cols})")
        else:
            print(f"{BAD} {key:<15} NOT FOUND")
            problems.append(key)

    # ---- preference CSV (best coverage) ----
    best = None
    print(f"{'-'*4} preference CSV candidates (coverage of {len(test_users)} test users) {'-'*4}")
    for p in pref_cands[:8]:
        try:
            d = pd.read_csv(p)
            d.columns = [c.strip().lower() for c in d.columns]
            ucol = "user" if "user" in d.columns else ("user_id" if "user_id" in d.columns else d.columns[0])
            users = set(d[ucol].astype(str).str.strip())
            cov = len(test_users & users) / max(len(test_users), 1) if test_users else 0
            print(f"      {cov:5.0%}  {p}")
            if best is None or cov > best[1]:
                best = (p, cov)
        except Exception as e:
            print(f"      ERR   {p}  ({type(e).__name__})")
    pref = best[0] if best else None
    found["pref_csv"] = pref
    if pref and best[1] > 0.95:
        print(f"{OK} pref_csv        {pref}  (coverage {best[1]:.0%})")
    elif pref:
        print(f"{WARN} pref_csv        {pref}  (coverage {best[1]:.0%} — verify this is the "
              f"CLEAN recurrent preference you evaluated with)")
    else:
        print(f"{BAD} pref_csv        NOT FOUND (pass --pref_csv)")
        problems.append("pref_csv")

    # ---- LoRA ----
    if lora and os.path.exists(os.path.join(lora, "adapter_config.json")):
        cfg = json.load(open(os.path.join(lora, "adapter_config.json")))
        base = cfg.get("base_model_name_or_path")
        print(f"{OK} peft_model_id   {lora}")
        print(f"        base={base}  r={cfg.get('r')}  alpha={cfg.get('lora_alpha')}")
    else:
        print(f"{WARN} peft_model_id   NOT FOUND — eval will use the BASE model "
              f"(pass --peft_model_id for your fine-tuned recommender)")

    # ---- env check ----
    print(f"{'-'*4} environment {'-'*4}")
    for m in ["torch", "transformers", "peft", "datasets", "sklearn"]:
        try:
            mod = __import__("sklearn" if m == "sklearn" else m)
            print(f"{OK} {m} {getattr(mod, '__version__', '?')}")
        except Exception:
            tag = WARN if m == "sklearn" else BAD
            print(f"{tag} {m} NOT INSTALLED in this env"
                  + ("  (AUC will be skipped)" if m == "sklearn" else
                     "  (needed for the GPU stages — activate your MLLM env)"))
            if m != "sklearn":
                problems.append(f"env:{m}")

    # ---- command block ----
    run = f"{root}/MLLM-MSR/Inference/microlens/attack/results/illusion_eps16"
    print("\n" + "=" * 72)
    print("READY-TO-RUN (from MLLM-MSR/Inference/microlens/attack/)")
    print("=" * 72)
    if {"covers_dir", "test_pairs_csv", "title_csv", "pairs_csv"} & set(problems):
        print("Fix the [FAIL] paths above first (pass them with --covers_dir etc.).")
    print(f"""# 1) popular-text target
python illusion_attack.py build_target \\
    --pairs_csv {pairs} \\
    --title_csv {titles} \\
    --top_n 20 --out_target {run}/popular_target.npz

# 2) adversarial covers — 8 GPUs in parallel (one shard each)
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python illusion_attack.py generate \\
    --src_dir {covers} \\
    --out_dir {run}/images --clean_resized_dir {run}/clean_resized \\
    --target {run}/popular_target.npz --items_csv {test_pairs} \\
    --eps 16 --iters 300 --batch_size 16 \\
    --num_shards 8 --shard_id $g --device cuda:0 &
done; wait
python illusion_attack.py embed_asr --manifest {run}/images   # merged embedding ASR

# 3) re-evaluate the final ranking with your fine-tuned LoRA
python eval_illusion_sft.py \\
    --peft_model_id {lora} \\
    --test_pairs_csv {test_pairs} \\
    --clean_image_dir {run}/clean_resized --attacked_image_dir {run}/images \\
    --title_csv {titles} --pref_csv {pref} \\
    --output_report {run}/recsys_asr_sft.json \\
    --candidates_per_user {cpu or 21} --batch_size 4 --num_proc 8""")

    print("\n" + ("PREFLIGHT OK — paths look good." if not problems
                  else f"PREFLIGHT found issues: {sorted(set(problems))}"))
    return 0 if not problems else 1


def main():
    ap = argparse.ArgumentParser(description="Validate paths for the illusion experiment")
    ap.add_argument("--root", default="/home/chenkuiyun/MLLM-attack")
    ap.add_argument("--covers_dir")
    ap.add_argument("--test_pairs_csv")
    ap.add_argument("--title_csv")
    ap.add_argument("--pairs_csv")
    ap.add_argument("--pref_csv")
    ap.add_argument("--peft_model_id")
    raise SystemExit(detect(ap.parse_args()))


if __name__ == "__main__":
    main()
