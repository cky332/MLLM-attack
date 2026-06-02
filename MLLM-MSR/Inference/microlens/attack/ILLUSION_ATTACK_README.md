# Adversarial-Illusion Attack on MLLM-MSR

Implements the **white-box adversarial-illusion attack** of

> Tingwei Zhang, Rishi Jha, Eugene Bagdasaryan, Vitaly Shmatikov.
> *Adversarial Illusions in Multi-Modal Embeddings.* USENIX Security 2025.
> <https://github.com/ebagdasa/adversarial_illusions>

against the recommendation judgment of **MLLM-MSR** ("Harnessing Multimodal
LLMs for Multimodal Sequential Recommendation", AAAI-25).

This is fundamentally different from the other attacks in this folder
(`generate_attacked_images.py`, `generate_hubness_attack.py`, …), which overlay
**visible text** on covers and attack the *image-summary* stage. The illusion
attack adds an **imperceptible, gradient-optimized L∞ perturbation** and attacks
the *recommendation* stage directly.

---

## 1. Why this works — where the attack surface is

MLLM-MSR decides whether a user will interact with a candidate item by prompting
LLaVA with **(user-preference text + candidate cover image + candidate title)**
and reading the probability of the answer `Yes`:

```
P(Yes) = softmax([logit("No"), logit("Yes")])[1]      # test_with_llava_sft.py
```

Items are ranked per user by `P(Yes)` (1 positive + 20 negatives → Recall@K /
NDCG@K / MRR@K).

LLaVA-1.6-Mistral-7B's **vision tower is exactly `openai/clip-vit-large-patch14-336`**.
So an adversary who controls a candidate's cover image controls the input to a
CLIP encoder whose gradients are public. That is precisely the setting the paper
exploits.

**Attack idea (`扰动候选商品图片，让候选商品图片与热门文本对齐`):** perturb the
cover image so its CLIP image embedding aligns with the text embedding of
**popular / trending** content. LLaVA then perceives a "popular" video and is
nudged toward `Yes`, lifting the item's `P(Yes)` and its rank.

---

## 2. The attack (faithful to the paper)

Objective — paper Eq. (3), optimized with L∞ PGD — paper Eq. (2):

```
min_δ   L_WB(x+δ, y_t) = 1 − cos( θ_img(x+δ),  θ_txt(y_t) )
s.t.    ‖δ‖_∞ ≤ ε,   x+δ ∈ [0,1]
```

* `θ_img`, `θ_txt` — CLIP image / text encoders (the LLaVA vision tower).
* `y_t` — the **popular-text target**. We align to the **centroid** of the CLIP
  text embeddings of the *top-N most-interacted item titles* (popularity comes
  from `MicroLens-50k_pairs.csv`). A custom target string is also supported.
* PGD update (`illusion_attack.pgd_illusion`): `δ ← δ − α·sign(∇_δ L)`, projected
  to `‖δ‖_∞ ≤ ε`, with `x+δ` clamped to valid pixels.
* Budget: `ε = 16/255` is the paper's standard bound (`{1,4,8,16,32}/255` swept).

---

## 3. Files

| File | Role |
|------|------|
| `illusion_attack.py` | CLIP image↔text illusion: `build_target` → `generate` (PGD on CLIP) → `embed_asr` |
| `illusion_attack_llava.py` | **feature-space illusion in LLaVA's own visual tokens** (use this — see §9) |
| `illusion_metrics.py` | numpy-only ASR definitions (shared, unit-tested) |
| `eval_illusion_sft.py` | **final re-ranking with your fine-tuned LoRA** (clean vs attacked) — use this if you've already trained |
| `eval_illusion_ranking.py` | same eval but with the **base** LLaVA (no adapter) |
| `preflight_illusion.py` | auto-detect/validate paths + env; prints ready-to-run commands |
| `run_illusion_experiment.sh` | end-to-end runner (3 steps) |
| `test_illusion_attack.py` | GPU-free tests for the metrics **and** the PGD algorithm |

`eval_illusion_ranking.py` reuses the LLaVA scoring (`score_batch`) from
`eval_item_ranking.py`, so its numbers are directly comparable to the rest of the
repo. The crucial difference from `eval_topk_ranking.py` (which perturbs the
*preference text*) is that here **user preferences are held fixed and only the
candidate image changes** — isolating the image attack.

---

## 4. If you've already run the full pipeline & fine-tuned the model (the common case)

**First, validate paths** (auto-detects covers / test_pairs / titles / popularity
pairs / the clean preference CSV / your LoRA, and prints the exact commands):

```bash
python preflight_illusion.py --root /home/chenkuiyun/MLLM-attack
```

You do **not** re-run training or regenerate user preferences. Only two new
things happen, and both are the "final-stage" only:

1. **Perturb the candidate images** — `illusion_attack.py generate` (Steps 1–2).
2. **Re-score the final ranking** with **your existing LoRA recommender** —
   `eval_illusion_sft.py`, which loads `PeftModel.from_pretrained(base, <your LoRA>)`
   exactly like `test/microlens/test_with_llava_sft.py`, and reports the same
   metrics (AUC, Recall/MRR/NDCG@{3,5,10}) for clean vs attacked, plus ASR.

```bash
# Step A — popular-text target (cheap) + adversarial covers  [GPU]
python illusion_attack.py build_target \
    --pairs_csv ../../data/microlens/MicroLens-50k_pairs.csv \
    --title_csv ../../data/microlens/MicroLens-50k_titles.csv \
    --top_n 20 --out_target results/illusion/popular_target.npz
python illusion_attack.py generate \
    --src_dir /path/to/MicroLens-50k_covers \
    --out_dir results/illusion/images --clean_resized_dir results/illusion/clean_resized \
    --target results/illusion/popular_target.npz \
    --items_csv /path/to/Split/test_pairs.csv --eps 16 --iters 300 --batch_size 16

# Step B — re-evaluate ONLY the final ranking with YOUR fine-tuned LoRA  [GPU]
python eval_illusion_sft.py \
    --peft_model_id /home/.../llava-v1.6-mistral-7b-hf-lora-recurrent-e4-r16 \
    --test_pairs_csv /path/to/Split/test_pairs.csv \
    --clean_image_dir results/illusion/clean_resized \
    --attacked_image_dir results/illusion/images \
    --title_csv ../../data/microlens/MicroLens-50k_titles.csv \
    --pref_csv  /path/to/user_preference_recurrent.csv \
    --candidates_per_user 21 --batch_size 4 --num_proc <#GPUs>
```

`run_illusion_experiment.sh` already wires this up — set `PEFT_MODEL_ID` at the
top to your adapter and it runs Step A then Step B.

> **Note on your saved test set.** `multi_col_dataset.py` drops the `item`
> column when it saves `MicroLens-50k-test`, and HF `save_to_disk` embeds image
> bytes, so the saved dataset can't be re-keyed to swap images by item. We
> therefore rebuild the identical scoring table from the **same CSVs** your test
> set was built from (`test_pairs.csv` + titles + `user_preference_recurrent.csv`),
> using the exact same prompt template. Recall/NDCG/MRR are set-based per user,
> so within-user ordering doesn't matter — the clean numbers reproduce your run.

---

## 5. Full end-to-end (if starting fresh, base or LoRA model)

One command:

```bash
# args: EPS(/255)  ITERS  TOP_N  N_ITEMS(0=all)  BATCH  DEVICE  NUM_PROC
bash run_illusion_experiment.sh 16 300 20 0 4 cuda:0 4
```

or the three steps explicitly:

```bash
# 1) Popular-text target (CLIP text encoder)
python illusion_attack.py build_target \
    --pairs_csv ../../data/microlens/MicroLens-50k_pairs.csv \
    --title_csv ../../data/microlens/MicroLens-50k_titles.csv \
    --top_n 20 --out_target results/illusion/popular_target.npz

# 2) Generate adversarial covers (PGD on CLIP image encoder)  [GPU]
python illusion_attack.py generate \
    --src_dir /path/to/MicroLens-50k_covers \
    --out_dir results/illusion/images \
    --clean_resized_dir results/illusion/clean_resized \
    --target  results/illusion/popular_target.npz \
    --items_csv /path/to/Split/test_pairs.csv \
    --eps 16 --alpha 1 --iters 300 --batch_size 16

# 3) Recommendation-level ASR on the real LLaVA judgment        [GPU]
python eval_illusion_ranking.py \
    --test_pairs_csv /path/to/Split/test_pairs.csv \
    --clean_image_dir   results/illusion/clean_resized \
    --attacked_image_dir results/illusion/images \
    --title_csv ../../data/microlens/MicroLens-50k_titles.csv \
    --pref_csv  /path/to/user_preference_recurrent.csv \
    --candidates_per_user 21 --topk 10 --batch_size 12
```

> **Fair baseline:** step 2 also writes the *resized-but-unperturbed* covers to
> `--clean_resized_dir`. Step 3 uses that as the clean baseline so the measured
> effect is the perturbation alone, not the 336×336 resize.

Prerequisites (produced by the MLLM-MSR pipeline in the repo README):
`user_preference_recurrent.csv` and a `test_pairs.csv` with `user,item,label`
(21 candidates/user), plus the `MicroLens-50k_covers` image folder.

---

## 6. Attack success rate — what is reported

**Embedding-level ASR** (`illusion_attack.py generate` → `images/summary.json`,
paper Table-1 style): fraction of covers whose adversarial embedding both
*increases* cosine alignment to the popular target **and** crosses an absolute
cosine threshold. Reported alongside mean cosine `clean → adv` and mean `‖δ‖_∞`.

**Recommendation-level ASR** (`eval_illusion_ranking.py` → `recsys_asr.json`):

| Metric | Meaning |
|--------|---------|
| **decision-flip ASR** | of attacked `(user,item)` pairs the clean model scored **No** (`P(Yes)<0.5`), the fraction flipped to **Yes**. This is the headline "攻击成功率". |
| mean `P(Yes)` lift | average change in `P(Yes)` on attacked pairs |
| **promotion ASR** | for users whose *positive* item was attacked, fraction pushed into top-K |
| mean rank Δ | average rank change of the attacked positive item (negative = promoted) |
| Recall@K / NDCG@K | global ranking, clean vs attacked |

---

## 7. Validation in this repo (no GPU needed)

```bash
python test_illusion_attack.py
```

Checks every ASR definition and the PGD algorithm itself: a numpy reference that
**mirrors `pgd_illusion`'s exact update rule** drives cosine alignment to a
random target from ≈0 to **0.9998** with budget (the loop converges), while
respecting `‖δ‖_∞ ≤ 16/255` and keeping pixels in `[0,1]`. If `torch` is present
it additionally runs the real `pgd_illusion` against a tiny linear encoder.

The torch/CLIP/LLaVA stages (`build_target`, `generate`, `eval_*`) require a GPU
and the model weights, so they run on the cluster, not here.

---

## 8. Notes, knobs, and honest limitations

* **Target choice.** `--top_n` controls how "popular" the target is; a single
  hand-written `--target_text "..."` is supported for a controlled target. A
  per-user-cluster target (push to a specific audience) is a natural extension —
  reuse the clustering in `targeted_recommendation.py`.
* **PNG, not JPEG.** Adversarial covers are saved losslessly (PNG); JPEG is a
  known partial defense (paper §6.1). The attack can be made JPEG-robust by
  optimizing through a differentiable JPEG, which is out of scope here.
* **LLaVA-Next any-res preprocessing.** This checkout's LLaVA config has
  `image_grid_pinpoints = [[336,672],[672,336],[672,672],[1008,336],[336,1008]]`,
  i.e. covers are re-tiled/up-sampled into several 336×336 CLIP tiles. The
  illusion is optimized at the base 336×336 input, so high-frequency perturbation
  can be partly attenuated when LLaVA up-samples to 672/1008 tiles. Embedding-level
  ASR (pure CLIP) will be near-perfect; if the recommendation-level ASR (through
  LLaVA) lags, raise `ε` (e.g. 32/255) or `--iters`, or optimize through the
  LlavaNext image processor + vision tower on all tiles (a natural follow-up).
* **Run in the right env.** The GPU stages need your MLLM conda env
  (`torch`, `transformers`, `peft`, `datasets`; `sklearn` only for AUC). The
  vision tower was confirmed to be `clip-vit-large-patch14-336` (336px, patch 14,
  24 layers, hidden 1024), which is exactly the encoder `illusion_attack.py`
  attacks.
* **Stronger variant.** This implements the paper's encoder-space attack
  (image embedding ↔ popular text). An end-to-end variant that backprops the
  `Yes`-token logit through LLaVA directly would likely raise the
  recommendation-level ASR further, at higher cost; the encoder-space attack is
  the paper's method and is task-agnostic.
* **Ethics.** For security evaluation of MLLM-based recommenders only.

---

## 9. Feature-space illusion (recommended — `illusion_attack_llava.py`)

**Pilot finding.** The CLIP image↔text illusion (§2) reaches ~99% embedding ASR
(mean cos to popular text 0.22→0.56 at ε=16/255) but barely moves the fine-tuned
recommender: P(Yes) 0.530→0.526, AUC/NDCG within noise, decision-flip ASR ~11%.
Reason: **LLaVA-Next does not read CLIP's pooled contrastive embedding**
(`get_image_features`). It feeds the LLM the vision tower's `vision_feature_layer`
(=-2) patch hidden states (CLS dropped) through `multi_modal_projector`. Aligning
the contrastive embedding therefore optimizes a representation LLaVA ignores; the
any-res up-sampling dilutes it further.

`illusion_attack_llava.py` fixes this by perturbing the cover so its **LLaVA
visual tokens** (exactly `projector(vision_tower(x).hidden_states[-2][:,1:])`)
align with those of **popular item covers** — "look like a popular video" in the
space that actually drives P(Yes). It backprops only through the vision tower +
projector (no 7B LLM), so it runs on one 24GB card.

```bash
RUN2=$PWD/results/illusion_llava_pilot
# 1) target = LLaVA features of the top-10 popular covers
python illusion_attack_llava.py build_target \
    --src_dir $COVERS --pairs_csv $PAIRS --title_csv $TITLE \
    --top_n 10 --out_target $RUN2/popular_target.pt --device cuda:0
# 2) perturb candidate covers to impersonate popular-cover features
CUDA_VISIBLE_DEVICES=0 python illusion_attack_llava.py generate \
    --src_dir $COVERS --out_dir $RUN2/images --clean_resized_dir $RUN2/clean_resized \
    --target $RUN2/popular_target.pt --items_csv $PILOT \
    --target_mode impersonate --eps 16 --iters 300 --batch_size 8 --device cuda:0
python illusion_attack_llava.py embed_asr --manifest $RUN2/images
# 3) same LoRA re-ranking eval as §4 (--num_proc = #GPUs, batch 1 on 24GB)
CUDA_VISIBLE_DEVICES=0,1,2 python eval_illusion_sft.py \
    --peft_model_id $LORA --test_pairs_csv $PILOT \
    --clean_image_dir $RUN2/clean_resized --attacked_image_dir $RUN2/images \
    --title_csv $TITLE --pref_csv $PREF --output_report $RUN2/recsys_asr_sft.json \
    --candidates_per_user 21 --batch_size 1 --num_proc 3
```

`--target_mode impersonate` (default) matches each candidate's per-token features
to a popular cover sampled from the top-N set; `--target_mode centroid` aligns the
mean-pooled feature to the popular centroid. If transfer is still weak, raise
`--eps 32`, or escalate to an end-to-end attack on LLaVA's `Yes`-logit (the
strongest option; backprops through the 7B LLM).
