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
| `illusion_attack.py` | core attack: `build_target` → `generate` (PGD on CLIP) → `embed_asr` |
| `illusion_metrics.py` | numpy-only ASR definitions (shared, unit-tested) |
| `eval_illusion_ranking.py` | recommendation-level ASR via the real LLaVA Yes/No judgment |
| `run_illusion_experiment.sh` | end-to-end runner (3 steps) |
| `test_illusion_attack.py` | GPU-free tests for the metrics **and** the PGD algorithm |

`eval_illusion_ranking.py` reuses the LLaVA scoring (`score_batch`) from
`eval_item_ranking.py`, so its numbers are directly comparable to the rest of the
repo. The crucial difference from `eval_topk_ranking.py` (which perturbs the
*preference text*) is that here **user preferences are held fixed and only the
candidate image changes** — isolating the image attack.

---

## 4. How to run (on the GPU cluster)

One command:

```bash
# args: EPS(/255)  ITERS  TOP_N  N_ITEMS(0=all)  BATCH  DEVICE
bash run_illusion_experiment.sh 16 300 20 0 12 cuda:0
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

## 5. Attack success rate — what is reported

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

## 6. Validation in this repo (no GPU needed)

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

## 7. Notes, knobs, and honest limitations

* **Target choice.** `--top_n` controls how "popular" the target is; a single
  hand-written `--target_text "..."` is supported for a controlled target. A
  per-user-cluster target (push to a specific audience) is a natural extension —
  reuse the clustering in `targeted_recommendation.py`.
* **PNG, not JPEG.** Adversarial covers are saved losslessly (PNG); JPEG is a
  known partial defense (paper §6.1). The attack can be made JPEG-robust by
  optimizing through a differentiable JPEG, which is out of scope here.
* **LLaVA-Next any-res preprocessing.** The illusion is optimized at the CLIP
  336×336 input. LLaVA-Next then re-tiles/normalizes the saved image, which can
  attenuate transfer from the pure CLIP embedding to LLaVA's `P(Yes)`. If the
  recommendation-level ASR is weaker than the embedding-level ASR, raise `ε`
  (e.g. 32/255) or `--iters`, or optimize through the LLaVA processor.
* **Stronger variant.** This implements the paper's encoder-space attack
  (image embedding ↔ popular text). An end-to-end variant that backprops the
  `Yes`-token logit through LLaVA directly would likely raise the
  recommendation-level ASR further, at higher cost; the encoder-space attack is
  the paper's method and is task-agnostic.
* **Ethics.** For security evaluation of MLLM-based recommenders only.
