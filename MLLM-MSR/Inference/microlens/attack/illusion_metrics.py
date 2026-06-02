#!/usr/bin/env python3
"""illusion_metrics.py — Attack-success-rate (ASR) metrics for the
adversarial-illusion attack on MLLM-MSR.

These are pure-numpy functions shared by ``illusion_attack.py`` (embedding-level
ASR) and ``eval_illusion_ranking.py`` (recommendation-level ASR). Keeping them
dependency-free (numpy only) means they can be unit-tested without a GPU,
torch, or model weights — see ``test_illusion_attack.py``.

Two families of ASR are reported, mirroring the two levels at which the attack
of Zhang et al., "Adversarial Illusions in Multi-Modal Embeddings" (USENIX
Security 2025) can be measured:

1. Embedding-level ASR (faithful to the paper's Table 1). The illusion is
   "successful" for an image if its perturbed CLIP embedding is aligned with
   the adversary-chosen popular-text target *more* than the clean image is, and
   above an absolute cosine threshold. This is task-agnostic: it measures the
   illusion itself, independent of the downstream recommender.

2. Recommendation-level ASR (the MLLM-MSR-specific goal). MLLM-MSR turns the
   candidate (user-preference + cover-image + title) into P(Yes) — the
   predicted probability the user interacts. We report:
     - decision-flip ASR: fraction of (user, item) pairs whose binary decision
       flips No -> Yes (P(Yes) crosses 0.5) after the image is perturbed;
     - mean P(Yes) lift;
     - rank-promotion ASR: fraction of attacked target items pushed into top-K,
       and mean rank improvement.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Logit -> probability
# ---------------------------------------------------------------------------
def yesno_softmax(yes_logits, no_logits):
    """P(Yes) from paired Yes/No logits, exactly as test_with_llava_sft.py:
    softmax over [no, yes] and take the Yes component. Numerically stable."""
    yl = np.asarray(yes_logits, dtype=np.float64)
    nl = np.asarray(no_logits, dtype=np.float64)
    m = np.maximum(yl, nl)
    eyes = np.exp(yl - m)
    enos = np.exp(nl - m)
    return eyes / (eyes + enos)


# ---------------------------------------------------------------------------
# 1. Embedding-level ASR (paper-faithful)
# ---------------------------------------------------------------------------
def embedding_alignment_asr(clean_cos, attacked_cos, cos_threshold=0.5):
    """Embedding-level ASR for an illusion attack.

    Args:
        clean_cos:    cos(theta_img(x_clean), target_text) per image.
        attacked_cos: cos(theta_img(x_adv),  target_text) per image.
        cos_threshold: absolute alignment a successful illusion must reach.

    A perturbed image counts as a success if it both (a) increases alignment
    over the clean image and (b) reaches the absolute threshold. We also report
    the looser "improved" rate (a only) and the stricter "threshold" rate (b
    only), so the caller can see which constraint binds.
    """
    clean_cos = np.asarray(clean_cos, dtype=np.float64)
    attacked_cos = np.asarray(attacked_cos, dtype=np.float64)
    n = clean_cos.size
    if n == 0:
        return {"n": 0}
    improved = attacked_cos > clean_cos
    reached = attacked_cos >= cos_threshold
    success = improved & reached
    return {
        "n": int(n),
        "cos_threshold": float(cos_threshold),
        "mean_cos_clean": float(clean_cos.mean()),
        "mean_cos_attacked": float(attacked_cos.mean()),
        "mean_cos_gain": float((attacked_cos - clean_cos).mean()),
        "asr_improved": float(improved.mean()),       # (a)
        "asr_threshold": float(reached.mean()),        # (b)
        "asr": float(success.mean()),                  # (a) AND (b)
    }


# ---------------------------------------------------------------------------
# 2. Recommendation-level ASR
# ---------------------------------------------------------------------------
def decision_flip_asr(clean_pyes, attacked_pyes, threshold=0.5,
                      direction="promote"):
    """Binary-decision-flip ASR over (user, item) pairs.

    MLLM-MSR's per-item decision is ``P(Yes) >= threshold``. A "promote" attack
    succeeds on a pair when the clean decision is No and the attacked decision
    is Yes. (Use direction="demote" for the opposite, e.g. burying a rival.)

    The headline ASR is conditioned on pairs that were *flippable* in the
    intended direction under the clean model (e.g. for "promote", pairs whose
    clean decision was No) — flipping an already-Yes pair is not a success.
    """
    c = np.asarray(clean_pyes, dtype=np.float64)
    a = np.asarray(attacked_pyes, dtype=np.float64)
    clean_yes = c >= threshold
    atk_yes = a >= threshold
    if direction == "promote":
        flippable = ~clean_yes
        flipped = flippable & atk_yes
    elif direction == "demote":
        flippable = clean_yes
        flipped = flippable & ~atk_yes
    else:
        raise ValueError(f"direction must be 'promote' or 'demote', got {direction}")
    n_flip = int(flippable.sum())
    return {
        "n_pairs": int(c.size),
        "threshold": float(threshold),
        "direction": direction,
        "n_flippable": n_flip,
        "n_flipped": int(flipped.sum()),
        # ASR over the flippable population (the meaningful denominator)
        "asr": float(flipped.sum() / n_flip) if n_flip else 0.0,
        # ASR over *all* pairs (includes already-satisfied pairs in denominator)
        "asr_over_all_pairs": float(flipped.sum() / c.size) if c.size else 0.0,
        "mean_pyes_clean": float(c.mean()) if c.size else 0.0,
        "mean_pyes_attacked": float(a.mean()) if a.size else 0.0,
        "mean_pyes_lift": float((a - c).mean()) if c.size else 0.0,
        "pct_pyes_increased": float((a > c).mean()) if c.size else 0.0,
    }


def positive_ranks(labels_grid, score_grid):
    """1-indexed rank of the positive item in each user's candidate list.

    Args:
        labels_grid: (n_users, K) 0/1 — exactly one 1 per row (the positive).
        score_grid:  (n_users, K) P(Yes) scores.
    Returns:
        (n_users,) int ranks; -1 where a row has no positive.
    Ties are broken by argsort's stable order, matching test_with_llava_sft.py.
    """
    labels_grid = np.asarray(labels_grid)
    score_grid = np.asarray(score_grid, dtype=np.float64)
    order = np.argsort(-score_grid, axis=1)
    n = labels_grid.shape[0]
    ranks = np.full(n, -1, dtype=int)
    for i in range(n):
        pos = np.where(labels_grid[i] == 1)[0]
        if pos.size == 0:
            continue
        ranks[i] = int(np.where(order[i] == pos[0])[0][0]) + 1
    return ranks


def rank_promotion_asr(labels_grid, clean_scores, attacked_scores, k=10):
    """Rank-promotion ASR for the *positive* (target) item of each user.

    Measures whether perturbing the positive item's cover image pushes it up the
    ranking. Success = the positive item enters top-K after the attack while it
    was outside top-K before (a strict "newly recommended" criterion).
    """
    labels_grid = np.asarray(labels_grid)
    rank_clean = positive_ranks(labels_grid, clean_scores)
    rank_atk = positive_ranks(labels_grid, attacked_scores)
    valid = (rank_clean > 0) & (rank_atk > 0)
    rc, ra = rank_clean[valid], rank_atk[valid]
    delta = ra - rc  # negative = improved (moved up)

    in_topk_clean = rc <= k
    in_topk_atk = ra <= k
    newly_in_topk = (~in_topk_clean) & in_topk_atk

    n_promotable = int((~in_topk_clean).sum())
    return {
        "k": int(k),
        "n_users": int(valid.sum()),
        "mean_rank_clean": float(rc.mean()) if rc.size else None,
        "mean_rank_attacked": float(ra.mean()) if ra.size else None,
        "mean_rank_delta": float(delta.mean()) if delta.size else None,  # <0 good
        "n_rank_improved": int((delta < 0).sum()),
        "n_rank_worsened": int((delta > 0).sum()),
        "topk_hit_rate_clean": float(in_topk_clean.mean()) if rc.size else 0.0,
        "topk_hit_rate_attacked": float(in_topk_atk.mean()) if ra.size else 0.0,
        "n_promotable": n_promotable,
        # ASR = of items not already in top-K, fraction pushed into top-K
        "promotion_asr": float(newly_in_topk.sum() / n_promotable) if n_promotable else 0.0,
    }


# ---------------------------------------------------------------------------
# Ranking metrics (kept identical to test_with_llava_sft.py for comparability)
# ---------------------------------------------------------------------------
def recall_at_k(y_true, y_prob, k):
    order = np.argsort(-np.asarray(y_prob, dtype=np.float64), axis=1)
    sorted_labels = np.take_along_axis(np.asarray(y_true), order, axis=1)
    return float(np.mean(np.sum(sorted_labels[:, :k], axis=1)))


def ndcg_at_k(y_true, y_prob, k):
    y_true = np.asarray(y_true)

    def dcg(scores):
        # Discounts sized to the actual width (handles k > #candidates safely).
        discounts = np.log2(np.arange(2, scores.shape[1] + 2))
        return np.sum((2 ** scores - 1) / discounts, axis=1)

    order = np.argsort(-np.asarray(y_prob, dtype=np.float64), axis=1)
    sorted_scores = np.take_along_axis(y_true, order, axis=1)[:, :k]
    ideal = np.sort(y_true, axis=1)[:, ::-1][:, :k]
    return float(np.mean(dcg(sorted_scores) / (dcg(ideal) + 1e-10)))


def mrr_at_k(y_true, y_prob, k):
    """Mean reciprocal rank @k, identical to test_with_llava_sft.py."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_prob, dtype=np.float64), axis=1)
    sorted_labels = np.take_along_axis(y_true, order, axis=1)
    rr = np.zeros(y_true.shape[0])
    for i, labels in enumerate(sorted_labels[:, :k]):
        pos = np.where(labels == 1)[0]
        if pos.size > 0:
            rr[i] = 1.0 / (pos[0] + 1)
    return float(np.mean(rr))
