#!/usr/bin/env python3
"""test_illusion_attack.py — runnable, GPU-free tests for the illusion attack.

Validates two things without needing torch, CLIP weights, or a GPU:

1. illusion_metrics.* — the ASR definitions (decision-flip, embedding alignment,
   rank promotion, ranking metrics).

2. The PGD optimisation *algorithm*. ``_pgd_numpy_reference`` below mirrors the
   exact update rule of ``illusion_attack.pgd_illusion`` (sign-gradient descent
   on L = 1 - cos, L_inf projection, pixel clamp). We run it against a random
   linear "encoder" and assert the illusion forms: cosine to the target rises
   sharply while ||delta||_inf stays within budget and pixels stay in [0,1].

If torch happens to be installed, we additionally exercise the REAL
``pgd_illusion`` against a tiny torch linear encoder (same assertions).

Run:  python test_illusion_attack.py
"""
import sys

import numpy as np

import illusion_metrics as M


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def _normalize(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# 1. metrics
# ---------------------------------------------------------------------------
def test_yesno_softmax():
    print("test_yesno_softmax")
    p = M.yesno_softmax([2.0, -1.0], [0.0, 1.0])
    _check("yes>no => p>0.5", p[0] > 0.5)
    _check("no>yes => p<0.5", p[1] < 0.5)
    # stable for large logits
    p2 = M.yesno_softmax([1000.0], [999.0])
    _check("numerically stable", np.isfinite(p2[0]) and 0 < p2[0] < 1)


def test_decision_flip_asr():
    print("test_decision_flip_asr")
    # clean: 3 of 4 are No (<.5). attacked flips two of those Nos to Yes.
    clean = np.array([0.2, 0.4, 0.45, 0.9])
    adv = np.array([0.8, 0.7, 0.49, 0.95])
    r = M.decision_flip_asr(clean, adv, threshold=0.5, direction="promote")
    _check("3 flippable (clean No)", r["n_flippable"] == 3)
    _check("2 flipped", r["n_flipped"] == 2)
    _check("ASR = 2/3", abs(r["asr"] - 2 / 3) < 1e-9)
    _check("lift reported", r["mean_pyes_lift"] > 0)
    # demote direction
    rd = M.decision_flip_asr(np.array([0.9, 0.8]), np.array([0.1, 0.85]),
                             direction="demote")
    _check("demote: 1 of 2 flipped", rd["n_flipped"] == 1 and rd["n_flippable"] == 2)


def test_embedding_alignment_asr():
    print("test_embedding_alignment_asr")
    clean = np.array([0.10, 0.20, 0.30, 0.60])
    adv = np.array([0.80, 0.70, 0.25, 0.95])  # img2 improved but only to .25
    r = M.embedding_alignment_asr(clean, adv, cos_threshold=0.5)
    # successes = improved AND >=0.5 => items 0 and 3 (item2 improved<thr, item1 not improved? .20->.70 yes improved & >=.5)
    # items: 0:.1->.8 ok; 1:.2->.7 ok; 2:.3->.25 not improved; 3:.6->.95 ok => 3/4
    _check("ASR = 3/4", abs(r["asr"] - 0.75) < 1e-9)
    _check("gain positive", r["mean_cos_gain"] > 0)
    _check("improved-only >= asr", r["asr_improved"] >= r["asr"])


def test_rank_and_ranking_metrics():
    print("test_rank_and_ranking_metrics")
    # 2 users, 4 candidates, positive is column 0.
    labels = np.array([[1, 0, 0, 0],
                       [1, 0, 0, 0]])
    # clean: positive scored low -> rank 4 / rank 3
    clean = np.array([[0.1, 0.9, 0.8, 0.7],
                      [0.2, 0.9, 0.1, 0.05]])
    # attacked: positive boosted to the top for both
    adv = np.array([[0.99, 0.9, 0.8, 0.7],
                    [0.99, 0.9, 0.1, 0.05]])
    rc = M.positive_ranks(labels, clean)
    ra = M.positive_ranks(labels, adv)
    _check("clean ranks (4,2)", rc[0] == 4 and rc[1] == 2)
    _check("attacked ranks (1,1)", ra[0] == 1 and ra[1] == 1)
    promo = M.rank_promotion_asr(labels, clean, adv, k=1)
    _check("both promoted into top-1", promo["promotion_asr"] == 1.0)
    _check("mean rank improved (delta<0)", promo["mean_rank_delta"] < 0)
    # recall@1: clean has 0 positives at top-1, attacked has 2/2
    _check("recall@1 clean=0", M.recall_at_k(labels, clean, 1) == 0.0)
    _check("recall@1 attacked=1", M.recall_at_k(labels, adv, 1) == 1.0)
    _check("ndcg@4 attacked>clean", M.ndcg_at_k(labels, adv, 4) > M.ndcg_at_k(labels, clean, 4))


# ---------------------------------------------------------------------------
# 2. PGD algorithm — numpy reference mirroring illusion_attack.pgd_illusion
# ---------------------------------------------------------------------------
def _pgd_numpy_reference(x0, target, W, eps, alpha, iters):
    """Mirror of pgd_illusion for a linear encoder f(x)=normalize(W @ vec(x)).

    Same update as the torch version: minimise L = 1 - cos(f(x+delta), target)
    via delta -= alpha*sign(grad); project to [-eps,eps]; clamp x+delta to [0,1].
    Gradient of cos wrt x is analytic for a linear+normalize encoder.
    """
    t = _normalize(target)
    x0 = x0.reshape(-1)
    delta = np.zeros_like(x0)
    cos_hist = []
    for _ in range(iters):
        xin = np.clip(x0 + delta, 0, 1)
        u = W @ xin                      # (D,)
        nu = np.linalg.norm(u) + 1e-12
        g = u / nu
        cos = float(g @ t)
        cos_hist.append(cos)
        # d cos / d u = (t - g (g·t)) / |u| ; d u / d x = W ; L = 1 - cos
        dcos_du = (t - g * (g @ t)) / nu
        dL_dx = -(W.T @ dcos_du)         # dL/dx = -dcos/dx
        delta = delta - alpha * np.sign(dL_dx)
        delta = np.clip(delta, -eps, eps)
        delta = np.clip(x0 + delta, 0, 1) - x0
    x_adv = np.clip(x0 + delta, 0, 1)
    return x_adv, cos_hist


def test_pgd_numpy_reference():
    print("test_pgd_numpy_reference (algorithm validation)")
    rng = np.random.default_rng(0)
    D, N = 32, 3 * 16 * 16     # embedding dim, flattened image dim
    W = rng.standard_normal((D, N)) / np.sqrt(N)
    x0 = rng.uniform(0, 1, size=N)
    target = rng.standard_normal(D)

    # (a) Standard paper budget eps=16/255: illusion forms + bound is respected.
    eps, alpha, iters = 16 / 255, 1 / 255, 300
    x_adv, cos_hist = _pgd_numpy_reference(x0, target, W, eps, alpha, iters)
    cos0, cosT = cos_hist[0], cos_hist[-1]
    print(f"    eps=16/255: cos start={cos0:+.4f} -> end={cosT:+.4f}  (iters={iters})")
    _check("alignment increased a lot", cosT > cos0 + 0.3)
    linf = float(np.max(np.abs(x_adv - x0)))
    _check("L_inf within budget (<=16/255)", linf <= eps + 1e-9)
    _check("pixels in [0,1]", x_adv.min() >= -1e-9 and x_adv.max() <= 1 + 1e-9)
    _check("alignment trends up (end is best)", cosT >= max(cos_hist) - 1e-6)

    # (b) Large budget: the optimiser fully forms the illusion (near-perfect
    #     alignment), proving the PGD loop converges to the target direction.
    x_adv2, cos_hist2 = _pgd_numpy_reference(x0, target, W, eps=1.0, alpha=2 / 255, iters=600)
    print(f"    eps=large : cos start={cos_hist2[0]:+.4f} -> end={cos_hist2[-1]:+.4f}")
    _check("large-budget alignment near-perfect (>0.9)", cos_hist2[-1] > 0.9)


# ---------------------------------------------------------------------------
# 3. optional: exercise the REAL torch pgd_illusion if torch is present
# ---------------------------------------------------------------------------
def test_pgd_illusion_torch_if_available():
    print("test_pgd_illusion_torch_if_available")
    try:
        import torch
    except Exception:
        print("    [SKIP] torch not installed in this environment")
        return
    from illusion_attack import pgd_illusion

    torch.manual_seed(0)
    D, C, H, Wd = 16, 3, 8, 8
    lin = torch.nn.Linear(C * H * Wd, D, bias=False)
    for p in lin.parameters():
        p.requires_grad_(False)

    def encode_image_01(x01):
        feats = lin(x01.reshape(x01.shape[0], -1))
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    x = torch.rand(2, C, H, Wd)
    target = torch.randn(D)
    with torch.no_grad():
        t = target / target.norm()
        cos0 = (encode_image_01(x) @ t).mean().item()
    x_adv, cos_final = pgd_illusion(
        x, target, encode_image_01, eps=16 / 255, alpha=1 / 255, iters=200,
        torch_mod=torch,
    )
    print(f"    cos: start={cos0:+.4f} end={float(np.mean(cos_final)):+.4f}")
    _check("real pgd increases alignment", float(np.mean(cos_final)) > cos0 + 0.2)
    linf = (x_adv - x).abs().max().item()
    _check("real pgd respects L_inf", linf <= 16 / 255 + 1e-6)
    _check("real pgd pixels valid", x_adv.min().item() >= -1e-6 and x_adv.max().item() <= 1 + 1e-6)


def main():
    tests = [
        test_yesno_softmax,
        test_decision_flip_asr,
        test_embedding_alignment_asr,
        test_rank_and_ranking_metrics,
        test_pgd_numpy_reference,
        test_pgd_illusion_torch_if_available,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  --> FAILED: {e}")
        print()
    if failed:
        print(f"RESULT: {failed} test(s) FAILED")
        sys.exit(1)
    print("RESULT: all tests PASSED")


if __name__ == "__main__":
    main()
