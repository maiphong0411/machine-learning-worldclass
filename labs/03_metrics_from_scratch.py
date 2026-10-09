"""
Lab 3 — Evaluation metrics from scratch (NumPy only)
=====================================================

Module: M04 — Evaluation, data & features (modules/04-evaluation-and-data.md)

Learning goals
--------------
1. Build a confusion matrix and derive precision, recall, F1 from it.
2. Trace a ROC curve by sweeping the threshold, integrate it with the trapezoid
   rule, and VERIFY that ROC-AUC equals P(score of random positive > score of
   random negative)  (the Mann–Whitney U interpretation).
3. Compute PR-AUC (average precision) and see why it drops under class imbalance
   while ROC-AUC does not.
4. Compute log-loss and see that it punishes confident mistakes.
5. Bin predictions into a reliability diagram and compute Expected Calibration
   Error (ECE); fix a miscalibrated model with histogram binning.
6. Compute DCG / NDCG@k for a ranked list.
7. Pick a decision threshold that minimizes expected business cost.

Run:  python3 labs/03_metrics_from_scratch.py      (exits 0 if all checks pass)
"""

import numpy as np

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 0. Synthetic data: an imbalanced binary problem (e.g. fraud, 5% positives)
# ---------------------------------------------------------------------------
def make_scores(n=20_000, pos_rate=0.05, separation=1.5, rng=rng):
    """Labels y in {0,1}; a 1-D feature z ~ N(-2,1) for negatives, N(-2+2*sep,1) for positives.

    We return the TRUE posterior p = P(y=1 | z) from Bayes' rule, so p is perfectly
    calibrated by construction. For two unit-variance Gaussians with means m0, m1 the
    log-odds is linear in z:  logit p = (m1-m0) z + (m0^2 - m1^2)/2 + logit(prior).
    """
    y = (rng.random(n) < pos_rate).astype(int)
    m0, m1 = -2.0, -2.0 + 2 * separation
    z = rng.normal(loc=np.where(y == 1, m1, m0), scale=1.0)
    logit = (m1 - m0) * z + (m0**2 - m1**2) / 2 + np.log(pos_rate / (1 - pos_rate))
    p = 1.0 / (1.0 + np.exp(-logit))
    return y, p


# ---------------------------------------------------------------------------
# 1. Confusion matrix, precision, recall, F1
# ---------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred):
    """Return (TP, FP, FN, TN) for binary labels/predictions in {0,1}."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return tp, fp, fn, tn


def precision_recall_f1(y_true, y_pred):
    tp, fp, fn, _ = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0  # convention: no predictions -> precision 1
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# 2. ROC curve + AUC
# ---------------------------------------------------------------------------
def roc_curve(y_true, scores):
    """Sweep the threshold from +inf down to -inf; return arrays FPR, TPR.

    Sorting once by descending score lets us compute all thresholds in O(n log n):
    each prefix of the sorted list is "everything predicted positive".
    Ties are handled by only emitting a point where the score changes.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    # indices where the next score differs (end of a tie group)
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, len(s_sorted) - 1]
    tps, fps = tps[idx], fps[idx]
    P, N = y_true.sum(), len(y_true) - y_true.sum()
    tpr = np.r_[0.0, tps / P]
    fpr = np.r_[0.0, fps / N]
    thresholds = np.r_[np.inf, s_sorted[idx]]
    return fpr, tpr, thresholds


def trapezoid_auc(x, y):
    """Area under a piecewise-linear curve using the trapezoid rule."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def auc_by_ranking_probability(y_true, scores):
    """P(score_pos > score_neg) + 0.5 * P(tie), computed via ranks (Mann–Whitney U).

    O(n log n) instead of the naive O(P*N) pair loop.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s_sorted = scores[order]
    # average ranks for ties
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    P = y_true.sum()
    N = n - P
    rank_sum_pos = ranks[y_true == 1].sum()
    U = rank_sum_pos - P * (P + 1) / 2.0
    return float(U / (P * N))


def auc_by_pair_sampling(y_true, scores, n_pairs=200_000, rng=rng):
    """Monte-Carlo estimate: draw random (pos, neg) pairs and count wins."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    a = rng.choice(pos, n_pairs)
    b = rng.choice(neg, n_pairs)
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


# ---------------------------------------------------------------------------
# 3. Precision–recall curve and average precision (PR-AUC)
# ---------------------------------------------------------------------------
def pr_curve(y_true, scores):
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = np.asarray(y_true)[order]
    s_sorted = scores[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, len(s_sorted) - 1]
    tps, fps = tps[idx], fps[idx]
    precision = tps / (tps + fps)
    recall = tps / y_sorted.sum()
    return precision, recall


def average_precision(y_true, scores):
    """AP = sum_k (R_k - R_{k-1}) * P_k  (step-wise, no optimistic interpolation)."""
    precision, recall = pr_curve(y_true, scores)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


# ---------------------------------------------------------------------------
# 4. Log-loss
# ---------------------------------------------------------------------------
def log_loss(y_true, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


# ---------------------------------------------------------------------------
# 5. Calibration: reliability bins, ECE, histogram-binning recalibration
# ---------------------------------------------------------------------------
def reliability_bins(y_true, p, n_bins=10):
    """Return per-bin (mean predicted prob, observed positive rate, count)."""
    edges = np.linspace(0, 1, n_bins + 1)
    bin_id = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf, acc, cnt = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = bin_id == b
        cnt[b] = m.sum()
        if cnt[b]:
            conf[b] = p[m].mean()
            acc[b] = y_true[m].mean()
    return conf, acc, cnt


def expected_calibration_error(y_true, p, n_bins=10):
    conf, acc, cnt = reliability_bins(y_true, p, n_bins)
    return float(np.sum(cnt / cnt.sum() * np.abs(acc - conf)))


def fit_histogram_binning(y_cal, p_cal, n_bins=10):
    """Simplest non-parametric calibrator: map each bin to its observed positive rate."""
    edges = np.linspace(0, 1, n_bins + 1)
    bin_id = np.clip(np.digitize(p_cal, edges[1:-1]), 0, n_bins - 1)
    table = np.array([y_cal[bin_id == b].mean() if np.any(bin_id == b) else (b + 0.5) / n_bins
                      for b in range(n_bins)])

    def calibrate(p):
        b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        return table[b]

    return calibrate


# ---------------------------------------------------------------------------
# 6. Ranking: DCG / NDCG@k
# ---------------------------------------------------------------------------
def dcg_at_k(relevances, k):
    rel = np.asarray(relevances, float)[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum((2 ** rel - 1) * discounts))


def ndcg_at_k(relevances_in_ranked_order, k):
    ideal = np.sort(relevances_in_ranked_order)[::-1]
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(relevances_in_ranked_order, k) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# 7. Cost-based threshold selection
# ---------------------------------------------------------------------------
def best_threshold_by_cost(y_true, p, cost_fp, cost_fn, grid=None):
    """Return (threshold, cost) minimizing total cost = c_FP*FP + c_FN*FN on this data."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 1001)
    costs = []
    for t in grid:
        tp, fp, fn, tn = confusion_matrix(y_true, p >= t)
        costs.append(cost_fp * fp + cost_fn * fn)
    costs = np.array(costs)
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


# ===========================================================================
# Main: run every piece and self-check
# ===========================================================================
def main():
    print("=" * 70)
    print("Lab 3: metrics from scratch")
    print("=" * 70)

    # --- tiny hand-checkable example --------------------------------------
    y_small = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    pred_small = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    tp, fp, fn, tn = confusion_matrix(y_small, pred_small)
    p_, r_, f_ = precision_recall_f1(y_small, pred_small)
    print(f"\n[1] tiny example  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"    precision={p_:.3f} recall={r_:.3f} F1={f_:.3f}")
    assert (tp, fp, fn, tn) == (2, 1, 1, 6)
    assert np.isclose(p_, 2 / 3) and np.isclose(r_, 2 / 3) and np.isclose(f_, 2 / 3)

    # --- ROC-AUC three ways -----------------------------------------------
    y, p = make_scores()
    print(f"\n[2] dataset: n={len(y)}, positive rate={y.mean():.3%}")
    fpr, tpr, _ = roc_curve(y, p)
    auc_trap = trapezoid_auc(fpr, tpr)
    auc_rank = auc_by_ranking_probability(y, p)
    auc_mc = auc_by_pair_sampling(y, p)
    print(f"    ROC-AUC trapezoid       = {auc_trap:.5f}")
    print(f"    ROC-AUC via ranks (U)   = {auc_rank:.5f}")
    print(f"    ROC-AUC via 200k pairs  = {auc_mc:.5f}   (Monte-Carlo, noisy)")
    assert abs(auc_trap - auc_rank) < 1e-9, "trapezoid AUC must equal Mann-Whitney AUC exactly"
    assert abs(auc_trap - auc_mc) < 0.01
    assert fpr[0] == 0 and tpr[0] == 0 and np.isclose(fpr[-1], 1) and np.isclose(tpr[-1], 1)

    # AUC is invariant to any strictly increasing transform of the scores
    auc_transformed = trapezoid_auc(*roc_curve(y, np.log(p) * 3 + 7)[:2])
    assert abs(auc_transformed - auc_trap) < 1e-9
    print("    AUC unchanged under monotone transform of scores: OK")

    # Small example with ties: compare with the brute-force pair loop
    ys = np.array([1, 0, 1, 0, 1, 0])
    ss = np.array([0.9, 0.9, 0.7, 0.3, 0.3, 0.1])
    brute = np.mean([1.0 if a > b else 0.5 if a == b else 0.0
                     for a in ss[ys == 1] for b in ss[ys == 0]])
    assert np.isclose(trapezoid_auc(*roc_curve(ys, ss)[:2]), brute)
    assert np.isclose(auc_by_ranking_probability(ys, ss), brute)
    print(f"    tie-handling check vs brute force pairs: AUC={brute:.4f} OK")

    # --- PR-AUC vs ROC-AUC under imbalance --------------------------------
    ap = average_precision(y, p)
    print(f"\n[3] PR-AUC (average precision) = {ap:.4f}   (random baseline = {y.mean():.4f})")
    y_bal, p_bal = make_scores(pos_rate=0.5)
    y_rare, p_rare = make_scores(pos_rate=0.005, n=100_000)
    for name, yy, pp in [("balanced 50%", y_bal, p_bal), ("rare 0.5%", y_rare, p_rare)]:
        a = trapezoid_auc(*roc_curve(yy, pp)[:2])
        print(f"    {name:12s}: ROC-AUC={a:.3f}  PR-AUC={average_precision(yy, pp):.3f}")
    # Same score distributions per class -> ROC-AUC ~ same, PR-AUC collapses with rarity
    roc_bal = trapezoid_auc(*roc_curve(y_bal, p_bal)[:2])
    roc_rare = trapezoid_auc(*roc_curve(y_rare, p_rare)[:2])
    assert abs(roc_bal - roc_rare) < 0.02
    assert average_precision(y_rare, p_rare) < average_precision(y_bal, p_bal) - 0.3
    assert ap > y.mean()  # better than random

    # --- Log-loss -----------------------------------------------------------
    ll = log_loss(y, p)
    ll_base = log_loss(y, np.full_like(p, y.mean()))
    print(f"    ECE of the true-posterior model = {expected_calibration_error(y, p):.4f} (should be ~0)")
    assert expected_calibration_error(y, p) < 0.01
    print(f"\n[4] log-loss model={ll:.4f}  vs constant-prior baseline={ll_base:.4f}")
    assert ll < ll_base
    # a single confident mistake is expensive
    print(f"    one confident mistake (y=1, p=0.001) costs {log_loss(np.array([1]), np.array([0.001])):.2f} nats")
    assert log_loss(np.array([1]), np.array([0.001])) > 6.9

    # --- Calibration ----------------------------------------------------------
    # make an over-confident model by sharpening the logits (AUC unchanged!)
    logit = np.log(p / (1 - p))
    p_over = 1 / (1 + np.exp(-(2.5 * logit + 2.0)))
    half = len(y) // 2
    ece_before = expected_calibration_error(y[half:], p_over[half:])
    calib = fit_histogram_binning(y[:half], p_over[:half], n_bins=15)
    ece_after = expected_calibration_error(y[half:], calib(p_over[half:]))
    conf, acc, cnt = reliability_bins(y[half:], p_over[half:])
    print("\n[5] reliability table for the over-confident model (test half):")
    print("    bin  mean_pred  frac_pos   count")
    for b in range(10):
        if cnt[b]:
            print(f"    {b:3d}  {conf[b]:9.3f}  {acc[b]:8.3f}  {cnt[b]:6d}")
    print(f"    ECE before={ece_before:.4f}  after histogram binning={ece_after:.4f}")
    assert ece_after < ece_before / 2
    assert abs(trapezoid_auc(*roc_curve(y, p_over)[:2]) - auc_trap) < 1e-9  # ranking unchanged

    # --- NDCG ---------------------------------------------------------------
    # worked example from the module text: graded relevance of results in ranked order
    rel = [3, 2, 0, 1]
    dcg = dcg_at_k(rel, 4)
    nd = ndcg_at_k(rel, 4)
    print(f"\n[6] NDCG worked example rel={rel}: DCG@4={dcg:.3f}  NDCG@4={nd:.3f}")
    assert np.isclose(dcg, 7 + 3 / np.log2(3) + 0 + 1 / np.log2(5))
    assert np.isclose(ndcg_at_k([3, 2, 1, 0], 4), 1.0)
    assert 0 < nd < 1

    # --- Cost-based threshold -------------------------------------------------
    # fraud: missing a fraud costs $500, blocking a good transaction costs $10
    c_fp, c_fn = 10.0, 500.0
    # the theory t* = c_FP/(c_FP+c_FN) assumes CALIBRATED probabilities; p is the true
    # posterior here, so it is calibrated by construction
    pc = p[half:]
    t_star, cost_star = best_threshold_by_cost(y[half:], pc, c_fp, c_fn)
    t_theory = c_fp / (c_fp + c_fn)
    _, cost_half = best_threshold_by_cost(y[half:], pc, c_fp, c_fn, grid=np.array([0.5]))
    print(f"\n[7] cost-based threshold: c_FP={c_fp}, c_FN={c_fn}")
    print(f"    theory t* = c_FP/(c_FP+c_FN) = {t_theory:.4f}")
    print(f"    empirical best t = {t_star:.3f}  cost=${cost_star:,.0f}  vs cost@0.5=${cost_half:,.0f}")
    assert cost_star < cost_half
    _, cost_at_theory = best_threshold_by_cost(y[half:], pc, c_fp, c_fn, grid=np.array([t_theory]))
    assert cost_at_theory <= 1.1 * cost_star  # theoretical threshold is near-optimal when calibrated

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Student exercises
# ---------------------------------------------------------------------------
# 1. (★) Implement F-beta and find the threshold maximizing F2 (recall-heavy, as in
#        medical screening). How does it compare to the F1-optimal threshold?
# 2. (★) Implement MRR and MAP@k for a list of queries; check MAP on the worked example
#        in the module text.
# 3. (★★) Replace histogram binning with Platt scaling: fit a, b in sigmoid(a*logit+b)
#        by gradient descent on the calibration half. Compare ECE.
# 4. (★★) Bootstrap a 95% confidence interval for ROC-AUC and PR-AUC on the rare dataset.
#        Which metric has the wider interval relative to its value, and why?
# 5. (★★★) Implement isotonic regression with the pool-adjacent-violators algorithm and
#        use it as a calibrator.
# 6. (★★) Make c_FN depend on the transaction amount (cost = amount) and choose a
#        per-transaction decision rule. Why is a single global threshold now suboptimal?
