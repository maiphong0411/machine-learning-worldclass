"""
Lab 8 — Matrix factorization for recommendation: SGD vs ALS, RMSE and recall@k.

Module: M09 Recommendation & ranking (modules/09-recommendation-and-ranking.md)

Learning goals
--------------
1. Generate a synthetic ratings matrix that is truly low-rank plus noise, with most
   entries missing and with *popularity skew* (a few items get most of the ratings),
   exactly like real recommender data.
2. Implement biased matrix factorization  r_hat = mu + b_u + b_i + p_u . q_i
   trained two ways:
     (a) stochastic gradient descent (SGD) over observed entries;
     (b) alternating least squares (ALS): fix items, solve a ridge regression per user
         in closed form; fix users, solve per item; repeat.
3. Evaluate with RMSE on held-out ratings (the Netflix Prize metric) AND with recall@k on a
   top-k recommendation task (the metric product teams actually care about).
4. Confirm that MF beats the obvious baselines: global mean / bias-only for RMSE and
   "recommend the most popular items" for recall@k.

Run:  python3 labs/08_matrix_factorization.py
Dependencies: numpy only. Runtime: a few seconds.
"""

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. Synthetic data
# ---------------------------------------------------------------------------
def make_ratings(n_users=500, n_items=300, rank=4, density=0.08, noise=0.3, seed=SEED):
    """Return (train, test) as arrays of (user, item, rating) triples.

    true rating = mu + b_u + b_i + u_vec . v_vec + noise, clipped to [1, 5].
    Observation probability is skewed by a Zipf-like item popularity, so a handful
    of items appear in many rows (the 'head') and most are rare (the 'long tail').
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1.0, (n_users, rank))
    V = rng.normal(0, 1.0, (n_items, rank))
    b_u = rng.normal(0, 0.3, n_users)
    b_i = rng.normal(0, 0.3, n_items)
    full = 3.5 + b_u[:, None] + b_i[None, :] + (U @ V.T) / np.sqrt(rank) * 0.9
    full = np.clip(full + rng.normal(0, noise, full.shape), 1.0, 5.0)

    popularity = 1.0 / (np.arange(1, n_items + 1) ** 0.6)      # Zipf-ish
    popularity = popularity[rng.permutation(n_items)]
    prob = popularity / popularity.mean() * density
    observed = rng.random((n_users, n_items)) < np.clip(prob[None, :], 0, 1)

    users, items = np.nonzero(observed)
    ratings = full[users, items]
    triples = np.c_[users, items, ratings]
    rng.shuffle(triples)
    n_test = int(0.2 * len(triples))
    test, train = triples[:n_test], triples[n_test:]
    return train, test, n_users, n_items


def split_cols(triples):
    return triples[:, 0].astype(int), triples[:, 1].astype(int), triples[:, 2]


def rmse(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


# ---------------------------------------------------------------------------
# 2. Baselines
# ---------------------------------------------------------------------------
def fit_biases(train, n_users, n_items, lam=5.0, iters=10):
    """Bias-only model r_hat = mu + b_u + b_i, fitted by alternating regularized means."""
    u, i, r = split_cols(train)
    mu = r.mean()
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    for _ in range(iters):
        res = r - mu - bu[u]
        bi = np.bincount(i, res, n_items) / (np.bincount(i, minlength=n_items) + lam)
        res = r - mu - bi[i]
        bu = np.bincount(u, res, n_users) / (np.bincount(u, minlength=n_users) + lam)
    return mu, bu, bi


# ---------------------------------------------------------------------------
# 3. Matrix factorization with SGD
# ---------------------------------------------------------------------------
def mf_sgd(train, n_users, n_items, k=8, lr=0.02, lam=0.05, epochs=40, seed=SEED):
    """Minimise sum_(u,i) (r - mu - b_u - b_i - p_u.q_i)^2 + lam(|p_u|^2 + |q_i|^2 + b^2).

    Per-observation gradient step (derived in M09, section 3):
        e    = r - r_hat
        b_u += lr (e - lam b_u)          b_i += lr (e - lam b_i)
        p_u += lr (e q_i - lam p_u)      q_i += lr (e p_u - lam q_i)
    """
    rng = np.random.default_rng(seed)
    u_all, i_all, r_all = split_cols(train)
    mu = r_all.mean()
    P = rng.normal(0, 0.1, (n_users, k))
    Q = rng.normal(0, 0.1, (n_items, k))
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    for _ in range(epochs):
        for idx in rng.permutation(len(r_all)):
            u, i, r = u_all[idx], i_all[idx], r_all[idx]
            pu = P[u].copy()
            e = r - (mu + bu[u] + bi[i] + pu @ Q[i])
            bu[u] += lr * (e - lam * bu[u])
            bi[i] += lr * (e - lam * bi[i])
            P[u] += lr * (e * Q[i] - lam * pu)
            Q[i] += lr * (e * pu - lam * Q[i])
    return mu, bu, bi, P, Q


# ---------------------------------------------------------------------------
# 4. Matrix factorization with ALS
# ---------------------------------------------------------------------------
def _als_half_step(rows, cols, targets, n_rows, F_cols, b_cols, lam):
    """For each row entity r solve ridge regression for [x_r, b_r]:

        min sum_{c in N(r)} (t - b_c - [x_r, b_r] . [f_c, 1])^2 + lam |[x_r, b_r]|^2
    Closed form: (A^T A + lam I)^{-1} A^T y   with A = [F_c, 1], y = t - b_c.
    """
    k = F_cols.shape[1]
    X = np.zeros((n_rows, k))
    b = np.zeros(n_rows)
    order = np.argsort(rows, kind="stable")
    rows_s, cols_s, t_s = rows[order], cols[order], targets[order]
    bounds = np.searchsorted(rows_s, np.arange(n_rows + 1))
    I = np.eye(k + 1)
    for r in range(n_rows):
        lo, hi = bounds[r], bounds[r + 1]
        if lo == hi:
            continue                      # cold-start row: leave at zero (prior mean)
        c = cols_s[lo:hi]
        A = np.c_[F_cols[c], np.ones(hi - lo)]
        y = t_s[lo:hi] - b_cols[c]
        sol = np.linalg.solve(A.T @ A + lam * I, A.T @ y)
        X[r], b[r] = sol[:k], sol[k]
    return X, b


def mf_als(train, n_users, n_items, k=8, lam=3.0, iters=12, seed=SEED):
    """Alternating least squares. Each half-step is an exact minimisation, so the
    training objective never increases — no learning rate to tune."""
    rng = np.random.default_rng(seed)
    u, i, r = split_cols(train)
    mu = r.mean()
    t = r - mu
    P = rng.normal(0, 0.1, (n_users, k))
    Q = rng.normal(0, 0.1, (n_items, k))
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    history = []
    for _ in range(iters):
        P, bu = _als_half_step(u, i, t, n_users, Q, bi, lam)   # fix items, solve users
        Q, bi = _als_half_step(i, u, t, n_items, P, bu, lam)   # fix users, solve items
        pred = mu + bu[u] + bi[i] + np.sum(P[u] * Q[i], axis=1)
        obj = np.sum((r - pred) ** 2) + lam * (np.sum(P ** 2) + np.sum(Q ** 2)
                                               + np.sum(bu ** 2) + np.sum(bi ** 2))
        history.append(obj)
    return mu, bu, bi, P, Q, history


def predict(model, u, i):
    mu, bu, bi, P, Q = model[:5]
    return np.clip(mu + bu[u] + bi[i] + np.sum(P[u] * Q[i], axis=1), 1.0, 5.0)


# ---------------------------------------------------------------------------
# 5. Top-k evaluation: recall@k
# ---------------------------------------------------------------------------
def recall_at_k(score_matrix, train, test, k=10, like_threshold=4.0):
    """For each user: rank all items they have NOT rated in train, take the top k,
    and measure the fraction of their held-out *liked* items (rating >= threshold)
    that appear in the top k. Average over users with at least one liked test item."""
    S = score_matrix.copy()
    tu, ti, _ = split_cols(train)
    S[tu, ti] = -np.inf                                   # never recommend seen items
    eu, ei, er = split_cols(test)
    liked = er >= like_threshold
    topk = np.argpartition(-S, k, axis=1)[:, :k]
    in_topk = np.zeros_like(S, dtype=bool)
    np.put_along_axis(in_topk, topk, True, axis=1)
    hits = np.bincount(eu[liked], in_topk[eu[liked], ei[liked]], S.shape[0])
    n_liked = np.bincount(eu[liked], minlength=S.shape[0])
    mask = n_liked > 0
    return float(np.mean(hits[mask] / n_liked[mask]))


# ---------------------------------------------------------------------------
# 6. Main with self-checks
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Lab 8: matrix factorization (SGD and ALS)")
    print("=" * 70)
    train, test, n_users, n_items = make_ratings()
    density = (len(train) + len(test)) / (n_users * n_items)
    counts = np.bincount(train[:, 1].astype(int), minlength=n_items)
    top10_share = np.sort(counts)[::-1][: n_items // 10].sum() / counts.sum()
    print(f"\n{n_users} users x {n_items} items, density={density:.1%}, "
          f"train={len(train)}, test={len(test)}")
    print(f"top 10% of items receive {top10_share:.0%} of ratings (popularity skew)")

    tu, ti, tr = split_cols(test)

    # --- Baselines ---------------------------------------------------------
    print("\n[1] RMSE on held-out ratings (lower is better)")
    mu = train[:, 2].mean()
    rmse_global = rmse(np.full_like(tr, mu), tr)
    b_mu, b_u, b_i = fit_biases(train, n_users, n_items)
    rmse_bias = rmse(np.clip(b_mu + b_u[tu] + b_i[ti], 1, 5), tr)
    print(f"  global mean            {rmse_global:.4f}")
    print(f"  bias-only (mu+bu+bi)   {rmse_bias:.4f}")

    # --- SGD ----------------------------------------------------------------
    sgd = mf_sgd(train, n_users, n_items)
    rmse_sgd = rmse(predict(sgd, tu, ti), tr)
    print(f"  MF-SGD (k=8)           {rmse_sgd:.4f}")

    # --- ALS ----------------------------------------------------------------
    als = mf_als(train, n_users, n_items)
    rmse_als = rmse(predict(als, tu, ti), tr)
    hist = als[5]
    print(f"  MF-ALS (k=8)           {rmse_als:.4f}")
    print("  ALS training objective per iteration:",
          " ".join(f"{h:.0f}" for h in hist[:6]), "...")

    assert rmse_sgd < 0.9 * rmse_global, "SGD-MF should clearly beat the global mean"
    assert rmse_als < 0.9 * rmse_global, "ALS-MF should clearly beat the global mean"
    assert rmse_sgd < rmse_bias and rmse_als < rmse_bias, "MF should beat bias-only"
    assert all(b <= a + 1e-6 for a, b in zip(hist, hist[1:])), \
        "ALS objective must be monotonically non-increasing"

    # --- Top-k recommendation ----------------------------------------------
    print("\n[2] Top-k recommendation: recall@10 of held-out liked items (higher is better)")
    K = 10
    pop_scores = np.tile(counts.astype(float), (n_users, 1))
    rng = np.random.default_rng(1)
    rand_scores = rng.random((n_users, n_items))
    r_rand = recall_at_k(rand_scores, train, test, K)
    r_pop = recall_at_k(pop_scores, train, test, K)

    def full_scores(model):
        m, bu, bi, P, Q = model[:5]
        return m + bu[:, None] + bi[None, :] + P @ Q.T

    # Ranking by predicted rating alone ignores that liked test items must also be
    # *observed*; real systems blend relevance with a popularity/exposure prior.
    # We show both the pure-MF ranking and a simple blend.
    log_pop = np.log1p(counts)[None, :]
    r_sgd = recall_at_k(full_scores(sgd), train, test, K)
    r_als = recall_at_k(full_scores(als), train, test, K)
    r_als_blend = recall_at_k(full_scores(als) + log_pop, train, test, K)
    print(f"  random                       {r_rand:.4f}")
    print(f"  most popular                 {r_pop:.4f}")
    print(f"  MF-SGD predicted rating      {r_sgd:.4f}")
    print(f"  MF-ALS predicted rating      {r_als:.4f}")
    print(f"  MF-ALS + log(popularity)     {r_als_blend:.4f}")
    assert r_pop > r_rand, "popularity should beat random"
    assert r_als_blend > r_pop, "personalised ranking should beat the popularity baseline"
    assert max(r_sgd, r_als) > r_rand * 2, "MF ranking should be far better than random"

    print("\nAll Lab 8 checks passed.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Student exercises
# ---------------------------------------------------------------------------
# 1. (★) Sweep the latent dimension k in {1, 2, 4, 8, 16, 32} for ALS. Plot test RMSE.
#    The true rank is 4: where is the minimum and why does RMSE rise for large k?
# 2. (★) Sweep lam for ALS. Relate the optimum to the bias-variance trade-off in M01.
# 3. (★★) Cold start: hold out ALL ratings of 50 users. What does MF predict for them?
#    Fold in a new user with 3 ratings using a single ALS half-step (no retraining).
# 4. (★★) Implicit feedback: convert ratings to binary "interacted" and implement
#    weighted ALS (Hu, Koren & Volinsky 2008) with confidence c = 1 + alpha * r over the
#    FULL matrix (missing = 0 with confidence 1). Compare recall@10.
# 5. (★★) Implement BPR (pairwise loss) with SGD: sample (u, i_pos, j_neg), maximise
#    log sigmoid(x_ui - x_uj). Compare its recall@10 with the RMSE-trained models.
# 6. (★★★) Why does the popularity blend help recall here? Relate it to "missing not at
#    random": an item must be *exposed* before it can be rated. Read about exposure /
#    propensity modelling and inverse propensity weighting.
