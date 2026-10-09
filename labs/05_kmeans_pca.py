"""
Lab 5 — k-means, PCA and anomaly detection (NumPy only)
========================================================

Module: M06 — Unsupervised learning & anomaly detection (modules/06-unsupervised-learning.md)

Learning goals
--------------
1. Implement k-means (Lloyd's algorithm) as coordinate descent on the inertia objective
   and VERIFY the objective never increases.
2. Implement k-means++ seeding and compare it with naive random seeding.
3. Use the "elbow" of inertia vs k to choose k.
4. Implement PCA via the SVD of the centered data matrix; compute explained-variance
   ratios and check they match the eigenvalues of the covariance matrix.
5. Reconstruct data from k principal components and measure reconstruction error.
6. Detect anomalies two ways: per-feature z-scores and PCA reconstruction error, and see
   why the second catches anomalies that break CORRELATIONS, not individual ranges.

Run:  python3 labs/05_kmeans_pca.py     (exits 0 if all checks pass)
"""

import numpy as np

rng = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# 1. k-means
# ---------------------------------------------------------------------------
def sq_dists(X, C):
    """(n, k) matrix of squared Euclidean distances, via ||x||^2 - 2 x.c + ||c||^2."""
    d = (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]
    return np.maximum(d, 0.0)


def init_random(X, k, rng):
    return X[rng.choice(len(X), k, replace=False)].copy()


def init_kmeans_pp(X, k, rng):
    """k-means++: first centre uniform; each next centre sampled with prob ∝ D(x)^2,
    where D(x) is the distance to the nearest centre chosen so far."""
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = sq_dists(X, np.array(centres)).min(1)
        centres.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centres)


def kmeans(X, k, init="++", n_iter=100, rng=rng):
    """Return centres, labels, history of inertia (objective after each half-step)."""
    C = init_kmeans_pp(X, k, rng) if init == "++" else init_random(X, k, rng)
    history = []
    labels = None
    for _ in range(n_iter):
        # Step A (assignment): fix centres, minimize J over assignments
        D = sq_dists(X, C)
        new_labels = D.argmin(1)
        history.append(float(D[np.arange(len(X)), new_labels].sum()))
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Step B (update): fix assignments, minimize J over centres -> cluster means
        for j in range(k):
            members = X[labels == j]
            if len(members):  # empty cluster: keep old centre (a common simple fix)
                C[j] = members.mean(0)
        history.append(float(((X - C[labels]) ** 2).sum()))
    return C, labels, history


def best_of(X, k, init, n_restarts, rng):
    runs = [kmeans(X, k, init, rng=rng) for _ in range(n_restarts)]
    return min(runs, key=lambda r: r[2][-1])


# ---------------------------------------------------------------------------
# 2. PCA via SVD
# ---------------------------------------------------------------------------
class PCA:
    def fit(self, X):
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        # Xc = U S V^T ; rows of V^T are principal directions; S^2/(n-1) are variances
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt
        self.explained_variance_ = S**2 / (len(X) - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        return self

    def transform(self, X, k):
        return (X - self.mean_) @ self.components_[:k].T

    def inverse_transform(self, Z):
        k = Z.shape[1]
        return Z @ self.components_[:k] + self.mean_

    def reconstruction_error(self, X, k):
        """Per-row squared error of projecting onto the top-k components and back."""
        Xh = self.inverse_transform(self.transform(X, k))
        return ((X - Xh) ** 2).sum(1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_blobs(n_per=200, rng=rng):
    """Four well-separated 2-D 'customer segments'."""
    centres = np.array([[0, 0], [6, 0], [0, 6], [6, 6]], float)
    X = np.vstack([c + rng.normal(0, 1.0, (n_per, 2)) for c in centres])
    y = np.repeat(np.arange(4), n_per)
    return X, y


def make_sensor_data(n=1000, rng=rng):
    """10 correlated sensors driven by 2 latent factors (e.g. temperature, load).

    Returns normal data plus two kinds of anomalies:
      - 'spike':      one sensor far out of range        (z-score catches it)
      - 'decoupled':  every sensor within normal range, but the usual correlation
                      between sensors is broken          (only PCA catches it)
    """
    W = rng.normal(size=(2, 10))
    Z = rng.normal(size=(n, 2))
    X = Z @ W + rng.normal(0, 0.1, (n, 10))
    # spikes
    spikes = X[:10].copy()
    spikes[np.arange(10), np.arange(10)] += 8 * X.std(0)
    # decoupled: random signs on each sensor of a typical row (values stay in range)
    decoupled = rng.normal(size=(10, 2)) @ W
    decoupled = decoupled * rng.choice([-1, 1], size=decoupled.shape)
    return X, spikes, decoupled


def main():
    print("=" * 70)
    print("Lab 5: k-means, PCA, anomaly detection")
    print("=" * 70)

    # --- k-means monotone objective ------------------------------------
    X, y = make_blobs()
    C, labels, hist = kmeans(X, 4, "++", rng=np.random.default_rng(1))
    print(f"\n[1] k-means++ on 4 blobs: converged in {len(hist) // 2 + 1} assignment steps")
    print(f"    inertia history (first 6): {np.round(hist[:6], 1)}")
    assert np.all(np.diff(hist) <= 1e-8), "inertia must never increase (coordinate descent)"
    # clusters should match the true segments up to relabeling
    purity = sum(np.bincount(y[labels == j]).max() for j in range(4) if np.any(labels == j)) / len(y)
    print(f"    cluster purity vs true segments = {purity:.3f}")
    assert purity > 0.95

    # --- k-means++ vs random init ----------------------------------------
    r = np.random.default_rng(3)
    final_pp = [kmeans(X, 4, "++", rng=r)[2][-1] for _ in range(30)]
    final_rand = [kmeans(X, 4, "random", rng=r)[2][-1] for _ in range(30)]
    best = min(final_pp + final_rand)
    bad_pp = np.mean(np.array(final_pp) > 1.05 * best)
    bad_rand = np.mean(np.array(final_rand) > 1.05 * best)
    print(f"\n[2] 30 runs each: fraction stuck in a bad local optimum (>5% above best)")
    print(f"    random init: {bad_rand:.2f}     k-means++ init: {bad_pp:.2f}")
    assert bad_pp <= bad_rand

    # --- elbow ----------------------------------------------------------
    inertias = [best_of(X, k, "++", 5, r)[2][-1] for k in range(1, 9)]
    print("\n[3] elbow: inertia vs k")
    for k, v in enumerate(inertias, 1):
        print(f"    k={k}: {v:9.1f}")
    drops = -np.diff(inertias)
    assert np.all(np.diff(inertias) < 1e-6), "best inertia decreases with k"
    # the biggest relative drop happens up to k=4, then diminishing returns
    assert drops[3] < 0.2 * drops[2], "after k=4 returns should diminish sharply"

    # --- PCA ------------------------------------------------------------
    Xs, spikes, decoupled = make_sensor_data()
    train, test = Xs[:800], Xs[800:]
    pca = PCA().fit(train)
    evr = pca.explained_variance_ratio_
    print(f"\n[4] PCA on 10 sensors: explained variance ratio = {np.round(evr[:4], 4)} ...")
    print(f"    top-2 components explain {evr[:2].sum():.2%}")
    assert evr[:2].sum() > 0.98, "data is generated from 2 latent factors"
    cov_eigs = np.sort(np.linalg.eigvalsh(np.cov(train, rowvar=False)))[::-1]
    assert np.allclose(cov_eigs, pca.explained_variance_), "SVD and covariance eigen-decomposition agree"
    print("    SVD singular values^2/(n-1) == eigenvalues of covariance: OK")
    comps = pca.components_
    assert np.allclose(comps @ comps.T, np.eye(10), atol=1e-8), "components are orthonormal"

    errs = [pca.reconstruction_error(test, k).mean() for k in range(0, 11)]
    print("    mean reconstruction error on test vs k:", np.round(errs[:5], 4), "...")
    assert np.all(np.diff(errs) <= 1e-10) and errs[-1] < 1e-20 + 1e-10

    # --- anomaly detection ---------------------------------------------
    mu, sd = train.mean(0), train.std(0)

    def z_flag(A):
        return (np.abs((A - mu) / sd).max(1) > 4)

    k = 2
    recon_train = pca.reconstruction_error(train, k)
    thresh = np.quantile(recon_train, 0.99)

    def pca_flag(A):
        return pca.reconstruction_error(A, k) > thresh

    print(f"\n[5] anomaly detection (PCA k={k}, threshold = 99th pct of train recon error)")
    rows = [("normal test", test), ("spikes", spikes), ("decoupled", decoupled)]
    rates = {}
    for name, A in rows:
        rates[name] = (z_flag(A).mean(), pca_flag(A).mean())
        print(f"    {name:12s}: z-score flags {rates[name][0]:.2f}   PCA-recon flags {rates[name][1]:.2f}")
    assert rates["normal test"][0] < 0.05 and rates["normal test"][1] < 0.05
    assert rates["spikes"][0] == 1.0 and rates["spikes"][1] == 1.0
    assert rates["decoupled"][0] <= 0.3, "decoupled rows look normal feature-by-feature"
    assert rates["decoupled"][1] >= 0.8, "...but break the correlation structure PCA learned"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Student exercises
# ---------------------------------------------------------------------------
# 1. (★) Add a feature with scale 1000x the others to the blobs. What happens to k-means?
#        Fix it with standardization.
# 2. (★) Compute the silhouette score for k=2..8 and compare its choice with the elbow.
# 3. (★★) Generate two concentric rings. Show that k-means fails and explain why in terms
#        of the objective (Voronoi cells are convex).
# 4. (★★) Implement a 1-D Gaussian mixture with EM on a bimodal dataset; compare its soft
#        responsibilities with k-means' hard assignments.
# 5. (★★) Choose the number of PCA components that keeps 95% variance; how does the anomaly
#        detector's recall on 'decoupled' change if you keep too many components (k=9)?
# 6. (★★★) Implement a tiny isolation forest (random feature, random threshold, record path
#        length) and compare its scores with the PCA detector on 'spikes' and 'decoupled'.
