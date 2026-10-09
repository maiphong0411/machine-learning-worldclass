"""
Lab 2 — Linear and logistic regression from scratch (NumPy only)
================================================================

Companion to M03 Linear & logistic regression (modules/03-linear-and-logistic-regression.md).

Learning goals
--------------
After finishing this lab you should be able to:
  1. Fit linear regression with the normal equation and with gradient descent,
     and show both reach the same weights.
  2. Implement ridge regression in closed form and see how lambda shrinks weights.
  3. Implement logistic regression (sigmoid + binary cross-entropy) trained by GD.
  4. Verify an analytic gradient against a numerical (finite-difference) gradient
     -- the single most useful debugging habit in ML.
  5. Report accuracy and log-loss, and explain the difference between them.

How to run
----------
    python3 labs/02_linear_logistic_regression.py

The script prints results and ends with assertions; exit code 0 means all passed.
"""

from __future__ import annotations

import numpy as np

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))])


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    """Fit mean/std on TRAIN only, then apply to both (no leakage)."""
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0) + 1e-12
    return (X_train - mu) / sd, (X_test - mu) / sd


def train_test_split(X, y, test_frac=0.25, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


# ---------------------------------------------------------------------------
# Part A — Linear regression
# ---------------------------------------------------------------------------

def make_linear_data(n=1000, seed=SEED):
    """Features on very different scales, like 'sqft' vs 'num_bedrooms'."""
    rng = np.random.default_rng(seed)
    sqft = rng.normal(1500, 400, n)
    bedrooms = rng.integers(1, 6, n).astype(float)
    age = rng.uniform(0, 50, n)
    X = np.column_stack([sqft, bedrooms, age])
    w_true = np.array([150.0, 10_000.0, -800.0])
    b_true = 50_000.0
    y = X @ w_true + b_true + rng.normal(0, 10_000, n)
    return X, y, w_true, b_true


def fit_normal_equation(Xb: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Solve (X^T X) w = X^T y. Never form the explicit inverse.
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ y)


def fit_ridge(Xb: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge: (X^T X + lam*I') w = X^T y, with I' not penalising the bias (last column)."""
    d = Xb.shape[1]
    I = np.eye(d)
    I[-1, -1] = 0.0
    return np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ y)


def fit_linreg_gd(Xb, y, lr=0.1, epochs=2000):
    n, d = Xb.shape
    w = np.zeros(d)
    losses = []
    for _ in range(epochs):
        r = Xb @ w - y
        losses.append(r @ r / n)
        w -= lr * (2.0 / n) * Xb.T @ r
    return w, np.array(losses)


def r2_score(y, yhat) -> float:
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def part_a():
    print("=" * 72)
    print("Part A: linear regression (synthetic house prices)")
    print("=" * 72)
    X, y, w_true, b_true = make_linear_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)

    # 1) Normal equation on raw features.
    w_ne_raw = fit_normal_equation(add_bias(X_tr), y_tr)
    print(f"true [w, b]               : {np.append(w_true, b_true)}")
    print(f"normal eq (raw features)  : {np.round(w_ne_raw, 1)}")

    # 2) GD on raw features is hopeless (sqft ~ 1500 dominates curvature) -> standardise.
    Xs_tr, Xs_te = standardize(X_tr, X_te)
    # Scale target too so a lr of 0.1 is sensible; GD on standardised problem.
    y_mu, y_sd = y_tr.mean(), y_tr.std()
    w_ne_std = fit_normal_equation(add_bias(Xs_tr), (y_tr - y_mu) / y_sd)
    w_gd_std, losses = fit_linreg_gd(add_bias(Xs_tr), (y_tr - y_mu) / y_sd, lr=0.1, epochs=500)
    print(f"normal eq (standardised)  : {np.round(w_ne_std, 4)}")
    print(f"GD        (standardised)  : {np.round(w_gd_std, 4)}")
    print(f"max |w_gd - w_ne|         : {np.max(np.abs(w_gd_std - w_ne_std)):.2e}")

    yhat_te = add_bias(Xs_te) @ w_gd_std * y_sd + y_mu
    r2 = r2_score(y_te, yhat_te)
    print(f"test R^2 (GD model)       : {r2:.3f}")

    # 3) Ridge: weights shrink as lambda grows.
    print("\nRidge on standardised features (target standardised):")
    print(f"{'lambda':>10}{'||w||_2 (no bias)':>20}{'train MSE':>12}")
    norms = []
    for lam in [0.0, 1.0, 10.0, 100.0, 1000.0]:
        w_r = fit_ridge(add_bias(Xs_tr), (y_tr - y_mu) / y_sd, lam)
        r = add_bias(Xs_tr) @ w_r - (y_tr - y_mu) / y_sd
        norms.append(np.linalg.norm(w_r[:-1]))
        print(f"{lam:>10.1f}{norms[-1]:>20.4f}{r @ r / len(r):>12.4f}")
    return dict(w_true=w_true, b_true=b_true, w_ne_raw=w_ne_raw, w_ne_std=w_ne_std,
                w_gd_std=w_gd_std, losses=losses, r2=r2, ridge_norms=norms)


# ---------------------------------------------------------------------------
# Part B — Logistic regression
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable: never exponentiate a large positive number.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss(w, Xb, y, lam=0.0) -> float:
    """Mean binary cross-entropy, computed from logits for stability, + L2 (no bias)."""
    z = Xb @ w
    # -[y log s(z) + (1-y) log(1-s(z))] = log(1+e^z) - y z
    loss = np.mean(np.logaddexp(0.0, z) - y * z)
    return float(loss + 0.5 * lam * np.sum(w[:-1] ** 2))


def log_loss_grad(w, Xb, y, lam=0.0) -> np.ndarray:
    """Analytic gradient: (1/n) X^T (sigmoid(Xw) - y) + lam * w (no bias)."""
    g = Xb.T @ (sigmoid(Xb @ w) - y) / len(y)
    reg = lam * w
    reg[-1] = 0.0
    return g + reg


def numerical_grad(f, w, eps=1e-6) -> np.ndarray:
    """Central differences: (f(w+eps e_i) - f(w-eps e_i)) / (2 eps)."""
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = eps
        g[i] = (f(w + e) - f(w - e)) / (2 * eps)
    return g


def make_two_class(n=1000, seed=SEED):
    """Two Gaussian blobs; true boundary is linear (Bayes-optimal is logistic)."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    X0 = rng.multivariate_normal([-1.0, -1.0], [[1.0, 0.3], [0.3, 1.0]], n0)
    X1 = rng.multivariate_normal([1.5, 1.0], [[1.0, 0.3], [0.3, 1.0]], n - n0)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0), np.ones(n - n0)])
    return X, y


def fit_logreg_gd(Xb, y, lr=0.5, epochs=2000, lam=0.0):
    w = np.zeros(Xb.shape[1])
    losses = []
    for _ in range(epochs):
        losses.append(log_loss(w, Xb, y, lam))
        w -= lr * log_loss_grad(w, Xb, y, lam)
    return w, np.array(losses)


def part_b():
    print("\n" + "=" * 72)
    print("Part B: logistic regression on two Gaussian blobs")
    print("=" * 72)
    X, y = make_two_class()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    Xb_tr, Xb_te = add_bias(X_tr), add_bias(X_te)

    # Gradient check at a random point, with and without L2.
    rng = np.random.default_rng(SEED)
    w_rand = rng.standard_normal(Xb_tr.shape[1])
    rel_errs = []
    for lam in [0.0, 0.1]:
        g_a = log_loss_grad(w_rand, Xb_tr, y_tr, lam)
        g_n = numerical_grad(lambda w: log_loss(w, Xb_tr, y_tr, lam), w_rand)
        rel = np.linalg.norm(g_a - g_n) / (np.linalg.norm(g_a) + np.linalg.norm(g_n))
        rel_errs.append(rel)
        print(f"gradient check lam={lam}: analytic={np.round(g_a, 5)} "
              f"numeric={np.round(g_n, 5)} rel_err={rel:.2e}")

    w, losses = fit_logreg_gd(Xb_tr, y_tr, lr=0.5, epochs=2000)
    p_te = sigmoid(Xb_te @ w)
    acc = np.mean((p_te >= 0.5) == y_te)
    ll = log_loss(w, Xb_te, y_te)
    base_ll = -np.mean(y_te * np.log(y_tr.mean()) + (1 - y_te) * np.log(1 - y_tr.mean()))
    print(f"\nlearned w = {np.round(w, 3)}  (boundary: w1*x1 + w2*x2 + b = 0)")
    print(f"train log-loss: start {losses[0]:.4f} -> end {losses[-1]:.4f}")
    print(f"test accuracy : {acc:.3f}")
    print(f"test log-loss : {ll:.4f}   (constant-prior baseline: {base_ll:.4f})")

    # Calibration sanity check: mean predicted prob in bins vs observed rate.
    print("\nReliability table (test):")
    print(f"{'bin':>12}{'n':>6}{'mean p':>9}{'frac pos':>10}")
    bins = np.linspace(0, 1, 6)
    max_gap = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p_te >= lo) & (p_te < hi if hi < 1 else p_te <= hi)
        if m.sum() >= 10:
            gap = abs(p_te[m].mean() - y_te[m].mean())
            max_gap = max(max_gap, gap)
            print(f"[{lo:.1f},{hi:.1f}){m.sum():>6}{p_te[m].mean():>9.3f}{y_te[m].mean():>10.3f}")
    return dict(rel_errs=rel_errs, losses=losses, acc=acc, ll=ll, base_ll=base_ll,
                max_gap=max_gap)


# ---------------------------------------------------------------------------
# Main + self-checks
# ---------------------------------------------------------------------------

def main():
    np.random.seed(SEED)
    a = part_a()
    b = part_b()

    print("\nRunning self-checks ...")
    # Linear regression: normal equation recovers the generating weights (loosely; noise is large).
    assert np.allclose(a["w_ne_raw"][:3], a["w_true"], rtol=0.15), a["w_ne_raw"]
    # GD and normal equation agree on the standardised problem.
    assert np.allclose(a["w_gd_std"], a["w_ne_std"], atol=1e-4), "GD != normal equation"
    assert np.all(np.diff(a["losses"]) <= 1e-12), "GD loss must not increase"
    assert a["r2"] > 0.7, f"test R^2 too low: {a['r2']}"
    # Ridge: weight norm is monotonically decreasing in lambda.
    assert np.all(np.diff(a["ridge_norms"]) < 0), "ridge norms should shrink with lambda"

    # Logistic regression.
    assert all(r < 1e-6 for r in b["rel_errs"]), f"gradient check failed: {b['rel_errs']}"
    assert b["losses"][-1] < b["losses"][0], "log-loss did not decrease"
    assert b["acc"] > 0.88, f"accuracy too low: {b['acc']}"
    assert b["ll"] < 0.5 * b["base_ll"], "model should halve the baseline log-loss"
    assert b["max_gap"] < 0.15, f"model badly miscalibrated: {b['max_gap']}"
    # Stable sigmoid must not overflow / produce NaN at extreme logits.
    s = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.allclose(s, [0.0, 0.5, 1.0]) and np.all(np.isfinite(s))

    print("All checks passed.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Student exercises (TODO)
# ---------------------------------------------------------------------------
# TODO 1 (★)   Run fit_linreg_gd on the RAW (unstandardised) features. Find the
#              largest lr that does not diverge and count epochs to converge.
#              Relate this to the condition number of X^T X (np.linalg.cond).
# TODO 2 (★)   Add polynomial features x, x^2, ..., x^k to a 1D problem and plot
#              train vs test MSE as k grows. Then add ridge and repeat.
# TODO 3 (★★)  Implement lasso with coordinate descent (soft-thresholding) and
#              show that some weights become exactly zero as lambda grows.
# TODO 4 (★★)  Extend logistic regression to softmax (3 classes). Gradient-check it.
# TODO 5 (★★)  Make the blobs perfectly separable. What happens to ||w|| without
#              L2? With L2? Explain using the log-loss formula.
# TODO 6 (★★★) Implement the hashing trick: map string features like
#              "user_country=DE" to one of 2^18 buckets with a hash and train
#              logistic regression with sparse SGD updates.
