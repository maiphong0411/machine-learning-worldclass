"""
Lab 4 — Decision trees, random forests and gradient boosting (NumPy only)
==========================================================================

Module: M05 — Trees & ensembles (modules/05-trees-and-ensembles.md)

Learning goals
--------------
1. Implement Gini impurity / entropy and the greedy "best split" search that CART uses.
2. Grow a CART tree recursively for classification (Gini) and regression (variance / MSE),
   with max_depth and min_samples_leaf as stopping rules.
3. See over-fitting: a deep single tree has ~0 training error but worse test error.
4. Build a random forest by bagging + random feature subsets, and estimate its error
   with out-of-bag (OOB) samples instead of a held-out set.
5. Implement gradient boosting for regression: each round fits a shallow tree to the
   residuals (the negative gradient of squared error) and the training loss decreases
   monotonically.

Run:  python3 labs/04_decision_tree_and_boosting.py     (exits 0 if all checks pass)
"""

import numpy as np

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1. Impurity measures
# ---------------------------------------------------------------------------
def gini(y, n_classes):
    """Gini = 1 - sum_k p_k^2 : probability two random draws have different labels."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=n_classes) / len(y)
    return float(1.0 - np.sum(p**2))


def entropy(y, n_classes):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=n_classes) / len(y)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# 2. Best split search (vectorised over thresholds via sorting + prefix sums)
# ---------------------------------------------------------------------------
def best_split(X, y, task, n_classes=None, feature_ids=None, min_samples_leaf=1):
    """Return (feature, threshold, gain) of the best axis-aligned split or None.

    For each feature: sort the values, then every split position i puts the first i
    sorted samples on the left. Prefix sums give the impurity of both sides for ALL
    positions in O(n) after the O(n log n) sort.
      - classification: weighted Gini of children
      - regression:     weighted variance (= MSE around the child mean) of children
    """
    n, d = X.shape
    if feature_ids is None:
        feature_ids = np.arange(d)
    if task == "clf":
        parent = gini(y, n_classes)
    else:
        parent = float(np.var(y))
    best = None
    best_gain = 1e-12
    for j in feature_ids:
        order = np.argsort(X[:, j], kind="mergesort")
        xs, ys = X[order, j], y[order]
        n_left = np.arange(1, n)  # left sizes for split after position i-1
        n_right = n - n_left
        if task == "clf":
            onehot = np.eye(n_classes)[ys]
            left_counts = np.cumsum(onehot, axis=0)[:-1]
            right_counts = onehot.sum(0) - left_counts
            g_left = 1 - np.sum((left_counts / n_left[:, None]) ** 2, axis=1)
            g_right = 1 - np.sum((right_counts / n_right[:, None]) ** 2, axis=1)
            child = (n_left * g_left + n_right * g_right) / n
        else:
            cs = np.cumsum(ys)[:-1]
            cs2 = np.cumsum(ys**2)[:-1]
            tot, tot2 = ys.sum(), (ys**2).sum()
            var_left = cs2 / n_left - (cs / n_left) ** 2
            var_right = (tot2 - cs2) / n_right - ((tot - cs) / n_right) ** 2
            child = (n_left * var_left + n_right * var_right) / n
        valid = (xs[1:] != xs[:-1]) & (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
        if not np.any(valid):
            continue
        gains = np.where(valid, parent - child, -np.inf)
        i = int(np.argmax(gains))
        if gains[i] > best_gain:
            best_gain = float(gains[i])
            best = (int(j), float((xs[i] + xs[i + 1]) / 2.0), best_gain)
    return best


# ---------------------------------------------------------------------------
# 3. CART tree
# ---------------------------------------------------------------------------
class Tree:
    """CART tree. task='clf' (Gini, leaf = class distribution) or 'reg' (MSE, leaf = mean)."""

    def __init__(self, task="clf", max_depth=5, min_samples_leaf=1, max_features=None, rng=None):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # None = all features; int = random subset per node
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def fit(self, X, y):
        self.n_classes = int(y.max()) + 1 if self.task == "clf" else None
        self.root = self._grow(X, y, depth=0)
        return self

    def _leaf(self, y):
        if self.task == "clf":
            return {"leaf": np.bincount(y, minlength=self.n_classes) / len(y)}
        return {"leaf": float(np.mean(y))}

    def _grow(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return self._leaf(y)
        if self.task == "clf" and np.all(y == y[0]):
            return self._leaf(y)
        d = X.shape[1]
        feats = None
        if self.max_features is not None and self.max_features < d:
            feats = self.rng.choice(d, self.max_features, replace=False)
        split = best_split(X, y, self.task, self.n_classes, feats, self.min_samples_leaf)
        if split is None:
            return self._leaf(y)
        j, t, gain = split
        m = X[:, j] <= t
        return {
            "feature": j,
            "threshold": t,
            "gain": gain,
            "n": len(y),
            "left": self._grow(X[m], y[m], depth + 1),
            "right": self._grow(X[~m], y[~m], depth + 1),
        }

    def _predict_one(self, x):
        node = self.root
        while "leaf" not in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["leaf"]

    def predict_value(self, X):
        """Regression: mean; classification: class-probability vector."""
        return np.array([self._predict_one(x) for x in X])

    def predict(self, X):
        out = self.predict_value(X)
        return out.argmax(1) if self.task == "clf" else out

    def feature_importance(self, d):
        """Mean-decrease-in-impurity: sum of n_node * gain over splits on each feature."""
        imp = np.zeros(d)
        stack = [self.root]
        while stack:
            node = stack.pop()
            if "leaf" in node:
                continue
            imp[node["feature"]] += node["n"] * node["gain"]
            stack += [node["left"], node["right"]]
        return imp / imp.sum() if imp.sum() > 0 else imp


# ---------------------------------------------------------------------------
# 4. Random forest (bagging + feature subsampling) with OOB error
# ---------------------------------------------------------------------------
class RandomForest:
    def __init__(self, n_trees=30, max_depth=8, max_features="sqrt", seed=0):
        self.n_trees, self.max_depth, self.max_features = n_trees, max_depth, max_features
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        n, d = X.shape
        k = max(1, int(np.sqrt(d))) if self.max_features == "sqrt" else self.max_features
        self.n_classes = int(y.max()) + 1
        self.trees = []
        oob_votes = np.zeros((n, self.n_classes))
        for _ in range(self.n_trees):
            idx = self.rng.integers(0, n, n)  # bootstrap sample (with replacement)
            oob = np.setdiff1d(np.arange(n), idx)  # ~36.8% of rows never drawn
            t = Tree("clf", self.max_depth, max_features=k, rng=self.rng).fit(X[idx], y[idx])
            # pad probabilities if the bootstrap missed a class
            self.trees.append(t)
            if len(oob):
                pv = t.predict_value(X[oob])
                oob_votes[oob, : pv.shape[1]] += pv
        seen = oob_votes.sum(1) > 0
        self.oob_error_ = float(np.mean(oob_votes[seen].argmax(1) != y[seen]))
        return self

    def predict(self, X):
        probs = np.zeros((len(X), self.n_classes))
        for t in self.trees:
            pv = t.predict_value(X)
            probs[:, : pv.shape[1]] += pv
        return probs.argmax(1)


# ---------------------------------------------------------------------------
# 5. Gradient boosting for regression (squared error)
# ---------------------------------------------------------------------------
class GradientBoostingRegressor:
    """F_0 = mean(y);  F_m = F_{m-1} + lr * h_m, where h_m fits r = y - F_{m-1}.

    For L = 1/2 (y - F)^2 the negative gradient dL/dF is exactly the residual y - F,
    so "fit the residuals" IS gradient descent in function space.
    """

    def __init__(self, n_rounds=100, learning_rate=0.1, max_depth=3):
        self.n_rounds, self.lr, self.max_depth = n_rounds, learning_rate, max_depth

    def fit(self, X, y, X_val=None, y_val=None):
        self.f0 = float(np.mean(y))
        self.trees = []
        F = np.full(len(y), self.f0)
        self.train_loss_ = [float(np.mean((y - F) ** 2))]
        self.val_loss_ = []
        Fv = np.full(len(y_val), self.f0) if X_val is not None else None
        for _ in range(self.n_rounds):
            residual = y - F  # pseudo-residual = negative gradient
            h = Tree("reg", self.max_depth, min_samples_leaf=5).fit(X, residual)
            F = F + self.lr * h.predict(X)
            self.trees.append(h)
            self.train_loss_.append(float(np.mean((y - F) ** 2)))
            if Fv is not None:
                Fv = Fv + self.lr * h.predict(X_val)
                self.val_loss_.append(float(np.mean((y_val - Fv) ** 2)))
        return self

    def predict(self, X):
        F = np.full(len(X), self.f0)
        for h in self.trees:
            F = F + self.lr * h.predict(X)
        return F


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_classification(n=600, d=6, rng=rng):
    """Two informative features with a non-linear (rule-like + disc) boundary, rest noise."""
    X = rng.normal(size=(n, d))
    r = X[:, 0] ** 2 + X[:, 1] ** 2
    y = (((X[:, 0] > 0.3) & (X[:, 1] > -0.5)) | (r < 0.5)).astype(int)
    flip = rng.random(n) < 0.08  # label noise -> deep trees will over-fit it
    y[flip] = 1 - y[flip]
    return X, y


def make_regression(n=500, rng=rng):
    X = rng.uniform(-3, 3, size=(n, 3))
    y = np.sin(X[:, 0]) * 2 + 0.5 * X[:, 1] ** 2 - X[:, 2] + rng.normal(0, 0.3, n)
    return X, y


def main():
    print("=" * 70)
    print("Lab 4: decision trees, random forests, gradient boosting")
    print("=" * 70)

    # --- impurity sanity --------------------------------------------------
    y_pure, y_mix = np.array([1, 1, 1, 1]), np.array([0, 1, 0, 1])
    print(f"\n[1] Gini(pure)={gini(y_pure, 2):.2f} Gini(50/50)={gini(y_mix, 2):.2f} "
          f"Entropy(50/50)={entropy(y_mix, 2):.2f} bits")
    assert gini(y_pure, 2) == 0 and np.isclose(gini(y_mix, 2), 0.5) and np.isclose(entropy(y_mix, 2), 1.0)

    # worked example from the module: 10 loan applicants, split on income
    income = np.array([20, 25, 30, 35, 40, 60, 70, 80, 90, 100], float)[:, None]
    default = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    j, t, g = best_split(income, default, "clf", 2)
    print(f"    loan example: best split income <= {t:.1f}, Gini gain = {g:.3f}")
    assert j == 0 and np.isclose(t, 50.0) and np.isclose(g, 0.48 - 0.5 * 0.32, atol=1e-9)

    # --- single trees: shallow vs deep -----------------------------------
    X, y = make_classification()
    Xtr, ytr, Xte, yte = X[:400], y[:400], X[400:], y[400:]
    print("\n[2] single CART trees (classification, Gini)")
    errs = {}
    for depth in [1, 3, 6, 20]:
        tree = Tree("clf", max_depth=depth).fit(Xtr, ytr)
        tr, te = np.mean(tree.predict(Xtr) != ytr), np.mean(tree.predict(Xte) != yte)
        errs[depth] = (tr, te)
        print(f"    depth={depth:2d}  train err={tr:.3f}  test err={te:.3f}")
    assert errs[20][0] < 0.01, "a fully grown tree should (nearly) memorize the training set"
    assert errs[20][1] > errs[20][0] + 0.1, "...and over-fit"
    imp = Tree("clf", max_depth=6).fit(Xtr, ytr).feature_importance(X.shape[1])
    print(f"    impurity-based importance: {np.round(imp, 3)}")
    assert imp[:2].sum() > 0.5, "the two informative features should dominate"

    # --- random forest -----------------------------------------------------
    rf = RandomForest(n_trees=40, max_depth=10, seed=1).fit(Xtr, ytr)
    rf_te = float(np.mean(rf.predict(Xte) != yte))
    print(f"\n[3] random forest (40 trees): test err={rf_te:.3f}  OOB err={rf.oob_error_:.3f}")
    assert rf_te < errs[20][1], "averaging decorrelated deep trees should beat one deep tree"
    assert abs(rf.oob_error_ - rf_te) < 0.1, "OOB error should approximate test error"

    # --- regression tree + gradient boosting -------------------------------
    Xr, yr = make_regression()
    Xr_tr, yr_tr, Xr_te, yr_te = Xr[:400], yr[:400], Xr[400:], yr[400:]
    stump_mse = float(np.mean((Tree("reg", 1).fit(Xr_tr, yr_tr).predict(Xr_te) - yr_te) ** 2))
    gb = GradientBoostingRegressor(n_rounds=150, learning_rate=0.1, max_depth=3)
    gb.fit(Xr_tr, yr_tr, Xr_te, yr_te)
    print("\n[4] gradient boosting (squared error, depth-3 trees, lr=0.1)")
    for m in [0, 1, 2, 5, 10, 25, 50, 100, 150]:
        v = f"{gb.val_loss_[m - 1]:.4f}" if m > 0 else "  -   "
        print(f"    round {m:3d}: train MSE={gb.train_loss_[m]:.4f}  val MSE={v}")
    tl = np.array(gb.train_loss_)
    assert np.all(np.diff(tl) <= 1e-10), "training loss must decrease every round"
    assert tl[-1] < 0.2 * tl[0]
    gb_mse = float(np.mean((gb.predict(Xr_te) - yr_te) ** 2))
    print(f"    test MSE: single stump={stump_mse:.3f}  boosted={gb_mse:.3f}  (noise var=0.09)")
    assert gb_mse < 0.5 * stump_mse
    assert np.isclose(gb.val_loss_[-1], gb_mse)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Student exercises
# ---------------------------------------------------------------------------
# 1. (★) Switch the classification criterion from Gini to entropy. Do the chosen splits
#        on the loan example change? Do test errors change materially?
# 2. (★) Plot (or print) RF test error vs number of trees (1, 5, 20, 80). Why does it
#        flatten instead of over-fitting as you add trees?
# 3. (★★) Add early stopping to GradientBoostingRegressor: stop when the validation MSE
#        has not improved for 10 rounds. Try learning_rate 1.0 vs 0.1 vs 0.01.
# 4. (★★) Implement gradient boosting for binary classification with log-loss:
#        pseudo-residual r = y - sigmoid(F); leaf value = sum(r) / sum(p(1-p)) (Newton step).
# 5. (★★) Implement permutation importance on the test set and compare it with the
#        impurity-based importance. Add a high-cardinality random ID feature: which method
#        is fooled?
# 6. (★★★) Implement histogram-based splitting (bin each feature into 32 quantile bins once,
#        then search only bin edges) and measure the speed-up on n=20,000 rows.
