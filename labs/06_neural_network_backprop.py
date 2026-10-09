"""
Lab 6 — A neural network from scratch: forward pass, manual backprop, gradient check.

Module: M07 Neural networks (modules/07-neural-networks.md)

Learning goals
--------------
1. Implement a 2-layer MLP (input -> hidden ReLU -> softmax output) with NumPy only.
2. Derive and code backpropagation by hand, layer by layer, using the chain rule.
3. Verify the analytic gradients against central finite differences (gradient check).
4. See *why* nonlinearity matters: a linear model cannot solve XOR; the MLP can.
5. Train on a non-linearly-separable 2-D "two spirals" dataset and reach high accuracy.

Run:  python3 labs/06_neural_network_backprop.py
Dependencies: numpy only. Runtime: a few seconds.
"""

import numpy as np

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1. Building blocks
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def softmax(logits):
    # Subtract the row max for numerical stability: softmax is shift-invariant.
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    """Mean negative log-likelihood of the true class. y holds integer labels."""
    n = y.shape[0]
    return -np.mean(np.log(probs[np.arange(n), y] + 1e-12))


def init_params(d_in, d_hidden, d_out, rng):
    """He initialisation for the ReLU layer: Var(W) = 2 / fan_in.

    Keeps the variance of activations roughly constant across layers so that
    signals (and gradients) neither explode nor vanish at the start of training.
    """
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / d_in), size=(d_in, d_hidden)),
        "b1": np.zeros(d_hidden),
        "W2": rng.normal(0.0, np.sqrt(2.0 / d_hidden), size=(d_hidden, d_out)),
        "b2": np.zeros(d_out),
    }


# ---------------------------------------------------------------------------
# 2. Forward and backward pass
# ---------------------------------------------------------------------------
def forward(params, X):
    """Return probabilities and a cache of intermediates needed by backward()."""
    z1 = X @ params["W1"] + params["b1"]        # (n, h)  pre-activation
    a1 = relu(z1)                               # (n, h)  hidden activation
    logits = a1 @ params["W2"] + params["b2"]   # (n, k)
    probs = softmax(logits)                     # (n, k)
    cache = (X, z1, a1, probs)
    return probs, cache


def loss_fn(params, X, y, weight_decay=0.0):
    probs, _ = forward(params, X)
    l2 = 0.5 * weight_decay * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
    return cross_entropy(probs, y) + l2


def backward(params, cache, y, weight_decay=0.0):
    """Manual backprop. Each line is one application of the chain rule.

    Key identity: for softmax + cross-entropy, dL/dlogits = (p - onehot(y)) / n.
    """
    X, z1, a1, probs = cache
    n = X.shape[0]
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y] = 1.0

    dlogits = (probs - onehot) / n               # (n, k)
    dW2 = a1.T @ dlogits                         # (h, k)
    db2 = dlogits.sum(axis=0)                    # (k,)
    da1 = dlogits @ params["W2"].T               # (n, h)
    dz1 = da1 * (z1 > 0)                         # (n, h)  ReLU gate: gradient 1 or 0
    dW1 = X.T @ dz1                              # (d, h)
    db1 = dz1.sum(axis=0)                        # (h,)

    dW1 += weight_decay * params["W1"]
    dW2 += weight_decay * params["W2"]
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ---------------------------------------------------------------------------
# 3. Gradient check: analytic vs numerical (central differences)
# ---------------------------------------------------------------------------
def gradient_check(params, X, y, weight_decay=0.0, eps=1e-5):
    _, cache = forward(params, X)
    analytic = backward(params, cache, y, weight_decay)
    max_rel_err = 0.0
    for name, P in params.items():
        num = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = P[idx]
            P[idx] = old + eps
            lp = loss_fn(params, X, y, weight_decay)
            P[idx] = old - eps
            lm = loss_fn(params, X, y, weight_decay)
            P[idx] = old
            num[idx] = (lp - lm) / (2 * eps)
        a = analytic[name]
        rel = np.linalg.norm(a - num) / (np.linalg.norm(a) + np.linalg.norm(num) + 1e-12)
        print(f"  grad check {name:>2}: relative error = {rel:.2e}")
        max_rel_err = max(max_rel_err, rel)
    return max_rel_err


# ---------------------------------------------------------------------------
# 4. Training loop (full-batch or mini-batch gradient descent with momentum)
# ---------------------------------------------------------------------------
def train(X, y, d_hidden, n_classes, lr, epochs, weight_decay=0.0,
          batch_size=None, seed=0, verbose_every=0):
    rng = np.random.default_rng(seed)
    params = init_params(X.shape[1], d_hidden, n_classes, rng)
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    n = X.shape[0]
    bs = batch_size or n
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            _, cache = forward(params, X[idx])
            grads = backward(params, cache, y[idx], weight_decay)
            for k in params:
                velocity[k] = 0.9 * velocity[k] - lr * grads[k]
                params[k] += velocity[k]
        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"    epoch {epoch + 1:4d}  loss={loss_fn(params, X, y, weight_decay):.4f}"
                  f"  acc={accuracy(params, X, y):.3f}")
    return params


def accuracy(params, X, y):
    probs, _ = forward(params, X)
    return float(np.mean(probs.argmax(axis=1) == y))


# ---------------------------------------------------------------------------
# 5. Datasets
# ---------------------------------------------------------------------------
def make_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y


def make_spirals(n_per_class=200, n_classes=2, noise=0.2, rng=RNG):
    """Interleaved spiral arms (CS231n style): impossible for a linear model."""
    X, y = [], []
    for c in range(n_classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(0.0, 4.5, n_per_class) + c * 2 * np.pi / n_classes
        t = t + rng.normal(0, noise, n_per_class)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, c))
    X = np.vstack(X)
    y = np.concatenate(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def linear_softmax_accuracy(X, y, n_classes, lr=0.5, epochs=500):
    """Baseline: softmax regression (no hidden layer) trained by gradient descent."""
    W = np.zeros((X.shape[1], n_classes))
    b = np.zeros(n_classes)
    n = len(y)
    onehot = np.eye(n_classes)[y]
    for _ in range(epochs):
        p = softmax(X @ W + b)
        g = (p - onehot) / n
        W -= lr * X.T @ g
        b -= lr * g.sum(axis=0)
    return float(np.mean((X @ W + b).argmax(axis=1) == y))


# ---------------------------------------------------------------------------
# 6. Main: run the experiments and self-check with asserts
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Lab 6: neural network backprop from scratch")
    print("=" * 70)

    # --- 6a. Gradient check on a small random problem ---------------------
    print("\n[1] Gradient check (analytic backprop vs central differences)")
    rng = np.random.default_rng(1)
    Xg = rng.normal(size=(8, 3))
    yg = rng.integers(0, 3, size=8)
    pg = init_params(3, 5, 3, rng)
    # Nudge pre-activations away from the ReLU kink at 0, where the numerical
    # derivative is ill-defined.
    pg["b1"] += 0.1
    err = gradient_check(pg, Xg, yg, weight_decay=1e-2)
    assert err < 1e-6, f"Gradient check failed: relative error {err:.2e}"
    print(f"  -> max relative error {err:.2e} < 1e-6  OK")

    # --- 6b. XOR: linear fails, MLP succeeds ------------------------------
    print("\n[2] XOR")
    Xx, yx = make_xor()
    lin_acc = linear_softmax_accuracy(Xx, yx, 2)
    print(f"  linear softmax regression accuracy: {lin_acc:.2f}  (cannot exceed 0.75)")
    assert lin_acc <= 0.75, "A linear classifier should not solve XOR"
    px = train(Xx, yx, d_hidden=8, n_classes=2, lr=0.1, epochs=2000, seed=3)
    mlp_acc = accuracy(px, Xx, yx)
    probs, _ = forward(px, Xx)
    print(f"  MLP (8 ReLU hidden units) accuracy: {mlp_acc:.2f}")
    print("  P(class=1) for inputs (0,0),(0,1),(1,0),(1,1):", np.round(probs[:, 1], 3))
    assert mlp_acc == 1.0, "MLP should solve XOR perfectly"

    # --- 6c. Two spirals --------------------------------------------------
    print("\n[3] Two spirals (non-linearly separable)")
    X, y = make_spirals(n_per_class=300, noise=0.15)
    n_train = int(0.8 * len(y))
    Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    lin_acc = linear_softmax_accuracy(Xtr, ytr, 2)
    print(f"  linear baseline accuracy (train set): {lin_acc:.3f}")
    ps = train(Xtr, ytr, d_hidden=64, n_classes=2, lr=0.05, epochs=600,
               weight_decay=1e-4, batch_size=64, seed=0, verbose_every=150)
    tr_acc, te_acc = accuracy(ps, Xtr, ytr), accuracy(ps, Xte, yte)
    print(f"  MLP train acc={tr_acc:.3f}  test acc={te_acc:.3f}")
    assert lin_acc < 0.80, "Linear model should struggle on spirals"
    assert te_acc > 0.90, f"MLP should reach >90% test accuracy on spirals, got {te_acc:.3f}"

    # --- 6d. Initialisation matters: tiny init kills the signal -----------
    print("\n[4] Initialisation: activation scale through a 10-layer ReLU stack")
    h = np.random.default_rng(2).normal(size=(256, 100))
    for name, scale in [("tiny N(0, 0.01^2)", lambda fan_in: 0.01),
                        ("He N(0, 2/fan_in)", lambda fan_in: np.sqrt(2.0 / fan_in))]:
        a = h.copy()
        r = np.random.default_rng(5)
        for _ in range(10):
            a = relu(a @ r.normal(0, scale(a.shape[1]), size=(a.shape[1], 100)))
        print(f"  {name:<20} std of layer-10 activations = {a.std():.2e}")
        if name.startswith("He"):
            assert 0.1 < a.std() < 10, "He init should keep activations O(1)"
        else:
            assert a.std() < 1e-6, "Tiny init should make activations vanish"

    print("\nAll Lab 6 checks passed.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Student exercises
# ---------------------------------------------------------------------------
# 1. (★) Replace ReLU with tanh in forward/backward (d tanh(z)/dz = 1 - tanh(z)^2).
#    Re-run the gradient check. Does the spiral accuracy change? Training speed?
# 2. (★) Set d_hidden=2 on the spirals. What accuracy do you get and why?
#    Plot (or print) accuracy vs d_hidden in {2, 4, 8, 16, 64}.
# 3. (★★) Add a second hidden layer. Write its backward pass and gradient-check it.
# 4. (★★) Implement inverted dropout on the hidden layer (scale by 1/keep_prob at
#    train time, identity at test time). Remember the mask in the cache for backprop.
# 5. (★★) Replace momentum SGD with Adam (see M02). Compare epochs to 95% train accuracy.
# 6. (★★★) Initialise all weights to the same constant. Show that every hidden unit
#    receives the identical gradient forever ("symmetry"), so the network behaves like
#    one unit. Explain why random initialisation breaks the symmetry.
