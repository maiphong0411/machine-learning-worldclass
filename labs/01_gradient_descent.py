"""
Lab 1 — Gradient descent from scratch (NumPy only)
==================================================

Companion to:
  * M01 What is learning?            (modules/01-ml-first-principles.md)
  * M02 Math toolkit: optimization   (modules/02-math-toolkit.md)

Learning goals
--------------
After finishing this lab you should be able to:
  1. Implement batch gradient descent, stochastic / mini-batch gradient descent,
     momentum and Adam using nothing but NumPy.
  2. Explain why an ill-conditioned quadratic bowl makes plain gradient descent
     zig-zag, and why momentum and Adam help.
  3. Train linear regression by minimising mean squared error and verify that the
     recovered weights match the closed-form (normal-equation) solution.
  4. Read a convergence table and reason about learning rate, noise and speed.

How to run
----------
    python3 labs/01_gradient_descent.py

The script prints convergence tables and ends with assertions. If it exits with
code 0, every check passed.
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# Part A — A 2D quadratic bowl
# ---------------------------------------------------------------------------
#
#   f(w) = 1/2 * w^T A w - b^T w
#   grad f(w) = A w - b
#   minimiser w* = A^{-1} b
#
# A is symmetric positive definite with eigenvalues 1 and 25, so the
# condition number is 25: the bowl is a long, thin valley.

A = np.array([[13.0, 12.0],
              [12.0, 13.0]])          # eigenvalues: 25 (along [1,1]) and 1 (along [1,-1])
b = np.array([1.0, -2.0])
W_STAR_BOWL = np.linalg.solve(A, b)


def bowl_loss(w: np.ndarray) -> float:
    return float(0.5 * w @ A @ w - b @ w)


def bowl_grad(w: np.ndarray) -> np.ndarray:
    return A @ w - b


class Optimizer:
    """Minimal optimizer interface: step(w, g) -> new w."""

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class GD(Optimizer):
    """Plain gradient descent: w <- w - lr * g."""

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, w, g):
        return w - self.lr * g


class Momentum(Optimizer):
    """Heavy-ball momentum: v <- beta*v + g ;  w <- w - lr*v."""

    def __init__(self, lr: float, beta: float = 0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, w, g):
        if self.v is None:
            self.v = np.zeros_like(w)
        self.v = self.beta * self.v + g
        return w - self.lr * self.v


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with bias correction."""

    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, w, g):
        if self.m is None:
            self.m, self.v = np.zeros_like(w), np.zeros_like(w)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def run_bowl(opt: Optimizer, steps: int = 300, w0=(-2.0, 2.0), noise: float = 0.0,
             rng: np.random.Generator | None = None):
    """Run an optimizer on the bowl. noise>0 simulates a stochastic gradient."""
    w = np.array(w0, dtype=float)
    losses = [bowl_loss(w)]
    for _ in range(steps):
        g = bowl_grad(w)
        if noise > 0:
            g = g + noise * rng.standard_normal(2)
        w = opt.step(w, g)
        losses.append(bowl_loss(w))
    return w, np.array(losses)


def steps_to_tol(losses: np.ndarray, f_star: float, tol: float = 1e-6) -> int | str:
    gap = losses - f_star
    idx = np.where(gap < tol)[0]
    return int(idx[0]) if len(idx) else ">max"


def part_a():
    print("=" * 72)
    print("Part A: 2D quadratic bowl, condition number = 25")
    print(f"  true minimiser w* = {W_STAR_BOWL}")
    print("=" * 72)
    f_star = bowl_loss(W_STAR_BOWL)
    rng = np.random.default_rng(SEED)

    # Max stable lr for GD is 2 / lambda_max = 2/25 = 0.08.
    configs = {
        "GD lr=0.07":           GD(lr=0.07),
        "GD lr=0.01 (too small)": GD(lr=0.01),
        "Momentum lr=0.01 b=0.9": Momentum(lr=0.01, beta=0.9),
        "Adam lr=0.1":          Adam(lr=0.1),
    }
    results = {}
    print(f"{'optimizer':<26}{'final loss gap':>16}{'steps to 1e-6':>16}{'||w-w*||':>12}")
    for name, opt in configs.items():
        w, losses = run_bowl(opt, steps=400)
        results[name] = (w, losses)
        print(f"{name:<26}{losses[-1] - f_star:>16.2e}{str(steps_to_tol(losses, f_star)):>16}"
              f"{np.linalg.norm(w - W_STAR_BOWL):>12.2e}")

    # Divergence demo: lr above 2/lambda_max.
    _, diverge = run_bowl(GD(lr=0.085), steps=60)
    print(f"\nGD lr=0.085 (> 2/25 = 0.08) after 60 steps: loss = {diverge[-1]:.3e}  <- diverges")

    # Noisy gradients (SGD-like) with a decaying learning rate.
    w = np.array([-2.0, 2.0])
    for t in range(2000):
        g = bowl_grad(w) + 1.0 * rng.standard_normal(2)
        w = w - (0.05 / (1 + 0.01 * t)) * g
    print(f"Noisy GD with decaying lr, 2000 steps: ||w-w*|| = {np.linalg.norm(w - W_STAR_BOWL):.3f}")
    return results, diverge, w


# ---------------------------------------------------------------------------
# Part B — Linear regression on synthetic data
# ---------------------------------------------------------------------------
#
#   y = X w_true + b_true + noise
#   L(w) = 1/n * sum_i (x_i^T w - y_i)^2      (bias folded into w via a ones column)
#   grad L = 2/n * X^T (X w - y)

def make_regression(n: int = 1000, d: int = 3, noise: float = 0.1, seed: int = SEED):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w_true = np.array([2.0, -3.0, 0.5])[:d]
    b_true = 1.0
    y = X @ w_true + b_true + noise * rng.standard_normal(n)
    Xb = np.hstack([X, np.ones((n, 1))])      # append bias column
    return Xb, y, np.append(w_true, b_true)


def mse(Xb, y, w) -> float:
    r = Xb @ w - y
    return float(r @ r / len(y))


def mse_grad(Xb, y, w) -> np.ndarray:
    return 2.0 / len(y) * Xb.T @ (Xb @ w - y)


def train_linreg(Xb, y, opt_factory, epochs: int, batch_size: int | None, seed: int = SEED):
    """batch_size=None -> full-batch GD; 1 -> pure SGD; k -> mini-batch."""
    rng = np.random.default_rng(seed)
    n, d = Xb.shape
    w = np.zeros(d)
    opt = opt_factory()
    history = [mse(Xb, y, w)]
    bs = n if batch_size is None else batch_size
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            w = opt.step(w, mse_grad(Xb[idx], y[idx], w))
        history.append(mse(Xb, y, w))
    return w, np.array(history)


def part_b():
    print("\n" + "=" * 72)
    print("Part B: linear regression, n=1000, d=3 (+bias), noise sd=0.1")
    print("=" * 72)
    Xb, y, w_true = make_regression()
    w_closed = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    print(f"true weights        : {np.round(w_true, 3)}")
    print(f"normal equation     : {np.round(w_closed, 3)}   MSE = {mse(Xb, y, w_closed):.5f}")

    configs = {
        "Batch GD (lr=0.1)":         (lambda: GD(0.1), 100, None),
        "SGD bs=1 (lr=0.01)":        (lambda: GD(0.01), 10, 1),
        "Mini-batch bs=32 (lr=0.05)": (lambda: GD(0.05), 20, 32),
        "Momentum bs=32 (lr=0.01)":  (lambda: Momentum(0.01, 0.9), 20, 32),
        "Adam bs=32 (lr=0.05)":      (lambda: Adam(0.05), 20, 32),
    }
    results = {}
    print(f"\n{'optimizer':<28}{'epochs':>7}{'updates':>9}{'final MSE':>12}{'||w-w_closed||':>16}")
    for name, (fac, epochs, bs) in configs.items():
        w, hist = train_linreg(Xb, y, fac, epochs, bs)
        updates = epochs * (1 if bs is None else int(np.ceil(len(y) / bs)))
        results[name] = (w, hist)
        print(f"{name:<28}{epochs:>7}{updates:>9}{hist[-1]:>12.5f}"
              f"{np.linalg.norm(w - w_closed):>16.4f}")
    return Xb, y, w_true, w_closed, results


# ---------------------------------------------------------------------------
# Main + self-checks
# ---------------------------------------------------------------------------

def main():
    np.random.seed(SEED)  # legacy global seed, for anyone who adds np.random.* calls
    bowl_results, diverge, w_noisy = part_a()
    Xb, y, w_true, w_closed, lin_results = part_b()

    print("\nRunning self-checks ...")
    # Bowl: GD with a good lr, momentum and Adam reach the minimiser.
    for name in ["GD lr=0.07", "Momentum lr=0.01 b=0.9", "Adam lr=0.1"]:
        w, losses = bowl_results[name]
        assert np.linalg.norm(w - W_STAR_BOWL) < 1e-2, f"{name} did not converge: {w}"
        assert losses[-1] < losses[0], f"{name}: loss did not decrease"
    # Momentum beats plain GD at the same small learning rate.
    gap_gd = bowl_results["GD lr=0.01 (too small)"][1][-1] - bowl_loss(W_STAR_BOWL)
    gap_mom = bowl_results["Momentum lr=0.01 b=0.9"][1][-1] - bowl_loss(W_STAR_BOWL)
    assert gap_mom < gap_gd, "momentum should beat GD at equal lr on an ill-conditioned bowl"
    # Too-large learning rate diverges.
    assert diverge[-1] > diverge[0], "lr > 2/lambda_max should diverge"
    # Noisy GD with decaying lr gets close.
    assert np.linalg.norm(w_noisy - W_STAR_BOWL) < 0.3

    # Linear regression: closed form recovers truth; every optimizer matches it.
    assert np.allclose(w_closed, w_true, atol=0.05), "normal equation far from true weights"
    for name, (w, hist) in lin_results.items():
        assert np.allclose(w, w_true, atol=0.1), f"{name}: weights {w} far from {w_true}"
        assert hist[-1] < 0.1 * hist[0], f"{name}: loss did not drop by 10x"
        assert hist[-1] < 0.05, f"{name}: final MSE too high ({hist[-1]})"
    # Full-batch GD loss must be monotonically non-increasing (small lr, convex problem).
    hist_gd = lin_results["Batch GD (lr=0.1)"][1]
    assert np.all(np.diff(hist_gd) <= 1e-12), "batch GD loss should never increase"

    print("All checks passed.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Student exercises (TODO)
# ---------------------------------------------------------------------------
# TODO 1 (★)   Change A to [[13, 12], [12, 13.5]] and recompute the max stable
#              learning rate 2/lambda_max with np.linalg.eigvalsh. Verify empirically.
# TODO 2 (★)   Add Nesterov momentum (look-ahead gradient) and compare it with
#              heavy-ball momentum on the bowl.
# TODO 3 (★★)  Implement RMSProp (Adam without the first moment). Which of Adam's
#              two moments matters more on this bowl? Ablate each.
# TODO 4 (★★)  Plot (or print) the trajectory w_t for GD lr=0.07 and explain the
#              zig-zag in terms of the eigenvectors of A.
# TODO 5 (★★)  For mini-batch SGD on Part B, measure final ||w - w_closed|| as a
#              function of batch size {1, 8, 32, 256, 1000} at a fixed number of
#              gradient evaluations. Explain the trade-off.
# TODO 6 (★★★) Add a feature with scale 100x the others to Part B. Show that GD
#              now needs a much smaller lr, then fix it with standardisation.
