"""
Lab 7 — Attention and embeddings from scratch.

Module: M08 Embeddings, sequences & transformers (modules/08-embeddings-and-transformers.md)

Learning goals
--------------
1. Implement scaled dot-product attention softmax(Q K^T / sqrt(d_k)) V with NumPy only.
2. Verify the two invariants every attention implementation must satisfy:
   (a) each row of the attention matrix is a probability distribution (sums to 1);
   (b) a causal mask stops position i from attending to any position j > i.
3. Show empirically WHY we divide by sqrt(d_k): unscaled dot products grow with d_k and
   saturate the softmax (near one-hot rows, vanishing gradients).
4. Implement multi-head attention (split, attend, concatenate, project) with shape asserts,
   and check that the causal model's output at position t does not change when future
   tokens change.
5. Build a tiny embedding table, compute cosine similarity, and do top-k retrieval
   (the core of semantic search and of two-tower retrieval in M09).

Run:  python3 labs/07_attention_and_embeddings.py
Dependencies: numpy only. Runtime: < 1 second.
"""

import numpy as np

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1. Scaled dot-product attention
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_mask(T):
    """Boolean (T, T) matrix: True where attention is ALLOWED (j <= i)."""
    return np.tril(np.ones((T, T), dtype=bool))


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q: (..., T_q, d_k), K: (..., T_k, d_k), V: (..., T_k, d_v).

    Returns (output (..., T_q, d_v), weights (..., T_q, T_k)).
    mask: boolean broadcastable to (..., T_q, T_k); False positions get -inf scores.
    """
    d_k = Q.shape[-1]
    assert K.shape[-1] == d_k, "Q and K must share the key dimension"
    assert K.shape[-2] == V.shape[-2], "K and V must have the same number of positions"
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)          # (..., T_q, T_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)                     # ~ -inf, avoids NaN
    weights = softmax(scores, axis=-1)
    out = weights @ V                                             # (..., T_q, d_v)
    return out, weights


# ---------------------------------------------------------------------------
# 2. Multi-head attention (self-attention over a sequence X)
# ---------------------------------------------------------------------------
def init_mha(d_model, n_heads, rng):
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    s = 1.0 / np.sqrt(d_model)
    return {
        "Wq": rng.normal(0, s, (d_model, d_model)),
        "Wk": rng.normal(0, s, (d_model, d_model)),
        "Wv": rng.normal(0, s, (d_model, d_model)),
        "Wo": rng.normal(0, s, (d_model, d_model)),
        "n_heads": n_heads,
    }


def split_heads(X, n_heads):
    """(B, T, d_model) -> (B, h, T, d_head)"""
    B, T, d = X.shape
    return X.reshape(B, T, n_heads, d // n_heads).transpose(0, 2, 1, 3)


def merge_heads(X):
    """(B, h, T, d_head) -> (B, T, h * d_head)"""
    B, h, T, dh = X.shape
    return X.transpose(0, 2, 1, 3).reshape(B, T, h * dh)


def multi_head_attention(params, X, causal=False):
    B, T, d_model = X.shape
    h = params["n_heads"]
    d_head = d_model // h
    Q = split_heads(X @ params["Wq"], h)
    K = split_heads(X @ params["Wk"], h)
    V = split_heads(X @ params["Wv"], h)
    assert Q.shape == (B, h, T, d_head), Q.shape
    mask = causal_mask(T)[None, None] if causal else None        # broadcast over B, h
    heads, weights = scaled_dot_product_attention(Q, K, V, mask)
    assert heads.shape == (B, h, T, d_head), heads.shape
    assert weights.shape == (B, h, T, T), weights.shape
    out = merge_heads(heads) @ params["Wo"]
    assert out.shape == (B, T, d_model), out.shape
    return out, weights


def sinusoidal_positions(T, d_model):
    """The original Transformer positional encoding (Vaswani et al., 2017)."""
    pos = np.arange(T)[:, None]
    i = np.arange(d_model // 2)[None, :]
    angles = pos / (10000 ** (2 * i / d_model))
    pe = np.zeros((T, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


# ---------------------------------------------------------------------------
# 3. Embeddings, cosine similarity and top-k retrieval
# ---------------------------------------------------------------------------
def l2_normalize(E, axis=-1):
    return E / (np.linalg.norm(E, axis=axis, keepdims=True) + 1e-12)


def cosine_top_k(query_vec, E, k, exclude=None):
    """Return indices and scores of the k rows of E most similar to query_vec.

    Exact (brute-force) search: O(N d). Production systems swap this for an
    approximate nearest neighbour index (HNSW, IVF-PQ) — see M09.
    """
    sims = l2_normalize(E) @ l2_normalize(query_vec)
    if exclude is not None:
        sims[exclude] = -np.inf
    idx = np.argpartition(-sims, k)[:k]           # O(N) partial selection
    idx = idx[np.argsort(-sims[idx])]             # sort only the k winners
    return idx, sims[idx]


def build_toy_vocab(rng, dim=16, noise=0.25):
    """Words are generated from latent 'topic' directions plus noise, mimicking how
    trained embeddings cluster semantically related words together."""
    topics = {
        "animals": ["cat", "dog", "kitten", "puppy", "horse"],
        "finance": ["bank", "loan", "credit", "interest", "mortgage"],
        "code":    ["python", "compiler", "bug", "function", "variable"],
    }
    centers = {t: rng.normal(size=dim) for t in topics}
    words, vecs, labels = [], [], []
    for t, ws in topics.items():
        for w in ws:
            words.append(w)
            vecs.append(centers[t] + noise * rng.normal(size=dim))
            labels.append(t)
    return words, np.array(vecs), labels


# ---------------------------------------------------------------------------
# 4. Main with self-checks
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Lab 7: attention and embeddings from scratch")
    print("=" * 70)

    # --- 4a. Basic attention: a soft dictionary lookup --------------------
    print("\n[1] Attention as a soft dictionary lookup")
    K = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])      # 3 keys
    V = np.array([[10.0], [20.0], [30.0]])                    # their values
    Q = np.array([[5.0, 0.0]])                                # query aligned with key 0
    out, w = scaled_dot_product_attention(Q, K, V)
    print("  weights:", np.round(w, 3), " output:", np.round(out, 3))
    assert w[0, 0] > 0.9 and abs(out[0, 0] - 10.0) < 1.0, "query should retrieve value 0"

    # --- 4b. Rows sum to one; causal mask works ---------------------------
    print("\n[2] Invariants: rows sum to 1, causal mask hides the future")
    T, d = 6, 8
    Q, K, V = (RNG.normal(size=(T, d)) for _ in range(3))
    _, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(T))
    assert np.allclose(w.sum(axis=-1), 1.0), "each row must be a distribution"
    upper = w[np.triu_indices(T, k=1)]
    assert np.all(upper < 1e-12), "causal mask leaked future positions"
    assert np.isclose(w[0, 0], 1.0), "first token can only attend to itself"
    print("  row sums:", np.round(w.sum(axis=-1), 6))
    print("  max weight above the diagonal:", upper.max())

    # --- 4c. Why divide by sqrt(d_k)? ------------------------------------
    print("\n[3] Why scale by sqrt(d_k): softmax saturation")
    for d_k in (4, 64, 512):
        q = RNG.normal(size=(200, d_k))
        k = RNG.normal(size=(50, d_k))
        raw = q @ k.T
        p_raw = softmax(raw)
        p_scaled = softmax(raw / np.sqrt(d_k))
        print(f"  d_k={d_k:4d}  std(q.k)={raw.std():6.2f}  "
              f"mean max-weight unscaled={p_raw.max(axis=1).mean():.3f}  "
              f"scaled={p_scaled.max(axis=1).mean():.3f}")
        if d_k == 512:
            assert p_raw.max(axis=1).mean() > 0.9, "unscaled softmax should be ~one-hot"
            assert p_scaled.max(axis=1).mean() < 0.5, "scaled softmax should stay soft"
            assert abs(raw.std() / np.sqrt(d_k) - 1) < 0.1, "q.k has std ~ sqrt(d_k)"

    # --- 4d. Multi-head attention with causal mask ------------------------
    print("\n[4] Multi-head causal self-attention")
    B, T, d_model, h = 2, 7, 32, 4
    params = init_mha(d_model, h, RNG)
    X = RNG.normal(size=(B, T, d_model)) + sinusoidal_positions(T, d_model)[None]
    out, w = multi_head_attention(params, X, causal=True)
    print(f"  input {X.shape} -> output {out.shape}, weights {w.shape}")
    assert np.allclose(w.sum(-1), 1.0)
    # Changing the LAST token must not change outputs at earlier positions.
    X2 = X.copy()
    X2[:, -1, :] += 100.0 * RNG.normal(size=(B, d_model))
    out2, _ = multi_head_attention(params, X2, causal=True)
    assert np.allclose(out[:, :-1], out2[:, :-1]), "causality violated"
    assert not np.allclose(out[:, -1], out2[:, -1]), "last position should change"
    # Without the mask, earlier positions DO see the change.
    out_nc, _ = multi_head_attention(params, X, causal=False)
    out_nc2, _ = multi_head_attention(params, X2, causal=False)
    assert not np.allclose(out_nc[:, 0], out_nc2[:, 0]), "bidirectional should see future"
    print("  perturbing the last token leaves earlier outputs unchanged  OK")

    # Permutation equivariance: without positional encodings attention is a set op.
    Xp = RNG.normal(size=(1, 5, d_model))
    perm = np.array([3, 0, 4, 1, 2])
    o, _ = multi_head_attention(params, Xp)
    op, _ = multi_head_attention(params, Xp[:, perm])
    assert np.allclose(o[:, perm], op), "attention without positions is permutation-equivariant"
    print("  without positional encoding, permuting inputs permutes outputs  OK")

    # --- 4e. Embedding similarity and top-k retrieval ---------------------
    print("\n[5] Embedding similarity and cosine top-k retrieval")
    words, E, labels = build_toy_vocab(np.random.default_rng(7))
    for q in ("cat", "loan", "compiler"):
        qi = words.index(q)
        idx, sims = cosine_top_k(E[qi], E, k=3, exclude=qi)
        neigh = [f"{words[i]}({s:.2f})" for i, s in zip(idx, sims)]
        print(f"  {q:>9} -> {', '.join(neigh)}")
        assert all(labels[i] == labels[qi] for i in idx), "neighbours should share topic"

    # Precision@3 over every word as a query.
    hits = 0
    for qi in range(len(words)):
        idx, _ = cosine_top_k(E[qi], E, k=3, exclude=qi)
        hits += sum(labels[i] == labels[qi] for i in idx)
    p_at_3 = hits / (3 * len(words))
    print(f"  mean precision@3 over all queries = {p_at_3:.3f}")
    assert p_at_3 == 1.0

    # One-hot vs embedding: one-hot vectors are all equally (dis)similar.
    onehot = np.eye(len(words))
    sims_oh = l2_normalize(onehot) @ l2_normalize(onehot).T
    assert np.allclose(sims_oh - np.eye(len(words)), 0), "one-hot: all pairs orthogonal"
    print("  one-hot: every pair of distinct words has cosine 0 (no notion of similarity)")

    print("\nAll Lab 7 checks passed.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Student exercises
# ---------------------------------------------------------------------------
# 1. (★) Remove the 1/sqrt(d_k) and re-run section [3]. Compute the softmax Jacobian
#    norm for a saturated row and explain what happens to gradients.
# 2. (★) Replace -1e9 with -np.inf in the mask. When does that produce NaN? (Hint: a row
#    in which every position is masked, e.g. padding.) How do real libraries handle it?
# 3. (★★) Add a full transformer block: x + MHA(LayerNorm(x)), then x + FFN(LayerNorm(x))
#    with a GELU FFN of width 4*d_model. Assert the output shape and causality still hold.
# 4. (★★) Implement a KV cache: generate tokens one at a time, appending K and V for each
#    new token, and assert that the outputs match the full-sequence causal computation.
# 5. (★★) Implement skip-gram with negative sampling on a toy corpus and check that words
#    that co-occur end up with high cosine similarity.
# 6. (★★★) Replace brute-force top-k with a tiny IVF index: k-means the embeddings into
#    C clusters, search only the n_probe nearest clusters. Measure recall@10 vs n_probe.
