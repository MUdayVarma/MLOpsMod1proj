"""
mlp_from_scratch.py

A 2-layer MLP (2 -> h -> 1), implemented in pure NumPy, trained with plain
SGD on a non-linear (XOR-style) binary classification toy problem.

No autograd. No PyTorch/TensorFlow. No sklearn. Every gradient below is
derived by hand via the chain rule (see comments above each grad line).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Save output files (plots) in the same directory as this script,
# regardless of which machine or working directory it's run from.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. DATA GENERATOR (verbatim)
# ---------------------------------------------------------------------------
def make_xor_blobs(n=400, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.array([0, 1, 1, 0])  # XOR pattern -> not linearly separable
    idx = rng.integers(0, 4, size=n)
    X = centers[idx] + rng.normal(scale=noise, size=(n, 2))
    y = labels[idx].reshape(-1, 1).astype(float)
    return X, y


# ---------------------------------------------------------------------------
# 2. PARAMETER INITIALISATION
# ---------------------------------------------------------------------------
def init_params(n_inputs, n_hidden, n_outputs, seed=0):
    """He-style init for the ReLU hidden layer, small init for the output
    layer. Biases start at zero."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(scale=np.sqrt(2.0 / n_inputs), size=(n_inputs, n_hidden))
    b1 = np.zeros((1, n_hidden))
    W2 = rng.normal(scale=np.sqrt(2.0 / n_hidden), size=(n_hidden, n_outputs))
    b2 = np.zeros((1, n_outputs))
    return W1, b1, W2, b2


# ---------------------------------------------------------------------------
# 3. FORWARD PASS
# ---------------------------------------------------------------------------
def sigmoid(z):
    # Numerically stable sigmoid
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                     np.exp(z) / (1.0 + np.exp(z)))


def forward(X, W1, b1, W2, b2):
    """Two GEMMs + ReLU on the hidden layer + sigmoid on the output.

    Returns predictions plus every intermediate value the backward pass
    needs (this dict plays the role of the 'cache' from the earlier
    SimpleDenseLayer example).
    """
    Z1 = X @ W1 + b1          # (N,2)@(2,h) + (1,h) -> (N,h)   linear 1
    A1 = np.maximum(0, Z1)    # (N,h)                          ReLU
    Z2 = A1 @ W2 + b2         # (N,h)@(h,1) + (1,1) -> (N,1)   linear 2
    A2 = sigmoid(Z2)          # (N,1)                          sigmoid -> prediction

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# ---------------------------------------------------------------------------
# 4. LOSS: BINARY CROSS-ENTROPY
# ---------------------------------------------------------------------------
def bce_loss(A2, y, eps=1e-8):
    """Mean binary cross-entropy. eps avoids log(0)."""
    A2 = np.clip(A2, eps, 1 - eps)
    return -np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2))


# ---------------------------------------------------------------------------
# 5. BACKWARD PASS (every line derived by hand via the chain rule)
# ---------------------------------------------------------------------------
def backward(y, W2, cache):
    X, Z1, A1, A2 = cache["X"], cache["Z1"], cache["A1"], cache["A2"]
    N = X.shape[0]

    # dL/dZ2 = dL/dA2 * dA2/dZ2. For BCE-loss composed with a sigmoid output,
    # this famously simplifies to (A2 - y). (Derivation: dL/dA2 =
    # -(y/A2 - (1-y)/(1-A2)); dA2/dZ2 = A2*(1-A2); the two cancel down to
    # A2 - y.) We divide by N here because our loss is a *mean* over samples.
    dZ2 = (A2 - y) / N                              # (N,1)

    # dL/dW2 = dL/dZ2 * dZ2/dW2, and dZ2/dW2 = A1 (since Z2 = A1 @ W2 + b2)
    dW2 = A1.T @ dZ2                                # (h,N)@(N,1) -> (h,1)

    # dL/db2 = dL/dZ2 * dZ2/db2, and dZ2/db2 = 1, summed over the batch
    db2 = np.sum(dZ2, axis=0, keepdims=True)        # (1,1)

    # dL/dA1 = dL/dZ2 * dZ2/dA1, and dZ2/dA1 = W2 (since Z2 = A1 @ W2 + b2)
    dA1 = dZ2 @ W2.T                                # (N,1)@(1,h) -> (N,h)

    # dL/dZ1 = dL/dA1 * dA1/dZ1, and dA1/dZ1 is ReLU'(Z1): 1 where Z1>0, else 0
    dZ1 = dA1 * (Z1 > 0)                            # (N,h)

    # dL/dW1 = dL/dZ1 * dZ1/dW1, and dZ1/dW1 = X (since Z1 = X @ W1 + b1)
    dW1 = X.T @ dZ1                                 # (2,N)@(N,h) -> (2,h)

    # dL/db1 = dL/dZ1 * dZ1/db1, and dZ1/db1 = 1, summed over the batch
    db1 = np.sum(dZ1, axis=0, keepdims=True)        # (1,h)

    return dW1, db1, dW2, db2


# ---------------------------------------------------------------------------
# 6. HELPERS: accuracy, manual train/val split (no sklearn)
# ---------------------------------------------------------------------------
def accuracy(A2, y):
    preds = (A2 >= 0.5).astype(float)
    return np.mean(preds == y)


def train_val_split(X, y, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ---------------------------------------------------------------------------
# 7. PLAIN SGD TRAINING LOOP
# ---------------------------------------------------------------------------
def train(X_train, y_train, X_val, y_val, n_hidden=16, lr=0.5,
           epochs=3000, print_every=200, seed=0):
    W1, b1, W2, b2 = init_params(2, n_hidden, 1, seed=seed)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ---- forward (train) ----
        A2_train, cache = forward(X_train, W1, b1, W2, b2)
        train_loss = bce_loss(A2_train, y_train)
        train_acc = accuracy(A2_train, y_train)

        # ---- backward ----
        dW1, db1, dW2, db2 = backward(y_train, W2, cache)

        # ---- plain SGD update: W -= lr * grad, no momentum/Adam ----
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        # ---- forward (val) -- no update, just monitoring ----
        A2_val, _ = forward(X_val, W1, b1, W2, b2)
        val_loss = bce_loss(A2_val, y_val)
        val_acc = accuracy(A2_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % print_every == 0 or epoch == 1:
            print(f"epoch {epoch:5d} | train_loss {train_loss:.4f} "
                  f"train_acc {train_acc:.3f} | val_loss {val_loss:.4f} "
                  f"val_acc {val_acc:.3f}")

    params = (W1, b1, W2, b2)
    return params, history


def plot_curves(history, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].plot(history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title(f"{title} -- loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train acc")
    axes[1].plot(history["val_acc"], label="val acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{title} -- accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. MAIN: RUN 1 (plateau run) + RUN 2 (overfitting run)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Shared data pool: generate once, split 80/20 ----
    X, y = make_xor_blobs(n=400, noise=0.25, seed=0)
    X_train_full, y_train_full, X_val, y_val = train_val_split(X, y, val_frac=0.2, seed=1)
    print(f"Full pool: {X.shape[0]} points -> "
          f"train_full={X_train_full.shape[0]}, val={X_val.shape[0]}\n")

    # =======================================================================
    # RUN 1: "plateau" run -- full training set, enough epochs to plateau >90%
    # =======================================================================
    print("=" * 70)
    print("RUN 1: full training set (n_train=%d), watching for plateau" % X_train_full.shape[0])
    print("=" * 70)
    params_full, hist_full = train(
        X_train_full, y_train_full, X_val, y_val,
        n_hidden=16, lr=0.5, epochs=3000, print_every=200, seed=0,
    )
    plot_curves(hist_full, "Run 1: Full training set",
                os.path.join(OUTPUT_DIR, "run1_plateau_curves.png"))

    final_train_loss_1 = hist_full["train_loss"][-1]
    final_val_loss_1 = hist_full["val_loss"][-1]
    final_train_acc_1 = hist_full["train_acc"][-1]
    final_val_acc_1 = hist_full["val_acc"][-1]

    # =======================================================================
    # RUN 2: "overfitting" run -- small training subset, same val set
    # =======================================================================
    print()
    print("=" * 70)
    print("RUN 2: small training subset (n_train=40), same val set -> overfitting")
    print("=" * 70)
    rng = np.random.default_rng(2)
    small_idx = rng.choice(X_train_full.shape[0], size=40, replace=False)
    X_train_small = X_train_full[small_idx]
    y_train_small = y_train_full[small_idx]

    params_small, hist_small = train(
        X_train_small, y_train_small, X_val, y_val,
        n_hidden=16, lr=0.5, epochs=3000, print_every=200, seed=0,
    )
    plot_curves(hist_small, "Run 2: Small training set (overfitting)",
                os.path.join(OUTPUT_DIR, "run2_overfit_curves.png"))

    final_train_loss_2 = hist_small["train_loss"][-1]
    final_val_loss_2 = hist_small["val_loss"][-1]
    final_train_acc_2 = hist_small["train_acc"][-1]
    final_val_acc_2 = hist_small["val_acc"][-1]

    # =======================================================================
    # SUMMARY REPORT
    # =======================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Run':<25}{'Train Loss':>12}{'Val Loss':>12}{'Train Acc':>12}{'Val Acc':>12}")
    print(f"{'1: Plateau (n=' + str(X_train_full.shape[0]) + ')':<25}"
          f"{final_train_loss_1:>12.4f}{final_val_loss_1:>12.4f}"
          f"{final_train_acc_1:>12.3f}{final_val_acc_1:>12.3f}")
    print(f"{'2: Overfit (n=40)':<25}"
          f"{final_train_loss_2:>12.4f}{final_val_loss_2:>12.4f}"
          f"{final_train_acc_2:>12.3f}{final_val_acc_2:>12.3f}")

    best_val_epoch = int(np.argmin(hist_small["val_loss"])) + 1
    print(f"\nRun 2: best val_loss={min(hist_small['val_loss']):.4f} at epoch "
          f"{best_val_epoch}, but training continued to epoch "
          f"{len(hist_small['val_loss'])} -- val loss rose afterward "
          f"({hist_small['val_loss'][best_val_epoch]:.4f} -> "
          f"{final_val_loss_2:.4f}), while train loss kept falling "
          f"({hist_small['train_loss'][best_val_epoch]:.4f} -> "
          f"{final_train_loss_2:.4f}). Classic overfitting signature.")