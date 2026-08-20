#!/usr/bin/env python3
"""
Week 1 live demo -- a neural network in ~40 lines of pure NumPy.

WHY THIS DEMO EXISTS
--------------------
The Week 1 slides claim that "a neural network is, computationally, a sequence
of matrix multiplies (GEMMs), a nonlinearity, a loss, and gradient descent."
This script *proves* it: no PyTorch, no autograd -- we hand-code the forward
pass, the backward pass (chain rule), and the SGD update, and watch the loss
fall live in the terminal.

It also reproduces the overfitting curve from Segment 2: run with --overfit to
see training loss keep dropping while validation loss turns back up.

RUN IT LIVE
-----------
    python3 train_from_scratch.py              # clean training run
    python3 train_from_scratch.py --overfit    # train vs val divergence
    python3 train_from_scratch.py --epochs 4000 --lr 0.5

Needs only NumPy. If matplotlib is present it also saves loss_curve.png;
otherwise it draws an ASCII loss curve so the demo always works on a bare
laptop or a cluster login node.
"""
import argparse
import numpy as np

# ----------------------------------------------------------------------
# 1. A toy dataset: two interleaving moons (a classic non-linear problem).
#    Hand-rolled so we need no scikit-learn.
# ----------------------------------------------------------------------
def make_moons(n=400, noise=0.20, seed=0):
    rng = np.random.default_rng(seed)
    n_out = n // 2
    n_in = n - n_out
    # outer moon
    to = np.linspace(0, np.pi, n_out)
    xo = np.c_[np.cos(to), np.sin(to)]
    # inner moon, shifted
    ti = np.linspace(0, np.pi, n_in)
    xi = np.c_[1 - np.cos(ti), 1 - np.sin(ti) - 0.5]
    X = np.vstack([xo, xi]).astype(np.float64)
    y = np.hstack([np.zeros(n_out), np.ones(n_in)]).reshape(-1, 1)
    X += rng.normal(scale=noise, size=X.shape)
    # standardise -- normalisation matters (Segment 2 of the slides)
    X = (X - X.mean(0)) / X.std(0)
    idx = rng.permutation(n)
    return X[idx], y[idx]


# ----------------------------------------------------------------------
# 2. The maths from the slides, spelled out.
# ----------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def bce(y_pred, y_true, eps=1e-9):
    """Binary cross-entropy -- the classification loss from the slides."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def accuracy(y_pred, y_true):
    return float(np.mean((y_pred > 0.5) == (y_true > 0.5)))


class TinyMLP:
    """A 2 -> H -> 1 network. Every line maps to a slide."""

    def __init__(self, in_dim=2, hidden=16, seed=1):
        rng = np.random.default_rng(seed)
        # He-ish initialisation
        self.W1 = rng.normal(scale=np.sqrt(2 / in_dim), size=(in_dim, hidden))
        self.b1 = np.zeros((1, hidden))
        self.W2 = rng.normal(scale=np.sqrt(2 / hidden), size=(hidden, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        # forward pass = two GEMMs + a nonlinearity (slide: "GEMM is the kernel")
        #
        # np.errstate: NumPy 2.x's SIMD matmul kernel can raise spurious
        # "divide by zero / overflow / invalid value in matmul" RuntimeWarnings
        # even when every input and output is finite -- it reads unused padding
        # lanes of the vector registers, which flips CPU FP-exception flags
        # without affecting the (bitwise-correct) result. Silence those flags
        # here; genuine blow-ups are still caught by the loss curve.
        self.X = X
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            self.z1 = X @ self.W1 + self.b1
            self.a1 = relu(self.z1)
            self.z2 = self.a1 @ self.W2 + self.b2
        self.y = sigmoid(self.z2)
        return self.y

    def backward(self, y_true):
        # backward pass = the chain rule, applied layer by layer.
        n = y_true.shape[0]
        # d(BCE + sigmoid)/dz2 conveniently collapses to (y_pred - y_true)
        dz2 = (self.y - y_true) / n
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            gW2 = self.a1.T @ dz2       # spurious-FPE guard (see forward())
            gb2 = dz2.sum(0, keepdims=True)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (self.z1 > 0)            # ReLU derivative
            gW1 = self.X.T @ dz1
        gb1 = dz1.sum(0, keepdims=True)
        return gW1, gb1, gW2, gb2

    def step(self, grads, lr):
        gW1, gb1, gW2, gb2 = grads
        # the SGD update rule from the slides: W <- W - lr * grad
        self.W1 -= lr * gW1
        self.b1 -= lr * gb1
        self.W2 -= lr * gW2
        self.b2 -= lr * gb2


# ----------------------------------------------------------------------
# 3. An ASCII loss curve so the demo always produces a picture.
# ----------------------------------------------------------------------
def ascii_curve(series, labels, width=56, height=12):
    allvals = np.concatenate(series)
    lo, hi = float(allvals.min()), float(allvals.max())
    hi = hi if hi > lo else lo + 1e-9
    marks = "*+o#"
    grid = [[" "] * width for _ in range(height)]
    for s_i, s in enumerate(series):
        xs = np.linspace(0, width - 1, len(s)).astype(int)
        for xi, v in zip(xs, s):
            row = int((1 - (v - lo) / (hi - lo)) * (height - 1))
            grid[row][xi] = marks[s_i % len(marks)]
    print()
    for r, line in enumerate(grid):
        axis = f"{hi + (lo-hi)*r/(height-1):5.2f} |"
        print(axis + "".join(line))
    print("      +" + "-" * width)
    print("       epochs " + "-> " * (width // 20))
    print("       legend: " + "  ".join(f"{marks[i]}={labels[i]}"
                                         for i in range(len(labels))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--overfit", action="store_true",
                    help="tiny noisy training set -> watch val loss diverge")
    args = ap.parse_args()

    if args.overfit:
        # small + noisy training set is the recipe for overfitting
        Xtr, ytr = make_moons(n=40, noise=0.45, seed=0)
        Xva, yva = make_moons(n=400, noise=0.45, seed=99)
        args.hidden = max(args.hidden, 64)   # over-capacity model
        print(">> OVERFIT MODE: 40 noisy training points, 64 hidden units\n")
    else:
        Xtr, ytr = make_moons(n=400, noise=0.20, seed=0)
        Xva, yva = make_moons(n=400, noise=0.20, seed=42)

    net = TinyMLP(in_dim=2, hidden=args.hidden)
    tr_hist, va_hist = [], []

    print(f"{'epoch':>6} | {'train_loss':>10} | {'val_loss':>9} | "
          f"{'train_acc':>9} | {'val_acc':>7}")
    print("-" * 56)
    for e in range(args.epochs + 1):
        yp = net.forward(Xtr)
        grads = net.backward(ytr)
        net.step(grads, args.lr)

        if e % max(1, args.epochs // 20) == 0:
            tr_loss = bce(net.forward(Xtr), ytr)
            va_loss = bce(net.forward(Xva), yva)
            tr_acc = accuracy(net.forward(Xtr), ytr)
            va_acc = accuracy(net.forward(Xva), yva)
            tr_hist.append(tr_loss)
            va_hist.append(va_loss)
            print(f"{e:>6} | {tr_loss:>10.4f} | {va_loss:>9.4f} | "
                  f"{tr_acc:>9.3f} | {va_acc:>7.3f}")

    print("\nFinal decision: the network learned a *curved* boundary that no")
    print("single linear layer could -- because of the ReLU nonlinearity.\n")

    # picture: matplotlib if available, else ASCII (always works)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = np.linspace(0, args.epochs, len(tr_hist))
        plt.plot(xs, tr_hist, label="train loss")
        plt.plot(xs, va_hist, label="val loss")
        plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.legend()
        plt.title("Overfitting" if args.overfit else "Training a NumPy MLP")
        plt.tight_layout(); plt.savefig("loss_curve.png", dpi=120)
        print("Saved loss_curve.png")
    except Exception:
        ascii_curve([np.array(tr_hist), np.array(va_hist)],
                    ["train", "val"])
        if args.overfit:
            print("\n>> Note how 'val' bottoms out and turns UP while 'train'")
            print("   keeps falling -- that gap IS overfitting (Segment 2).")


if __name__ == "__main__":
    main()
