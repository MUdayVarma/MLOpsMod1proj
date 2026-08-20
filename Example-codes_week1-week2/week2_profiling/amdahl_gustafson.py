#!/usr/bin/env python3
"""amdahl_gustafson.py -- Week 2: "Amdahl's Law" & "Gustafson's Law" slides.

Reproduces the two speedup curves from the slides and prints a table so the
audience sees *why* a small serial fraction caps speedup (Amdahl) yet big
clusters still pay off on bigger problems (Gustafson / weak scaling).

    Amdahl   (fixed problem):  S_p = 1 / ((1-f) + f/p)
    Gustafson(scaled problem): S_p = p - (1-f)(p-1)

No third-party deps required. If matplotlib is installed you also get a PNG;
otherwise a text table is printed (the number is the point, not the picture).

Run:
    python3 amdahl_gustafson.py                # f defaults to 0.95
    python3 amdahl_gustafson.py --f 0.75
"""
import argparse


def amdahl(f, p):
    """Fixed-size speedup. p -> infinity gives the ceiling 1/(1-f)."""
    return 1.0 / ((1.0 - f) + f / p)


def gustafson(f, p):
    """Scaled-size speedup: the problem grows with the number of workers."""
    return p - (1.0 - f) * (p - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float, default=0.95,
                    help="parallelisable fraction (0..1), default 0.95")
    args = ap.parse_args()
    f = args.f
    workers = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print(f"Parallelisable fraction f = {f}")
    print(f"Amdahl ceiling (p -> inf): {1.0 / (1.0 - f):.1f}x  "
          f"<-- a {100*(1-f):.0f}% serial part caps you here, forever\n")
    print(f"{'workers':>8} | {'Amdahl':>10} | {'Gustafson':>10} | {'ideal':>6}")
    print("-" * 44)
    for p in workers:
        print(f"{p:>8} | {amdahl(f, p):>9.2f}x | {gustafson(f, p):>9.2f}x | {p:>5}x")

    print("\nTakeaways for the room:")
    print(" * Amdahl (strong scaling): fixed job -> serial fraction wins in the end.")
    print(" * Gustafson (weak scaling): grow the job with the machine -> stays useful.")
    print(' * This is why frontier models "trained on 1024 GPUs" weak-scale:')
    print("   more GPUs to fit a bigger model/batch, not to finish a fixed job faster.")

    # Optional plot -- purely a bonus if matplotlib happens to be present.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(1, 257))
        plt.figure(figsize=(6, 4))
        plt.plot(xs, [amdahl(f, p) for p in xs], label=f"Amdahl (f={f})")
        plt.plot(xs, [gustafson(f, p) for p in xs], label=f"Gustafson (f={f})")
        plt.plot(xs, xs, "k--", alpha=0.4, label="ideal")
        plt.xlabel("# workers"); plt.ylabel("speedup")
        plt.legend(); plt.title("Amdahl vs Gustafson"); plt.tight_layout()
        plt.savefig("amdahl_gustafson.png", dpi=120)
        print("\n[plot written to amdahl_gustafson.png]")
    except Exception:
        print("\n[matplotlib not installed -- skipping plot; the table is the point]")


if __name__ == "__main__":
    main()
