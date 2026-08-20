#!/usr/bin/env python3
"""roofline.py -- Week 2: "Arithmetic Intensity", "The Roofline Model",
"Reading a Roofline" slides.

We take two textbook ML kernels and *measure* them, then place them on a
roofline:

  * SAXPY  (y = a*x + y)  -- 2 FLOPs per element, ~24 bytes moved per element.
                            LOW arithmetic intensity => MEMORY-bound.
  * GEMM   (C = A @ B)    -- O(N^3) FLOPs, O(N^2) bytes.
                            HIGH arithmetic intensity => COMPUTE-bound.

The slide definition:  AI = FLOPs performed / Bytes moved from memory.
Low AI sits on the slanted (bandwidth) roof; high AI reaches the flat
(compute) roof. This demo shows the two kernels landing in different regimes
on the *same* machine -- the core diagnostic skill of the week.

Needs: numpy. matplotlib optional (text summary always prints).

Run:
    python3 roofline.py
    python3 roofline.py --n 6000 --vec 40000000
"""
import argparse
import time
import warnings
import numpy as np

# Silence spurious numpy-2.0/Accelerate first-call FP warnings (macOS).
warnings.filterwarnings("ignore")
np.seterr(all="ignore")


def timeit(fn, warmup=3, iters=10):
    """Warm up (caches, allocation), then time -- the Week 2 timing recipe."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def bench_saxpy(vec):
    """Memory-bound: read x, read y, write y => ~3 arrays * 8 bytes = 24 B/elem."""
    x = np.random.rand(vec)
    y = np.random.rand(vec)
    a = 2.0
    dt = timeit(lambda: np.add(y, a * x, out=y))
    flops = 2.0 * vec              # one multiply + one add per element
    bytes_moved = 24.0 * vec       # x read, y read, y write (float64)
    return dt, flops, bytes_moved


def bench_gemm(n):
    """Compute-bound: 2*N^3 FLOPs, ~3*N^2*8 bytes touched => AI grows with N."""
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    dt = timeit(lambda: A @ B, warmup=2, iters=3)
    flops = 2.0 * n ** 3
    bytes_moved = 3.0 * n * n * 8.0
    return dt, flops, bytes_moved


def report(name, dt, flops, bytes_moved):
    gflops = flops / dt / 1e9
    ai = flops / bytes_moved
    verdict = "MEMORY-bound (low AI)" if ai < 5 else "COMPUTE-bound (high AI)"
    print(f"\n{name}")
    print(f"  time/iter        : {dt*1e3:8.3f} ms")
    print(f"  achieved GFLOP/s : {gflops:8.1f}")
    print(f"  arithmetic inten.: {ai:8.3f} FLOP/byte")
    print(f"  classification   : {verdict}")
    return ai, gflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="GEMM matrix dim")
    ap.add_argument("--vec", type=int, default=20_000_000, help="SAXPY length")
    args = ap.parse_args()

    print("Measuring two ML kernels on THIS machine (warmup + averaged)...")
    ai_s, gf_s = report("SAXPY  y=a*x+y", *bench_saxpy(args.vec))
    ai_g, gf_g = report("GEMM   C=A@B", *bench_gemm(args.n))

    print("\n" + "=" * 60)
    print("Roofline reading:")
    print(f"  SAXPY sits far LEFT (AI={ai_s:.2f}) on the slanted memory roof.")
    print(f"  GEMM  sits RIGHT   (AI={ai_g:.1f}) up near the flat compute roof.")
    print("  => To speed SAXPY: raise AI (fuse ops, lower precision).")
    print("  => To speed GEMM : better math / Tensor Cores; it is already compute-bound.")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Rough machine roofs inferred from the GEMM run (illustrative, not spec).
        peak_gflops = max(gf_g * 1.1, gf_s * 1.1)
        peak_bw_gbs = max((gf_s / ai_s), 1.0)  # GFLOP/s / (FLOP/B) = GB/s
        ai_axis = np.logspace(-2, 3, 200)
        roof = np.minimum(peak_bw_gbs * ai_axis, peak_gflops)
        plt.figure(figsize=(6, 4))
        plt.loglog(ai_axis, roof, "k-", label="roofline")
        plt.loglog([ai_s], [gf_s], "o", label="SAXPY (memory-bound)")
        plt.loglog([ai_g], [gf_g], "s", label="GEMM (compute-bound)")
        plt.xlabel("Arithmetic intensity (FLOP/byte)")
        plt.ylabel("GFLOP/s"); plt.legend(); plt.title("Measured roofline")
        plt.tight_layout(); plt.savefig("roofline.png", dpi=120)
        print("\n[plot written to roofline.png]")
    except Exception:
        print("\n[matplotlib not installed -- text roofline above is sufficient]")


if __name__ == "__main__":
    main()
