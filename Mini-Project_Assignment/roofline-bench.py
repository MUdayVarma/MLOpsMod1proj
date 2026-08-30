"""
roofline_bench.py

Roofline-model micro-benchmark: SAXPY (y = a*x + y) and GEMM (C = A @ B),
timed with proper warmup + averaged repeats, then classified as
memory-bound or compute-bound using arithmetic intensity (AI = FLOPs/byte).

Methodology: Measure -> Classify -> Change one thing -> Re-measure.
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Assumed machine "ridge point" (peak GFLOP/s / peak GB/s) used purely to
# turn AI into a memory-bound / compute-bound verdict. This is a rough,
# documented assumption for a modern multi-core CPU running BLAS-backed
# NumPy (~200 GFLOP/s double-precision peak, ~20 GB/s sustained DRAM
# bandwidth) -- NOT measured on this specific machine.
# ---------------------------------------------------------------------------
ASSUMED_PEAK_GFLOPS = 200.0
ASSUMED_PEAK_GBPS = 20.0
RIDGE_POINT_AI = ASSUMED_PEAK_GFLOPS / ASSUMED_PEAK_GBPS  # FLOPs/byte


def classify(ai):
    if ai >= RIDGE_POINT_AI:
        return "compute-bound"
    return "memory-bound"


# ---------------------------------------------------------------------------
# Timing helper: warmup (untimed) then average over multiple timed reps.
# "Measure, don't guess" -- never time a single cold call.
# ---------------------------------------------------------------------------
def time_kernel(fn, warmup=5, iters=15, inner_reps=1):
    """warmup: untimed calls to reach steady state (avoids first-touch page
    faults / allocator warmup skewing the result).
    iters: number of independently-timed samples to average.
    inner_reps: for kernels so fast that a single call is near clock
    resolution, run `inner_reps` calls inside one timed block and divide --
    this is still 'average over timed iterations', just batched so the
    per-call signal isn't swamped by timer granularity noise.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(inner_reps):
            fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) / inner_reps)
    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# SAXPY: y = a*x + y
# ---------------------------------------------------------------------------
def bench_saxpy(n, warmup=5, iters=15, dtype=np.float64):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(dtype)
    y = rng.standard_normal(n).astype(dtype)
    a = 2.5

    def run():
        # in-place update so we're actually doing the SAXPY, not building
        # a new array each call
        y[:] = a * x + y

    mean_s, std_s = time_kernel(run, warmup=warmup, iters=iters)

    flops = 2 * n                 # 1 multiply + 1 add per element
    bytes_moved = 24 * n          # read x (8B) + read y (8B) + write y (8B), float64
    ai = flops / bytes_moved
    gflops = (flops / mean_s) / 1e9

    return {
        "kernel": "SAXPY", "n": n, "mean_s": mean_s, "std_s": std_s,
        "flops": flops, "bytes": bytes_moved, "ai": ai, "gflops": gflops,
    }


# ---------------------------------------------------------------------------
# GEMM: C = A @ B  (N x N x N)
# ---------------------------------------------------------------------------
def bench_gemm(n, warmup=5, iters=10, dtype=np.float64, inner_reps=1):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(dtype)
    B = rng.standard_normal((n, n)).astype(dtype)

    def run():
        C = A @ B
        return C

    mean_s, std_s = time_kernel(run, warmup=warmup, iters=iters, inner_reps=inner_reps)

    flops = 2 * n ** 3                 # 2N^3 FLOPs for an N x N x N matmul
    bytes_moved = 3 * n ** 2 * 8       # read A, read B, write C (float64, naive count)
    ai = flops / bytes_moved
    gflops = (flops / mean_s) / 1e9

    return {
        "kernel": "GEMM", "n": n, "mean_s": mean_s, "std_s": std_s,
        "flops": flops, "bytes": bytes_moved, "ai": ai, "gflops": gflops,
    }


def report(result):
    print(f"{result['kernel']} (N={result['n']}):")
    print(f"  time/iter     : {result['mean_s']*1e3:.3f} ms  (std {result['std_s']*1e3:.3f} ms)")
    print(f"  FLOPs         : {result['flops']:.3e}")
    print(f"  bytes moved   : {result['bytes']:.3e}")
    print(f"  AI (FLOP/byte): {result['ai']:.4f}")
    print(f"  achieved      : {result['gflops']:.2f} GFLOP/s")
    print(f"  classification: {classify(result['ai'])} "
          f"(ridge point assumed at {RIDGE_POINT_AI:.1f} FLOP/byte)")
    print()


# ---------------------------------------------------------------------------
# MEASURE -> CLASSIFY -> CHANGE ONE THING -> RE-MEASURE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SAXPY_N = 20_000_000     # kept fixed across both runs (not the variable under test)
    GEMM_N_BASELINE = 16     # small GEMM -> baseline (small enough that call
                              # overhead is not yet fully amortized by BLAS)
    GEMM_N_LARGER = 512      # the "one thing" we change

    print("=" * 70)
    print("STEP 1: MEASURE (baseline)")
    print("=" * 70)
    saxpy_baseline = bench_saxpy(SAXPY_N)
    report(saxpy_baseline)
    gemm_baseline = bench_gemm(GEMM_N_BASELINE, inner_reps=200)
    report(gemm_baseline)

    print("=" * 70)
    print("STEP 2: CLASSIFY")
    print("=" * 70)
    print(f"SAXPY is {classify(saxpy_baseline['ai'])}: each element requires only "
          f"2 FLOPs but 24 bytes of DRAM traffic (AI={saxpy_baseline['ai']:.3f}), "
          f"so the CPU stalls waiting on memory bandwidth long before it runs out "
          f"of arithmetic throughput.")
    print(f"GEMM (N={GEMM_N_BASELINE}) is {classify(gemm_baseline['ai'])}: AI "
          f"grows linearly with N (AI=N/12), so data gets reused O(N) times "
          f"from cache per byte fetched from DRAM, shifting the bottleneck "
          f"toward the compute units as N grows.")
    print()

    print("=" * 70)
    print("STEP 3: CHANGE ONE THING -> increase GEMM N "
          f"({GEMM_N_BASELINE} -> {GEMM_N_LARGER}); SAXPY size held fixed as control")
    print("=" * 70)
    print()

    print("=" * 70)
    print("STEP 4: RE-MEASURE")
    print("=" * 70)
    gemm_larger = bench_gemm(GEMM_N_LARGER, warmup=3, iters=5)
    report(gemm_larger)

    print("=" * 70)
    print("DELTA vs THEORY")
    print("=" * 70)
    ai_up = gemm_larger["ai"] > gemm_baseline["ai"]
    gflops_up = gemm_larger["gflops"] > gemm_baseline["gflops"]
    print(f"AI:      {gemm_baseline['ai']:.2f} -> {gemm_larger['ai']:.2f} "
          f"FLOP/byte ({'up' if ai_up else 'down'})")
    print(f"GFLOP/s: {gemm_baseline['gflops']:.2f} -> {gemm_larger['gflops']:.2f} "
          f"({'up' if gflops_up else 'down'})")
    matched = ai_up and gflops_up
    print(f"\nTheory predicted both AI and achieved GFLOP/s would rise as N grows "
          f"(roofline: higher AI moves the kernel rightward, toward/along the "
          f"compute-bound ceiling). Observed result "
          f"{'MATCHES' if matched else 'DOES NOT MATCH'} theory.")

    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Kernel':<18}{'N':>8}{'AI':>10}{'GFLOP/s':>12}{'Class':>16}")
    for r in (saxpy_baseline, gemm_baseline, gemm_larger):
        label = r["kernel"] if r is not gemm_larger else "GEMM (bigger N)"
        print(f"{label:<18}{r['n']:>8}{r['ai']:>10.3f}{r['gflops']:>12.2f}"
              f"{classify(r['ai']):>16}")