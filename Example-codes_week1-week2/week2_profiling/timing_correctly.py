#!/usr/bin/env python3
"""timing_correctly.py -- Week 2: "First Rule: Measure, Don't Guess" and
"How to Time Correctly" slides.

The slide shows a WRONG timing (single call, no warmup) vs a RIGHT one
(warmup, then average many iterations, report variance). On a GPU the extra
sin is forgetting torch.cuda.synchronize() because kernel launches are async;
we emulate the same lesson on CPU where the JIT/cache warmup and one-shot
noise cause the same mistakes.

Needs: numpy only.

Run:
    python3 timing_correctly.py
"""
import time
import statistics
import warnings
import numpy as np

# numpy 2.0 + Apple Accelerate can emit spurious FP warnings on the first
# matmul while it probes the BLAS backend. Harmless; silence for a clean demo.
warnings.filterwarnings("ignore")
np.seterr(all="ignore")


def workload():
    """A representative compute chunk (a small GEMM)."""
    A = workload.A
    B = workload.B
    return A @ B


workload.A = np.random.rand(1200, 1200)
workload.B = np.random.rand(1200, 1200)


def wrong_way():
    """One cold call. First call pays allocation + cache-warm + library init."""
    t0 = time.perf_counter()
    workload()
    return time.perf_counter() - t0


def right_way(warmup=5, iters=30):
    """Warm up, then average many iterations and report the spread."""
    for _ in range(warmup):
        workload()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        workload()
        samples.append(time.perf_counter() - t0)
    return statistics.mean(samples), statistics.pstdev(samples), min(samples)


def main():
    cold = wrong_way()
    mean, std, best = right_way()

    print("WRONG (single cold call, no warmup):")
    print(f"    {cold*1e3:8.3f} ms   <- includes one-time warmup/allocation noise\n")
    print("RIGHT (warmup, then 30 timed iters):")
    print(f"    mean {mean*1e3:8.3f} ms   std {std*1e3:.3f} ms   best {best*1e3:.3f} ms\n")
    print(f"The cold measurement is {cold/mean:.2f}x the true steady-state cost.")
    print("Report mean +/- std over many iters, never a single shot.\n")
    print("On a GPU the same slide adds ONE more rule:")
    print("    for _ in range(10): model(x)         # warmup")
    print("    torch.cuda.synchronize(); t0 = ...    # kernels are async!")
    print("    for _ in range(50): model(x)")
    print("    torch.cuda.synchronize(); dt = ...    # sync BEFORE stopping the clock")
    print("Without synchronize() you would time only the launch, not the work.")


if __name__ == "__main__":
    main()
