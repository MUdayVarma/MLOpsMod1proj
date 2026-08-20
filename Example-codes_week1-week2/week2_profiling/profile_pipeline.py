#!/usr/bin/env python3
"""profile_pipeline.py -- Week 2: "Case Study: Diagnosing a Slow Training Run"
and "Common ML Bottlenecks" slides.

A mock training loop that is deliberately I/O-bound: a slow "data loader"
starves the "GPU" (compute stub), so utilisation is low and bursty. We use
Python's own cProfile to SHOW that the time is in the loader -- exactly the
slide's "profiler shows 55% of step in the data loader => confirmed I/O-bound".
Then we apply the slide's fix (prefetch / overlap the loader with compute using
a background thread) and re-measure the speedup.

No third-party deps: this is pure standard library so it runs anywhere.

Run:
    python3 profile_pipeline.py                 # runs slow, profiles, then fixed
    python3 profile_pipeline.py --profile-only  # just the cProfile breakdown
"""
import argparse
import cProfile
import pstats
import io
import time
import threading
import queue


STEPS = 8


def load_batch():
    """Stand-in for a slow data loader (decode/augment/read from disk)."""
    time.sleep(0.06)          # the I/O / preprocessing cost
    return sum(i * i for i in range(2000))   # a little CPU work too


def compute_step(_batch):
    """Stand-in for the GPU forward+backward (the work we WANT dominating)."""
    time.sleep(0.04)          # the useful compute
    return 1


# ---------- BASELINE: loader and compute run serially (GPU starves) ----------
def train_serial(steps=STEPS):
    for _ in range(steps):
        batch = load_batch()     # GPU idle while we wait here
        compute_step(batch)


# ---------- FIX: background prefetch overlaps loading with compute ----------
def train_prefetched(steps=STEPS):
    q = queue.Queue(maxsize=2)

    def producer():
        for _ in range(steps):
            q.put(load_batch())   # load NEXT batch while GPU computes current
        q.put(None)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    while True:
        batch = q.get()
        if batch is None:
            break
        compute_step(batch)
    t.join()


def wall(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-only", action="store_true")
    args = ap.parse_args()

    print("Profiling the SERIAL pipeline with cProfile (where does time go?)...\n")
    pr = cProfile.Profile()
    pr.enable()
    train_serial()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(6)
    # Trim cProfile's header noise for a clean live view.
    for line in s.getvalue().splitlines():
        if "load_batch" in line or "compute_step" in line or "cumtime" in line:
            print("   " + line.strip())
    print("\n=> load_batch dominates cumulative time: this run is I/O-bound.")

    if args.profile_only:
        return

    print("\nApplying the slide's fix: prefetch/overlap the loader with compute.\n")
    t_serial = wall(train_serial)
    t_fixed = wall(train_prefetched)
    print(f"  serial (loader blocks compute) : {t_serial:6.3f} s")
    print(f"  prefetched (overlapped)        : {t_fixed:6.3f} s")
    print(f"  speedup                        : {t_serial / t_fixed:6.2f}x")
    print("\nSame lesson as the slide: low, bursty GPU util + spare memory =>")
    print("suspect data starvation => overlap the loader => reclaim the idle time.")


if __name__ == "__main__":
    main()
