# Week 2 — Profiling ML Workloads & Parallel Computing Basics

Live demos for the Week 2 deck (`slides/week2/week2_hpc_fundamentals.tex`).
Everything here runs on a laptop; the lessons transfer directly to the GPU
tools named on the slides (nvidia-smi, Nsight, PyTorch profiler).

## Prerequisites

- A C compiler (`clang` on macOS, `gcc` on Linux) — for the cache demo only.
- Python 3.8+ with **numpy**. `pip install numpy`.
- `matplotlib` is **optional** (every script prints a text table without it).

## Demos

| File | Slide it backs | One-line objective |
| --- | --- | --- |
| `cache_locality.c` | "Why Data Layout Matters: Row vs Column Access" | Same FLOPs, different memory layout → measurable slowdown. |
| `amdahl_gustafson.py` | "Amdahl's Law", "Gustafson's Law", "Strong vs Weak Scaling" | Why a serial fraction caps speedup, and why big clusters still pay off. |
| `roofline.py` | "Arithmetic Intensity", "The Roofline Model", "Reading a Roofline" | Measure SAXPY vs GEMM and classify each as memory- or compute-bound. |
| `timing_correctly.py` | "Measure, Don't Guess", "How to Time Correctly" | Warmup + averaging + (on GPU) synchronize — the right way to benchmark. |
| `profile_pipeline.py` | "Diagnosing a Slow Training Run", "Common ML Bottlenecks" | Use a profiler to find an I/O-bound loop, then overlap the loader to fix it. |

## Run them

```bash
make run                        # compile + run the cache-locality demo
./cache_locality 8192           # bigger matrix => bigger cache-miss penalty

python3 amdahl_gustafson.py            # f=0.95 (5% serial => 20x ceiling)
python3 amdahl_gustafson.py --f 0.75   # heavier serial fraction

python3 roofline.py                    # SAXPY (memory-bound) vs GEMM (compute-bound)
python3 timing_correctly.py            # wrong vs right benchmarking
python3 profile_pipeline.py            # cProfile finds the bottleneck, then fix it
```
