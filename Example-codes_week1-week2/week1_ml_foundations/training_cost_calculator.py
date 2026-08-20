#!/usr/bin/env python3
"""
Week 1 live demo -- "why one machine is not enough", quantified.

WHY THIS DEMO EXISTS
--------------------
Segment 6 of the Week 1 slides gives two back-of-envelope laws:

    compute (FLOPs) ~= 6 * P * T          (P = params, T = training tokens)
    memory  (bytes) ~= 16 * P             (FP16 weights + grads + Adam moments)

This script turns those laws into a table for real model sizes and shows the
wall-clock time on 1 GPU versus 1,024 GPUs. The punchline the instructor lands:
a run that takes YEARS on one GPU takes DAYS on a thousand -- *if* you can make
a thousand GPUs cooperate. That "if" is the whole point of Modules 1-3.

RUN IT LIVE
-----------
    python3 training_cost_calculator.py
    python3 training_cost_calculator.py --params 7e9 --tokens 2e12
    python3 training_cost_calculator.py --gpu-tflops 312 --util 0.4 --ngpu 2048

Pure standard library -- runs anywhere Python does.
"""
import argparse

# Reference accelerator: ~an A100-class GPU doing dense FP16/BF16 math.
# 312 TFLOP/s peak; realistic sustained utilisation for large-model training
# is well under peak (30-50%), which is itself a core lesson of this course.
DEFAULT_GPU_TFLOPS = 312.0
DEFAULT_UTIL = 0.40
DEFAULT_GPU_MEM_GB = 80.0

# (name, params, training tokens) -- rough public figures, illustrative only.
CATALOG = [
    ("Small MLP (this class)", 5e4, 1e6),
    ("BERT-base",              1.1e8, 2.5e11),
    ("GPT-2 (1.5B)",           1.5e9, 4e10),
    ("LLaMA-7B",               7e9, 1e12),
    ("LLaMA-70B",              7e10, 2e12),
    ("GPT-3 (175B)",           1.75e11, 3e11),
    ("Frontier (~500B)",       5e11, 1.5e13),
]


def human_flops(f):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if f < 1000:
            return f"{f:6.1f}{unit}"
        f /= 1000.0
    return f"{f:.1f}Y"


def human_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    if seconds < 86400:
        return f"{seconds/3600:.1f} h"
    if seconds < 86400 * 365:
        return f"{seconds/86400:.1f} days"
    return f"{seconds/86400/365:.2f} years"


def analyse(name, P, T, gpu_flops, util, ngpu, gpu_mem_bytes):
    flops = 6 * P * T
    mem_bytes = 16 * P
    eff = gpu_flops * util
    t1 = flops / eff                      # seconds on 1 GPU
    tN = t1 / ngpu                        # ideal linear scaling on N GPUs
    gpus_for_mem = max(1, -(-int(mem_bytes) // int(gpu_mem_bytes)))  # ceil
    return {
        "name": name, "flops": flops, "mem_gb": mem_bytes / 1e9,
        "t1": t1, "tN": tN, "gpus_for_mem": gpus_for_mem,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, help="override: parameter count P")
    ap.add_argument("--tokens", type=float, help="override: training tokens T")
    ap.add_argument("--gpu-tflops", type=float, default=DEFAULT_GPU_TFLOPS)
    ap.add_argument("--util", type=float, default=DEFAULT_UTIL,
                    help="sustained fraction of peak (0-1)")
    ap.add_argument("--ngpu", type=int, default=1024)
    ap.add_argument("--gpu-mem-gb", type=float, default=DEFAULT_GPU_MEM_GB)
    args = ap.parse_args()

    gpu_flops = args.gpu_tflops * 1e12
    gpu_mem_bytes = args.gpu_mem_gb * 1e9

    print("=" * 78)
    print("  TRAINING COST CALCULATOR   (compute = 6*P*T,  memory = 16*P bytes)")
    print(f"  GPU: {args.gpu_tflops:.0f} TFLOP/s peak x {args.util:.0%} util "
          f"= {args.gpu_tflops*args.util:.0f} TFLOP/s sustained, "
          f"{args.gpu_mem_gb:.0f} GB each")
    print(f"  Cluster size for scaling column: {args.ngpu} GPUs")
    print("=" * 78)

    if args.params and args.tokens:
        models = [("custom", args.params, args.tokens)]
    else:
        models = CATALOG

    header = (f"{'model':22} {'FLOPs':>9} {'mem':>9} "
              f"{'GPUs(mem)':>9} {'1 GPU':>10} {args.ngpu:>6} GPUs")
    print(header)
    print("-" * 78)
    for name, P, T in models:
        r = analyse(name, P, T, gpu_flops, args.util, args.ngpu, gpu_mem_bytes)
        print(f"{r['name']:22} {human_flops(r['flops']):>9} "
              f"{r['mem_gb']:>7.0f}GB {r['gpus_for_mem']:>9} "
              f"{human_time(r['t1']):>10} {human_time(r['tN']):>11}")

    print("-" * 78)
    print("Read across a large row:")
    print("  * 'mem' already exceeds one GPU  -> the MEMORY wall (Module 3).")
    print("  * '1 GPU' time is years          -> the COMPUTE wall (parallelism).")
    print("  * the last column only helps IF 1,024 GPUs actually cooperate")
    print("    at near-linear efficiency -- which is Weeks 3-4 and Modules 2-3.")
    print("=" * 78)


if __name__ == "__main__":
    main()
