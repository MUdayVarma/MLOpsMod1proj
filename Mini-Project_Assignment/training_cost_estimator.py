"""
training_cost_estimator.py

Back-of-envelope training cost estimate using two standard scaling laws.
Model: Phi-3-mini.
  P = 3.8e9 params, T = 3.3e12 tokens
  Source: Phi-3 Technical Report (Abdin et al., 2024, Microsoft),
  https://arxiv.org/abs/2404.14219 -- abstract states "phi-3-mini, a 3.8
  billion parameter language model trained on 3.3 trillion tokens".
"""

# ---- 1. Inputs (hardcoded) ------------------------------------------------
MODEL_NAME = "Phi-3-mini"
P = 3.8e9     # parameter count (Phi-3-mini, per Microsoft's technical report)
T = 3.3e12    # training tokens (Phi-3-mini, per Microsoft's technical report)

print(f"Model                  : {MODEL_NAME}")
print(f"P(Parameters)          : {P:.2e} params")
print(f"T(Tokens)              : {T:.2e} tokens")

# ---- 2. Law 1: compute -----------------------------------------------------
# 6 * P * T FLOPs total training compute (standard Kaplan et al. / Chinchilla
# approximation: ~2 FLOPs/param/token forward, ~4 FLOPs/param/token backward)
total_flops = 6 * P * T

# ---- 2. Law 2: optimizer memory -------------------------------------------
# 16 bytes/param: 2 (FP16 weights) + 2 (FP16 grads) + 4+4 (FP32 Adam m,v) + 4
# (FP32 master weights copy) = 16 bytes, the standard mixed-precision Adam figure
optimizer_bytes = 16 * P
optimizer_gb = optimizer_bytes / 1e9

print(f"Total training FLOPs   : {total_flops:.3e} FLOPs")
print(f"Optimizer memory       : {optimizer_gb:.1f} GB")

# ---- 3. Wall-clock time on 1 GPU and 1,024 GPUs ----------------------------
# A100 peak BF16 Tensor Core throughput: 312 TFLOP/s (dense, no sparsity).
# Source: NVIDIA A100 datasheet, nvidia.com (a100-80gb-datasheet, "Peak
# BFLOAT16 Tensor Core: 312 TF").
peak_flops_per_gpu = 312e12   # FLOP/s
utilization = 0.40            # assumed achieved fraction of peak (typical real-world MFU)
effective_flops_per_gpu = peak_flops_per_gpu * utilization

time_1_gpu_s = total_flops / effective_flops_per_gpu
time_1024_gpu_s = time_1_gpu_s / 1024   # assumes perfect linear scaling

print(f"GPU assumed            : A100, 312 TFLOP/s peak BF16, {utilization:.0%} utilization "
      f"-> {effective_flops_per_gpu:.3e} FLOP/s effective")
print(f"Wall-clock, 1 GPU      : {time_1_gpu_s/86400:.1f} days = {time_1_gpu_s/86400/365:.1f} years")
print(f"Wall-clock, 1,024 GPUs : {time_1024_gpu_s/3600:.1f} hours = {time_1024_gpu_s/86400:.1f} days")