# Additional Resources — 

External reading and reference material to accompany the Week 1 (Fundamentals
of AI/ML) and Week 2 (HPC Fundamentals & Profiling) sessions and Assignment 1.

## Week 1 — ML/DL foundations

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) (Karpathy) — builds backprop, an MLP, and a GPT from scratch, in the same "no magic" spirit as the course demo.
- [3Blue1Brown — Neural Networks series](https://www.3blue1brown.com/lessons/backpropagation) — the best visual intuition for forward pass, gradient descent, and backprop.
- [CS231n notes — Backpropagation](https://cs231n.github.io/optimization-2/) (Stanford) — the chain-rule derivation students are asked to do by hand in Assignment 1, Part A.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020) — the paper behind the 6PT/16P back-of-envelope laws.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., "Chinchilla," 2022) — refines the scaling-law numbers used in Week 1.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — the Transformer paper referenced for why self-attention parallelizes on GPUs.

## Week 2 — HPC fundamentals & profiling

- [The Roofline Model (Berkeley Lab)](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/) — the origin of the arithmetic-intensity/roofline framework used in `roofline.py`.
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) (Ulrich Drepper, 2007) — the definitive deep dive on cache lines, locality, and the memory hierarchy behind the cache-locality demo.
- [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law) / [Gustafson's Law](https://en.wikipedia.org/wiki/Gustafson%27s_law) (Wikipedia) — clean reference derivations for the strong/weak scaling material.
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — SIMT execution model, memory hierarchy, and Tensor Cores in more depth.
- [Nsight Systems documentation](https://docs.nvidia.com/nsight-systems/) — the real profiler the slides point to beyond `nvidia-smi`.
- [TOP500](https://top500.org/) — current supercomputer rankings referenced in the "exascale" discussion.

## General references

Not week-specific, but load-bearing for the assignment code:

- [NumPy documentation](https://numpy.org/doc/stable/) — for Assignment 1 Parts A/C implementations.
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html) — for when learners move beyond the from-scratch NumPy versions.
