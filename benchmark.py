import numpy as np
import time

# Create matrices
a = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1000)

print("Running matrix multiplication...")
c = a @ b                 

print("Calculating row means...")
row_means = a.mean(axis=1)

print("Applying boolean mask...")
mask = a > 0.5            
a[mask] = 0.0

print("\n--- Starting Speed Comparison ---")

# Measure slow Python list comprehension
start_time = time.time()
slow = [x*x for x in range(10_000_000)]
slow_duration = time.time() - start_time
print(f"Standard Python List took: {slow_duration:.4f} seconds")

# Measure fast NumPy vectorisation
start_time = time.time()
fast = np.arange(10_000_000) ** 2
fast_duration = time.time() - start_time
print(f"NumPy Vectorisation took:  {fast_duration:.4f} seconds")

# Calculate performance boost
speedup = slow_duration / fast_duration
print(f"\nResult: NumPy was {speedup:.1f}x faster!")

