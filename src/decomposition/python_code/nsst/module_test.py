import numpy as np
from scipy.signal import convolve2d
import time

def conv2(x, y, mode='same'):
    return np.rot90(
        convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode),
        2
    )

def conv2_same_matlab(x, k):
    full = convolve2d(x, k, mode='full')
    kh, kw = k.shape
    h, w = x.shape
    return full[kh//2:kh//2+h, kw//2:kw//2+w]

# Fixed input
A = np.array([
    [1, 2, 3, 4],
    [4, 5, 6, 7],
    [7, 8, 9, 10],
    [10, 11, 12, 13]
], dtype=float)

K = np.array([
    [1, 2],
    [3, 4]
], dtype=float)

N = 1000000
# 🔴 Warm-up (important for fair timing)
for _ in range(1000):
    conv2(A, K)
    conv2_same_matlab(A, K)
    convolve2d(A, K, mode='same')

# 🔵 Benchmark conv2 (rot90 version)
start = time.perf_counter()
for _ in range(N):
    conv2(A, K)
end = time.perf_counter()

time_conv2 = end - start

# 🟢 Benchmark full+crop version
start = time.perf_counter()
for _ in range(N):
    conv2_same_matlab(A, K)
end = time.perf_counter()

time_custom = end - start

# 🟡 Benchmark scipy's convolve2d with mode='same
start = time.perf_counter()
for _ in range(N):
  convolve2d(A, K, mode='same')
end = time.perf_counter()
time_scipy_same = end - start

print(f"\nconv2 (rot90) time: {time_conv2:.4f} sec")
print(f"conv2_same_matlab time: {time_custom:.4f} sec")
print(f"scipy convolve2d (mode='same') time: {time_scipy_same:.4f} sec")

print(f"\nSpeedup: {time_conv2 / time_custom:.2f}x")