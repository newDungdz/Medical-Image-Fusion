# Test atrousdec on a 9x9 non-random matrix
import numpy as np
from nsst_dec import atrousdec

# 15x15 test image: smooth gradient + a central impulse
N = 27
x = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        x[i, j] = i + j  # smooth gradient
x[N//2, N//2] += 10            # central impulse to stress detail bands


print("Input matrix:")
print(x)
print()

# Run decomposition
fname = '9-7'
Nlevels = 2
y = atrousdec(x, fname, Nlevels)

# Inspect outputs
print(f"Decomposition: fname='{fname}', Nlevels={Nlevels}")
print(f"Number of subbands: {len(y)}\n")

for k, band in enumerate(y):
    label = "Lowpass (coarsest)" if k == 0 else f"Detail level {k} ({'finest' if k == Nlevels else 'coarser'})"
    print(f"y[{k}] — {label}")
    print(f"  Shape : {band.shape}")
    print(f"  Min   : {band.min():.6f}")
    print(f"  Max   : {band.max():.6f}")
    print(f"  Sum   : {band.sum():.6f}")
    print()

# Sanity check: energy conservation (approximate for frame)
input_energy = np.sum(x ** 2)
output_energy = sum(np.sum(b ** 2) for b in y)
print(f"Input energy  : {input_energy:.4f}")
print(f"Output energy : {output_energy:.4f}")
print(f"Ratio (out/in): {output_energy / input_energy:.4f}  (should be ~1 for tight frame)")