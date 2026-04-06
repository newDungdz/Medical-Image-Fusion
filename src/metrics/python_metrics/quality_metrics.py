import numpy as np
from scipy import signal
from scipy.ndimage import sobel, convolve, uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import sobel

# ──────────────────────────────────────────────
# MLI_error  –  Mean Luminance Intensity Error
# ──────────────────────────────────────────────
def mli_error(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    F = F.astype(np.float64)

    mli_a = A.mean()
    mli_b = B.mean()
    mli_f = F.mean()

    mli_ref = (mli_a + mli_b) / 2
    return float(abs(mli_f - mli_ref) / mli_ref)

# ──────────────────────────────────────────────
# SD  –  Standard Deviation
# ──────────────────────────────────────────────
def sd(F: np.ndarray) -> float:
    """Contrast via pixel intensity standard deviation."""
    F = F.astype(np.float64)
    m, n = F.shape
    u = F.mean()
    return float(np.sqrt(((F - u) ** 2).sum() / (m * n)))


# ──────────────────────────────────────────────
# AG  –  Average Gradient
# ──────────────────────────────────────────────
def ag(img: np.ndarray) -> float:
    """Sharpness via average gradient magnitude."""
    img = img.astype(np.float64)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    g = []
    for k in range(img.shape[2]):
        band = img[:, :, k]
        dzdx, dzdy = np.gradient(band)
        s = np.sqrt((dzdx ** 2 + dzdy ** 2) / 2)
        r, c = band.shape
        g.append(s.sum() / ((r - 1) * (c - 1)))
    return float(np.mean(g))

# ──────────────────────────────────────────────
# MSE  –  Mean Squared Error
# ──────────────────────────────────────────────
def mse(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """Average squared pixel error between fused and source images."""
    A, B, F = A / 255.0, B / 255.0, F / 255.0
    m, n = F.shape
    mse_af = ((F - A) ** 2).sum() / (m * n)
    mse_bf = ((F - B) ** 2).sum() / (m * n)
    return float(0.5 * mse_af + 0.5 * mse_bf)


# ──────────────────────────────────────────────
# PSNR  –  Peak Signal-to-Noise Ratio
# ──────────────────────────────────────────────
def psnr(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """Signal-to-noise ratio in dB derived from MSE."""
    return float(20 * np.log10(255 / np.sqrt(mse(A, B, F))))



