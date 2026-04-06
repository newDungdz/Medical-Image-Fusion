import numpy as np
from scipy import signal
from scipy.ndimage import sobel, convolve, uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import sobel


# ──────────────────────────────────────────────
# QABF  –  Quality of Image Fusion (Edge-based)
# ──────────────────────────────────────────────
def qabf(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """Edge information transfer quality from source images to fused image."""
    L = 1; Tg = 0.9994; kg = -15; Dg = 0.5; Ta = 0.9879; ka = -22; Da = 0.8

    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)

    def _edge(img):
        img = img.astype(np.float64)
        Sx = signal.convolve2d(img, h3, mode='same')
        Sy = signal.convolve2d(img, h1, mode='same')
        g = np.sqrt(Sx ** 2 + Sy ** 2)
        a = np.where(Sx == 0, np.pi / 2, np.arctan2(Sy, Sx))
        return g, a

    gA, aA = _edge(A)
    gB, aB = _edge(B)
    gF, aF = _edge(F)

    def _Qxf(gX, aX, gF, aF):
        GAF = np.where(gX > gF, gF / (gX + 1e-10), np.where(gX == gF, gF, gX / (gF + 1e-10)))
        AAF = 1 - np.abs(aX - aF) / (np.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        return QgAF * QaAF

    QAF = _Qxf(gA, aA, gF, aF)
    QBF = _Qxf(gB, aB, gF, aF)
    deno = (gA + gB).sum()
    nume = (QAF * gA + QBF * gB).sum()
    return float(nume / deno)

# ──────────────────────────────────────────────
# SF  –  Spatial Frequency
# ──────────────────────────────────────────────
def sf(F: np.ndarray) -> float:
    """Overall activity level via row and column frequency."""
    F = F.astype(np.float64)
    rf = np.diff(F, axis=0)
    cf = np.diff(F, axis=1)
    rf1 = np.sqrt((rf ** 2).mean())
    cf1 = np.sqrt((cf ** 2).mean())
    return float(np.sqrt(rf1 ** 2 + cf1 ** 2))
