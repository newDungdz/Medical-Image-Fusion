import numpy as np
from scipy import signal
from scipy.ndimage import sobel, convolve, uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import sobel

# ──────────────────────────────────────────────
# CC  –  Correlation Coefficient
# ──────────────────────────────────────────────
def cc(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """Mean Pearson correlation between fused and each source image."""
    def _corr(X, Y):
        Xm, Ym = X - X.mean(), Y - Y.mean()
        return (Xm * Ym).sum() / (np.sqrt((Xm ** 2).sum() * (Ym ** 2).sum()) + 1e-10)
    rAF = _corr(A.astype(np.float64), F.astype(np.float64))
    rBF = _corr(B.astype(np.float64), F.astype(np.float64))
    return float(np.mean([rAF, rBF]))
# endregion




