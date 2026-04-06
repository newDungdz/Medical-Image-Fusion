import numpy as np
from scipy import signal
from scipy.ndimage import sobel, convolve, uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import sobel

# # ── Helper: 2-D 'valid' convolution (replicates MATLAB filter2 'valid') ───────
def _conv_valid(img: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Correlate img with win using 'valid' border handling (no padding)."""
    from scipy.signal import correlate2d
    return correlate2d(img, win, mode='valid')

# ──────────────────────────────────────────────
# VIF  –  Visual Information Fidelity (fixed)
# ──────────────────────────────────────────────
def vif(ref: np.ndarray, dist: np.ndarray) -> float:
    """
    Visual Information Fidelity (VIF) — direct Python port of vifp_mscale.m
    (Sheikh & Bovik, 2006).

    Parameters
    ----------
    ref  : np.ndarray  – Reference image  (2-D float or uint8)
    dist : np.ndarray  – Distorted image  (2-D float or uint8)

    Returns
    -------
    float  – VIF score  (1.0 = perfect fidelity, lower = more distortion)
    """
    ref  = ref.astype(np.float64)
    dist = dist.astype(np.float64)

    sigma_nsq = 2.0        # HVS noise variance — matches MATLAB constant
    EPS       = 1e-10

    num = 0.0
    den = 0.0

    for scale in range(1, 5):                      # scale = 1..4

        # ── 1. Scale-dependent Gaussian window (matches MATLAB exactly) ──────
        N   = 2 ** (4 - scale + 1) + 1            # 17, 9, 5, 3
        sig = N / 5.0

        k      = np.arange(N) - N // 2
        g1d    = np.exp(-k**2 / (2 * sig**2))
        g1d   /= g1d.sum()
        win    = np.outer(g1d, g1d)                # (N×N) Gaussian kernel

        # ── 2. Pre-filter + subsample for scales 2-4 (matches MATLAB) ────────
        #      MATLAB: filter2(win, img, 'valid')  then  img(1:2:end, 1:2:end)
        if scale > 1:
            ref  = _conv_valid(ref,  win)[::2, ::2]
            dist = _conv_valid(dist, win)[::2, ::2]

        # ── 3. Local statistics via 'valid' convolution ───────────────────────
        mu1     = _conv_valid(ref,       win)
        mu2     = _conv_valid(dist,      win)

        mu1_sq  = mu1 * mu1
        mu2_sq  = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = _conv_valid(ref  * ref,  win) - mu1_sq
        sigma2_sq = _conv_valid(dist * dist, win) - mu2_sq
        sigma12   = _conv_valid(ref  * dist, win) - mu1_mu2

        # Clamp negative variances (numerical noise)
        sigma1_sq = np.maximum(sigma1_sq, 0)
        sigma2_sq = np.maximum(sigma2_sq, 0)

        # ── 4. Distortion-channel gain g and residual noise sv_sq ─────────────
        g     = sigma12 / (sigma1_sq + EPS)
        sv_sq = sigma2_sq - g * sigma12

        # ── 5. Edge-case masking (mirrors MATLAB conditionals exactly) ─────────
        # Where reference variance is negligible → no signal to compare
        g    [sigma1_sq < EPS] = 0
        sv_sq[sigma1_sq < EPS] = sigma2_sq[sigma1_sq < EPS]
        sigma1_sq[sigma1_sq < EPS] = 0

        # Where distorted variance is negligible → no information transferred
        g    [sigma2_sq < EPS] = 0
        sv_sq[sigma2_sq < EPS] = 0

        # Negative gain is non-physical → clamp to zero
        sv_sq[g < 0] = sigma2_sq[g < 0]
        g    [g < 0] = 0

        # Residual noise floor
        sv_sq[sv_sq <= EPS] = EPS

        # ── 6. VIF information ratio (log10, matches MATLAB) ──────────────────
        num += np.sum(np.log10(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    return float(num / (den + EPS))
