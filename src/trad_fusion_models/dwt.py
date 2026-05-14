"""
Image Fusion Module
===================
Multi-mode wavelet-based image fusion supporting three strategies:

    1. ``grayscale``  — fuse two grayscale images directly via DWT
    2. ``ycbcr``      — fuse luma (Y) of a grayscale + RGB pair; keep Cb/Cr
    3. ``hsi``        — fuse intensity (I) of a grayscale + RGB pair; keep H/S

All three modes share the same core wavelet engine
(:func:`_wavelet_fuse_channels`), so swapping the colour-space wrapper
is the only difference between strategies 2 and 3.

Usage
-----
    from image_fusion import fuse

    # Strategy 1 – pure grayscale
    out = fuse(gray1, gray2, mode="grayscale")

    # Strategy 2 – YCbCr  (gray as Y₁, RGB supplies Y₂ + Cb + Cr)
    out = fuse(gray, rgb, mode="ycbcr")

    # Strategy 3 – HSI    (gray as I₁, RGB supplies I₂ + H + S)
    out = fuse(gray, rgb, mode="hsi")

    # All modes accept optional wavelet / level kwargs
    out = fuse(gray, rgb, mode="hsi", wavelet="db4", level=3)

API (low-level)
---------------
    wavelet_fusion(img1, img2)            → grayscale fused ndarray
    ycbcr_fusion(gray_img, rgb_img, ...)  → RGB fused ndarray
    hsi_fusion(gray_img, rgb_img, ...)    → RGB fused ndarray
"""

import numpy as np
import pywt
from color_model_ulti import rgb_to_ycbcr, ycbcr_to_rgb, rgb_to_hsi, hsi_to_rgb

# ══════════════════════════════════════════════════════════════════════
# 1.  Shared wavelet engine
# ══════════════════════════════════════════════════════════════════════

def _wavelet_fuse_channels(
    ch1: np.ndarray,
    ch2: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
) -> np.ndarray:
    """
    Fuse two single-channel float arrays via multi-level DWT.

    Rules
    -----
    * Approximation (LL) band  →  arithmetic mean of both inputs
    * Detail bands (LH, HL, HH) →  coefficient with greater |value|

    Parameters
    ----------
    ch1, ch2 : float32 ndarray, shape (H, W)
        Same spatial size; values in [0, 255] or [0, 1] — units don't
        matter as long as both channels use the *same* scale.
    wavelet  : PyWavelets wavelet name (default ``'db1'``).
    level    : decomposition levels (default ``1``).

    Returns
    -------
    float32 ndarray, shape (H, W), values clipped to [0, 255].
    """
    coeffs1 = pywt.wavedec2(ch1, wavelet, level=level)
    coeffs2 = pywt.wavedec2(ch2, wavelet, level=level)

    # LL band: average
    fused: list = [(coeffs1[0] + coeffs2[0]) / 2.0]

    # Detail bands: max absolute energy
    for c1_tuple, c2_tuple in zip(coeffs1[1:], coeffs2[1:]):
        fused.append(
            tuple(
                np.where(np.abs(d1) >= np.abs(d2), d1, d2)
                for d1, d2 in zip(c1_tuple, c2_tuple)
            )
        )

    reconstructed = pywt.waverec2(fused, wavelet)
    return np.clip(reconstructed, 0, 255).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 4.  Internal: normalise grayscale input
# ══════════════════════════════════════════════════════════════════════

def _to_gray2d(img: np.ndarray, target_scale: float = 255.0) -> np.ndarray:
    """
    Squeeze a grayscale image to shape (H, W) and ensure it is float32
    in [0, *target_scale*].

    Accepts
    -------
    * (H, W)       — already 2-D
    * (H, W, 1)    — single-channel 3-D
    * (H, W, 3)    — already-gray BGR/RGB stored as 3-channel
    * uint8 or float32 input

    Parameters
    ----------
    target_scale : 255.0 for YCbCr mode (Y lives in [0, 255]);
                   1.0   for HSI mode    (I lives in [0, 1]).
    """
    arr = np.squeeze(img)

    if arr.ndim == 3:
        # Treat 3-channel input as a pre-converted grayscale: take mean
        arr = arr.mean(axis=-1)

    if arr.ndim != 2:
        raise ValueError(
            f"Grayscale image must be 2-D after squeezing, got shape {img.shape}"
        )

    arr = arr.astype(np.float32)

    # Detect whether the array is already in [0, 1] or [0, 255]
    if arr.max() <= 1.0 and target_scale == 255.0:
        arr = arr * 255.0
    elif arr.max() > 1.0 and target_scale == 1.0:
        arr = arr / 255.0

    return arr


# ══════════════════════════════════════════════════════════════════════
# 5.  Public fusion functions
# ══════════════════════════════════════════════════════════════════════

def wavelet_fusion(
    img1: np.ndarray,
    img2: np.ndarray,
    wavelet: str = "db1",
    level: int = 1,
) -> np.ndarray:
    """
    **Mode 1 — grayscale-only DWT fusion.**

    Fuse two single-channel images directly in the spatial/wavelet domain.
    No colour space involved; the output is a grayscale uint8 image.

    Parameters
    ----------
    img1, img2 : ndarray, shape (H, W) or (H, W, 1)
        Grayscale images (uint8 or float32).
    wavelet    : PyWavelets wavelet name (default ``'db1'``).
    level      : DWT decomposition level (default ``1``).

    Returns
    -------
    ndarray : uint8 grayscale image, shape (H, W).
    """
    ch1 = _to_gray2d(img1, target_scale=255.0)
    ch2 = _to_gray2d(img2, target_scale=255.0)

    return _wavelet_fuse_channels(ch1, ch2, wavelet=wavelet, level=level).astype(np.uint8)


def ycbcr_fusion(
    gray_img: np.ndarray,
    rgb_img: np.ndarray,
    wavelet: str = "db1",
    level: int = 1,
) -> np.ndarray:
    """
    **Mode 2 — YCbCr luma fusion.**

    Fuse a grayscale image with an RGB image by operating exclusively on
    the Y (luma) channel.  Cb and Cr are taken from *rgb_img* unchanged,
    preserving all colour information of the RGB source.

    Parameters
    ----------
    gray_img : ndarray, shape (H, W) or (H, W, 1)
        Grayscale source (uint8 or float32).  Treated as Y₁.
    rgb_img  : ndarray, shape (H, W, 3), uint8
        Colour source.  Its Y channel becomes Y₂; Cb and Cr are kept.
    wavelet  : PyWavelets wavelet name (default ``'db1'``).
    level    : DWT decomposition level (default ``1``).

    Returns
    -------
    ndarray : uint8 RGB fused image, same spatial dimensions as inputs.

    Notes
    -----
    Both images must share the same (H, W) spatial dimensions.
    The Y channel lives in [0, 255] in BT.601 full-swing encoding, so
    :func:`_to_gray2d` is called with ``target_scale=255``.
    """
    # Grayscale → float32 [0, 255]  (Y₁)
    Y1 = _to_gray2d(gray_img, target_scale=255.0)

    # RGB → YCbCr; extract Y₂, keep Cb/Cr
    Y2, Cb, Cr = rgb_to_ycbcr(rgb_img)

    # Fuse only the luma channels
    Y_fused = _wavelet_fuse_channels(Y1, Y2, wavelet=wavelet, level=level)

    # Reconstruct RGB with fused Y and original chroma
    return ycbcr_to_rgb(Y_fused, Cb, Cr)


def hsi_fusion(
    gray_img: np.ndarray,
    rgb_img: np.ndarray,
    wavelet: str = "db1",
    level: int = 1,
) -> np.ndarray:
    """
    **Mode 3 — HSI intensity fusion.**

    Fuse a grayscale image with an RGB image by operating exclusively on
    the I (intensity) channel.  H and S are taken from *rgb_img* unchanged,
    preserving all hue and saturation information of the RGB source.

    Parameters
    ----------
    gray_img : ndarray, shape (H, W) or (H, W, 1)
        Grayscale source (uint8 or float32).  Treated as I₁.
    rgb_img  : ndarray, shape (H, W, 3), uint8
        Colour source.  Its I channel becomes I₂; H and S are kept.
    wavelet  : PyWavelets wavelet name (default ``'db1'``).
    level    : DWT decomposition level (default ``1``).

    Returns
    -------
    ndarray : uint8 RGB fused image, same spatial dimensions as inputs.

    Notes
    -----
    Both images must share the same (H, W) spatial dimensions.
    The I channel lives in [0, 1] in HSI encoding, so
    :func:`_to_gray2d` is called with ``target_scale=1`` and the fused
    result is re-normalised to [0, 1] before calling :func:`hsi_to_rgb`.
    """
    # Grayscale → float32 [0, 1]  (I₁)
    I1 = _to_gray2d(gray_img, target_scale=1.0)

    # RGB → HSI; extract I₂, keep H/S
    H, S, I2 = rgb_to_hsi(rgb_img)

    # Scale I channels to [0, 255] for the shared wavelet engine, then
    # scale the result back to [0, 1] for hsi_to_rgb.
    I_fused = _wavelet_fuse_channels(
        I1 * 255.0,
        I2 * 255.0,
        wavelet=wavelet,
        level=level,
    ) / 255.0

    # Reconstruct RGB with fused I and original hue/saturation
    return hsi_to_rgb(H, S, I_fused)


# ══════════════════════════════════════════════════════════════════════
# 6.  Unified entry point
# ══════════════════════════════════════════════════════════════════════

_MODES = {
    "grayscale": wavelet_fusion,
    "ycbcr":     ycbcr_fusion,
    "hsi":       hsi_fusion,
}


def fuse(
    img1: np.ndarray,
    img2: np.ndarray,
    mode: str = "grayscale",
    wavelet: str = "db1",
    level: int = 1,
) -> np.ndarray:
    """
    Unified fusion entry point.

    Parameters
    ----------
    img1    : first image (grayscale in all modes).
    img2    : second image:
                * ``'grayscale'`` → also grayscale
                * ``'ycbcr'`` or ``'hsi'`` → uint8 RGB
    mode    : one of ``'grayscale'``, ``'ycbcr'``, ``'hsi'``.
    wavelet : PyWavelets wavelet name (default ``'db1'``).
    level   : DWT decomposition levels (default ``1``).

    Returns
    -------
    Fused image:
        * ``'grayscale'`` → uint8 (H, W)   grayscale
        * ``'ycbcr'``     → uint8 (H, W, 3) RGB
        * ``'hsi'``       → uint8 (H, W, 3) RGB

    Examples
    --------
    >>> import cv2
    >>> from image_fusion import fuse
    >>>
    >>> gray = cv2.imread("mri.png",  cv2.IMREAD_GRAYSCALE)
    >>> rgb  = cv2.imread("pet.png")
    >>> rgb  = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    >>>
    >>> out_gray  = fuse(gray, gray2,  mode="grayscale")
    >>> out_ycbcr = fuse(gray, rgb,    mode="ycbcr",  wavelet="db4", level=2)
    >>> out_hsi   = fuse(gray, rgb,    mode="hsi",    wavelet="db4", level=2)
    """
    if mode not in _MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: {list(_MODES.keys())}"
        )
    return _MODES[mode](img1, img2, wavelet=wavelet, level=level)


# ══════════════════════════════════════════════════════════════════════
# 7.  CLI / quick demo
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import cv2

    # Load inputs
    gray_bgr = cv2.imread("data/AANLIB/PET-MRI/MRI/25015.png", cv2.IMREAD_GRAYSCALE)
    rgb_bgr  = cv2.imread("data/AANLIB/PET-MRI/PET/25015.png")

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    # Fuse
    mode = "ycbcr"  # choose from "grayscale", "ycbcr", "hsi"
    result = fuse(gray_bgr, rgb, mode=mode)

    # Save — convert RGB → BGR for OpenCV if needed
    if result.ndim == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"data/test-sample/{mode}_fused.png", result)
    print(f"✓  {mode.upper()} fused image saved → {mode}_fused.png")
    print(f"   shape={result.shape}  dtype={result.dtype}")