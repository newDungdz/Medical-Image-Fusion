"""
Laplacian Pyramid Image Fusion Module
======================================
Multi-mode Laplacian-pyramid-based image fusion supporting three strategies:

    1. ``grayscale``  — fuse two grayscale images directly via Laplacian pyramid
    2. ``ycbcr``      — fuse luma (Y) of a grayscale + RGB pair; keep Cb/Cr
    3. ``hsi``        — fuse intensity (I) of a grayscale + RGB pair; keep H/S

All three modes share the same core Laplacian engine
(:func:`_laplacian_fuse_channels`), so swapping the colour-space wrapper
is the only difference between strategies 2 and 3.

Usage
-----
    from laplacian_fusion import fuse

    # Strategy 1 – pure grayscale
    out = fuse(gray1, gray2, mode="grayscale")

    # Strategy 2 – YCbCr  (gray as Y₁, RGB supplies Y₂ + Cb + Cr)
    out = fuse(gray, rgb, mode="ycbcr")

    # Strategy 3 – HSI    (gray as I₁, RGB supplies I₂ + H + S)
    out = fuse(gray, rgb, mode="hsi")

    # All modes accept an optional levels kwarg
    out = fuse(gray, rgb, mode="hsi", levels=4)

API (low-level)
---------------
    laplacian_fusion(img1, img2)            → grayscale fused ndarray
    ycbcr_fusion(gray_img, rgb_img, ...)    → RGB fused ndarray
    hsi_fusion(gray_img, rgb_img, ...)      → RGB fused ndarray
"""

import cv2
import numpy as np
from color_model_ulti import rgb_to_ycbcr, ycbcr_to_rgb, rgb_to_hsi, hsi_to_rgb


# ══════════════════════════════════════════════════════════════════════
# 1.  Pyramid helpers
# ══════════════════════════════════════════════════════════════════════

def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    """Return a Gaussian pyramid with *levels* downsampling steps."""
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        gp.append(img)
    return gp


def _build_laplacian_pyramid(gp: list) -> list:
    """Convert a Gaussian pyramid into a Laplacian pyramid."""
    lp = []
    for i in range(len(gp) - 1):
        size = (gp[i].shape[1], gp[i].shape[0])
        up = cv2.pyrUp(gp[i + 1], dstsize=size)
        lp.append(gp[i] - up)
    lp.append(gp[-1])   # coarsest (base / approximation) layer
    return lp


def _reconstruct_from_laplacian(lp: list) -> np.ndarray:
    """Collapse a Laplacian pyramid back to a full-resolution image."""
    img = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        size = (lp[i].shape[1], lp[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + lp[i]
    return img


# ══════════════════════════════════════════════════════════════════════
# 2.  Shared Laplacian engine
# ══════════════════════════════════════════════════════════════════════

def _laplacian_fuse_channels(
    ch1: np.ndarray,
    ch2: np.ndarray,
    levels: int = 4,
) -> np.ndarray:
    """
    Fuse two single-channel float arrays via Laplacian pyramid.

    Rules
    -----
    * Base (coarsest) layer  →  arithmetic mean of both inputs
    * Detail layers          →  coefficient with greater |value|

    Parameters
    ----------
    ch1, ch2 : float32 ndarray, shape (H, W)
        Same spatial size; values in [0, 255] — both must use the same scale.
    levels   : number of pyramid levels (default ``4``).

    Returns
    -------
    float32 ndarray, shape (H, W), values clipped to [0, 255].
    """
    lp1 = _build_laplacian_pyramid(_build_gaussian_pyramid(ch1, levels))
    lp2 = _build_laplacian_pyramid(_build_gaussian_pyramid(ch2, levels))

    fused_pyramid = []
    for i in range(len(lp1)):
        if i == len(lp1) - 1:
            # Base layer: average (mirrors LL-band averaging in wavelet mode)
            fused_pyramid.append((lp1[i] + lp2[i]) / 2.0)
        else:
            # Detail layers: max absolute energy
            fused_pyramid.append(
                np.where(np.abs(lp1[i]) >= np.abs(lp2[i]), lp1[i], lp2[i])
            )

    reconstructed = _reconstruct_from_laplacian(fused_pyramid)
    return np.clip(reconstructed, 0, 255).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 3.  Internal: normalise grayscale input
# ══════════════════════════════════════════════════════════════════════

def _to_gray2d(img: np.ndarray, target_scale: float = 255.0) -> np.ndarray:
    """
    Squeeze a grayscale image to shape (H, W) and ensure it is float32
    in [0, *target_scale*].

    Accepts
    -------
    * (H, W)       — already 2-D
    * (H, W, 1)    — single-channel 3-D
    * (H, W, 3)    — already-gray BGR/RGB stored as 3-channel (mean taken)
    * uint8 or float32 input

    Parameters
    ----------
    target_scale : 255.0 for YCbCr mode (Y lives in [0, 255]);
                   1.0   for HSI mode    (I lives in [0, 1]).
    """
    arr = np.squeeze(img)

    if arr.ndim == 3:
        arr = arr.mean(axis=-1)

    if arr.ndim != 2:
        raise ValueError(
            f"Grayscale image must be 2-D after squeezing, got shape {img.shape}"
        )

    arr = arr.astype(np.float32)

    if arr.max() <= 1.0 and target_scale == 255.0:
        arr = arr * 255.0
    elif arr.max() > 1.0 and target_scale == 1.0:
        arr = arr / 255.0

    return arr


# ══════════════════════════════════════════════════════════════════════
# 4.  Public fusion functions
# ══════════════════════════════════════════════════════════════════════

def laplacian_fusion(
    img1: np.ndarray,
    img2: np.ndarray,
    levels: int = 4,
) -> np.ndarray:
    """
    **Mode 1 — grayscale-only Laplacian pyramid fusion.**

    Fuse two single-channel images directly in the pyramid domain.
    No colour space involved; the output is a grayscale uint8 image.

    Parameters
    ----------
    img1, img2 : ndarray, shape (H, W) or (H, W, 1)
        Grayscale images (uint8 or float32).
    levels     : Laplacian pyramid depth (default ``4``).

    Returns
    -------
    ndarray : uint8 grayscale image, shape (H, W).
    """
    ch1 = _to_gray2d(img1, target_scale=255.0)
    ch2 = _to_gray2d(img2, target_scale=255.0)

    return _laplacian_fuse_channels(ch1, ch2, levels=levels).astype(np.uint8)


def ycbcr_fusion(
    gray_img: np.ndarray,
    rgb_img: np.ndarray,
    levels: int = 4,
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
    levels   : Laplacian pyramid depth (default ``4``).

    Returns
    -------
    ndarray : uint8 RGB fused image, same spatial dimensions as inputs.

    Notes
    -----
    Both images must share the same (H, W) spatial dimensions.
    The Y channel lives in [0, 255] in BT.601 full-swing encoding.
    """
    Y1 = _to_gray2d(gray_img, target_scale=255.0)

    Y2, Cb, Cr = rgb_to_ycbcr(rgb_img)

    Y_fused = _laplacian_fuse_channels(Y1, Y2, levels=levels)

    return ycbcr_to_rgb(Y_fused, Cb, Cr)


def hsi_fusion(
    gray_img: np.ndarray,
    rgb_img: np.ndarray,
    levels: int = 4,
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
    levels   : Laplacian pyramid depth (default ``4``).

    Returns
    -------
    ndarray : uint8 RGB fused image, same spatial dimensions as inputs.

    Notes
    -----
    Both images must share the same (H, W) spatial dimensions.
    The I channel lives in [0, 1] in HSI encoding, so the channel is
    scaled up to [0, 255] for the shared engine then scaled back to [0, 1]
    before calling :func:`hsi_to_rgb`.
    """
    I1 = _to_gray2d(gray_img, target_scale=1.0)

    H, S, I2 = rgb_to_hsi(rgb_img)

    # Scale to [0, 255] for the shared engine, then back to [0, 1]
    I_fused = _laplacian_fuse_channels(
        I1 * 255.0,
        I2 * 255.0,
        levels=levels,
    ) / 255.0

    return hsi_to_rgb(H, S, I_fused)


# ══════════════════════════════════════════════════════════════════════
# 5.  Unified entry point
# ══════════════════════════════════════════════════════════════════════

_MODES = {
    "grayscale": laplacian_fusion,
    "ycbcr":     ycbcr_fusion,
    "hsi":       hsi_fusion,
}


def fuse(
    img1: np.ndarray,
    img2: np.ndarray,
    mode: str = "grayscale",
    levels: int = 4,
) -> np.ndarray:
    """
    Unified Laplacian fusion entry point.

    Parameters
    ----------
    img1   : first image (grayscale in all modes).
    img2   : second image:
               * ``'grayscale'`` → also grayscale
               * ``'ycbcr'`` or ``'hsi'`` → uint8 RGB
    mode   : one of ``'grayscale'``, ``'ycbcr'``, ``'hsi'``.
    levels : Laplacian pyramid depth (default ``4``).

    Returns
    -------
    Fused image:
        * ``'grayscale'`` → uint8 (H, W)    grayscale
        * ``'ycbcr'``     → uint8 (H, W, 3) RGB
        * ``'hsi'``       → uint8 (H, W, 3) RGB

    Examples
    --------
    >>> import cv2
    >>> from laplacian_fusion import fuse
    >>>
    >>> gray = cv2.imread("mri.png",  cv2.IMREAD_GRAYSCALE)
    >>> rgb  = cv2.imread("pet.png")
    >>> rgb  = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    >>>
    >>> out_gray  = fuse(gray, gray2, mode="grayscale")
    >>> out_ycbcr = fuse(gray, rgb,   mode="ycbcr",  levels=4)
    >>> out_hsi   = fuse(gray, rgb,   mode="hsi",    levels=4)
    """
    if mode not in _MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: {list(_MODES.keys())}"
        )
    return _MODES[mode](img1, img2, levels=levels)


# ══════════════════════════════════════════════════════════════════════
# 6.  CLI / quick demo
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    gray_bgr = cv2.imread("data/AANLIB/PET-MRI/MRI/25015.png", cv2.IMREAD_GRAYSCALE)
    rgb_bgr  = cv2.imread("data/AANLIB/PET-MRI/PET/25015.png")

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    mode = "ycbcr"  # choose from "grayscale", "ycbcr", "hsi"
    result = fuse(gray_bgr, rgb, mode=mode, levels=4)

    if result.ndim == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"data/test-sample/{mode}_laplacian_fused.png", result)
    print(f"✓  {mode.upper()} Laplacian fused image saved → data/test-sample/{mode}_laplacian_fused.png")
    print(f"   shape={result.shape}  dtype={result.dtype}")