import numpy as np

def rgb_to_ycbcr(img_rgb: np.ndarray):
    """BT.601 full-swing — returns float32 Y, Cb, Cr each in [0, 255]."""
    rgb = img_rgb.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y  =  0.299   * R + 0.587   * G + 0.114   * B
    Cb = -0.16874 * R - 0.33126 * G + 0.5     * B + 128.0
    Cr =  0.5     * R - 0.41869 * G - 0.08131 * B + 128.0
    return Y.astype(np.float32), Cb.astype(np.float32), Cr.astype(np.float32)


def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """Reconstruct uint8 RGB from float32 YCbCr channels."""
    cb, cr = Cb - 128.0, Cr - 128.0
    R = Y                   + 1.402   * cr
    G = Y - 0.34414 * cb   - 0.71414 * cr
    B = Y + 1.772   * cb
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)


def rgb_to_hsi(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an RGB image to HSI color space (pure NumPy – no cv2 shortcut).

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image, either uint8 [0, 255] or float32 [0.0, 1.0].
        Shape must be (H, W, 3).

    Returns
    -------
    H : float32 ndarray, shape (H, W)  — Hue in radians [0, 2π]
    S : float32 ndarray, shape (H, W)  — Saturation     [0, 1]
    I : float32 ndarray, shape (H, W)  — Intensity      [0, 1]
    """
    rgb = img_rgb.astype(np.float32) / 255.0 if img_rgb.dtype == np.uint8 else img_rgb.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Intensity
    I = (R + G + B) / 3.0

    # Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 0, 1.0 - min_rgb / (I + 1e-10), 0.0)

    # Hue
    num   = 0.5 * ((R - G) + (R - B))
    den   = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))

    H = np.where(B <= G, theta, 2 * np.pi - theta)

    return H.astype(np.float32), S.astype(np.float32), I.astype(np.float32)


def hsi_to_rgb(H: np.ndarray, S: np.ndarray, I: np.ndarray) -> np.ndarray:
    """
    Reconstruct an RGB image from HSI channels.

    Parameters
    ----------
    H : float32 ndarray — Hue in radians [0, 2π]
    S : float32 ndarray — Saturation     [0, 1]
    I : float32 ndarray — Intensity      [0, 1]

    Returns
    -------
    np.ndarray : uint8 RGB image, shape (H, W, 3)
    """
    H = H.copy()
    R = np.zeros_like(I)
    G = np.zeros_like(I)
    B = np.zeros_like(I)

    # Sector 1: 0 ≤ H < 2π/3
    mask1 = (H >= 0) & (H < 2 * np.pi / 3)
    B[mask1] = I[mask1] * (1 - S[mask1])
    R[mask1] = I[mask1] * (1 + S[mask1] * np.cos(H[mask1]) / np.cos(np.pi / 3 - H[mask1]))
    G[mask1] = 3 * I[mask1] - (R[mask1] + B[mask1])

    # Sector 2: 2π/3 ≤ H < 4π/3
    mask2 = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    H2      = H[mask2] - 2 * np.pi / 3
    R[mask2] = I[mask2] * (1 - S[mask2])
    G[mask2] = I[mask2] * (1 + S[mask2] * np.cos(H2) / np.cos(np.pi / 3 - H2))
    B[mask2] = 3 * I[mask2] - (R[mask2] + G[mask2])

    # Sector 3: 4π/3 ≤ H < 2π
    mask3 = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)
    H3      = H[mask3] - 4 * np.pi / 3
    G[mask3] = I[mask3] * (1 - S[mask3])
    B[mask3] = I[mask3] * (1 + S[mask3] * np.cos(H3) / np.cos(np.pi / 3 - H3))
    R[mask3] = 3 * I[mask3] - (G[mask3] + B[mask3])

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def rgb_to_hsv(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an RGB image to HSV color space (pure NumPy – no cv2 shortcut).

    The Hue formula uses the cosine / arccos approach (same as the HSI
    model) so both color spaces share the same H channel and can be
    compared directly.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image, either uint8 [0, 255] or float32 [0.0, 1.0].
        Shape must be (H, W, 3).

    Returns
    -------
    H : float32 ndarray, shape (H, W)  — Hue in radians [0, 2π]
    S : float32 ndarray, shape (H, W)  — Saturation     [0, 1]
    V : float32 ndarray, shape (H, W)  — Value          [0, 1]
    """
    rgb = img_rgb.astype(np.float32) / 255.0 if img_rgb.dtype == np.uint8 else img_rgb.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Value — brightest channel
    V = np.maximum(np.maximum(R, G), B)

    # Saturation — chroma relative to peak; 0 where V == 0 (pure black)
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(V > 0, (V - min_rgb) / (V + 1e-10), 0.0)

    # Hue — cosine-based formula, identical to HSI for direct comparison
    num   = 0.5 * ((R - G) + (R - B))
    den   = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    H     = np.where(B <= G, theta, 2 * np.pi - theta)

    return H.astype(np.float32), S.astype(np.float32), V.astype(np.float32)


def hsv_to_rgb(H: np.ndarray, S: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Reconstruct an RGB image from HSV channels.

    Uses the standard 60°-sector HSV → RGB decomposition:
        p = V(1 − S)
        q = V(1 − fS)
        t = V(1 − (1−f)S)
    where f is the fractional part of H / 60°.

    Parameters
    ----------
    H : float32 ndarray — Hue in radians [0, 2π]
    S : float32 ndarray — Saturation     [0, 1]
    V : float32 ndarray — Value          [0, 1]

    Returns
    -------
    np.ndarray : uint8 RGB image, shape (H, W, 3)
    """
    H_deg = np.degrees(H) % 360.0           # radians → [0, 360)
    Hi    = (H_deg / 60.0).astype(int) % 6  # sector index 0-5
    f     = (H_deg / 60.0) - (H_deg / 60.0).astype(int)

    p = V * (1.0 - S)
    q = V * (1.0 - f * S)
    t = V * (1.0 - (1.0 - f) * S)

    R = np.select([Hi == 0, Hi == 1, Hi == 2, Hi == 3, Hi == 4, Hi == 5],
                  [V,       q,       p,       p,       t,       V      ], default=V)
    G = np.select([Hi == 0, Hi == 1, Hi == 2, Hi == 3, Hi == 4, Hi == 5],
                  [t,       V,       V,       q,       p,       p      ], default=V)
    B = np.select([Hi == 0, Hi == 1, Hi == 2, Hi == 3, Hi == 4, Hi == 5],
                  [p,       p,       t,       V,       V,       q      ], default=V)

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)
