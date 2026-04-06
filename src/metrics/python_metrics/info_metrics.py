import numpy as np
from scipy import signal
from scipy.ndimage import sobel, convolve, uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import sobel
from scipy.fft import dctn

# ──────────────────────────────────────────────
# ! EN  –  Entropy
# ──────────────────────────────────────────────
def en(F: np.ndarray, grey_level: int = 256) -> float:
    F = F.astype(np.int32).ravel()
    hist = np.bincount(F, minlength=grey_level)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ──────────────────────────────────────────────
# ! MI  –  Mutual Information
# ──────────────────────────────────────────────
def _joint_entropy(A: np.ndarray, B: np.ndarray, grey_level: int = 256) -> float:
    A = A.astype(np.int32).ravel()
    B = B.astype(np.int32).ravel()

    joint = A * grey_level + B
    hist = np.bincount(joint, minlength=grey_level * grey_level)

    p = hist / hist.sum()
    p = p[p > 0]

    return float(-(p * np.log2(p)).sum())


def mi(A: np.ndarray, B: np.ndarray, F: np.ndarray, grey_level: int = 256) -> float:
    """Total mutual information between fused and both source images."""
    ha = en(A, grey_level)
    hb = en(B, grey_level)
    hf = en(F, grey_level)
    hfa = _joint_entropy(F, A, grey_level)
    hfb = _joint_entropy(F, B, grey_level)
    mifa = ha + hf - hfa
    mifb = hb + hf - hfb
    return float(mifa + mifb)

# ──────────────────────────────────────────────
# ! FMI  –  Feature Mutual Information (Missing)
# ──────────────────────────────────────────────
#region FMI  –  Feature Mutual Information

# Helpers
def _rerange(im: np.ndarray) -> np.ndarray:
    """Rescale image values to [0, 1]. Matches MATLAB rerange()."""
    im = im.astype(np.float64)
    mn, mx = im.min(), im.max()
    if mx == mn:
        return np.ones_like(im)
    return (im - mn) / (mx - mn)
 
 
def _extract_feature(im: np.ndarray, feature: str) -> np.ndarray:
    """Apply the requested feature extraction to a 2-D image."""
    im = im.astype(np.float64)
 
    if feature == "none":
        return im
 
    elif feature == "gradient":
        # MATLAB gradient() on 2-D returns x-gradient (along columns).
        _, gx = np.gradient(im)
        return gx
 
    elif feature == "edge":
        # Sobel magnitude — matches MATLAB edge(im) default (Sobel)
        sx = sobel(im, axis=1)
        sy = sobel(im, axis=0)
        return np.hypot(sx, sy)
 
    elif feature == "dct":
        return dctn(im, norm="ortho")
 
    elif feature == "wavelet":
        try:
            import pywt
        except ImportError:
            raise ImportError(
                "PyWavelets (pywt) is required for the 'wavelet' feature. "
                "Install it with:  pip install PyWavelets"
            )
        cA, (cH, cV, cD) = pywt.dwt2(im, "dmey")
        combined = np.block([[cA, cH], [cV, cD]])
        return _rerange(combined)
 
    else:
        raise ValueError(
            f"Unknown feature '{feature}'. Choose from: "
            "'gradient', 'edge', 'dct', 'wavelet', 'none'."
        )
 
 
# Per-patch normalised mutual information
def _patch_mi(sub_x: np.ndarray, sub_y: np.ndarray) -> float:
    """
    Compute normalised mutual information between two (w x w) patches.
    Matches the inner loop logic of the MATLAB FMI implementation exactly.
    Returns a value in [0, 1].
    """
    # Identical patches -> perfect transfer
    if np.array_equal(sub_x, sub_y):
        return 1.0
 
    l = sub_x.size     # (2*hw + 1)^2
 
    # Normalise each patch to a marginal PDF
    def _to_pdf(patch: np.ndarray) -> np.ndarray:
        mn, mx = patch.min(), patch.max()
        p = np.ones(l, dtype=np.float64) if mx == mn else \
            ((patch.ravel() - mn) / (mx - mn))
        s = p.sum()
        return p / s if s != 0 else p
 
    xPdf = _to_pdf(sub_x)
    yPdf = _to_pdf(sub_y)
 
    # PDF -> CDF
    xCdf = np.cumsum(xPdf)
    yCdf = np.cumsum(yPdf)
 
    # Pearson correlation between marginal PDFs
    xTemp = xPdf - xPdf.mean()
    yTemp = yPdf - yPdf.mean()
    dot   = (xTemp * yTemp).sum()
    norm  = np.sqrt((xTemp ** 2).sum() * (yTemp ** 2).sum())
    c     = 0.0 if norm == 0 else float(dot / norm)
 
    # Population std-devs (index-weighted, matching MATLAB)
    idx  = np.arange(1, l + 1, dtype=np.float64)
    ex   = (idx       * xPdf).sum()
    ex2  = (idx ** 2  * xPdf).sum()
    ey   = (idx       * yPdf).sum()
    ey2  = (idx ** 2  * yPdf).sum()
    xSd  = np.sqrt(max(ex2 - ex ** 2, 0.0))
    ySd  = np.sqrt(max(ey2 - ey ** 2, 0.0))
 
    # 2-D CDF grids (l x l)
    xC  = xCdf[:, None]        # (l, 1) rows = i
    yC  = yCdf[None, :]        # (1, l) cols = j
    xCm = xCdf[:-1, None]      # (l-1, 1)  i-1 shifted
    yCm = yCdf[None, :-1]      # (1, l-1)  j-1 shifted
 
    def _accum_entropy(H: float, jpdf_arr: np.ndarray) -> float:
        """Add -p*log2(p) for every strictly positive element."""
        pos = jpdf_arr > 0
        if not np.any(pos):
            return H
        v = jpdf_arr[pos]
        H += float(np.real((-v * np.log2(v)).sum()))
        return H
 
    def _joint_entropy_upper(phi: float) -> float:
        """Frechet upper-bound copula blend -> joint entropy."""
        def _minFG(a, b):
            return 0.5 * (a + b - np.abs(a - b))
 
        mFG   = _minFG(xC,  yC)    # (l, l)
        mFGim = _minFG(xCm, yC)    # (l-1, l)
        mFGjm = _minFG(xC,  yCm)   # (l, l-1)
        mFGij = _minFG(xCm, yCm)   # (l-1, l-1)
 
        H = 0.0
 
        # (0,0) corner
        jpdf = phi * mFG[0, 0] + (1 - phi) * xPdf[0] * yPdf[0]
        if jpdf > 0:
            H += float(np.real(-jpdf * np.log2(jpdf)))
 
        # i-boundary (i >= 1, j = 0)
        up    = mFG[1:, 0] - mFGim[:, 0]
        jpdf_ = phi * up + (1 - phi) * xPdf[1:] * yPdf[0]
        H     = _accum_entropy(H, jpdf_)
 
        # j-boundary (i = 0, j >= 1)
        up    = mFG[0, 1:] - mFGjm[0, :]
        jpdf_ = phi * up + (1 - phi) * xPdf[0] * yPdf[1:]
        H     = _accum_entropy(H, jpdf_)
 
        # interior (i >= 1, j >= 1)
        up    = mFG[1:, 1:] - mFGim[:, 1:] - mFGjm[1:, :] + mFGij
        jpdf_ = phi * up + (1 - phi) * xPdf[1:, None] * yPdf[None, 1:]
        H     = _accum_entropy(H, jpdf_)
 
        return H
 
    def _joint_entropy_lower(theta: float) -> float:
        """Frechet lower-bound copula blend -> joint entropy."""
        def _maxFG(a, b):
            return 0.5 * (a + b - 1 + np.abs(a + b - 1))
 
        mFG   = _maxFG(xC,  yC)
        mFGim = _maxFG(xCm, yC)
        mFGjm = _maxFG(xC,  yCm)
        mFGij = _maxFG(xCm, yCm)
 
        H = 0.0
 
        jpdf = theta * mFG[0, 0] + (1 - theta) * xPdf[0] * yPdf[0]
        if jpdf > 0:
            H += float(np.real(-jpdf * np.log2(jpdf)))
 
        lo    = mFG[1:, 0] - mFGim[:, 0]
        jpdf_ = theta * lo + (1 - theta) * xPdf[1:] * yPdf[0]
        H     = _accum_entropy(H, jpdf_)
 
        lo    = mFG[0, 1:] - mFGjm[0, :]
        jpdf_ = theta * lo + (1 - theta) * xPdf[0] * yPdf[1:]
        H     = _accum_entropy(H, jpdf_)
 
        lo    = mFG[1:, 1:] - mFGim[:, 1:] - mFGjm[1:, :] + mFGij
        jpdf_ = theta * lo + (1 - theta) * xPdf[1:, None] * yPdf[None, 1:]
        H     = _accum_entropy(H, jpdf_)
 
        return H
 
    # Compute joint entropy using the appropriate Frechet bound
    if c >= 0:
        if c == 0 or xSd == 0 or ySd == 0:
            phi = 0.0
        else:
            covUp  = (0.5 * (xC + yC - np.abs(xC - yC)) - xC * yC).sum()
            corrUp = covUp / (xSd * ySd)
            phi    = float(c / corrUp) if corrUp != 0 else 0.0
        jointEntropy = _joint_entropy_upper(phi)
 
    else:
        if xSd == 0 or ySd == 0:
            theta = 0.0
        else:
            covLo  = (0.5 * (xC + yC - 1 + np.abs(xC + yC - 1)) - xC * yC).sum()
            corrLo = covLo / (xSd * ySd)
            theta  = float(c / corrLo) if corrLo != 0 else 0.0
        jointEntropy = _joint_entropy_lower(theta)
 
    # Marginal entropies
    mx_  = xPdf > 0
    xH   = float(-(xPdf[mx_] * np.log2(xPdf[mx_])).sum())
    my_  = yPdf > 0
    yH   = float(-(yPdf[my_] * np.log2(yPdf[my_])).sum())
 
    # Normalised MI
    mi    = xH + yH - jointEntropy
    denom = xH + yH
    if mi == 0 or denom == 0:
        return 0.0
    return float(2.0 * mi / denom)
 
 
# Main function 
def FMI_metrics(
    ima: np.ndarray,
    imb: np.ndarray,
    imf: np.ndarray,
    feature: str = "edge",
    w: int = 3,
) -> float:
    """
    Compute the Feature Mutual Information (FMI) score for image fusion.
 
    Parameters
    ----------
    ima     : First source image  (H x W), uint8 or float.
    imb     : Second source image (H x W), same shape as ima.
    imf     : Fused image         (H x W), same shape as ima.
    feature : Feature extraction — one of:
                'none'      raw pixels (no extraction)
                'gradient'  x-direction central-difference gradient
                'edge'      Sobel magnitude                 [default]
                'dct'       2-D orthonormal DCT
                'wavelet'   discrete Meyer wavelet (requires PyWavelets)
    w       : Window size passed exactly as in MATLAB (default 3).
              Internally converted to half-width: hw = floor(w/2).
 
    Returns
    -------
    nfmi : float
        Normalised Feature Mutual Information in [0, 1].
        Higher means more source information was preserved in the fused image.
 
    Examples
    --------
    >>> import numpy as np
    >>> A = np.random.rand(128, 128) * 255
    >>> B = np.random.rand(128, 128) * 255
    >>> F = 0.5 * A + 0.5 * B
    >>> score = FMI_metrics(A, B, F, feature='edge', w=3)
    >>> print(f"FMI = {score:.4f}")
    """
    if ima.shape != imb.shape:
        raise ValueError("Source images must have the same shape.")
    if ima.shape != imf.shape:
        raise ValueError("Source and fused images must have the same shape.")
 
    ima = ima.astype(np.float64)
    imb = imb.astype(np.float64)
    imf = imf.astype(np.float64)
 
    aFeat = _extract_feature(ima, feature)
    bFeat = _extract_feature(imb, feature)
    fFeat = _extract_feature(imf, feature)
 
    # MATLAB does:  w = floor(w/2)  before the sliding-window loop
    hw = int(np.floor(w / 2))
 
    m, n = aFeat.shape
    fmi_map = np.ones((m - 2 * hw, n - 2 * hw), dtype=np.float64)
 
    for p in range(hw, m - hw):
        for q in range(hw, n - hw):
            aSub = aFeat[p - hw: p + hw + 1, q - hw: q + hw + 1]
            bSub = bFeat[p - hw: p + hw + 1, q - hw: q + hw + 1]
            fSub = fFeat[p - hw: p + hw + 1, q - hw: q + hw + 1]
 
            fmi_af = _patch_mi(aSub.copy(), fSub.copy())
            fmi_bf = _patch_mi(bSub.copy(), fSub.copy())
 
            fmi_map[p - hw, q - hw] = (fmi_af + fmi_bf) / 2.0
 
    return float(np.nanmean(fmi_map))

# endregion


# ──────────────────────────────────────────────
# SCD  –  Sum of Correlations of Differences
# ──────────────────────────────────────────────
def scd(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """Fusion quality via correlation of residual differences."""
    def _corr2(X, Y):
        X, Y = X.astype(np.float64), Y.astype(np.float64)
        Xm, Ym = X - X.mean(), Y - Y.mean()
        denom = np.sqrt((Xm ** 2).sum() * (Ym ** 2).sum())
        return float((Xm * Ym).sum() / (denom + 1e-10))
    return float(_corr2(F - B, A) + _corr2(F - A, B))
