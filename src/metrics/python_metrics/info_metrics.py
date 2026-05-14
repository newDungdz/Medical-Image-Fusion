import numpy as np
from scipy.ndimage import sobel
from scipy.fft import dctn
import pywt
import cv2
from skimage.util import view_as_windows

EPS = 1e-12


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
# ! NCIE - Nonlinear Correlation Information Entropy
# ──────────────────────────────────────────────
#region NCIE - Nonlinear Correlation Information Entropy

def ncie(im1, im2, fim):
    """
    NCIE (Nonlinear Correlation Information Entropy) metric for image fusion.
    
    Parameters:
        im1  : First input image (numpy array)
        im2  : Second input image (numpy array)
        fim  : Fused image (numpy array)
    
    Returns:
        res  : NCIE metric value
    
    Reference: Performance evaluation of image fusion techniques, Chapter 19,
               pp.469-492, in Image Fusion: Algorithms and Applications, by Qiang Wang
    """
    im1 = _normalize(im1)
    im2 = _normalize(im2)
    fim = _normalize(fim)

    b = 256
    K = 3

    NCCxy = _NCC(im1, im2)
    NCCxf = _NCC(im1, fim)
    NCCyf = _NCC(im2, fim)

    R = np.array([
        [1,      NCCxy, NCCxf],
        [NCCxy,  1,     NCCyf],
        [NCCxf,  NCCyf, 1    ]
    ])

    eigenvalues = np.linalg.eigvals(R)

    # HR calculation
    HR = np.sum(eigenvalues * np.log2(eigenvalues / K) / K)
    HR = -HR / np.log2(b)

    NCIE = 1 - HR
    return NCIE.real  # eigenvalues are real for symmetric matrix, but cast just in case


def _NCC(im1, im2):
    """
    NCC (Nonlinear Correlation Coefficient) between two images.
    Similar to mutual information but normalized differently.
    
    Parameters:
        im1 : First image, values in range [0, 255]
        im2 : Second image, values in range [0, 255]
    
    Returns:
        res : NCC value
    """
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    N = 256
    b = 256

    # Joint histogram using numpy for efficiency
    h, _, _ = np.histogram2d(
        im1.ravel(), im2.ravel(),
        bins=N, range=[[0, N], [0, N]]
    )

    # Normalize to probability
    h = h / h.sum()

    im1_marg = h.sum(axis=0)   # marginal for im1 (sum columns)
    im2_marg = h.sum(axis=1)   # marginal for im2 (sum rows)

    # Entropy calculations (safe log: 0*log(0) = 0)
    H_x = -np.sum(im1_marg * np.log2(im1_marg + (im1_marg == 0)))
    H_y = -np.sum(im2_marg * np.log2(im2_marg + (im2_marg == 0)))
    H_xy = -np.sum(h * np.log2(h + (h == 0)))

    H_x  /= np.log2(b)
    H_y  /= np.log2(b)
    H_xy /= np.log2(b)

    return H_x + H_y - H_xy


def _normalize(data):
    """
    Normalize image data to [0, 255] integer range.
    
    Parameters:
        data : Input image (numpy array)
    
    Returns:
        Normalized image rounded to integers in [0, 255]
    """
    data = data.astype(np.float64)
    d_max = data.max()
    d_min = data.min()

    if d_max == 0 and d_min == 0:
        return data

    normalized = (data - d_min) / (d_max - d_min)
    return np.round(normalized * 255)

# ──────────────────────────────────────────────
# ! FMI  –  Feature Mutual Information 
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
    # print("Original:",im)
    if feature == "none":
        return im
 
    elif feature == "gradient":
        # MATLAB gradient() on 2-D returns x-gradient (along columns).
        _, gx = np.gradient(im)
        return gx
 
    elif feature == "edge":
        # Sobel magnitude — matches MATLAB edge(im) default (Sobel)
        sx = sobel(im, axis=1)/8
        sy = sobel(im, axis=0)/8
        # Mimic Matlab thresholding
        b = (sx**2 + sy**2)
        b= np.float64(b)
        cutoff = 4 * np.mean(b)
        # print('cutoff:', cutoff)
        edges = np.uint8(b > cutoff) 
        thinned_edges = cv2.ximgproc.thinning(edges * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        return thinned_edges
        
    elif feature == "dct":
        return dctn(im, norm="ortho")
 
    elif feature == "wavelet":

        cA, (cH, cV, cD) = pywt.dwt2(im, "dmey")
        # print(cH)
        combined = np.block([[cA, cH], [cV, cD]])
        return _rerange(combined)
 
    else:
        raise ValueError(
            f"Unknown feature '{feature}'. Choose from: "
            "'gradient', 'edge', 'dct', 'wavelet', 'none'."
        )
 
 
# Per-patch normalised mutual information
def _batch_patch_mi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Vectorized equivalent of _patch_mi applied to all patches at once.

    Parameters
    ----------
    x : (B, L) — all patches from one feature map, flattened (column-major to match MATLAB)
    y : (B, L) — all patches from another feature map, same layout

    Returns
    -------
    mi : (B,) normalised mutual information for each patch pair
    """
    B, L = x.shape

    # ── identical-patch shortcut (B,) bool mask ──────────────────────────────
    identical = np.all(x == y, axis=1)

    # ── per-patch normalisation → marginal PDF ──────────────────────────────
    def _to_pdf(p: np.ndarray) -> np.ndarray:
        mn = p.min(axis=1, keepdims=True)
        mx = p.max(axis=1, keepdims=True)
        flat = np.where(mx == mn, np.ones_like(p), (p - mn) / (mx - mn + EPS))
        s = flat.sum(axis=1, keepdims=True)
        return flat / np.where(s == 0, 1.0, s)

    xPdf = _to_pdf(x)   # (B, L)
    yPdf = _to_pdf(y)   # (B, L)

    # ── CDFs ────────────────────────────────────────────────────────────────
    xCdf = np.cumsum(xPdf, axis=1)   # (B, L)
    yCdf = np.cumsum(yPdf, axis=1)

    # ── Pearson r between marginal PDFs ─────────────────────────────────────
    xm = xPdf - xPdf.mean(axis=1, keepdims=True)
    ym = yPdf - yPdf.mean(axis=1, keepdims=True)
    dot  = (xm * ym).sum(axis=1)                              # (B,)
    norm = np.sqrt((xm**2).sum(axis=1) * (ym**2).sum(axis=1))
    c    = np.where(norm == 0, 0.0, dot / norm)               # (B,)

    # ── population std-devs (index-weighted) ────────────────────────────────
    idx = np.arange(1, L + 1, dtype=np.float64)               # (L,)
    ex  = (idx       * xPdf).sum(axis=1)
    ex2 = (idx**2    * xPdf).sum(axis=1)
    ey  = (idx       * yPdf).sum(axis=1)
    ey2 = (idx**2    * yPdf).sum(axis=1)
    xSd = np.sqrt(np.maximum(ex2 - ex**2, 0.0))               # (B,)
    ySd = np.sqrt(np.maximum(ey2 - ey**2, 0.0))

    # ── 2-D CDF grids broadcast-free using pre-expanded slices ───────────────
    # xCdf[:, :, None] → (B, L, 1), yCdf[:, None, :] → (B, 1, L)
    xC  = xCdf[:, :,  None]    # (B, L,   1)  i
    yC  = yCdf[:, None, :]     # (B, 1,   L)  j
    xCm = xCdf[:, :-1, None]   # (B, L-1, 1)  i-1
    yCm = yCdf[:, None, :-1]   # (B, 1, L-1)  j-1
    def _accum_H(jpdf: np.ndarray) -> np.ndarray:
        """Sum -p*log2|p| over last two axes → (B,).  Mirrors _accum_entropy."""
        
        mask = jpdf != 0
        out  = np.zeros(B, dtype=np.float64)
        # safe log2: only evaluate where mask is true
        safe = np.where(mask, np.abs(jpdf), 1.0)
        out  = -(np.where(mask, jpdf, 0.0) * np.log2(safe)).sum(axis=(-2, -1))
        # print(out)
        return out

    def _joint_entropy_upper_batch(phi: np.ndarray) -> np.ndarray:
        """Fréchet upper-bound copula, batched. phi: (B,)"""
        ph = phi[:, None, None]   # broadcast over (L, L) grids

        def _min(a, b): return 0.5 * (a + b - np.abs(a - b))

        mFG   = _min(xC,  yC)     # (B, L,   L  )
        mFGim = _min(xCm, yC)     # (B, L-1, L  )
        mFGjm = _min(xC,  yCm)    # (B, L,   L-1)
        mFGij = _min(xCm, yCm)    # (B, L-1, L-1)

        H = np.zeros(B)

        # (0,0) corner
        jp = ph[:, 0, 0] * mFG[:, 0, 0] + (1 - phi) * xPdf[:, 0] * yPdf[:, 0]
        pos = jp > 0
        H[pos] += (-jp[pos] * np.log2(jp[pos])).real

        # i-boundary (i>=1, j=0): shape (B, L-1)
        up  = mFG[:, 1:, 0] - mFGim[:, :, 0]
        jp_ = ph[:, :, 0] * up + (1 - ph[:, :, 0]) * xPdf[:, 1:] * yPdf[:, 0:1]
        H  += _accum_H(jp_[:, :, None])[:] * 0  # placeholder — reshape to (B,L-1,1)

        # easier: accumulate over axis=-1 only, keep (B, L-1) flat
        def _H1d(arr):
            mask = arr != 0
            safe = np.where(mask, np.abs(arr), 1.0)
            return -(np.where(mask, arr, 0.0) * np.log2(safe)).sum(axis=-1)

        # redo with 1-D accumulation
        H = np.zeros(B)
        jp = mFG[:, 0, 0] * phi + (1 - phi) * xPdf[:, 0] * yPdf[:, 0]
        pos = jp > 0
        H[pos] += (-jp[pos] * np.log2(jp[pos]))

        up   = mFG[:, 1:, 0] - mFGim[:, :, 0]               # (B, L-1)
        jp_  = ph[:, :, 0] * up + (1-ph[:, :, 0]) * xPdf[:, 1:] * yPdf[:, :1]
        H   += _H1d(jp_)

        up   = mFG[:, 0, 1:] - mFGjm[:, 0, :]               # (B, L-1)
        jp_  = ph[:, 0, :] * up + (1-ph[:, 0, :]) * xPdf[:, :1] * yPdf[:, 1:]
        H   += _H1d(jp_)

        up   = mFG[:,1:,1:] - mFGim[:,:,1:] - mFGjm[:,1:,:] + mFGij  # (B,L-1,L-1)
        jp_  = ph * up + (1-ph) * xPdf[:, 1:, None] * yPdf[:, None, 1:]
        H   += _accum_H(jp_)

        return H

    def _joint_entropy_lower_batch(theta: np.ndarray) -> np.ndarray:
        """Fréchet lower-bound copula, batched. theta: (B,)"""
        th = theta[:, None, None]

        def _max(a, b): return 0.5 * (a + b - 1 + np.abs(a + b - 1))

        mFG   = _max(xC,  yC)
        mFGim = _max(xCm, yC)
        mFGjm = _max(xC,  yCm)
        mFGij = _max(xCm, yCm)

        def _H1d(arr):
            mask = arr != 0
            safe = np.where(mask, np.abs(arr), 1.0)
            return -(np.where(mask, arr, 0.0) * np.log2(safe)).sum(axis=-1)

        H = np.zeros(B)
        jp = mFG[:, 0, 0] * theta + (1-theta) * xPdf[:, 0] * yPdf[:, 0]

        nz = jp != 0
        H[nz] += -jp[nz] * np.log2(np.abs(jp[nz]))

        lo  = mFG[:, 0, 1:] - mFGjm[:, 0, :]
        jp_ = th[:, 0, :] * lo + (1-th[:, 0, :]) * xPdf[:, :1] * yPdf[:, 1:]
        H  += _H1d(jp_)

        lo  = mFG[:, 1:, 0] - mFGim[:, :, 0]
        jp_ = th[:, :, 0] * lo + (1-th[:, :, 0]) * xPdf[:, 1:] * yPdf[:, :1]
        H  += _H1d(jp_)

        lo  = mFG[:,1:,1:] - mFGim[:,:,1:] - mFGjm[:,1:,:] + mFGij
        jp_ = th * lo + (1-th) * xPdf[:,1:,None] * yPdf[:,None,1:]
        H  += _accum_H(jp_)

        return H

    # ── route each patch to upper or lower copula ────────────────────────────
    pos_mask = c >= 0    # (B,)

    # phi for positive-c patches
    xC2d = xCdf[:, :, None]   # reuse
    yC2d = yCdf[:, None, :]
    def _min2(a, b): return 0.5 * (a + b - np.abs(a - b))
    def _max2(a, b): return 0.5 * (a + b - 1 + np.abs(a + b - 1))

    # ── phi (upper copula) ───────────────────────────────────────────────────────
    covUp  = (_min2(xC2d, yC2d) - xC2d * yC2d).sum(axis=(-2, -1))
    sd_prod = xSd * ySd

    # Step 1: safe corrUp
    corrUp = np.zeros_like(covUp)
    valid_corr = sd_prod != 0
    corrUp[valid_corr] = covUp[valid_corr] / sd_prod[valid_corr]

    # Step 2: safe phi (match scalar logic exactly)
    phi = np.zeros_like(c)
    valid_phi = (c != 0) & (xSd != 0) & (ySd != 0) & (corrUp != 0)
    phi[valid_phi] = c[valid_phi] / corrUp[valid_phi]

    # ── theta (lower copula) ─────────────────────────────────────────────────────
    covLo  = (_max2(xC2d, yC2d) - xC2d * yC2d).sum(axis=(-2, -1))
    sd_prod = xSd * ySd
        
    corrLo = np.zeros_like(covLo)
    valid_corr = sd_prod != 0
    corrLo[valid_corr] = covLo[valid_corr] / sd_prod[valid_corr]    
    theta = np.zeros_like(c)
    
    valid_theta = (xSd != 0) & (ySd != 0) & (corrLo != 0)
    theta[valid_theta] = c[valid_theta] / corrLo[valid_theta]
    # compute both branches, select by mask (avoids conditionals over B)
    H_upper = _joint_entropy_upper_batch(phi)
    H_lower = _joint_entropy_lower_batch(theta)
    jointH  = np.where(pos_mask, H_upper, H_lower)   # (B,)

    # ── marginal entropies ───────────────────────────────────────────────────
    def _marginal_H(pdf):
        mask = pdf > 0
        safe = np.where(mask, pdf, 1.0)
        return -(np.where(mask, pdf, 0.0) * np.log2(safe)).sum(axis=1)   # (B,)

    xH = _marginal_H(xPdf)
    yH = _marginal_H(yPdf)

    # ── normalised MI ─────────────────────────────────────────────────────────────
    mi_val = xH + yH - jointH
    denom  = xH + yH
    nmi    = np.where(
        (mi_val == 0) | (denom == 0),
        0.0,
        mi_val / np.where(denom == 0, 1.0, denom) * 2.0
    )
    # apply identical-patch override
    nmi = np.where(identical, 1.0, nmi)
    return nmi

def fmi(
    ima: np.ndarray,
    imb: np.ndarray,
    imf: np.ndarray,
    feature: str = "none",
    w: int = 3,
) -> float:
    if ima.shape != imb.shape or ima.shape != imf.shape:
        raise ValueError("All images must have the same shape.")

    ima = ima.astype(np.float64)
    imb = imb.astype(np.float64)
    imf = imf.astype(np.float64)

    aFeat = _extract_feature(ima, feature)
    bFeat = _extract_feature(imb, feature)
    fFeat = _extract_feature(imf, feature)

    # Guard: all features must be 2D arrays after extraction
    for name, arr in [("aFeat", aFeat), ("bFeat", bFeat), ("fFeat", fFeat)]:
        if arr.ndim != 2:
            raise ValueError(
                f"{name} has ndim={arr.ndim} after feature='{feature}' extraction; "
                f"expected 2D. Shape: {arr.shape}"
            )

    hw    = int(np.floor(w / 2))
    wsize = 2 * hw + 1

    # view_as_windows: (M-2hw, N-2hw, wsize, wsize)
    # skimage's view_as_windows is safer than numpy's sliding_window_view
    # for non-contiguous or oddly-strided arrays (e.g. wavelet output)
    aFeat = np.ascontiguousarray(aFeat)
    bFeat = np.ascontiguousarray(bFeat)
    fFeat = np.ascontiguousarray(fFeat)

    aW = view_as_windows(aFeat, (wsize, wsize))   # (M, N, wsize, wsize)
    bW = view_as_windows(bFeat, (wsize, wsize))
    fW = view_as_windows(fFeat, (wsize, wsize))

    M, N = aW.shape[:2]
    B    = M * N

    def _prep(W):
        # (M, N, wsize, wsize) → (B, wsize*wsize), column-major patch flatten
        return W.reshape(B, wsize, wsize).reshape(B, wsize * wsize, order='F')

    aP = _prep(aW)
    bP = _prep(bW)
    fP = _prep(fW)

    fmi_af = _batch_patch_mi(aP, fP)
    fmi_bf = _batch_patch_mi(bP, fP)

    fmi_map = ((fmi_af + fmi_bf) / 2.0).reshape(M, N)
    return float(np.nanmean(fmi_map))

# endregion


# ──────────────────────────────────────────────
# SCD  –  Sum of Correlations of Differences
# ──────────────────────────────────────────────
def scd(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """Fusion quality via correlation of residual differences."""
    def _corr2(X, Y):
        Xm, Ym = X - X.mean(), Y - Y.mean()

        denom = np.sqrt((Xm * Xm).sum() * (Ym * Ym).sum())
        return float((Xm * Ym).sum() / (denom + 1e-10))
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    F = F.astype(np.float64)
    # print(_corr2(F - B, A), _corr2(F - A, B))
    return float(_corr2(F - B, A) + _corr2(F - A, B))

if __name__ == "__main__":
    import cv2
    from PIL import Image
    from time import time
    def load_gray(path):
        return np.array(Image.open(path).convert("L"))
    A = load_gray('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/4010.png')
    B = load_gray('data/AANLIB/MyDatasets/SPECT-MRI/test/SPECT/4010.png')
    F = load_gray('data/Fused_results/SPECT-MRI/ASFE-Fusion/4010.png')
    I_1 = np.array([
        [80, 20, 85],
        [75, 25, 78],
        [80, 22, 88]
    ], dtype=np.float64)

    I_2 = np.array([
        [30, 110, 35],
        [28, 120, 32],
        [26, 115, 30]
    ], dtype=np.float64)

    I_F = np.array([
        [58, 70, 60],
        [55, 78, 57],
        [62, 75, 61]
    ], dtype=np.float64)
    print(ncie(I_1, I_2, I_F))

    # print("SCD:", scd(A, B, F))