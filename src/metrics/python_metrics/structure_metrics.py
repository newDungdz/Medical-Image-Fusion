import numpy as np
from scipy import signal
from scipy.ndimage import sobel, convolve, uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import sobel

# ──────────────────────────────────────────────
# Structural Similarity (SSIM)
# ──────────────────────────────────────────────
# ---------------------------------------------------------------------------
# Gaussian kernel  (mirrors images.internal.createGaussianKernel)
# ---------------------------------------------------------------------------
 
def _create_gaussian_kernel(sigma, hsize):
    """
    Create a normalised 1-D Gaussian kernel (used for separable filtering).
 
    Parameters
    ----------
    sigma : float
    hsize : int
 
    Returns
    -------
    h : ndarray, shape (hsize,)
    """
    sigma  = float(sigma)
    hsize  = int(hsize)
    radius = (hsize - 1) / 2.0
    x      = np.arange(-radius, radius + 1)
    h      = np.exp(-0.5 * (x * x) / (sigma * sigma))
 
    # Suppress near-zero components (mirrors MATLAB)
    h[h < np.finfo(float).eps * h.max()] = 0.0
 
    s = h.sum()
    if s != 0:
        h = h / s
 
    return h
 
 
# ---------------------------------------------------------------------------
# imgaussfilt  (separable spatial path – sufficient for ssim)
# ---------------------------------------------------------------------------
def _scipy_pad_mode(padding):
    mapping = {'replicate': 'nearest', 'symmetric': 'reflect', 'circular': 'wrap'}
    if isinstance(padding, str):
        return mapping.get(padding, 'nearest')
    return 'constant'
 
 
def _imgaussfilt(A, sigma, filt_size, padding='replicate'):
    """
    2-D Gaussian filter – separable spatial implementation matching MATLAB.
 
    sigma     : scalar or 2-element [sigma_row, sigma_col]
    filt_size : scalar or 2-element [size_row, size_col]
    """
    sigma     = np.broadcast_to(np.atleast_1d(np.asarray(sigma,     dtype=float)), (2,)).copy()
    filt_size = np.broadcast_to(np.atleast_1d(np.asarray(filt_size, dtype=int)),   (2,)).copy()
 
    hcol = _create_gaussian_kernel(sigma[0], filt_size[0])  # column (vertical) kernel
    hrow = _create_gaussian_kernel(sigma[1], filt_size[1])  # row (horizontal) kernel
 
    pad_mode = _scipy_pad_mode(padding)
 
    # Separable convolution: apply hcol along axis-0, hrow along axis-1
    out = convolve(A.astype(float, copy=False), hcol[:, np.newaxis], mode=pad_mode)
    out = convolve(out,                          hrow[np.newaxis, :], mode=pad_mode)
    return out
 
 
# ---------------------------------------------------------------------------
# Core algorithm  (mirrors ssimalgo.m)
# ---------------------------------------------------------------------------
 
def _guarded_divide_and_exponent(num, den, C, exponent):
    if C > 0:
        component = num / den
    else:
        component = np.ones_like(num)
        nz = den != 0
        component[nz] = num[nz] / den[nz]
 
    if exponent != int(exponent):          # fractional exponent: clamp negatives
        component = np.maximum(component, 0.0)
 
    if exponent != 1:
        component = component ** exponent
 
    return component
 
 
def _ssimalgo(A, ref, gauss_fn, exponents, C, num_spatial_dims):
    """Direct translation of ssimalgo.m."""
 
    mux2 = gauss_fn(A)
    muy2 = gauss_fn(ref)
    muxy = mux2 * muy2
    mux2 = mux2 ** 2
    muy2 = muy2 ** 2
 
    sigmax2 = np.maximum(gauss_fn(A   ** 2) - mux2, 0.0)
    sigmay2 = np.maximum(gauss_fn(ref ** 2) - muy2, 0.0)
    sigmaxy = gauss_fn(A * ref) - muxy
 
    # Special case: equation 13 (Wang 2004)
    if C[2] == C[1] / 2 and np.array_equal(exponents, [1.0, 1.0, 1.0]):
        num = (2.0 * muxy + C[0]) * (2.0 * sigmaxy + C[1])
        den = (mux2 + muy2 + C[0]) * (sigmax2 + sigmay2 + C[1])
        if C[0] > 0 and C[1] > 0:
            ssimmap = num / den
        else:
            ssimmap = np.ones_like(A)
            nz = den != 0
            ssimmap[nz] = num[nz] / den[nz]
 
    else:
        # General case: equation 12
        ssimmap = np.ones_like(A) if exponents[0] == 0 else \
                  _guarded_divide_and_exponent(2.0 * muxy + C[0],
                                               mux2 + muy2 + C[0],
                                               C[0], exponents[0])
        sigmaxsigmay = None
        if exponents[1] > 0:
            sigmaxsigmay = np.sqrt(sigmax2 * sigmay2)
            ssimmap = ssimmap * _guarded_divide_and_exponent(
                2.0 * sigmaxsigmay + C[1], sigmax2 + sigmay2 + C[1], C[1], exponents[1])
 
        if exponents[2] > 0:
            if sigmaxsigmay is None:
                sigmaxsigmay = np.sqrt(sigmax2 * sigmay2)
            ssimmap = ssimmap * _guarded_divide_and_exponent(
                sigmaxy + C[2], sigmaxsigmay + C[2], C[2], exponents[2])
 
    # Mean over spatial dimensions (axes 0 and 1 for 2-D spatial)
    axis    = tuple(range(num_spatial_dims))
    ssimval = ssimmap.mean(axis=axis)
    return ssimval, ssimmap
 
 
# ---------------------------------------------------------------------------
# Dynamic-range helper
# ---------------------------------------------------------------------------
 
def _dynamic_range_from_dtype(dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.max - info.min)
    return 1.0          # float32 / float64
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def ssim(A, ref,
         dynamic_range=None,
         regularization_constants=None,
         exponents=None,
         radius=1.5):
    """
    Structural Similarity Index (SSIM).
 
    Python port of MATLAB's ``ssim(A, ref, ...)`` (Image Processing Toolbox).
 
    Parameters
    ----------
    A, ref : array-like
        Images to compare.  Must have identical shape and dtype.
        Supported dtypes: uint8, uint16, int16, float32, float64.
        2-D (grayscale) or 3-D (rows × cols × channels).
 
    dynamic_range : float, optional
        Peak signal range.  Defaults to the theoretical range of the dtype
        (255 for uint8, 65535 for uint16, 1.0 for floats).
 
    regularization_constants : sequence of 3 floats, optional
        [C1, C2, C3].  If omitted, follows MATLAB default:
        [(0.01·DR)², (0.03·DR)², (0.03·DR)²/2].
 
    exponents : sequence of 3 floats, optional
        Exponents for luminance, contrast, and structure terms.
        Default [1, 1, 1].
 
    radius : float, optional
        Standard deviation of the Gaussian weighting window.  Default 1.5.
 
    Returns
    -------
    ssimval : float or ndarray
        Mean SSIM.  Scalar for 2-D input;
        1-D array (one per channel) for 3-D input.
    ssimmap : ndarray
        Local SSIM map, same shape as the input.
 
    Notes
    -----
    * int16 inputs are offset by ``intmin('int16')`` (−32768) before
      processing, exactly as MATLAB does.
    * Other integer types are cast to float64.
    * The Gaussian filter uses 'replicate' (nearest-neighbour) boundary
      padding, matching MATLAB's default.
    """
    A   = np.asarray(A)
    ref = np.asarray(ref)
 
    if A.dtype != ref.dtype:
        raise TypeError("A and ref must have the same dtype "
                        f"(got {A.dtype} and {ref.dtype}).")
    if A.shape != ref.shape:
        raise ValueError("A and ref must have the same shape "
                         f"(got {A.shape} and {ref.shape}).")
    if A.ndim < 2 or A.ndim > 3:
        raise ValueError("Only 2-D and 3-D inputs are supported.")
 
    if exponents is None:
        exponents = np.array([1.0, 1.0, 1.0])
    else:
        exponents = np.asarray(exponents, dtype=float)
        if exponents.shape != (3,):
            raise ValueError("exponents must have exactly 3 elements.")
 
    if dynamic_range is None:
        dynamic_range = _dynamic_range_from_dtype(A.dtype)
    DR = float(dynamic_range)
 
    # --- dtype handling (mirrors ssimParseInputs.m) ---
    if A.dtype == np.int16:
        offset = float(np.iinfo(np.int16).min)   # -32768
        A   = A.astype(float) - offset
        ref = ref.astype(float) - offset
    elif np.issubdtype(A.dtype, np.integer):
        A   = A.astype(float)
        ref = ref.astype(float)
    else:
        A   = A.astype(float, copy=False)
        ref = ref.astype(float, copy=False)
 
    # --- regularisation constants ---
    if regularization_constants is None:
        C = np.array([(0.01 * DR) ** 2,
                      (0.03 * DR) ** 2,
                      (0.03 * DR) ** 2 / 2.0])
    else:
        C = np.asarray(regularization_constants, dtype=float)
        if C.shape != (3,):
            raise ValueError("regularization_constants must have exactly 3 elements.")
    # print(C)
    # --- filter size (mirrors ssimParseInputs.m) ---
    filt_radius = int(np.ceil(radius * 3))   # 3 std-devs cover >99 % of area
    filt_size   = 2 * filt_radius + 1
 
    num_spatial_dims = 2   # rows and columns are always spatial
 
    # For 3-D input MATLAB processes slices independently via the 2-D gauss filter
    if A.ndim == 3:
        n_channels = A.shape[2]
 
        def gauss_fn_3d(X):
            out = np.empty_like(X)
            for c in range(n_channels):
                out[:, :, c] = _imgaussfilt(X[:, :, c], radius, filt_size)
            return out
        gauss_fn = gauss_fn_3d
    else:
        gauss_fn = lambda X: _imgaussfilt(X, radius, filt_size)
 
    ssimval, ssimmap = _ssimalgo(A, ref, gauss_fn, exponents, C, num_spatial_dims)
    return ssimval
# ──────────────────────────────────────────────
# MS_SSIM  –  Multi-Scale SSIM for MEF
# ──────────────────────────────────────────────
def _mef_ssim(seq, fi, wsize=11, K=0.03):
    """
    Single-scale MEF-SSIM.
    seq : H×W×N stack of source images
    fi  : fused image
    """

    seq = seq.astype(np.float64)
    fi = fi.astype(np.float64)

    H, W, N = seq.shape
    bd = wsize // 2
    C = (K * 255) ** 2

    # mean of each source image
    mu = uniform_filter(seq, size=(wsize, wsize, 1))[bd:-bd, bd:-bd, :]

    # variance → edge strength
    sigma = uniform_filter(seq ** 2, size=(wsize, wsize, 1))[bd:-bd, bd:-bd, :] - mu ** 2
    ed = np.sqrt(np.maximum(wsize ** 2 * sigma, 0)) + 1e-3

    # gaussian window
    ax = np.arange(-5, 6)
    g = np.exp(-(ax ** 2) / (2 * 1.5 ** 2))
    gwin = np.outer(g, g)
    gwin /= gwin.sum()
    
    
    qmap = np.zeros((H - 2 * bd, W - 2 * bd))

    for i in range(bd, H - bd):
        for j in range(bd, W - bd):

            patch = seq[i-bd:i+bd+1, j-bd:j+bd+1, :]
            vecs = patch.reshape(wsize*wsize, N)

            mu_local = mu[i-bd, j-bd, :]
            ed_local = ed[i-bd, j-bd, :]

            # structure consistency
            denom = sum(np.linalg.norm(vecs[:,k] - mu_local[k]) for k in range(N))
            numerator = np.linalg.norm(vecs.sum(axis=1) - vecs.sum(axis=1).mean())

            R = (numerator + 1e-10) / (denom + 1e-10)
            R = np.clip(R, 1e-10, 1-1e-10)

            p = min(np.tan(np.pi/2 * R), 10)

            wk = (ed_local / wsize) ** p
            wk = wk / (wk.sum() + 1e-10)

            maxEd = ed_local.max()

            rblock = sum(
                wk[k] * (vecs[:,k] - mu_local[k]) / (ed_local[k] + 1e-10)
                for k in range(N)
            )

            rblock = rblock.reshape(wsize, wsize)

            nrm = np.linalg.norm(rblock)
            if nrm > 0:
                rblock = rblock / nrm * maxEd

            fblock = fi[i-bd:i+bd+1, j-bd:j+bd+1]

            rv = rblock.ravel()
            fv = fblock.ravel()
            gw = gwin.ravel()

            mu1 = (gw * rv).sum()
            mu2 = (gw * fv).sum()

            s1 = (gw * (rv - mu1) ** 2).sum()
            s2 = (gw * (fv - mu2) ** 2).sum()
            s12 = (gw * (rv - mu1) * (fv - mu2)).sum()

            qmap[i-bd, j-bd] = (2 * s12 + C) / (s1 + s2 + C)

    return qmap.mean()


def ms_ssim(img_seq, fI, K=0.03, level=3):
    """
    Multi-scale MEF-SSIM

    img_seq : H×W×N stack of source images
    fI      : fused image
    """

    weight = np.array([0.0448, 0.2856, 0.3001])
    weight = weight[:level]
    weight = weight / weight.sum()

    down = np.ones((2,2)) / 4

    img_seq = img_seq.astype(np.float64)
    fI = fI.astype(np.float64)
    Q = np.zeros(level)

    for l in range(level):

        Q[l] = _mef_ssim(img_seq, fI)

        if l < level-1:

            seq_new = np.zeros(
                ((img_seq.shape[0]+1)//2,
                 (img_seq.shape[1]+1)//2,
                 img_seq.shape[2])
            )

            for i in range(img_seq.shape[2]):
                d = signal.convolve2d(
                    img_seq[:,:,i],
                    down,
                    mode='same',
                    boundary='symm'
                )
                seq_new[:,:,i] = d[::2, ::2]

            img_seq = seq_new

            d = signal.convolve2d(fI, down, mode='same', boundary='symm')
            fI = d[::2, ::2]

    return float(np.prod(Q ** weight))

if __name__ == "__main__":
    import cv2
    def load_gray(path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    A = load_gray('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/3015.png')
    B = load_gray('data/AANLIB/MyDatasets/SPECT-MRI/test/SPECT/3015.png')
    F = load_gray('data/Fused_results/SPECT-MRI/ASFE-Fusion/3015.png')
    
    ssimval1 = ssim(A, F)
    ssimval2 = ssim(B, F)
    
    print("SSIM1:", ssimval1)
    print("SSIM2:", ssimval2)