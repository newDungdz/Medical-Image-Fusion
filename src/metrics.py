import numpy as np
from scipy import signal
from scipy.ndimage import sobel, convolve, uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import sobel

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
# SD  –  Standard Deviation
# ──────────────────────────────────────────────
def sd(F: np.ndarray) -> float:
    """Contrast via pixel intensity standard deviation."""
    F = F.astype(np.float64)
    m, n = F.shape
    u = F.mean()
    return float(np.sqrt(((F - u) ** 2).sum() / (m * n)))


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


# ──────────────────────────────────────────────
# EN  –  Entropy
# ──────────────────────────────────────────────
def en(F: np.ndarray, grey_level: int = 256) -> float:
    F = F.astype(np.int32).ravel()
    hist = np.bincount(F, minlength=grey_level)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

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


# ──────────────────────────────────────────────
# MI  –  Mutual Information
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


# ── Helper: 2-D 'valid' convolution (replicates MATLAB filter2 'valid') ───────
def _conv_valid(img: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Correlate img with win using 'valid' border handling (no padding)."""
    from scipy.signal import correlate2d
    return correlate2d(img, win, mode='valid')

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


def MS_SSIM(img_seq, fI, K=0.03, level=3):
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


# ──────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import cv2
    def load_gray(path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    image_idx = 1
    model = 'DenseFuse'
    
    A = load_gray(f"Image-Fusion/General Evaluation Metric/Image/Source-Image/TNO/ir/0{image_idx}.png")
    B = load_gray(f"Image-Fusion/General Evaluation Metric/Image/Source-Image/TNO/vi/0{image_idx}.png")
    F = load_gray(f"Image-Fusion/General Evaluation Metric/Image/Algorithm/{model}_TNO/0{image_idx}.png")

    # A = np.array([[10, 20, 30, 40],
    #               [50, 60, 70, 80],
    #               [90, 100, 110, 120]], dtype=np.float64)
    # B = np.array([[15, 25, 35, 45],
    #               [55, 65, 75, 85],
    #               [95, 105, 115, 125]], dtype=np.float64)
    # F = np.array([[12, 22, 32, 42],
    #               [52, 62, 72, 82],
    #               [92, 102, 112, 122]], dtype=np.float64)
    
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(A, cmap='gray')
    # axes[0].set_title('Source Image A (IR)')
    # axes[0].axis('off')
    # axes[1].imshow(B, cmap='gray')
    # axes[1].set_title('Source Image B (VI)')
    # axes[1].axis('off')
    # axes[2].imshow(F, cmap='gray')
    # axes[2].set_title('Fused Image F')
    # axes[2].axis('off')
    # plt.tight_layout()
    # plt.show()
    
    print("Image shapes:", A.shape, "Image range:", A.min(), "-", A.max(), "-", A.dtype)

    print(f"EN    : {en(F):.4f}")
    print(f"MI    : {mi(A, B, F):.4f}")
    print(f"SD    : {sd(F):.4f}")
    print(f"SF    : {sf(F):.4f}")
    print(f"MSE   : {mse(A, B, F):.4f}")
    print(f"PSNR  : {psnr(A, B, F):.4f}")
    print(f"VIF   : {vif(A, F):.4f}")
    print(f"AG    : {ag(F):.4f}")
    print(f"SCD   : {scd(A, B, F):.4f}")
    print(f"CC    : {cc(A, B, F):.4f}")
    print(f"QABF  : {qabf(A, B, F):.4f}")
    
    print(f"MS_SSIM: {MS_SSIM(np.stack([A, B], axis=2), F):.4f}")
