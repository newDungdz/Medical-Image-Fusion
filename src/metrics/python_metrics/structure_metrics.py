import numpy as np
from scipy import signal
from scipy.ndimage import sobel, convolve, uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import sobel

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
