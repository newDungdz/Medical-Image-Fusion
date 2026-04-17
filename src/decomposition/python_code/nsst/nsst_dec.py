import numpy as np
from scipy.signal import convolve2d
from shearing_filters_myer import shearing_filters_myer
from processing_ultis import stats, symext, upsample2df, atrousc
from atrousfilters import atrousfilters


def atrousdec(x, fname, Nlevels):
    """
    2-D nonsubsampled (undecimated) à trous wavelet decomposition.

    Decomposes image x into Nlevels + 1 subbands — one lowpass and
    Nlevels bandpass (detail) subbands — all at full resolution.

    Parameters
    ----------
    x       : ndarray, shape (m, n)  — input image
    fname   : str                    — filter name (see atrousfilters)
    Nlevels : int                    — number of decomposition levels

    Returns
    -------
    y : list of length (Nlevels + 1)
        y[0]          : coarsest lowpass subband
        y[1], …, y[N] : detail subbands, coarse → fine  (y[N] is finest)
    """
    h0, h1, g0, g1 = atrousfilters(fname)

    y = [None] * (Nlevels + 1)

    # ── Level 1 (finest detail) — plain convolution ─────────────────────────
    shift = np.array([1, 1])
    ext_x_h0 = symext(x, h0, shift)
    ext_x_h1 = symext(x, h1, shift)
    y0 = convolve2d(ext_x_h0, h0, mode='valid')
    y1 = convolve2d(ext_x_h1, h1, mode='valid')

    y[Nlevels] = y1      # finest detail  (y{Nlevels+1} in MATLAB, 1-based)
    x_curr = y0
    I2 = np.eye(2)

    # ── Levels 2 … Nlevels — à trous convolution ────────────────────────────
    for i in range(1, Nlevels):            # i = 1 : Nlevels-1  (MATLAB)
        L     = 2 ** i
        shift = -2 ** (i - 1) * np.array([1, 1]) + 2   # delay compensation

        h0_up = upsample2df(h0, i)
        h1_up = upsample2df(h1, i)

        y0 = atrousc(symext(x_curr, h0_up, shift), h0, I2 * L)
        y1 = atrousc(symext(x_curr, h1_up, shift), h1, I2 * L)

        y[Nlevels - i] = y1    # coarser detail levels (y{Nlevels-i+1} MATLAB)
        x_curr = y0

    y[0] = x_curr              # coarsest lowpass
    return y


def nsst_dec(x, shear_parameters, lpfilt):
    """
    Nonsubsampled Shearlet Transform (NSST) decomposition.

    Combines an à trous Laplacian Pyramid with Meyer directional filters
    to separate the image by both scale and orientation — at full resolution.

    Parameters
    ----------
    x : ndarray, shape (m, n)
        Input image (grayscale, float preferred).

    shear_parameters : dict with keys:
        'dcomp' : list of int, length = number_of_levels
            dcomp[i] means level i+1 has 2^dcomp[i] directional subbands.
        'dsize' : list of int, same length as dcomp
            Spatial support size of the shearing filter at each level.

    lpfilt : str
        Filter name for the lowpass/highpass pyramid (see atrousfilters).


    Returns
    -------
    dst : list of length (number_of_levels + 1)
        dst[0]            : ndarray (m, n)     — coarsest lowpass subband
        dst[i]            : ndarray (m, n, K)  — level i detail subbands
                            where K = 2^dcomp[i-1] is the number of directions

    shear_f : list of length number_of_levels
        shear_f[i] : ndarray (dsize[i], dsize[i], 2^dcomp[i])
                     Precomputed shearing filters for level i.
                     Can be reused on other images of the same size.

    """
    x = np.asarray(x, dtype=float)

    level = len(shear_parameters['dcomp'])

    # ── Step 1: à trous Laplacian Pyramid decomposition ─────────────────────
    y = atrousdec(x, lpfilt, level)   # list of (level+1) full-res subbands

    # ── Step 2: directional filtering of each detail subband ────────────────
    dst     = [None] * (level + 1)
    shear_f = [None] * level

    dst[0] = y[0]   # coarsest lowpass passed through directly
    stats(y[0], 'y')
    for i in range(level):
        dcomp_i = shear_parameters['dcomp'][i]
        dsize_i = shear_parameters['dsize'][i]
        n_dirs  = 2 ** dcomp_i

        # Build (or retrieve) shearing filters for this level
        # Scaled by sqrt(dsize) to normalise energy across directions
        shear_f[i] = (
            shearing_filters_myer(dsize_i, dcomp_i) * np.sqrt(dsize_i)
        )

        print()
        # Apply each directional filter to the i-th detail subband
        detail = np.zeros((*x.shape, n_dirs))
        for k in range(n_dirs):
            detail[:, :, k] = convolve2d(
                y[i + 1], shear_f[i][:, :, k], mode='same'
            )
            stats(detail[:, :, k], f"Level {i+1} detail dir {k+1}")
        print()
        dst[i + 1] = detail

    return dst, shear_f

if __name__ == "__main__": 
    import cv2
    import matplotlib.pyplot as plt

    # Custom test image: smooth gradient + a central impulse

    img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    # N = 63
    # img = np.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    #         img[i, j] = i + j  # smooth gradient
    # img[N//2, N//2] += 10   
    params = {'dcomp': [2, 2, 2], 'dsize': [16, 16, 16]}

    lpfilt = '9-7'   # or gaussian if error

    # print("Decomposing the image using NSST...")
    dst, shear_f = nsst_dec(img, params, lpfilt=lpfilt)
    # print("Reconstructing the image from NSST coefficients...")
    # print("\n\n")
    # stats(dst[0], "dst[0] - lowpass")
    # count = 1
    # for band in dst[1:]:
    #     for d in range(band.shape[2]):
    #         stats(band[:, :, d], f"dst {count} dir {d+1}")
    #     count += 1
        