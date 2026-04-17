import numpy as np
from atrousfilters import atrousfilters
from processing_ultis import symext, upsample2df, atrousc
from scipy.signal import convolve2d

def atrousrec(y, fname):
    """
    Inverse 2-D atrous decomposition (NS pyramid reconstruction)

    Parameters
    ----------
    y : list of ndarray
        Decomposition coefficients:
        y[0] = coarse (lowpass)
        y[1:] = detail bands (from coarse → fine order)

    fname : str
        Filter name (passed to atrousfilters)

    Returns
    -------
    x : ndarray
        Reconstructed image
    """

    Nlevels = len(y) - 1

    # Load filters
    h0, h1, g0, g1 = atrousfilters(fname)

    # ------------------------------------------------------------
    # First Nlevels - 1 levels (coarse → fine reconstruction)
    # ------------------------------------------------------------
    x = y[0]

    I2 = np.eye(2)

    for i in range(Nlevels - 1, 0, -1):
        # MATLAB: y{Nlevels-i+1}
        y1 = y[Nlevels - i]

        # Delay correction
        shift = - (2 ** (i - 1)) * np.array([1, 1]) + 2

        L = 2 ** i

        # Upsampled filters
        g0_up = upsample2df(g0, i)
        g1_up = upsample2df(g1, i)

        # Reconstruction step
        x = (
            atrousc(symext(x, g0_up, shift), g0, L * I2)
            +
            atrousc(symext(y1, g1_up, shift), g1, L * I2)
        )

    # ------------------------------------------------------------
    # Final level (standard convolution, no atrous)
    # ------------------------------------------------------------
    shift = np.array([1, 1])

    x = (
        convolve2d(symext(x, g0, shift), g0, mode='valid')
        +
        convolve2d(symext(y[Nlevels], g1, shift), g1, mode='valid')
    )

    return x

def nsst_rec1(dst, lpfilt):
    """
    Inverse nonsubsampled shearlet transform (Easley version)

    Parameters
    ----------
    dst : list
        Shearlet coefficients:
        dst[0] = lowpass (2D array)
        dst[i] = 3D array (H, W, directions)

    lpfilt : str
        Lowpass filter name (passed to atrousrec)

    Returns
    -------
    x : ndarray
        Reconstructed image
    """

    level = len(dst) - 1

    # Initialize list of pyramid coefficients
    y = [None] * (level + 1)

    # Lowpass stays unchanged
    y[0] = dst[0]

    # Sum directional subbands
    for i in range(level):
        # MATLAB: sum(dst{i+1}, 3)
        # Python: sum over axis=2
        y[i + 1] = np.real(np.sum(dst[i + 1], axis=2))

    # Reconstruct via atrous pyramid
    x = np.real(atrousrec(y, lpfilt))

    return x

if __name__ == "__main__":
    from processing_ultis import stats

    # ── Hand-crafted 60×60 smooth gradient with a single spike ───────────────
    N = 60
    img = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            img[i, j] = i + j
    img[N//2, N//2] += 10

    stats(img, "original")
    
    print("=== atrousrec ===")

    y1 = [img / 2, img / 2]
    r1 = atrousrec(y1, 'pyr')
    stats(r1, "1-level")

    y2 = [img / 4, img / 4, img / 2]
    r2 = atrousrec(y2, 'pyr')
    stats(r2, "2-level")

    print("\n=== nsst_rec1 ===")

    band1 = np.stack([img / 4, np.rot90(img) / 4], axis=2)
    dst1  = [img / 2, band1]
    s1    = nsst_rec1(dst1, 'pyr')
    stats(s1, "1-level 2-dir")

    band_c = np.stack([img * np.cos(d * np.pi / 4) * 0.2 for d in range(4)], axis=2)
    band_f = np.stack([img * np.sin(d * np.pi / 4) * 0.1 for d in range(4)], axis=2)
    dst2   = [img * 0.1, band_c, band_f]
    s2     = nsst_rec1(dst2, 'pyr')
    stats(s2, "2-level 4-dir")
