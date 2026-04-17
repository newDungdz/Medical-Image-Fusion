import numpy as np
from scipy.signal import convolve


def ldfilter(fname):
    """
    Generate filter for the ladder structure network.

    Parameters:
        fname (str): Filter name. Available options:
                     'pkva12' or 'pkva': length-12 filter
                     'pkva8':            length-8 filter
                     'pkva6':            length-6 filter

    Returns:
        f (np.ndarray): Symmetric impulse response filter
    """
    if fname in ('pkva12', 'pkva'):
        v = np.array([0.6300, -0.1930, 0.0972, -0.0526, 0.0272, -0.0144])

    elif fname == 'pkva8':
        v = np.array([0.6302, -0.1924, 0.0930, -0.0403])

    elif fname == 'pkva6':
        v = np.array([0.6261, -0.1794, 0.0688])

    else:
        raise ValueError(f"Unrecognized ladder structure filter name: '{fname}'")

    # Symmetric impulse response
    f = np.concatenate([v[::-1], v])
    return f


def wfilters(wname, filter_type=None):
    """
    Wavelet filters — Python equivalent of MATLAB's wfilters().

    Parameters:
        wname (str):       Wavelet name, e.g. 'db2', 'sym4', 'bior2.2', 'coif1'.
                           Supports all wavelets available in PyWavelets.
        filter_type (str, optional): Which pair of filters to return:
                           'd' -> (Lo_D, Hi_D)  decomposition filters
                           'r' -> (Lo_R, Hi_R)  reconstruction filters
                           'l' -> (Lo_D, Lo_R)  lowpass filters
                           'h' -> (Hi_D, Hi_R)  highpass filters
                           None -> (Lo_D, Hi_D, Lo_R, Hi_R)  all four filters

    Returns:
        Tuple of 2 or 4 np.ndarray filters, depending on filter_type.
    """
    import pywt

    wname = wname.strip()

    try:
        wavelet = pywt.Wavelet(wname)
    except Exception:
        raise ValueError(f"Unrecognized or unsupported wavelet name: '{wname}'")

    Lo_D = np.array(wavelet.dec_lo)
    Hi_D = np.array(wavelet.dec_hi)
    Lo_R = np.array(wavelet.rec_lo)
    Hi_R = np.array(wavelet.rec_hi)

    if filter_type is None:
        return Lo_D, Hi_D, Lo_R, Hi_R

    filter_type = filter_type.strip().lower()[0]

    if filter_type == 'd':
        return Lo_D, Hi_D
    elif filter_type == 'r':
        return Lo_R, Hi_R
    elif filter_type == 'l':
        return Lo_D, Lo_R
    elif filter_type == 'h':
        return Hi_D, Hi_R
    else:
        raise ValueError(
            f"Invalid filter_type '{filter_type}'. Expected 'd', 'r', 'l', or 'h'."
        )


def fliplr(x):
    """
    Flip a 1D or 2D array in the left/right direction.

    Parameters:
        x (array-like): Input array.

    Returns:
        np.ndarray: Array flipped along axis=0 (1D) or axis=1 (2D).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return np.flip(x)
    return np.flip(x, axis=1)


def pfilters(fname):
    """
    Generate filters for the Laplacian pyramid.

    Parameters:
        fname (str): Name of the filter set. Available options:
                     '9-7'  or '9/7'  : CDF 9/7 biorthogonal filters (JPEG 2000)
                     '5-3'  or '5/3'  : CDF 5/3 biorthogonal filters
                     'Burt'           : Burt-Adelson filters
                     'pkva'           : Phoong-Kim-Vaidyanathan-Ansari ladder filters
                     Any PyWavelets wavelet name (e.g. 'db2', 'sym4', 'bior2.2')

    Returns:
        h (np.ndarray): 1D lowpass analysis filter
        g (np.ndarray): 1D lowpass synthesis filter
    """
    if fname in ('9-7', '9/7'):
        # CDF 9/7 — used in JPEG 2000
        h = np.array([
            .037828455506995,
            -.023849465019380,
            -.11062440441842,
            .37740285561265
        ])
        h = np.concatenate([h, [.85269867900940], fliplr(h)])

        g = np.array([
            -.064538882628938,
            -.040689417609558,
            .41809227322221
        ])
        g = np.concatenate([g, [.78848561640566], fliplr(g)])

    elif fname in ('5-3', '5/3'):
        # CDF 5/3 — used in lossless JPEG 2000
        h = np.array([-1, 2, 6, 2, -1], dtype=float) / (4 * np.sqrt(2))
        g = np.array([1, 2, 1], dtype=float) / (2 * np.sqrt(2))

    elif fname == 'Burt':
        # Burt-Adelson pyramid filters
        h = np.array([0.6, 0.25, -0.05])
        h = np.sqrt(2) * np.concatenate([h[::-1][:-1], h])

        g = np.array([17/28, 73/280, -3/56, -3/280])
        g = np.sqrt(2) * np.concatenate([g[::-1][:-1], g])

    elif fname == 'pkva':
        # Phoong-Kim-Vaidyanathan-Ansari ladder structure filters
        beta = ldfilter(fname)

        lf = len(beta)
        n = lf / 2

        if n != int(n):
            raise ValueError('The input allpass filter must be even length')
        n = int(n)

        # beta(z^2): upsample by inserting zeros between coefficients
        beta2 = np.zeros(2 * lf - 1)
        beta2[::2] = beta

        # H(z): analysis lowpass filter
        h = beta2.copy()
        h[2*n - 1] += 1     # MATLAB h(2n) -> Python h[2n-1]
        h = h / 2

        # G(z): synthesis lowpass filter
        g = -convolve(beta2, h)
        g[4*n - 2] += 1     # MATLAB g(4n-1) -> Python g[4n-2]
        g[1::2] = -g[1::2]  # MATLAB g(2:2:end) -> Python g[1::2] (0-based odd indices)

        # Normalize to preserve energy
        h = h * np.sqrt(2)
        g = g * np.sqrt(2)

    else:
        # Fall back to PyWavelets for any standard named wavelet
        h, g = wfilters(fname, 'l')

    return h, g


if __name__ == '__main__':
    test_cases = ['9/7', '5/3', 'Burt', 'pkva', 'db2', 'sym4']

    for name in test_cases:
        try:
            h, g = pfilters(name)
            print(f"[{name}]")
            print(f"  h (len={len(h)}): {np.round(h, 6)}")
            print(f"  g (len={len(g)}): {np.round(g, 6)}")
        except Exception as e:
            print(f"[{name}] ERROR: {e}")