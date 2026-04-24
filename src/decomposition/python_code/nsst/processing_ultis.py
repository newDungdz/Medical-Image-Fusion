import numpy as np
from scipy.signal import convolve2d

def symext(x, h, shift):
    """
    Symmetrically (mirror) extend image x to accommodate filter h.

    After convolution in 'valid' mode on the extended image, the output
    has the same spatial size as x — without border artefacts.

    Parameters
    ----------
    x     : ndarray, shape (m, n)  — input image
    h     : ndarray, shape (p, q)  — 2-D filter
    shift : array-like, length 2   — delay compensation [s1, s2]

    Returns
    -------
    yT : ndarray, shape (m+p-1, n+q-1)  — symmetrically extended image
    """
    m, n = x.shape
    p, q = h.shape
    p2   = p // 2
    q2   = q // 2
    s1   = int(shift[0])
    s2   = int(shift[1])

    ss = p2 - s1 + 1    # left/right extension width
    rr = q2 - s2 + 1    # top/bottom extension height

    # ── Horizontal extension ────────────────────────────────────────────────
    # Left  : mirror of first ss columns
    left = x[:, :ss][:, ::-1]
    # Right : mirror of last (p + s1) columns  [= x(:,n:-1:n-p-s1+1) in MATLAB]
    right = x[:, n - p - s1:n][:, ::-1]
    yT = np.hstack([left, x, right])

    # ── Vertical extension ──────────────────────────────────────────────────
    # Top    : mirror of first rr rows
    top = yT[:rr, :][::-1, :]
    # Bottom : mirror of last (q + s2) rows  [= yT(m:-1:m-q-s2+1,:) in MATLAB]
    bottom = yT[m - q - s2:m, :][::-1, :]
    yT = np.vstack([top, yT, bottom])

    # ── Trim to required size ───────────────────────────────────────────────
    yT = yT[:m + p - 1, :n + q - 1]
    return yT


def upsample2df(h, power):
    """
    Upsample 2-D filter h by a factor of 2^power (insert zeros).

    Parameters
    ----------
    h     : ndarray, shape (m, n)
    power : int  — upsampling power (result is dilated by 2^power)

    Returns
    -------
    ho : ndarray, shape (2^power*(m-1)+1, 2^power*(n-1)+1)
         Filter with 2^power - 1 zeros inserted between every pair of
         original coefficients along each axis.
    """
    m, n  = h.shape
    step  = 2 ** power
    rows = step * m      
    cols = step * n
    ho   = np.zeros((rows, cols))
    ho[::step, ::step] = h
    return ho


def atrousc(signal, h, M):
    """
    Efficient à trous ('with holes') convolution — Python equivalent of the
    atrousc.c MEX file.

    Computes the 'valid' convolution of `signal` with filter `h` upsampled
    by the diagonal matrix M = diag(M0, M3).  Rather than explicitly
    building the upsampled filter, it is assembled here and passed to
    scipy's convolve2d.

    Parameters
    ----------
    signal : ndarray, shape (R, C)  — symmetrically-extended image patch
    h      : ndarray, shape (p, q)  — base filter (NOT yet upsampled)
    M      : ndarray, shape (2, 2)  — diagonal upsampling matrix

    Returns
    -------
    out : ndarray — 'valid' convolution result (same size as original image)
    """
    M0 = int(round(M[0, 0]))
    M3 = int(round(M[1, 1]))

    p, q = h.shape

    h_up = np.zeros(((p - 1) * M0 + 1,
                     (q - 1) * M3 + 1))
    h_up[::M0, ::M3] = h

    result = convolve2d(signal, h_up, mode='valid')

    # Match C code's output crop: skip first (M0-1) rows and (M3-1) cols
    return result[M0-1:, M3-1:]


def conv2_same_matlab(x, k):
    """
    Simulate MATLAB's conv2 with 'same' output size by performing a full convolution
    and then cropping the result to match the input size.
    """
    full = convolve2d(x, k, mode='full')
    kh, kw = k.shape
    h, w = x.shape
    return full[kh//2:kh//2+h, kw//2:kw//2+w]


def stats(arr, label):
    a = np.asarray(arr, dtype=float)
    print(f"  [{label}] shape={a.shape}  min={a.min():.4f}  max={a.max():.4f}  mean={a.mean():.4f}  sum={np.sum(a**2):.4f}")
