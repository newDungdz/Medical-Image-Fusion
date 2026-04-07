import pywt
import numpy as np

def dwt2(image, wavelet='db1'):
    """
    Perform 2D Discrete Wavelet Transform
    
    Returns:
        LL : Approximation (low-low)
        LH : Horizontal details
        HL : Vertical details
        HH : Diagonal details
    """
    coeffs2 = pywt.dwt2(image, wavelet)
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

def idwt2(LL, LH, HL, HH, wavelet='db1'):
    coeffs2 = (LL, (LH, HL, HH))
    return pywt.idwt2(coeffs2, wavelet)

def multilevel_dwt(image, wavelet='db1', level=3):
    """
    Multi-level wavelet decomposition
    
    Returns:
        coeffs[0] = LL_n
        coeffs[1:] = (LH, HL, HH) from each level
    """
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    return coeffs

def multilevel_idwt(coeffs, wavelet='db1'):
    return pywt.waverec2(coeffs, wavelet)

if __name__ == "__main__":
    img = np.random.rand(256, 256)

    coeffs = multilevel_dwt(img, wavelet='db2', level=3)
    recon = multilevel_idwt(coeffs, wavelet='db2')

    error = np.mean((img - recon) ** 2)
    print("Reconstruction error:", error)