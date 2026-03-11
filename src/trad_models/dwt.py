import numpy as np
import pywt

def wavelet_fusion(img1, img2, wavelet='db1', level=1):
    """
    Simple wavelet image fusion.

    Parameters
    ----------
    img1 : ndarray
        First grayscale image
    img2 : ndarray
        Second grayscale image
    wavelet : str
        Wavelet type (default 'db1')
    level : int
        Decomposition level

    Returns
    -------
    fused : ndarray
        Fused image
    """

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Wavelet decomposition
    coeffs1 = pywt.wavedec2(img1, wavelet, level=level)
    coeffs2 = pywt.wavedec2(img2, wavelet, level=level)

    fused_coeffs = []

    # Fuse approximation coefficients (LL)
    cA1, cA2 = coeffs1[0], coeffs2[0]
    fused_cA = (cA1 + cA2) / 2
    fused_coeffs.append(fused_cA)

    # Fuse detail coefficients
    for i in range(1, len(coeffs1)):
        c1 = coeffs1[i]
        c2 = coeffs2[i]

        fused_detail = []
        for d1, d2 in zip(c1, c2):
            fused = np.where(np.abs(d1) > np.abs(d2), d1, d2)
            fused_detail.append(fused)

        fused_coeffs.append(tuple(fused_detail))

    # Reconstruction
    fused_img = pywt.waverec2(fused_coeffs, wavelet)

    return np.clip(fused_img, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    import cv2

    img1 = cv2.imread("test_sample/mri-ct/mri.png", 0)
    img2 = cv2.imread("test_sample/mri-ct/ct.png", 0)

    fused = wavelet_fusion(img1, img2)

    cv2.imwrite("test_sample/mri-ct/fused_dwt.png", fused)