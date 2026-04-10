import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

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
def visualize_dwt(coeffs, level=1):
    """
    Visualize the 4 transform channels (LL, LH, HL, HH) at a specific level
    """
    
    cA = coeffs[0]  # Approximation (LL)
    cD = coeffs[level]  # Details (LH, HL, HH)
    
    cH, cV, cD_diag = cD
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(cA, cmap='gray')
    axes[0, 0].set_title('LL (Approximation)')
    axes[0, 1].imshow(cH, cmap='gray')
    axes[0, 1].set_title('LH (Horizontal)')
    axes[1, 0].imshow(cV, cmap='gray')
    axes[1, 0].set_title('HL (Vertical)')
    axes[1, 1].imshow(cD_diag, cmap='gray')
    axes[1, 1].set_title('HH (Diagonal)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img = cv2.imread('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/4010.png', cv2.IMREAD_GRAYSCALE)

    coeffs = multilevel_dwt(img, wavelet='db2', level=3)

    visualize_dwt(coeffs, level=3)
    
    recon = multilevel_idwt(coeffs, wavelet='db2')
    cv2.imshow('Reconstructed Image', recon.astype(np.uint8))
    error = np.mean((img - recon) ** 2)
    print("Reconstruction error:", error)