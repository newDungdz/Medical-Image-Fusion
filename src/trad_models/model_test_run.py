import os

import cv2

from dwt import wavelet_fusion
from pyramid import laplacian_fusion
from pca import pca_fusion
from fuzzy import fuzzy_fusion
import numpy as np

def apply_pet_colormap(img):
    """
    img: fused grayscale image (0–255)
    """
    img = img.astype(np.uint8)
    colored = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return colored


fuse1 = 'mri'
fuse2 = 'pet'

img1 = cv2.imread(f"test_sample/{fuse1}-{fuse2}/{fuse1}.png", 0)
img2 = cv2.imread(f"test_sample/{fuse1}-{fuse2}/{fuse2}.png", 0)

cv2.imwrite(f"test_sample/{fuse1}-{fuse2}/colored_mri.png", apply_pet_colormap(img1))
cv2.imwrite(f"test_sample/{fuse1}-{fuse2}/grayscale_pet.png", img2)

# cv2.imshow('MRI Image', img1)
# cv2.imshow('PET Image', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

fused_dwt = wavelet_fusion(img1, img2)
fused_laplacian = laplacian_fusion(img1, img2)
fused_pca = pca_fusion(img1, img2)
fused_fuzzy = fuzzy_fusion(img1, img2)

os.makedirs(f"test_sample/{fuse1}-{fuse2}/output", exist_ok=True)

cv2.imwrite(f"test_sample/{fuse1}-{fuse2}/output/fused_dwt.png", fused_dwt)
cv2.imwrite(f"test_sample/{fuse1}-{fuse2}/output/fused_laplacian.png", fused_laplacian)
cv2.imwrite(f"test_sample/{fuse1}-{fuse2}/output/fused_pca.png", fused_pca)
cv2.imwrite(f"test_sample/{fuse1}-{fuse2}/output/fused_fuzzy.png", fused_fuzzy)

colored_dwt = apply_pet_colormap(fused_dwt)
colored_laplacian = apply_pet_colormap(fused_laplacian)
colored_pca = apply_pet_colormap(fused_pca)
colored_fuzzy = apply_pet_colormap(fused_fuzzy)

cv2.imwrite(f"test_sample/{fuse1}-{fuse2}/output/colored_dwt.png", colored_dwt)
cv2.imwrite(f"test_sample/{fuse1}-{fuse2}/output/colored_laplacian.png", colored_laplacian)
cv2.imwrite(f"test_sample/{fuse1}-{fuse2}/output/colored_pca.png", colored_pca)
cv2.imwrite(f"test_sample/{fuse1}-{fuse2}/output/colored_fuzzy.png", colored_fuzzy)

print("Fusion completed. Fused images saved in output folder.")