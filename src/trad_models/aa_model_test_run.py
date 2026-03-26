import os
import cv2
import dwt, laplacian, pca, fuzzy
import numpy as np

import matplotlib.pyplot as plt

def show_fusion_results(img1, img2, results: dict):
    """
    img1: grayscale (MRI)
    img2: RGB (PET)
    results: dict {name: fused_image}
    """

    n = 2 + len(results)  # MRI + PET + fused images
    plt.figure(figsize=(4*n, 5))

    # MRI
    plt.subplot(1, n, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("MRI (Grayscale)")
    plt.axis('off')

    # PET
    plt.subplot(1, n, 2)
    plt.imshow(img2)
    plt.title("PET (RGB)")
    plt.axis('off')

    # Fused images
    for i, (name, img) in enumerate(results.items(), start=3):
        plt.subplot(1, n, i)
        plt.imshow(img)
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

fuse1 = 'MRI'
fuse2 = 'PET'
img1 = cv2.imread(f"data/AANLIB/{fuse2}-{fuse1}/{fuse1}/25015.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f"data/AANLIB/{fuse2}-{fuse1}/{fuse2}/25015.png", cv2.IMREAD_COLOR)

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# cv2.imshow('MRI Image', img1)
# cv2.imshow('PET Image', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

color_model="ycbcr"

fused_dwt = dwt.fuse(img1, img2, mode=color_model)
fused_laplacian = laplacian.fuse(img1, img2, mode=color_model)

results = {
    "DWT": fused_dwt,
    "Laplacian": fused_laplacian,
}

show_fusion_results(img1, img2, results)
print("Fusion completed. Fused images saved in output folder.")