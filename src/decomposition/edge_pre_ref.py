"""
This is a reference implementation of Bilateral/Guided filter. 
Since the code use tools and short, it just a notes for future used, not a module
"""
import cv2
from cv2.ximgproc import guidedFilter
import numpy as np
import matplotlib.pyplot as plt

def visualize_detail(detail):
    vis = detail + 128  # center around gray
    return np.clip(vis, 0, 255).astype(np.uint8)

# img = cv2.imread('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/4010.png', cv2.IMREAD_GRAYSCALE)

img = np.array([[10, 30, 200, 200],
                [10, 30, 200, 200],
                [10, 100, 200, 200],
                [10, 100, 150, 100]], dtype=np.float32)

img2 = np.array([[50, 52, 49, 48],
                [51, 50, 52, 49],
                [48, 49, 51, 50],
                [47, 48, 49, 51]], dtype=np.float32)

# img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)

# d = kernel size
# sigmaColor = intensity similarity (σ_r)
# sigmaSpace = spatial distance (σ_s)
bilat_filtered = cv2.bilateralFilter(img, d=3, sigmaColor=50, sigmaSpace=1, borderType=cv2.BORDER_REPLICATE)

# guide image = same as input (common case)
guided_filtered = guidedFilter(
    guide=img,
    src=img2,
    radius=3,
    eps=0.01
)

img_f = img.astype(np.float32)

bilat_base = bilat_filtered.astype(np.float32)
guided_base = guided_filtered.astype(np.float32)

bilat_detail = img_f - bilat_base
guided_detail = img_f - guided_base

print("Bilateral Base:\n", bilat_base)
print("Guided Base:\n", guided_base)

# fig, axes = plt.subplots(1, 5, figsize=(15, 3))

# axes[0].imshow(img, cmap='gray')
# axes[0].set_title('Original')
# axes[0].axis('off')

# axes[1].imshow(bilat_base, cmap='gray')
# axes[1].set_title('Bilateral Base')
# axes[1].axis('off')

# axes[2].imshow(visualize_detail(bilat_detail), cmap='gray')
# axes[2].set_title('Bilateral Detail')
# axes[2].axis('off')

# axes[3].imshow(guided_base, cmap='gray')
# axes[3].set_title('Guided Base')
# axes[3].axis('off')

# axes[4].imshow(visualize_detail(guided_detail), cmap='gray')
# axes[4].set_title('Guided Detail')
# axes[4].axis('off')

# plt.tight_layout()
# plt.show()