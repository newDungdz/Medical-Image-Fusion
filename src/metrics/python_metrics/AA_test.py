import cv2
from structure_metrics import *
from img_metrics import *
from info_metrics import *
from quality_metrics import *
from structure_metrics import ms_ssim
from visual_metrics import *    


def load_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

image_idx = 1
model = 'DenseFuse'

A = load_gray(f"Image-Fusion/General Evaluation Metric/Image/Source-Image/TNO/ir/0{image_idx}.png")
B = load_gray(f"Image-Fusion/General Evaluation Metric/Image/Source-Image/TNO/vi/0{image_idx}.png")
F = load_gray(f"Image-Fusion/General Evaluation Metric/Image/Algorithm/{model}_TNO/0{image_idx}.png")

# A = np.array([[10, 20, 30, 40],
#               [50, 60, 70, 80],
#               [90, 100, 110, 120]], dtype=np.float64)
# B = np.array([[15, 25, 35, 45],
#               [55, 65, 75, 85],
#               [95, 105, 115, 125]], dtype=np.float64)
# F = np.array([[12, 22, 32, 42],
#               [52, 62, 72, 82],
#               [92, 102, 112, 122]], dtype=np.float64)

# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(A, cmap='gray')
# axes[0].set_title('Source Image A (IR)')
# axes[0].axis('off')
# axes[1].imshow(B, cmap='gray')
# axes[1].set_title('Source Image B (VI)')
# axes[1].axis('off')
# axes[2].imshow(F, cmap='gray')
# axes[2].set_title('Fused Image F')
# axes[2].axis('off')
# plt.tight_layout()
# plt.show()

print("Image shapes:", A.shape, "Image range:", A.min(), "-", A.max(), "-", A.dtype)

print(f"EN    : {en(F):.4f}")
print(f"MI    : {mi(A, B, F):.4f}")
print(f"SD    : {sd(F):.4f}")
print(f"SF    : {sf(F):.4f}")
print(f"MSE   : {mse(A, B, F):.4f}")
print(f"PSNR  : {psnr(A, B, F):.4f}")
print(f"VIF   : {vif(A, F):.4f}")
print(f"AG    : {ag(F):.4f}")
print(f"SCD   : {scd(A, B, F):.4f}")
print(f"QABF  : {qabf(A, B, F):.4f}")

print(f"MS_SSIM: {ms_ssim(np.stack([A, B], axis=2), F):.4f}")