import numpy as np
from nsst_dec import nsst_dec
from nsst_rec import nsst_rec1
import cv2
import matplotlib.pyplot as plt
from processing_ultis import stats


img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
params = {'dcomp': [2, 2, 2], 'dsize': [16, 16, 16]}
stats(img, "original")

lpfilt = 'maxflat'   # or gaussian if error

# print("Decomposing the image using NSST...")
dst, shear_f = nsst_dec(img, params, lpfilt=lpfilt)
# print("Reconstructing the image from NSST coefficients...")
stats(dst[0], "dst[0] - lowpass")
count = 1
for band in dst[1:]:
    for d in range(band.shape[2]):
        stats(band[:, :, d], f"dst {count} dir {d+1}")
    count += 1
rec = nsst_rec1(dst, lpfilt)
stats(rec, "reconstructed")

# print("Complete.") 
error = np.linalg.norm(img - rec) / np.linalg.norm(img)
print(f"Relative reconstruction error: {error:.2e}")


# # visualize the first level of shearlet coefficients
# fig, axes = plt.subplots(3, 4, figsize=(12, 3))
# axes[0, 0].imshow(img, cmap='gray')
# axes[0, 0].set_title('Original Image')
# axes[0, 0].axis('off')
# axes[0, 1].imshow(dst[0], cmap='gray')
# axes[0, 1].set_title('Lowpass Subband')
# axes[0, 1].axis('off')
# axes[0, 2].imshow(rec, cmap='gray')
# axes[0, 2].set_title('Reconstructed Image')
# axes[0, 2].axis('off')

# for i in range(3):
#     axes[1, i].imshow(shear_f[i][:, :, i], cmap='gray')
#     axes[1, i].set_title(f'Shearlet Direction {i+1}')
#     axes[1, i].axis('off')

# for i in range(4):
#     axes[2, i].imshow(dst[1][:, :, i], cmap='gray')
#     axes[2, i].set_title(f'Shearlet Coefficients - Direction {i+1}')
#     axes[2, i].axis('off')

# plt.tight_layout()
# plt.show()
