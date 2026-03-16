import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib 

# --------------------------------------------------
# Load one SPECT image
# --------------------------------------------------
image_path = pathlib.Path("data/AANLIB/SPECT-MRI/SPECT/3008.png")

img_bgr = cv2.imread(str(image_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

Y, Cr, Cb = cv2.split(img_ycrcb)

# --------------------------------------------------
# Print stats
# --------------------------------------------------
print(f"Image: {image_path.name}")
print(f"Shape: {img_bgr.shape}")
print(f"Y  — min: {Y.min():3d}, max: {Y.max():3d}, mean: {Y.mean():.1f}, std: {Y.std():.2f}")
print(f"Cr — min: {Cr.min():3d}, max: {Cr.max():3d}, mean: {Cr.mean():.1f}, std: {Cr.std():.2f}")
print(f"Cb — min: {Cb.min():3d}, max: {Cb.max():3d}, mean: {Cb.mean():.1f}, std: {Cb.std():.2f}")
print()
print("→ Neutral Cr/Cb would be ~128 with std ≈ 0 (grayscale image)")
print(f"→ This image Cr std={Cr.std():.2f}, Cb std={Cb.std():.2f} — ", end="")
print("HAS color" if Cr.std() > 5 and Cb.std() > 5 else "EFFECTIVELY GRAYSCALE")

# --------------------------------------------------
# Plot all channels
# --------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img_rgb)
axes[0].set_title(f"Original RGB\n{image_path.name}", fontsize=9)

axes[1].imshow(Y, cmap='gray')
axes[1].set_title(f"Y (Luminance)\nstd={Y.std():.2f}")

axes[2].imshow(Cr, cmap='RdBu_r')
axes[2].set_title(f"Cr (Red chroma)\nstd={Cr.std():.2f}")

axes[3].imshow(Cb, cmap='RdBu_r')
axes[3].set_title(f"Cb (Blue chroma)\nstd={Cb.std():.2f}")

for ax in axes:
    ax.axis('off')

plt.suptitle("YCrCb Channel Inspection", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()