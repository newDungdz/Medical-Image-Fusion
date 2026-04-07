"""
This is a reference implementation of Bilateral/Guided filter. 
Since the code use tools and short, it just a notes for future used, not a module
"""
import cv2
from cv2.ximgproc import guidedFilter

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# d = kernel size
# sigmaColor = intensity similarity (σ_r)
# sigmaSpace = spatial distance (σ_s)
filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)



img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# guide image = same as input (common case)
filtered = guidedFilter(
    guide=img,
    src=img,
    radius=8,
    eps=1e-3
)