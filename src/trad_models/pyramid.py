import cv2
import numpy as np


def build_gaussian_pyramid(img, levels):
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        gp.append(img)
    return gp


def build_laplacian_pyramid(gp):
    lp = []
    for i in range(len(gp) - 1):
        size = (gp[i].shape[1], gp[i].shape[0])
        up = cv2.pyrUp(gp[i+1], dstsize=size)
        lap = gp[i] - up
        lp.append(lap)
    lp.append(gp[-1])  # base layer
    return lp


def reconstruct_from_laplacian(lp):
    img = lp[-1]
    for i in range(len(lp)-2, -1, -1):
        size = (lp[i].shape[1], lp[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = img + lp[i]
    return img


def laplacian_fusion(img1, img2, levels=4):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    gp1 = build_gaussian_pyramid(img1, levels)
    gp2 = build_gaussian_pyramid(img2, levels)

    lp1 = build_laplacian_pyramid(gp1)
    lp2 = build_laplacian_pyramid(gp2)

    fused_pyramid = []

    for i in range(len(lp1)):
        if i == len(lp1) - 1:
            fused = (lp1[i] + lp2[i]) / 2
        else:
            fused = np.where(
                np.abs(lp1[i]) > np.abs(lp2[i]),
                lp1[i],
                lp2[i]
            )
        fused_pyramid.append(fused)

    fused_img = reconstruct_from_laplacian(fused_pyramid)

    return np.clip(fused_img, 0, 255).astype(np.uint8)

if __name__ == "__main__":

    img1 = cv2.imread("test_sample/mri-ct/mri.png", 0)
    img2 = cv2.imread("test_sample/mri-ct/ct.png", 0)

    fused = laplacian_fusion(img1, img2)

    cv2.imwrite("test_sample/mri-ct/fused_laplacian.png", fused)