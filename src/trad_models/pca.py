import numpy as np


def pca_fusion(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Flatten images
    x1 = img1.flatten()
    x2 = img2.flatten()

    # Stack into matrix
    X = np.vstack((x1, x2))

    # Covariance matrix
    C = np.cov(X)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eig(C)

    # Principal component
    idx = np.argmax(eigvals)
    weights = eigvecs[:, idx]

    # Normalize weights
    weights = weights / np.sum(weights)

    # Fuse images
    fused = weights[0] * img1 + weights[1] * img2

    return np.clip(fused, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    import cv2

    img1 = cv2.imread("test_sample/mri-ct/mri.png", 0)
    img2 = cv2.imread("test_sample/mri-ct/ct.png", 0)

    fused = pca_fusion(img1, img2)

    cv2.imwrite("test_sample/mri-ct/fused/fused_pca.png", fused)
    