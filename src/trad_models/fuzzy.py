import numpy as np


def fuzzy_membership(x):
    """
    Simple linear membership function
    maps intensity [0,255] → [0,1]
    """
    return x / 255.0


def fuzzy_fusion(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # fuzzification
    m1 = fuzzy_membership(img1)
    m2 = fuzzy_membership(img2)

    # fuzzy rule (choose stronger membership)
    fused_m = np.maximum(m1, m2)

    # defuzzification
    fused = fused_m * 255.0

    return fused.astype(np.uint8)


if __name__ == "__main__":
    import cv2
    
    img1 = cv2.imread("test_sample/mri-ct/mri.png", 0)
    img2 = cv2.imread("test_sample/mri-ct/ct.png", 0)

    fused = fuzzy_fusion(img1, img2)

    cv2.imwrite("test_sample/mri-ct/fused_fuzzy.png", fused)