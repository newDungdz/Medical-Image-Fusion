import numpy as np
import cv2

def gaussian_pyramid(image, levels):
    """
    Build Gaussian pyramid using standard definition:
    blur + downsample (factor 2)
    """
    G = image.astype(np.float32)
    pyramid = [G]

    for _ in range(levels - 1):
        # OpenCV pyrDown implements:
        # Gaussian blur + downsample (correct reference behavior)
        G = cv2.pyrDown(G)
        pyramid.append(G)

    return pyramid


def reconstruct_gaussian(gaussian_pyr):
    """
    Reconstruct image from Gaussian pyramid
    """
    current = gaussian_pyr[-1]

    for i in reversed(range(len(gaussian_pyr) - 1)):
        current = cv2.pyrUp(current)

        h, w = gaussian_pyr[i].shape[:2]
        current = current[:h, :w]

    return current

def laplacian_pyramid(image, levels):
    """
    Build Laplacian pyramid based on Gaussian pyramid
    L_i = G_i - Expand(G_{i+1})
    """
    gaussian_pyr = gaussian_pyramid(image, levels)
    laplacian_pyr = []

    for i in range(levels - 1):
        G_current = gaussian_pyr[i]
        G_next = gaussian_pyr[i + 1]

        # Expand (upsample)
        G_next_up = cv2.pyrUp(G_next)

        # Match size (important for odd dimensions)
        h, w = G_current.shape[:2]
        G_next_up = G_next_up[:h, :w]

        L = G_current - G_next_up
        laplacian_pyr.append(L)

    # Last level is same as Gaussian top
    laplacian_pyr.append(gaussian_pyr[-1])

    return laplacian_pyr

def reconstruct_laplacian(laplacian_pyr):
    """
    Reconstruct image from Laplacian pyramid
    """
    current = laplacian_pyr[-1]

    for i in reversed(range(len(laplacian_pyr) - 1)):
        current = cv2.pyrUp(current)

        h, w = laplacian_pyr[i].shape[:2]
        current = current[:h, :w]

        current = current + laplacian_pyr[i]

    return current


def visualize_pyramid(pyramid, title="Pyramid"):
    """
    Visualize pyramid levels by stacking them horizontally
    """
    # Normalize for visualization
    normalized = []
    for level in pyramid:
        norm = cv2.normalize(level, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        normalized.append(norm)
    
    # Pad and concatenate
    h0, w0 = normalized[0].shape[:2]
    canvas = np.zeros((h0, w0), dtype=np.uint8)
    canvas[:h0, :w0] = normalized[0]
    
    x_offset = w0
    for i in range(1, len(normalized)):
        h, w = normalized[i].shape[:2]
        padded = np.zeros((h0, w), dtype=np.uint8)
        padded[:h, :w] = normalized[i]
        canvas = np.hstack([canvas, padded])
    
    cv2.imshow(title, canvas)

if __name__ == "__main__":
    img = cv2.imread("test.png", 0).astype(np.float32)

    gaus = gaussian_pyramid(img, levels=4)
    recon_gaus = reconstruct_gaussian(gaus)
    error = np.mean((img - recon_gaus) ** 2)
    print("Reconstruction error:", error)
    visualize_pyramid(gaus, "Gaussian Pyramid")
    
    lap = laplacian_pyramid(img, levels=4)
    recon = reconstruct_laplacian(lap)

    error = np.mean((img - recon) ** 2)
    print("Reconstruction error:", error)
    
    visualize_pyramid(lap, "Laplacian Pyramid")
    cv2.waitKey(0)
    cv2.destroyAllWindows()