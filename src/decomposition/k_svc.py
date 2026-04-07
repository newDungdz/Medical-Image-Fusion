from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view

def extract_patches(image, patch_size=8, stride=1):
    patches = sliding_window_view(image, (patch_size, patch_size))
    patches = patches[::stride, ::stride]
    
    h, w, _, _ = patches.shape
    patches = patches.reshape(-1, patch_size * patch_size)
    
    return patches.T, h, w

def omp_sparse_coding(Y, D, sparsity):
    """
    Solve: Y ≈ D * X
    
    Y: (n_features, n_samples)
    D: (n_features, n_atoms)
    
    Returns:
        X: (n_atoms, n_samples)
    """
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity, fit_intercept=False)
    
    X = []
    for i in range(Y.shape[1]):
        omp.fit(D, Y[:, i])
        X.append(omp.coef_)
    
    return np.array(X).T


def ksvd(Y, D, sparsity, n_iter=10):
    """
    K-SVD dictionary learning
    
    Y: (n_features, n_samples)
    D: (n_features, n_atoms)
    """
    for iteration in range(n_iter):
        # Step 1: Sparse coding
        X = omp_sparse_coding(Y, D, sparsity)
        
        # Step 2: Dictionary update
        for k in range(D.shape[1]):
            # Find samples using atom k
            omega = np.where(X[k, :] != 0)[0]
            
            if len(omega) == 0:
                continue
            
            # Compute residual (excluding atom k)
            Dk = D.copy()
            Dk[:, k] = 0
            
            R = Y[:, omega] - Dk @ X[:, omega]
            
            # SVD update
            U, S, Vt = np.linalg.svd(R, full_matrices=False)
            
            D[:, k] = U[:, 0]
            X[k, omega] = S[0] * Vt[0, :]
        
        # Optional: normalize atoms
        D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-8)
        
        print(f"Iteration {iteration+1} done")
    
    return D, X

def initialize_dictionary(Y, n_atoms):
    """
    Initialize dictionary by sampling data
    """
    indices = np.random.choice(Y.shape[1], n_atoms, replace=False)
    D = Y[:, indices]
    
    # Normalize atoms
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-8)
    
    return D

def reconstruct_patches(D, X):
    """
    Reconstruct patches from dictionary and sparse codes
    
    D: (n_features, n_atoms)
    X: (n_atoms, n_samples)
    
    Returns:
        Y_hat: (n_features, n_samples)
    """
    return D @ X

def reconstruct_image(patches, image_shape, patch_size=8, stride=1):
    """
    Overlap-add reconstruction
    """
    H, W = image_shape
    img = np.zeros((H, W))
    weight = np.zeros((H, W))
    
    idx = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = patches[:, idx].reshape(patch_size, patch_size)
            
            img[i:i+patch_size, j:j+patch_size] += patch
            weight[i:i+patch_size, j:j+patch_size] += 1
            
            idx += 1
    
    return img / (weight + 1e-8)

if __name__ == "__main__":
    # Step 1: Extract patches
    image = cv2.imread("test.png", 0).astype(np.float32)

    Y, h, w = extract_patches(image, patch_size=8, stride=2)

    # Step 2: Initialize dictionary
    D_init = initialize_dictionary(Y, n_atoms=128)

    # Step 3: Train K-SVD
    D, X = ksvd(Y, D_init, sparsity=5, n_iter=5)

    # Step 4: Reconstruct patches
    Y_hat = reconstruct_patches(D, X)

    # Step 5: Reconstruct image
    recon_img = reconstruct_image(Y_hat, image.shape, patch_size=8, stride=2)