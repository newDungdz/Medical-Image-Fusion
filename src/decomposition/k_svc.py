from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
import cv2
import warnings
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


def extract_patches(image, patch_size=8, stride=1):
    patches = sliding_window_view(image, (patch_size, patch_size))
    patches = patches[::stride, ::stride]
    h, w, _, _ = patches.shape
    patches = patches.reshape(-1, patch_size * patch_size)
    return patches.T, h, w


def normalize_dictionary(D):
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    return D / np.maximum(norms, 1e-8)


def replace_dead_atoms(D, Y):
    n_atoms = D.shape[1]
    G = D.T @ D
    np.fill_diagonal(G, 0)

    for k in range(n_atoms):
        atom_norm = np.linalg.norm(D[:, k])
        is_dead = atom_norm < 1e-6
        is_duplicate = np.any(np.abs(G[k]) > 0.99)

        if is_dead or is_duplicate:
            idx = np.random.randint(0, Y.shape[1])
            new_atom = Y[:, idx].copy()
            new_norm = np.linalg.norm(new_atom)
            if new_norm > 1e-8:
                D[:, k] = new_atom / new_norm
            else:
                D[:, k] = np.random.randn(D.shape[0])
                D[:, k] /= np.linalg.norm(D[:, k])

            G[k, :] = D[:, k].T @ D
            G[:, k] = G[k, :]
            np.fill_diagonal(G, 0)

    return D


def omp_sparse_coding(Y, D, sparsity):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity, fit_intercept=False)
    X = np.zeros((D.shape[1], Y.shape[1]), dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(Y.shape[1]):
            omp.fit(D, Y[:, i])
            X[:, i] = omp.coef_

    return X


def reconstruct_patches(D, X):
    return D @ X


def reconstruct_image(patches, image_shape, patch_size=8, stride=1):
    H, W = image_shape
    img = np.zeros((H, W))
    weight = np.zeros((H, W))

    idx = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = patches[:, idx].reshape(patch_size, patch_size)
            img[i:i + patch_size, j:j + patch_size] += patch
            weight[i:i + patch_size, j:j + patch_size] += 1
            idx += 1

    return img / np.maximum(weight, 1e-8)


def initialize_dictionary(Y, n_atoms):
    indices = np.random.choice(Y.shape[1], n_atoms, replace=False)
    D = Y[:, indices].copy()
    return normalize_dictionary(D)


def ksvd(Y, D, sparsity, n_iter=10, image_shape=None, patch_size=8,
         stride=2, save_every=1, save_dir="ksvd_progress"):
    """
    K-SVD with per-iteration error logging and image snapshots.

    save_every : int  — save a reconstructed image every N iterations
    save_dir   : str  — folder to write snapshot PNGs into
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    D = normalize_dictionary(D.copy())

    history = []          # list of dicts, one per iteration
    snapshots = {}        # {iteration: reconstructed_image}

    for iteration in range(n_iter):
        # ── Step 1: Sparse coding ──────────────────────────────────────────
        X = omp_sparse_coding(Y, D, sparsity)

        # ── Step 2: Dictionary update ──────────────────────────────────────
        for k in range(D.shape[1]):
            omega = np.where(X[k, :] != 0)[0]
            if len(omega) == 0:
                continue

            Dk = D.copy()
            Dk[:, k] = 0
            R_k = Y[:, omega] - Dk @ X[:, omega]

            U, S, Vt = np.linalg.svd(R_k, full_matrices=False)
            D[:, k] = U[:, 0]
            X[k, omega] = S[0] * Vt[0, :]

        # ── Step 3: Normalize & replace bad atoms ─────────────────────────
        D = normalize_dictionary(D)
        D = replace_dead_atoms(D, Y)

        # ── Step 4: Reconstruction error ──────────────────────────────────
        Y_hat = D @ X
        residual = Y - Y_hat

        rmse          = np.sqrt(np.mean(residual ** 2))
        total_frob    = np.linalg.norm(residual, 'fro')
        relative_err  = total_frob / (np.linalg.norm(Y, 'fro') + 1e-8)
        mean_sparsity = np.mean(np.sum(X != 0, axis=0))

        print(
            f"[Iter {iteration + 1:>2}/{n_iter}] "
            f"RMSE: {rmse:.6f} | "
            f"Relative error: {relative_err:.4%} | "
            f"Frobenius: {total_frob:.4f} | "
            f"Avg sparsity: {mean_sparsity:.2f}/{sparsity}"
        )

        history.append({
            "iteration":    iteration + 1,
            "rmse":         rmse,
            "relative_err": relative_err,
            "frobenius":    total_frob,
            "sparsity":     mean_sparsity,
        })

        # ── Step 5: Save snapshot every `save_every` iterations ───────────
        if (iteration + 1) % save_every == 0 and image_shape is not None:
            recon = reconstruct_image(Y_hat, image_shape,
                                      patch_size=patch_size, stride=stride)
            # Clip to valid pixel range before saving
            recon_clipped = np.clip(recon, 0, 255).astype(np.uint8)

            path = os.path.join(save_dir, f"iter_{iteration + 1:03d}.png")
            cv2.imwrite(path, recon_clipped)

            snapshots[iteration + 1] = recon   # keep float for PSNR / display

    return D, X, history, snapshots


def visualize_progress(original, snapshots, history, save_path="ksvd_comparison.png"):
    """
    Grid: original image  +  one panel per saved snapshot.
    Bottom row: RMSE and relative-error curves.
    """
    snap_iters  = sorted(snapshots.keys())
    n_snaps     = len(snap_iters)

    # ── Layout: 2 rows ────────────────────────────────────────────────────
    # Row 0: original + snapshots
    # Row 1: RMSE curve + relative-error curve
    fig = plt.figure(figsize=(4 * (n_snaps + 1), 9))
    gs  = fig.add_gridspec(2, max(n_snaps + 1, 2),
                           height_ratios=[2, 1], hspace=0.4, wspace=0.3)

    # ── Row 0: images ─────────────────────────────────────────────────────
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original, cmap="gray", vmin=0, vmax=255)
    ax_orig.set_title("Original", fontsize=11, fontweight="bold")
    ax_orig.axis("off")

    for col, it in enumerate(snap_iters, start=1):
        rec = snapshots[it]

        # PSNR
        mse_val = np.mean((original.astype(np.float64) - rec) ** 2)
        psnr    = 10 * np.log10(255 ** 2 / mse_val) if mse_val > 0 else float("inf")

        ax = fig.add_subplot(gs[0, col])
        ax.imshow(np.clip(rec, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Iter {it}\nPSNR {psnr:.2f} dB", fontsize=10)
        ax.axis("off")

    # ── Row 1: error curves ───────────────────────────────────────────────
    iters        = [h["iteration"]    for h in history]
    rmse_vals    = [h["rmse"]         for h in history]
    rel_err_vals = [h["relative_err"] for h in history]

    # Mark which iterations have snapshots
    snap_set = set(snap_iters)

    ax_rmse = fig.add_subplot(gs[1, :n_snaps // 2 + 1])
    ax_rmse.plot(iters, rmse_vals, marker="o", color="steelblue", linewidth=2)
    for it in snap_iters:
        ax_rmse.axvline(it, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_rmse.set_title("RMSE per iteration")
    ax_rmse.set_xlabel("Iteration")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.grid(True, alpha=0.3)

    ax_rel = fig.add_subplot(gs[1, n_snaps // 2 + 1:])
    ax_rel.plot(iters, [v * 100 for v in rel_err_vals],
                marker="s", color="tomato", linewidth=2)
    for it in snap_iters:
        ax_rel.axvline(it, color="orange", linestyle="--", linewidth=0.8, alpha=0.7,
                       label="snapshot" if it == snap_iters[0] else "")
    ax_rel.set_title("Relative Error per iteration")
    ax_rel.set_xlabel("Iteration")
    ax_rel.set_ylabel("Relative Error (%)")
    ax_rel.legend(fontsize=8)
    ax_rel.grid(True, alpha=0.3)

    fig.suptitle("K-SVD Reconstruction Progress", fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Comparison figure saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PATCH_SIZE = 8
    STRIDE     = 2
    N_ATOMS    = 128
    SPARSITY   = 5
    N_ITER     = 10
    SAVE_EVERY = 2          # snapshot every 2 iterations

    image = cv2.imread("test.png", 0).astype(np.float32)

    Y, h, w = extract_patches(image, patch_size=PATCH_SIZE, stride=STRIDE)
    D_init  = initialize_dictionary(Y, n_atoms=N_ATOMS)

    D, X, history, snapshots = ksvd(
        Y, D_init,
        sparsity    = SPARSITY,
        n_iter      = N_ITER,
        image_shape = image.shape,
        patch_size  = PATCH_SIZE,
        stride      = STRIDE,
        save_every  = SAVE_EVERY,
        save_dir    = "ksvd_progress",
    )

    visualize_progress(image, snapshots, history, save_path="ksvd_comparison.png")