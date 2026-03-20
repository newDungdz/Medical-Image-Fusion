"""
Convert an image to a grayscale matrix and export it as a text file.
Usage: python image_to_grayscale_matrix.py <image_path> [output_path]
"""

import sys
import numpy as np
from PIL import Image


def image_to_grayscale_matrix(image_path: str) -> np.ndarray:
    """Load an image and convert it to a 2D grayscale matrix (0-255)."""
    img = Image.open(image_path).convert("L")  # "L" = 8-bit grayscale
    return np.array(img)


def save_matrix_as_text(matrix: np.ndarray, output_path: str, separator: str = " ") -> None:
    """Save the grayscale matrix to a text file with values separated by `separator`."""
    rows, cols = matrix.shape
    with open(output_path, "w") as f:
        # Header with dimensions
        f.write(f"# Grayscale matrix — rows: {rows}, cols: {cols}\n")
        for row in matrix:
            f.write(separator.join(str(v) for v in row) + "\n")


def main():
    image_path = "25015.png"
    output_path =  "grayscale_matrix.txt"

    print(f"Loading image: {image_path}")
    matrix = image_to_grayscale_matrix(image_path)

    rows, cols = matrix.shape
    print(f"Matrix size: {rows} rows × {cols} cols")
    print(f"Value range: {matrix.min()} – {matrix.max()}")

    save_matrix_as_text(matrix, output_path)
    print(f"Matrix saved to: {output_path}")


if __name__ == "__main__":
    main()