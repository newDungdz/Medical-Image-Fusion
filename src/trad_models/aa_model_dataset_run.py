from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2

import dwt, laplacian

def fuse_images(img1_path, img2_path, color_model, fuse_model):
    """
    Fuse two images together.
    You can customize this function based on your fusion model.
    """
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path))
    
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    fused = fuse_model(img1, img2, mode=color_model)

    return fused

def process_image_pairs(root_folder, color_model, fuse_model, output_folder_path, output_folder_name="FUSED"):
    """
    Process all pairs of images from two subfolders and save fused results.
    """
    root_path = Path(root_folder)
    
    # Get subfolders
    subfolders = [f for f in root_path.iterdir() if f.is_dir()]

    
    subfolders = [f for f in subfolders if not (f.name.upper() == "FUSED" or "FUSED" in f.name.upper() or f.name == "eval_results")]
    
    if len(subfolders) != 2:
        print(f"Error: Expected 2 image folders after filtering, found {len(subfolders)}")
        return
    
    folder1, folder2 = subfolders[0], subfolders[1]
    
    # Create FUSED output folder
    output_folder_path.mkdir(exist_ok=True)
    fused_folder = output_folder_path / output_folder_name
    fused_folder.mkdir(exist_ok=True)
    
    # Get image names from first folder
    image_names = {f.name for f in folder1.iterdir() if f.is_file()}
    
    # Process each pair
    for img_name in tqdm(image_names, desc="Processing image pairs"):
        img1_path = folder1 / img_name
        img2_path = folder2 / img_name
        
        if img1_path.exists() and img2_path.exists():
            try:
                fused_img = fuse_images(img1_path, img2_path, color_model, fuse_model)
                output_path = fused_folder / img_name
                if fused_img.ndim == 3:
                    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), fused_img)
                # print(f"Fused: {img_name}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        else:
            print(f"Pair not found for: {img_name}")
    print(f"Finished processing. Fused images saved to: {fused_folder}")


if __name__ == "__main__":
    root_output_folder = Path("data/Fused_results/CT-MRI")
    root_folder = "data/AANLIB/MyDatasets/CT-MRI/test"
    
    # color_model = "grayscale"
    # process_image_pairs(root_folder, color_model, dwt.fuse, root_output_folder, f"DWT-{color_model}")
    # process_image_pairs(root_folder, color_model, laplacian.fuse, root_output_folder, f"PYRAMID-{color_model}")
    
    # root_output_folder = Path("data/Fused_results/PET-MRI")
    # root_folder = "data/AANLIB/MyDatasets/PET-MRI/test"
    # process_image_pairs(root_folder, color_model, dwt.fuse, root_output_folder, f"DWT-{color_model}")
    # process_image_pairs(root_folder, color_model, laplacian.fuse, root_output_folder, f"PYRAMID-{color_model}")

    # root_output_folder = Path("data/Fused_results/SPECT-MRI")
    # root_folder = "data/AANLIB/MyDatasets/SPECT-MRI/test"
    # process_image_pairs(root_folder, color_model, dwt.fuse, root_output_folder, f"DWT-{color_model}")
    # process_image_pairs(root_folder, color_model, laplacian.fuse, root_output_folder, f"PYRAMID-{color_model}")
    
    # =================================
    # YCbCr FUSION
    # =================================
    color_model = "ycbcr"

    root_output_folder = Path("data/Fused_results/PET-MRI")
    root_folder = "data/AANLIB/MyDatasets/PET-MRI/test"
    process_image_pairs(root_folder, color_model, dwt.fuse, root_output_folder, f"DWT-{color_model}")
    process_image_pairs(root_folder, color_model, laplacian.fuse, root_output_folder, f"PYRAMID-{color_model}")

    root_output_folder = Path("data/Fused_results/SPECT-MRI")
    root_folder = "data/AANLIB/MyDatasets/SPECT-MRI/test"
    process_image_pairs(root_folder, color_model, dwt.fuse, root_output_folder, f"DWT-{color_model}")
    process_image_pairs(root_folder, color_model, laplacian.fuse, root_output_folder, f"PYRAMID-{color_model}")
    
    # =================================
    # HSI FUSION
    # =================================
    color_model = "hsi"
    root_output_folder = Path("data/Fused_results/PET-MRI")
    root_folder = "data/AANLIB/MyDatasets/PET-MRI/test"
    process_image_pairs(root_folder, color_model, dwt.fuse, root_output_folder, f"DWT-{color_model}")
    process_image_pairs(root_folder, color_model, laplacian.fuse, root_output_folder, f"PYRAMID-{color_model}")

    root_output_folder = Path("data/Fused_results/SPECT-MRI")
    root_folder = "data/AANLIB/MyDatasets/SPECT-MRI/test"
    process_image_pairs(root_folder, color_model, dwt.fuse, root_output_folder, f"DWT-{color_model}")
    process_image_pairs(root_folder, color_model, laplacian.fuse, root_output_folder, f"PYRAMID-{color_model}")