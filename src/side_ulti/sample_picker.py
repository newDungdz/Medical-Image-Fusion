import os
import random
import shutil
from pathlib import Path

def pick_and_copy_images(parent_folder, output_folder, pick_mode='top'):
    """
    Pick images from two subfolders with same filenames and copy to output folder.
    
    Args:
        parent_folder: Path to folder containing 2 subfolders (e.g., 'MRI', 'PET')
        output_folder: Path where processed images will be saved
        pick_mode: 'top' for first image or 'random' for random selection
    """
    parent_folder = Path(parent_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get subfolders
    subfolders = [f for f in parent_folder.iterdir() if f.is_dir()]
    
    if len(subfolders) != 2:
        print(f"Error: Expected 2 subfolders, found {len(subfolders)}")
        return
    
    # Get images from both subfolders
    for subfolder in subfolders:
        images = sorted([f for f in subfolder.iterdir() if f.is_file()])
        
        if not images:
            print(f"No images found in {subfolder}")
            continue
        
        # Select image based on mode
        if pick_mode == 'random':
            selected_image = random.choice(images)
        else:  # 'top'
            selected_image = images[0]
        
        # Copy and rename
        folder_name = subfolder.name.lower()
        new_filename = f"{folder_name}{selected_image.suffix}"
        output_path = output_folder / new_filename
        
        shutil.copy2(selected_image, output_path)
        print(f"Copied: {selected_image.name} -> {new_filename}")

# Example usage
if __name__ == "__main__":
    pick_and_copy_images(
        parent_folder="Havard-Medical-Image-Fusion-Datasets/PET-MRI",
        output_folder="test_sample/mri-pet",
        pick_mode='top'  # or 'random'
    )