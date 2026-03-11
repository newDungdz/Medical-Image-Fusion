import os
from pathlib import Path

def merge_files(folder_path, output_file, exceptions=None):
    """
    Merge all files in a folder into a single output file.
    
    Args:
        folder_path: Path to the folder containing files to merge
        output_file: Path to the output merged file
        exceptions: List of filenames to exclude (e.g., ['file1.txt', 'file2.txt'])
    """
    if exceptions is None:
        exceptions = []
    
    with open(output_file, 'w+') as outf:
        for filename in os.listdir(folder_path):
            # Skip exceptions
            if filename in exceptions:
                continue
            
            filepath = os.path.join(folder_path, filename)
            # Only process files, not directories
            if os.path.isfile(filepath) and filepath.endswith('.m'):
                try:
                    with open(filepath, 'r') as inf:
                        content = inf.read()
                        if 'function' not in content:  # Skip files that don't contain 'function'
                            continue
                        outf.write(f"\n--- {filename} ---\n")
                        outf.write(content)
                        outf.write("\n\n")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

# Example usage:
if __name__ == "__main__":
    merge_files(
        folder_path="Image-Fusion/General Evaluation Metric/Evaluation",
        output_file="./merged_output.txt",
        exceptions=["analysis_Reference.m", "Evaluation_for_Multi_Algorithm.m", "Evaluation_for_Single_Algorithm.m", "Y2RGB.m"]
    )