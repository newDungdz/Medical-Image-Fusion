import zipfile
import os
from tqdm import tqdm

import pandas as pd

def extract_zip(zip_path, extract_to, img_type):
  os.makedirs(extract_to, exist_ok=True)
  extracting_list = []
  print(f"Processing {zip_path}...")
  with zipfile.ZipFile(zip_path, 'r') as z:
      names = z.namelist()
      total_size = 0

      for name in tqdm(names, desc="Counting"):
          parts = name.split('/')
          info = z.getinfo(name)
          image_id = int(parts[4][1:])
          if img_type == "mri":
            if image_id in mri_choosen_image:
              # skip directories
              extracting_list.append(name)
              if not name.endswith('/'):
                  total_size += info.file_size
          elif img_type == "pet":
            if image_id in pet_choosen_image:
              extracting_list.append(name)
              # skip directories
              if not name.endswith('/'):
                  total_size += info.file_size

      print(f"Total files: {len(extracting_list)}")
      print(f"Total size: {total_size / (1024**3):.2f} GB")

      for name in tqdm(extracting_list, desc="Extracting"):
          parts = name.split('/')
          z.extract(name, extract_to)
      print(f"Done Extracting {zip_path} to {extract_to}")



root_path = "data/ADNI/"

mri_df = pd.read_csv(root_path + "All_Subjects_Key_MRI_27Mar2026.csv")
pet_df = pd.read_csv(root_path + "All_Subjects_Key_PET_27Mar2026.csv")

# Ensure date columns are datetime
mri_df["image_date"] = pd.to_datetime(mri_df["image_date"])
pet_df["image_date"] = pd.to_datetime(pet_df["image_date"])

# Step 1: merge on subject_id only (cartesian product per subject)
merged = pd.merge(
    mri_df,
    pet_df,
    on="subject_id",
    suffixes=("_mri", "_pet")
)

# Step 2: filter where date difference is within 30 days
merged["date_diff_days"] = (merged["image_date_mri"] - merged["image_date_pet"]).dt.days.abs()
merged = merged[merged["date_diff_days"] <= 7]

merged["visit_match"] = merged["image_visit_mri"] == merged["image_visit_pet"]

merged = merged[merged["visit_match"] == True]
mri_choosen_image = set(merged["image_id_mri"].to_list())
pet_choosen_image = set(merged["image_id_pet"].to_list())

for i in range(3, 6):
  zip_path = f"E:\\Dung\\ADNI Data\\T1w_with_Amyloid_and_Tau_PET_MRI_{i}.zip"
  extract_to =  "MRI"

  extract_zip(zip_path, extract_to, "mri")
