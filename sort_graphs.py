

import os
import shutil
from glob import glob

"""
sort_contrast_path = "graphs/contrasts/by_contrasts"
sort_sub_path = "graphs/contrasts/by_subjects"

os.makedirs(sort_sub_path, exist_ok=True)
os.makedirs(sort_contrast_path, exist_ok=True)


for file in glob("graphs/contrasts/*"):
    if not os.path.isfile(file):
        continue

    filename = file.split("/")[-1]
    subject_id = filename.split("-")[1]
    contrast_id = filename.split("-")[3].split(".")[0]

    os.makedirs(f"{sort_contrast_path}/{contrast_id}", exist_ok=True)
    os.makedirs(f"{sort_sub_path}/{subject_id}", exist_ok=True)

    shutil.copy(file, f"{sort_contrast_path}/{contrast_id}")
    shutil.copy(file, f"{sort_sub_path}/{subject_id}")
"""

for file in glob("brute_force/*/regions/*"):
    if not os.path.isfile(file):
        continue

    ftype = file.split("\\")[1]
    fname = file.split("\\")[3]

    shutil.copy(file, f"brute_force/all_regions/{ftype}_{fname}")

