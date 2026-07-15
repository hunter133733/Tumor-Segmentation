import os

# code to extract .nii and .nii.gz files from dataset parent folder
def collect_nii_files(root_dir):
    """
    Walk through root_dir and return all .nii and .nii.gz files.
    """
    nii_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                full_path = os.path.join(dirpath, fname)
                nii_files.append(full_path)

    return nii_files

# Put the path to the parent folder here (will vary depending on your folder structure): 
root_folder = r"PATH/TO/PKG - BraTS-TCGA-LGG/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations"

files = collect_nii_files(root_folder)

print(f"Found {len(files)} NIfTI files.")
