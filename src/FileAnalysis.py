# FileAnalysis.py
# Goal:
# - Take the BraTS-TCGA-LGG dataset on my computer
# - Turn it into nnU-Net v2 format (imagesTr, imagesTs, labelsTr)
# - Seperate patients into training and validation/test sets
#
# Needed in ordered to run inference (formated Naming scheme)

# Imports----------------------------------------------------------------------------
import os
import random
import shutil
import nibabel as nib
import numpy as np

from pathlib import Path
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

# Basic settings------------------------------------------------------------------
DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_TCGA_LGG"

NUM_TRAIN = 40
NUM_TOTAL = 60

# Folder with the original TCGA-LGG data (Data should be in folder with this file)
HERE = Path(__file__).resolve().parent

TCGA_ROOT = (
    HERE
    / "PKG - BraTS-TCGA-LGG"
    / "BraTS-TCGA-LGG"
    / "Pre-operative_TCGA_LGG_NIfTI_and_Segmentations"
)

if not TCGA_ROOT.exists():
    print("WARNING: TCGA files not found")

# nnU-Net folder structure (I just put everything in ./nnUNet_data)
NNUNET_BASE = HERE / "nnUNet_data"
NNUNET_RAW = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_BASE / "nnUNet_results"

for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

os.environ["nnUNet_raw"] = str(NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
os.environ["nnUNet_results"] = str(NNUNET_RESULTS)

# Helper Function ---------------------------------------------------------------------------------
def pick_file_ci(files, must_contain, must_not_contain=None):
    if must_not_contain is None:
        must_not_contain = []

    for f in files:
        name = f.name.lower()
        ok = True
        for s in must_contain:
            if s.lower() not in name:
                ok = False
                break
        for s in must_not_contain:
            if s.lower() in name:
                ok = False
                break
        if ok:
            return f

    print("Couldn't find file: pick_file_ci()")

# Main work----------------------------------------------------------------------
def prepare_nnunet_dataset():
    # Find all patient folders
    patient_dirs = []
    for p in TCGA_ROOT.iterdir():
        if p.is_dir() and p.name.startswith("TCGA"):
            patient_dirs.append(p)

    patient_dirs = sorted(patient_dirs)

    if len(patient_dirs) == 0:
        print("No patient folders found under TCGA_ROOT.")

    if len(patient_dirs) < NUM_TOTAL:
        num_total = len(patient_dirs)
    else:
        num_total = NUM_TOTAL

    # Random split
    rng = random.Random(340)
    rng.shuffle(patient_dirs)
    chosen = patient_dirs[:num_total]

    train_dirs = chosen[:NUM_TRAIN]
    val_dirs = chosen[NUM_TRAIN:]

    # Create nnU-Net dataset folders
    dataset_dir = NNUNET_RAW / DATASET_NAME
    imagesTr = dataset_dir / "imagesTr"
    imagesTs = dataset_dir / "imagesTs"
    labelsTr = dataset_dir / "labelsTr"

    for d in [dataset_dir, imagesTr, imagesTs, labelsTr]:
        d.mkdir(parents=True, exist_ok=True)

    train_ids = []
    val_ids = []

    # Inner helper to copy one patient
    def copy_patient(patient_dir: Path, is_train: bool):
        case_id = patient_dir.name
        nii_files = list(patient_dir.glob("*.nii.gz"))
        if len(nii_files) == 0:
            raise FileNotFoundError(f"No .nii.gz files in {patient_dir}")
        
        try:
            seg_path = pick_file_ci(nii_files, ["glistrboost"])
        except FileNotFoundError:
            seg_path = pick_file_ci(nii_files, ["seg"])

        # Modalities
        flair_path = pick_file_ci(nii_files, ["flair"])
        t1gd_path = pick_file_ci(nii_files, ["t1", "gd"])
        t2_path   = pick_file_ci(nii_files, ["t2"])
        t1_path   = pick_file_ci(nii_files, ["t1"], ["gd"])

        if is_train:
            img_out_dir = imagesTr
            lab_out_dir = labelsTr
        else:
            img_out_dir = imagesTs
            lab_out_dir = None

        # Copy images with nnU-Net channel names
        shutil.copy2(t1_path,   img_out_dir / f"{case_id}_0000.nii.gz")
        shutil.copy2(t1gd_path, img_out_dir / f"{case_id}_0001.nii.gz")
        shutil.copy2(t2_path,   img_out_dir / f"{case_id}_0002.nii.gz")
        shutil.copy2(flair_path,img_out_dir / f"{case_id}_0003.nii.gz")

        # map all label 4's to  3
        if lab_out_dir is not None:
            seg_nii = nib.load(seg_path)
            seg_data = seg_nii.get_fdata().astype(np.int16)

            # BraTS uses 0,1,2,4
            seg_data[seg_data == 4] = 3

            out_label_path = lab_out_dir / f"{case_id}.nii.gz"
            new_nii = nib.Nifti1Image(seg_data, affine=seg_nii.affine, header=seg_nii.header)
            nib.save(new_nii, out_label_path)

        return case_id

    # Copy all training patients
    print("Copying training patients into imagesTr and labelsTr...")
    for p in train_dirs:
        cid = copy_patient(p, is_train=True)
        train_ids.append(cid)

    # Copy all val/test patients
    print("Copying val/test patients into imagesTs...")
    for p in val_dirs:
        cid = copy_patient(p, is_train=False)
        val_ids.append(cid)

    # Make dataset.json for nnU-Net to understands the dataset
    print("Generating dataset.json...")
    channel_names = {
        0: "T1",
        1: "T1ce",
        2: "T2",
        3: "FLAIR",
    }

    # labels after remapping: 0,1,2,3
    labels = {
        "background": 0,
        "NCR_NET": 1,
        "ED": 2,
        "ET": 3,
    }

    generate_dataset_json(
        output_folder=str(dataset_dir),
        channel_names=channel_names,
        labels=labels,
        num_training_cases=len(train_ids),
        file_ending=".nii.gz",
        dataset_name="TCGA_LGG",
        description="TCGA-LGG dataset converted to nnUNetv2 format",
    )

    return train_ids, val_ids

if __name__ == "__main__":
    train_ids, val_ids = prepare_nnunet_dataset()
    print("Training case IDs:")
    print(train_ids)
    print("Val/test case IDs:")
    print(val_ids)

