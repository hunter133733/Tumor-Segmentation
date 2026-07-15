# SamAnalysis.py
#
# Goal: 
# - export 2D tumor slices for ALL 4 MRI modes:
#   flair, t1, t1ce (T1Gd), t2 into a structure so SAM can read easily it:

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# directory that contains SamAnalysis.py
HERE = Path(__file__).resolve().parent

# where the TCGA-LGG data lives, relative to this file
DATA_ROOT = (
    HERE
    / "PKG - BraTS-TCGA-LGG"
    / "BraTS-TCGA-LGG"
    / "Pre-operative_TCGA_LGG_NIfTI_and_Segmentations"
)

# where to write the exported slices
OUT_ROOT = HERE / "sam_slices"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# one subfolder per modality
MODALITIES = ["flair", "t1", "t1ce", "t2"]
for m in MODALITIES:
    (OUT_ROOT / m / "images").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / m / "masks").mkdir(parents=True, exist_ok=True)

# ID's of patients
patient_ids = ["TCGA-CS-4942", "TCGA-DU-7298"]

def pick_file_ci(files, must_contain, must_not_contain=None):
    if must_not_contain is None:
        must_not_contain = []
    for f in files:
        name = f.name.lower()
        if all(s.lower() in name for s in must_contain) and not any(
            s.lower() in name for s in must_not_contain
        ):
            return f


for pid in patient_ids:
    pdir = DATA_ROOT / pid
    nii_files = list(pdir.glob("*.nii.gz"))
    if not nii_files:
        print("No NIfTI files for", pid)
        continue

    # segmentation
    try:
        seg_path = pick_file_ci(nii_files, ["glistrboost"])
    except FileNotFoundError:
        seg_path = pick_file_ci(nii_files, ["seg"])

    # modalities
    flair_path = pick_file_ci(nii_files, ["flair"])
    t2_path    = pick_file_ci(nii_files, ["t2"])
    t1ce_path  = pick_file_ci(nii_files, ["t1", "gd"])
    t1_path    = pick_file_ci(nii_files, ["t1"], ["gd"])

    # load volumes
    flair_vol = nib.load(flair_path).get_fdata()
    t2_vol    = nib.load(t2_path).get_fdata()
    t1ce_vol  = nib.load(t1ce_path).get_fdata()
    t1_vol    = nib.load(t1_path).get_fdata()
    seg_vol   = nib.load(seg_path).get_fdata().astype(np.int16)

    # binary tumor mask
    seg_bin = (seg_vol > 0).astype(np.uint8)

    #volumes should have same shape
    assert flair_vol.shape == t1_vol.shape == t1ce_vol.shape == t2_vol.shape == seg_bin.shape

    volumes = {
        "flair": flair_vol,
        "t1":    t1_vol,
        "t1ce":  t1ce_vol,
        "t2":    t2_vol,
    }

    # iterate over slices
    for z in range(flair_vol.shape[2]):
        if seg_bin[..., z].sum() == 0:
            continue

        mask_slice = seg_bin[..., z]

        for mod_name, vol in volumes.items():
            slice_img = vol[..., z]

            # normalize to [0,1], then 0-255, then RGB. If we dont normalize, then
            # the images will be washed out and SAM model will perform much worse. 
            sl = slice_img
            sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
            rgb = (sl * 255).astype(np.uint8)
            rgb = np.stack([rgb, rgb, rgb], axis=-1)

            img_path = OUT_ROOT / mod_name / "images" / f"{pid}_z{z:03d}.png"
            mask_path = OUT_ROOT / mod_name / "masks" / f"{pid}_z{z:03d}_mask.png"

            plt.imsave(img_path, rgb)
            plt.imsave(mask_path, mask_slice, cmap="gray")

    print(f"Done patient {pid}")

print("All selected patients processed.")
