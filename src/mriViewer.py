import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

'''
Loads a 3D MRI NIfTI (.nii or .nii.gz) scan and displays the middle slices
in axial, coronal, and sagittal views for quick visualization.

CLI Usage: 

python3 mriViewer.py {path_to_.nii.gz_file}

Install:

pip install nibabel matplotlib numpy
'''

def show_slices(img_path):
    img = nib.load(img_path)
    data = img.get_fdata()

    x_mid, y_mid, z_mid = np.array(data.shape) // 2

    axial     = data[:, :, z_mid]     # top-down view
    coronal   = data[:, y_mid, :]     # front-back view
    sagittal  = data[x_mid, :, :]     # left-right view

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(axial.T, origin="lower")
    axes[0].set_title("Axial")

    axes[1].imshow(coronal.T, origin="lower")
    axes[1].set_title("Coronal")

    axes[2].imshow(sagittal.T, origin="lower")
    axes[2].set_title("Sagittal")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_mri.py <path_to_nii_or_nii.gz>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print("File not found:", path)
        sys.exit(1)

    show_slices(str(path))
