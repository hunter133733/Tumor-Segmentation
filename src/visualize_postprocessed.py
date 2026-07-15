import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Paths to your files
seg_path = "/home/cgutwin/temp/340PROJ_RESULTS_NNUNET/100ep_pp/output/pp/TCGA-DU-7014.nii.gz"
img_path = "/home/cgutwin/dev/cmpt/2025_3_project_07/src/nnUNet_raw/nnUNet_raw/Dataset501_TCGA_LGG/imagesTs/TCGA-DU-7014_0000.nii.gz"   # nnU-Net output

# Load NIfTI
img_nii = nib.load(img_path)
seg_nii = nib.load(seg_path)

img = img_nii.get_fdata()
seg = seg_nii.get_fdata().astype(int)

print("Image shape:", img.shape)
print("Seg shape:", seg.shape)

# Choose a slice axis and index (here: axial slice)
axis = 2
slice_idx = img.shape[axis] // 2  # middle slice

img_slice = np.take(img, slice_idx, axis=axis)
seg_slice = np.take(seg, slice_idx, axis=axis)

# Basic normalization for display
vmin, vmax = np.percentile(img_slice, [1, 99])

# Create a simple colormap for labels:
# 0=background transparent, 1,2,3,... = coloured
max_label = int(seg_slice.max())
colors = np.zeros((max_label + 1, 4))
colors[0] = [0, 0, 0, 0]      # fully transparent for background
if max_label >= 1:
    colors[1] = [1, 0, 0, 0.4]  # class 1: red, semi-transparent
if max_label >= 2:
    colors[2] = [0, 1, 0, 0.4]  # class 2: green
if max_label >= 3:
    colors[3] = [0, 0, 1, 0.4]  # class 3: blue

seg_cmap = ListedColormap(colors)

plt.figure(figsize=(6, 6))
plt.imshow(img_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
plt.imshow(seg_slice.T, cmap=seg_cmap, interpolation="none", origin="lower")
plt.title(f"Slice {slice_idx} (axis={axis})")
plt.axis("off")
plt.savefig("pp.png")
