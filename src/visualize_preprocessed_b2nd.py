import os

import blosc2
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as np_t

DIR_PREPROCESSED_DS = "./nnUNet_data/nnUNet_preprocessed"

# Hardcode the image to use for now
DS_NAME = "Dataset501_TCGA_LGG"
PKL_FILE = "TCGA-CS-4944.pkl"
B2ND_FILE = "TCGA-CS-4944.b2nd"
SEG_FILE = "TCGA-CS-4944_seg.b2nd"

pref = os.path.join(DIR_PREPROCESSED_DS, DS_NAME, "nnUNetPlans_2d")
path_pkl = os.path.join(pref, PKL_FILE)
path_b2nd = os.path.join(pref, B2ND_FILE)
path_seg = os.path.join(pref, SEG_FILE)

# Load the b2nd files which contain the images
img: np_t.NDArray[np.floating] = blosc2.open(path_b2nd)[:]  # type: ignore
print(f"loaded {path_b2nd}")
print(img.shape)

# Load the segmentation information
seg: np_t.NDArray[np.integer] = blosc2.open(path_seg)[:]    # type: ignore
print(f"loaded {path_seg}")
print(seg.shape)

# Pick the middle slice to visualize
slice = img.shape[1] // 2                     
img_slice_mid: np_t.NDArray[np.floating] = img[0, slice]
seg_slice_mid = seg[0, slice]

# Animate the main MRI image sliding the cross-section up.
fig, ax = plt.subplots()
im = ax.imshow(img[0,0], cmap="gray")

def update_slice_z(frame):
    im.set_data(img[0, frame])
    return [im]

ani = FuncAnimation(fig, update_slice_z, frames=img.shape[1], interval=3000, blit=True)
ani.save("slices.gif", writer="Pillow", fps=30)

# Create a composite plot of three images: MRI, segmentation, overlay of the segmentation
plt.figure()
plt.subplot(1,3,1)
plt.imshow(img_slice_mid, cmap="gray")

# Segmentation
plt.subplot(1,3,2)
plt.imshow(seg_slice_mid, cmap="gray")

# Overlay
plt.subplot(1,3,3)
plt.imshow(img_slice_mid, cmap="gray")
plt.imshow(np.ma.masked_where(seg_slice_mid == 0, seg_slice_mid), alpha=0.5)

plt.savefig("t.png")
