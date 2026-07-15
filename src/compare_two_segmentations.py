#!/usr/bin/env python3
import argparse

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def load_nifti(path):
    """Load a NIfTI file and return its data as a numpy array."""
    img = nib.load(path)
    data = img.get_fdata()
    return data


def get_slice(data, axis=2, index=None):
    """
    Extract a 2D slice from a 3D volume.

    axis:
        0 -> sagittal (x)
        1 -> coronal (y)
        2 -> axial   (z)  [default]
    index:
        if None, use the middle slice along that axis.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D data, got shape {data.shape}")

    if index is None:
        index = data.shape[axis] // 2

    if axis == 0:
        slice2d = data[index, :, :]
    elif axis == 1:
        slice2d = data[:, index, :]
    elif axis == 2:
        slice2d = data[:, :, index]
    else:
        raise ValueError("axis must be 0, 1, or 2")

    return slice2d, index


def overlay_segmentation(ax, img_slice, seg_slice, title, vmax=None,
                         cmap_img="gray", cmap_seg="tab10", alpha=0.5):
    """
    Show an image slice with a segmentation overlay.
    seg_slice is assumed to be integer labels (0 = background).
    """
    ax.imshow(img_slice.T, cmap=cmap_img, origin="lower", vmax=vmax)
    seg = seg_slice.T

    # Mask out background
    seg_masked = np.ma.masked_where(seg == 0, seg)

    ax.imshow(seg_masked, cmap=cmap_seg, alpha=alpha, origin="lower")
    ax.set_title(title)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two segmentation volumes side-by-side on a single slice."
    )
    parser.add_argument("--image", required=True,
                        help="Path to original image .nii/.nii.gz")
    parser.add_argument("--seg_a", required=True,
                        help="Path to first segmentation .nii/.nii.gz")
    parser.add_argument("--seg_b", required=True,
                        help="Path to second segmentation .nii/.nii.gz")

    parser.add_argument(
        "--axis", type=int, default=2, choices=[0, 1, 2],
        help="Axis along which to take the slice (0=sagittal, 1=coronal, 2=axial). Default: 2"
    )
    parser.add_argument(
        "--slice-index", type=int, default=None,
        help="Index of slice along the chosen axis. Default: middle slice."
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Optional upper limit for grayscale intensity."
    )
    parser.add_argument(
        "--title-a", type=str, default="Segmentation A",
        help="Title for the first segmentation panel."
    )
    parser.add_argument(
        "--title-b", type=str, default="Segmentation B",
        help="Title for the second segmentation panel."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="If set, save the figure to this path instead of showing it."
    )

    args = parser.parse_args()

    # Load data
    img = load_nifti(args.image)
    seg_a = load_nifti(args.seg_a)
    seg_b = load_nifti(args.seg_b)

    # Sanity checks
    shape = img.shape
    for arr, name in [(seg_a, "seg_a"), (seg_b, "seg_b")]:
        if arr.shape != shape:
            raise ValueError(
                f"Shape mismatch: image {shape}, {name} {arr.shape}"
            )

    # Extract same slice from all
    img_slice, idx = get_slice(img, axis=args.axis, index=args.slice_index)
    seg_a_slice, _ = get_slice(seg_a, axis=args.axis, index=idx)
    seg_b_slice, _ = get_slice(seg_b, axis=args.axis, index=idx)

    axis_name = {0: "sagittal (x)", 1: "coronal (y)", 2: "axial (z)"}[args.axis]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Image only
    axes[0].imshow(img_slice.T, cmap="gray", origin="lower", vmax=args.vmax)
    axes[0].set_title(f"Image ({axis_name} slice {idx})")
    axes[0].axis("off")

    # Image + segmentation A
    overlay_segmentation(
        axes[1],
        img_slice,
        seg_a_slice,
        title=args.title_a,
        vmax=args.vmax,
        cmap_seg="Blues",
        alpha=0.5,
    )

    # Image + segmentation B
    overlay_segmentation(
        axes[2],
        img_slice,
        seg_b_slice,
        title=args.title_b,
        vmax=args.vmax,
        cmap_seg="Reds",
        alpha=0.5,
    )

    fig.suptitle("Comparison of Two Segmentations", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output is not None:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()

