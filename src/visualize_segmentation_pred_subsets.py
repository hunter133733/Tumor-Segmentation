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
        description="Visualize test image with predictions from 10%, 25%, and 90% nnU-Net models."
    )
    parser.add_argument("--image", required=True,
                        help="Path to original image .nii/.nii.gz")
    parser.add_argument("--pred_10", required=True,
                        help="Path to 10%% subset prediction .nii/.nii.gz")
    parser.add_argument("--pred_25", required=True,
                        help="Path to 25%% subset prediction .nii/.nii.gz")
    parser.add_argument("--pred_90", required=True,
                        help="Path to 90%% subset prediction .nii/.nii.gz")

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
        "--title-10", type=str, default="Pred (10% data)",
        help="Title for the 10%% subset prediction panel."
    )
    parser.add_argument(
        "--title-25", type=str, default="Pred (25% data)",
        help="Title for the 25%% subset prediction panel."
    )
    parser.add_argument(
        "--title-90", type=str, default="Pred (90% data)",
        help="Title for the 90%% subset prediction panel."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="If set, save the figure to this path instead of showing it."
    )

    args = parser.parse_args()

    # Load data
    img = load_nifti(args.image)
    pred_10 = load_nifti(args.pred_10)
    pred_25 = load_nifti(args.pred_25)
    pred_90 = load_nifti(args.pred_90)

    # Check shapes match image
    shape = img.shape
    for arr, name in [(pred_10, "pred_10"), (pred_25, "pred_25"), (pred_90, "pred_90")]:
        if arr.shape != shape:
            raise ValueError(
                f"Shape mismatch: image {shape}, {name} {arr.shape}"
            )

    # Extract same slice from all
    img_slice, idx = get_slice(img, axis=args.axis, index=args.slice_index)
    pred_10_slice, _ = get_slice(pred_10, axis=args.axis, index=idx)
    pred_25_slice, _ = get_slice(pred_25, axis=args.axis, index=idx)
    pred_90_slice, _ = get_slice(pred_90, axis=args.axis, index=idx)

    axis_name = {0: "sagittal (x)", 1: "coronal (y)", 2: "axial (z)"}[args.axis]

    # Figure: 1 row, 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Panel 1: image only
    axes[0].imshow(img_slice.T, cmap="gray", origin="lower", vmax=args.vmax)
    axes[0].set_title(f"Image ({axis_name} slice {idx})")
    axes[0].axis("off")

    # Panels 2–4: overlays
    overlay_segmentation(
        axes[1],
        img_slice,
        pred_10_slice,
        title=args.title_10,
        vmax=args.vmax,
        cmap_seg="Blues",
        alpha=0.5,
    )
    overlay_segmentation(
        axes[2],
        img_slice,
        pred_25_slice,
        title=args.title_25,
        vmax=args.vmax,
        cmap_seg="Greens",
        alpha=0.5,
    )
    overlay_segmentation(
        axes[3],
        img_slice,
        pred_90_slice,
        title=args.title_90,
        vmax=args.vmax,
        cmap_seg="Purples",
        alpha=0.5,
    )

    fig.suptitle("Comparison of training subset size predictions on 10 epochs", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output is not None:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()

