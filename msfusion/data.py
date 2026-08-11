"""Volume loading and ground-truth preparation.

Everything the pipelines need from disk is loaded exactly once, into a single
:class:`Volumes` object, which is then passed explicitly to the rest of the code.
The original scripts kept these as module-level globals; making them an object is
what allows the pipeline functions to be imported and tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, zoom

from .config import DataConfig, PathConfig


@dataclass
class Volumes:
    """All image / mask arrays on the common ``(H, D, W)`` grid."""

    #: raw, uncalibrated CT-reconstruction intensity -- deliberately not rescaled
    xray: np.ndarray
    neutron: np.ndarray
    #: X-ray-derived segmentation labels (0 = matrix/background, 1..3 = clast types)
    seg: np.ndarray
    #: where the neutron detector actually recorded signal
    neutron_signal_mask: np.ndarray
    #: ROI the X-ray branch is scored on: dilated non-zero segmentation
    roi_xray: np.ndarray
    #: ROI the neutron branch is scored on: X-ray ROI intersected with neutron signal
    roi_neutron: np.ndarray
    #: independently drawn neutron segmentation -- a "double confirmation" GT, never trained on.
    #: Optional: all zeros if no such file was supplied.
    neutron_seg_labels: np.ndarray
    neutron_gt_clast: np.ndarray
    #: whether an independent neutron segmentation was actually provided
    has_neutron_seg: bool = False

    label_names: Dict[int, str]
    class_ids: List[int]

    @property
    def shape(self):
        return self.xray.shape

    def label_stack(self) -> np.ndarray:
        """One-hot ``(C, H, D, W)`` stack of the configured clast classes."""
        return np.stack([(self.seg == c) for c in self.class_ids], axis=0).astype(np.uint8)

    def modality(self, name: str) -> np.ndarray:
        """Look a modality up by the name used in the branch registry."""
        if name == "xray":
            return self.xray
        if name == "neutron":
            return self.neutron
        raise KeyError(f"unknown modality {name!r} (expected 'xray' or 'neutron')")

    def roi(self, name: str) -> np.ndarray:
        if name == "xray":
            return self.roi_xray
        if name == "neutron":
            return self.roi_neutron
        raise KeyError(f"unknown ROI {name!r} (expected 'xray' or 'neutron')")


def add_background_channel(class_stack: np.ndarray) -> np.ndarray:
    """Prepend an explicit background channel to a ``(C, ...)`` one-hot stack, giving ``(1+C, ...)``.

    Required for softmax outputs: with only the C mutually exclusive clast channels a plain
    softmax would force every voxel -- including pure matrix -- to commit to one of them, since
    softmax normalises to sum=1 across channels with no "none of the above" option. Channel 0
    is background throughout the codebase.
    """
    background = (~class_stack.any(axis=0, keepdims=True)).astype(np.uint8)
    return np.concatenate([background, class_stack], axis=0)


def load_volumes(paths: PathConfig, data_cfg: DataConfig, verbose: bool = True) -> Volumes:
    """Read the four NIfTI files and derive the masks the pipelines depend on."""

    xray = nib.load(paths.xray).get_fdata(dtype=np.float32)

    neutron = nib.load(paths.neutron).get_fdata(dtype=np.float32)
    neutron_signal_mask = neutron > 0

    seg_raw = nib.load(paths.seg).get_fdata()
    seg = zoom(np.round(seg_raw).astype(np.uint8), data_cfg.seg_zoom, order=0)
    del seg_raw

    roi_xray = binary_dilation(seg != 0, iterations=data_cfg.roi_dilation_iters)
    roi_neutron = roi_xray & neutron_signal_mask

    if paths.neutron_seg is None:
        if verbose:
            print(
                "no independent neutron segmentation supplied -- skipping the double-confirmation "
                "export (this is optional and does not affect training or the reported metrics)"
            )
        neutron_seg = np.zeros(xray.shape, dtype=np.uint8)
        has_neutron_seg = False
    else:
        has_neutron_seg = True
        neutron_seg = nib.load(paths.neutron_seg).get_fdata(dtype=np.float32)
    if has_neutron_seg and neutron_seg.shape != xray.shape:
        factors = tuple(t / s for t, s in zip(xray.shape, neutron_seg.shape))
        if verbose:
            print(
                f"{paths.neutron_seg.name} at {neutron_seg.shape} -- upsampling by {factors} "
                f"to match the {xray.shape} grid"
            )
        neutron_seg = zoom(np.round(neutron_seg).astype(np.uint8), factors, order=0)
    else:
        neutron_seg = np.round(neutron_seg).astype(np.uint8)

    lo, hi = min(data_cfg.label_names), max(data_cfg.label_names)
    neutron_gt_clast = (neutron_seg >= lo) & (neutron_seg <= hi)
    if neutron_gt_clast.shape != xray.shape:
        raise ValueError(
            f"neutron clast GT shape {neutron_gt_clast.shape} != image grid {xray.shape}"
        )

    vols = Volumes(
        xray=xray,
        neutron=neutron,
        seg=seg,
        neutron_signal_mask=neutron_signal_mask,
        roi_xray=roi_xray,
        roi_neutron=roi_neutron,
        neutron_seg_labels=neutron_seg,
        neutron_gt_clast=neutron_gt_clast,
        label_names=data_cfg.label_names,
        class_ids=list(data_cfg.class_ids),
        has_neutron_seg=has_neutron_seg,
    )

    if verbose:
        summarise(vols, data_cfg)
    return vols


def summarise(vols: Volumes, data_cfg: DataConfig) -> None:
    """Print the coverage statistics the original scripts logged at start-up."""
    print(f"xray: {vols.xray.shape}   neutron: {vols.neutron.shape}")
    print(f"  X-ray ROI coverage   : {vols.roi_xray.mean() * 100:.2f}%")
    print(f"  neutron ROI coverage : {vols.roi_neutron.mean() * 100:.2f}%")
    if vols.has_neutron_seg:
        print(
            f"  neutron-drawn clast GT (any type): {vols.neutron_gt_clast.mean() * 100:.4f}% "
            f"of volume (independent of the X-ray-derived labels)"
        )
    for c, name in data_cfg.label_names.items():
        print(f"  {name:10s}: {(vols.seg == c).mean() * 100:.4f}% of volume")

    has_signal = vols.neutron_signal_mask.reshape(vols.neutron_signal_mask.shape[0], -1).any(axis=1)
    valid = np.where(has_signal)[0]
    auto_start, auto_end = int(valid.min()), int(valid.max()) + 1
    print(
        f"  auto-detected neutron FOV: slices {auto_start}-{auto_end} "
        f"({auto_end - auto_start} slices) -- restricted to "
        f"{data_cfg.neutron_h_start}-{data_cfg.neutron_h_end} "
        f"({data_cfg.neutron_h_end - data_cfg.neutron_h_start} slices) to exclude edge artefacts. "
        f"X-ray keeps its full native range 0-{vols.xray.shape[0]}."
    )
