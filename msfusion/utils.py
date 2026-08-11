"""Small shared helpers: GPU selection, padding, ROI masking and plotting."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# GPU selection -- must run before torch is imported
# ---------------------------------------------------------------------------
def select_gpu(verbose: bool = True) -> Optional[int]:
    """Pin ``CUDA_VISIBLE_DEVICES`` to the GPU with the most free memory.

    Called from the entry point *before* torch is imported, since torch caches the visible-device
    list at import time. A scheduler-provided ``CUDA_VISIBLE_DEVICES`` is always respected.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        if verbose:
            print(f"CUDA_VISIBLE_DEVICES already set by scheduler: {os.environ['CUDA_VISIBLE_DEVICES']}")
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip().split("\n")
        free = [int(x) for x in out]
    except (OSError, subprocess.CalledProcessError, ValueError):
        if verbose:
            print("nvidia-smi unavailable -- leaving CUDA_VISIBLE_DEVICES unset")
        return None
    idx = free.index(max(free))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    if verbose:
        print(f"GPU selected: {idx}  ({free[idx]} MiB free)  all: {free}")
    return idx


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Padding (2D pipeline)
# ---------------------------------------------------------------------------
def pad_to(arr: np.ndarray, h: int, w: int, value: float = 0) -> np.ndarray:
    """Symmetrically constant-pad the last two axes of ``arr`` up to ``(h, w)``."""
    H, W = arr.shape[-2], arr.shape[-1]
    if h < H or w < W:
        raise ValueError(f"target ({h},{w}) smaller than source ({H},{W})")
    top, bottom = (h - H) // 2, (h - H) - (h - H) // 2
    left, right = (w - W) // 2, (w - W) - (w - W) // 2
    pad_width = [(0, 0)] * (arr.ndim - 2) + [(top, bottom), (left, right)]
    return np.ascontiguousarray(np.pad(arr, pad_width, mode="constant", constant_values=value))


def padded_image_stack(volume: np.ndarray, h0: int, h1: int, pad_size) -> np.ndarray:
    """Pad slices ``[h0:h1)`` of a volume and return them with a leading channel axis."""
    out = np.zeros((h1 - h0, *pad_size), dtype=np.float32)
    for i in range(h0, h1):
        out[i - h0] = pad_to(volume[i], *pad_size)
    return out[None]


# ---------------------------------------------------------------------------
# ROI handling
# ---------------------------------------------------------------------------
BACKGROUND_LOGIT, FOREGROUND_LOGIT = 20.0, -20.0


def outside_roi_fill(out):
    """Logit tensor used to overwrite predictions outside the ROI mask.

    Strongly favours the background channel (index 0), so the softmax output there matches the
    guaranteed target (background=1, every clast channel=0). This replaces the older sigmoid
    trick of pushing all channels towards zero, which has no meaning under a softmax.
    """
    import torch

    fill = torch.full_like(out, FOREGROUND_LOGIT)
    fill[:, 0] = BACKGROUND_LOGIT
    return fill


def apply_roi(out, roi):
    """Keep model logits inside the ROI, force background outside it."""
    import torch

    return torch.where(roi.expand_as(out) > 0, out, outside_roi_fill(out))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_convergence_plot(train_losses, val_losses, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="train")
    if val_losses is not None:
        ax.plot(val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_per_slice_plot(xs, series, title: str, out_path: Path) -> None:
    """``series`` maps a method name to its per-slice Dice array."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 4.5))
    for name, ys in series.items():
        ax.plot(xs, ys, alpha=0.8, linewidth=1.1, label=f"{name}  mean={np.nanmean(ys):.3f}")
    ax.axhline(0.5, color="lightgray", linestyle=":", linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Slice index (H)")
    ax.set_ylabel("Dice")
    ax.set_title("Binary clast detection")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_per_class_plot(class_scores, class_names, title: str, out_path: Path) -> None:
    """Grouped bar chart of Dice per clast type, one group of bars per method.

    ``class_scores`` maps a method name to ``{class_name: metric_dict}``.
    """
    import matplotlib.pyplot as plt

    methods = list(class_scores)
    n_methods = len(methods)
    x = np.arange(len(class_names))
    width = min(0.8 / max(n_methods, 1), 0.25)

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(class_names) * n_methods / 2), 4.5))
    for i, method in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * width
        heights = [class_scores[method][c]["dice"] for c in class_names]
        ax.bar(x + offset, heights, width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Dice")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[saved] {out_path}")
