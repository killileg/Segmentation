"""Segmentation metrics: Dice / F1 / precision / recall / IoU, plus fragmentation rate."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

EPS = 1e-8


def confusion(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int]:
    """Return ``(tp, fp, fn)`` for two boolean-castable arrays of identical shape."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return tp, fp, fn


def prf1_iou(pred: np.ndarray, gt: np.ndarray, eps: float = EPS) -> Dict[str, float]:
    """Precision, recall, F1 (== Dice for a binary mask) and IoU."""
    tp, fp, fn = confusion(pred, gt)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"dice": f1, "f1": f1, "precision": precision, "recall": recall, "iou": iou}


def fragment_rate(pred: np.ndarray, black_mask: np.ndarray, eps: float = EPS) -> float:
    """Fraction of predicted voxels that fall where the neutron detector saw nothing.

    A high value means the model is hallucinating clasts into dead detector regions.
    """
    pred = pred.astype(bool)
    n_pred = pred.sum()
    if not n_pred:
        return 0.0
    return float((pred & black_mask.astype(bool)).sum()) / (n_pred + eps)


def per_slice_dice(pred_vol: np.ndarray, gt_vol: np.ndarray) -> np.ndarray:
    """Dice computed independently per slice along axis 0.

    Slices where both prediction and ground truth are empty are ``nan`` rather than 1.0, so they
    don't inflate the mean.
    """
    out = []
    for i in range(pred_vol.shape[0]):
        p = pred_vol[i] > 0.5
        g = gt_vol[i] > 0.5
        tp = (p & g).sum()
        fp = (p & ~g).sum()
        fn = (~p & g).sum()
        out.append(2 * tp / (2 * tp + fp + fn + EPS) if (g.any() or p.any()) else np.nan)
    return np.array(out)


def binarise(pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Collapse a ``(C, ...)`` per-class probability map to a single any-clast boolean mask."""
    return (pred > threshold).any(axis=0)


def argmax_labels(pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a ``(C, ...)`` probability map to hard class indices.

    Returns ``-1`` where no class exceeds ``threshold`` (i.e. background), otherwise the index of
    the winning clast class. Because the network outputs a softmax over background + C clast
    channels and the background channel is dropped before this point, taking the argmax over the
    remaining channels is equivalent to asking "which clast type, given that this is a clast".
    """
    winner = pred.argmax(axis=0)
    detected = pred.max(axis=0) > threshold
    return np.where(detected, winner, -1)


def per_class_metrics(
    pred: np.ndarray,
    gt_stack: np.ndarray,
    class_names: Sequence[str],
    threshold: float = 0.5,
    mode: str = "argmax",
) -> Dict[str, Dict[str, float]]:
    """Score each clast class separately.

    Args:
        pred: ``(C, ...)`` per-class probabilities, background channel already dropped.
        gt_stack: ``(C, ...)`` one-hot ground truth in the same class order.
        class_names: human-readable name per channel.
        threshold: probability above which a voxel counts as detected.
        mode: ``"argmax"`` assigns each detected voxel to its single winning class, which matches
            how a softmax model is meant to be read and makes the classes mutually exclusive.
            ``"threshold"`` scores each channel independently against its own threshold, which
            is more forgiving and can be useful for diagnosing whether a class is being detected
            at all but losing the argmax to a neighbour.

    Returns:
        Mapping from class name to its metric dict.
    """
    if mode not in ("argmax", "threshold"):
        raise ValueError(f"mode must be 'argmax' or 'threshold', got {mode!r}")

    out: Dict[str, Dict[str, float]] = {}
    labels = argmax_labels(pred, threshold) if mode == "argmax" else None
    for c, name in enumerate(class_names):
        pred_c = (labels == c) if mode == "argmax" else (pred[c] > threshold)
        out[name] = prf1_iou(pred_c, gt_stack[c])
    return out


def confusion_matrix(
    pred: np.ndarray, gt_stack: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Class-confusion counts over clast voxels, shape ``(C+1, C+1)``.

    Row 0 / column 0 are background; rows are ground truth, columns are prediction. Useful for
    seeing *which* class a misclassified clast was assigned to, which the per-class Dice alone
    doesn't reveal.
    """
    C = gt_stack.shape[0]
    pred_lab = argmax_labels(pred, threshold) + 1          # background becomes 0
    gt_lab = np.zeros(gt_stack.shape[1:], dtype=np.int64)
    for c in range(C):
        gt_lab[gt_stack[c].astype(bool)] = c + 1

    matrix = np.zeros((C + 1, C + 1), dtype=np.int64)
    for g in range(C + 1):
        mask = gt_lab == g
        if not mask.any():
            continue
        counts = np.bincount(pred_lab[mask], minlength=C + 1)
        matrix[g] = counts[: C + 1]
    return matrix
