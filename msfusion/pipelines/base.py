"""Shared orchestration for the 2D and 3D pipelines.

The two dimensionalities differ only in how data is batched (per-slice with padding vs. random
volumetric patches) and how inference is tiled. Everything else -- which branches to train, the
freeze-then-fuse staging, evaluation, reporting -- is identical, and lives here.

Stages:
    1. Train the minimal set of base branches required by the selected methods.
    2. Freeze them and cache their probabilities (and logits, if a meta-learner needs them)
       over the validation and test ranges.
    3. Train one meta-learner per selected meta method, on the frozen validation predictions.
    4. Score every selected method on the test range and write metrics, plots and predictions.
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from ..config import Config
from ..data import Volumes, add_background_channel
from ..methods import BRANCHES, METHODS, BranchSpec, meta_methods, needs_logits, required_branches
from ..metrics import (
    binarise,
    confusion_matrix,
    fragment_rate,
    per_class_metrics,
    per_slice_dice,
    prf1_iou,
)
from ..models import build_backbone, count_parameters
from ..splits import Splits, build_splits
from ..utils import save_convergence_plot, save_per_class_plot, save_per_slice_plot


class BasePipeline(ABC):
    """Template method pattern: subclasses supply the dimensionality-specific pieces."""

    #: 2 or 3 -- set by the subclass
    spatial_dims: int = 2

    def __init__(self, cfg: Config, volumes: Volumes, methods: Sequence[str]):
        self.cfg = cfg
        self.vols = volumes
        self.methods = list(methods)
        self.branch_names = required_branches(self.methods)

        self.splits: Dict[str, Splits] = build_splits(
            volume_h=volumes.shape[0],
            neutron_h_start=cfg.data.neutron_h_start,
            neutron_h_end=cfg.data.neutron_h_end,
            k_folds=cfg.split.k_folds,
            fold_index=cfg.split.fold_index,
            gap=cfg.split.gap,
        )
        self.adjust_splits()

        # Fusion is anchored to the neutron window: averaging or stacking X-ray with neutron only
        # makes sense where the neutron branch actually has a prediction to contribute.
        self.val_h0, self.val_h1 = self.splits["neutron"].val
        self.test_h0, self.test_h1 = self.splits["neutron"].test
        self.xs = np.arange(self.test_h0, self.test_h1)

        self.label_stack = volumes.label_stack()                     # (C, ...) clast-only
        self.label_stack_bg = add_background_channel(self.label_stack)  # (1+C, ...) training target

        self.trained: Dict[str, torch.nn.Module] = {}
        self.val_probs: Dict[str, np.ndarray] = {}
        self.val_logits: Dict[str, np.ndarray] = {}
        self.test_probs: Dict[str, np.ndarray] = {}
        self.test_logits: Dict[str, np.ndarray] = {}

    # -- hooks ------------------------------------------------------------
    def adjust_splits(self) -> None:
        """Optional post-processing of the computed splits (3D drops sub-patch-size ranges)."""

    @abstractmethod
    def build_branch_data(self, spec: BranchSpec):
        """Return whatever :meth:`fit_branch` needs for this branch (files, transforms, ...)."""

    @abstractmethod
    def fit_branch(self, model, data, checkpoint: Path, tag: str):
        """Train one base branch. Returns ``(model, train_losses, val_losses)``."""

    @abstractmethod
    def predict_range(self, model, spec: BranchSpec, h0: int, h1: int, want_logits: bool):
        """Run inference over absolute slice range ``[h0, h1)``.

        Returns ``(probs, logits_or_None)``, each shaped ``(C, h1-h0, ...)`` with the background
        channel dropped and ROI masking applied.
        """

    @abstractmethod
    def fit_meta(self, branch_names: Sequence[str], tag: str, checkpoint: Path):
        """Train a meta-learner on the cached validation predictions of ``branch_names``."""

    @abstractmethod
    def apply_meta(self, meta_model, branch_names: Sequence[str]) -> np.ndarray:
        """Apply a trained meta-learner over the test range."""

    @abstractmethod
    def test_targets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(gt_stack, neutron_gt_labels, dead_detector_mask)`` over the test range."""

    # -- stages -----------------------------------------------------------
    def run(self) -> Dict[str, np.ndarray]:
        self.report_plan()
        self.train_branches()
        self.cache_branch_predictions()
        method_preds = self.build_method_predictions()
        self.evaluate(method_preds)
        return method_preds

    def report_plan(self) -> None:
        cfg = self.cfg
        print("=" * 72)
        print(f"Fold {cfg.split.fold_index} / {cfg.split.k_folds}   ({cfg.dim.upper()})")
        print(f"  X-ray   splits — {self.splits['xray']}")
        print(f"  Neutron splits — {self.splits['neutron']}")
        print(f"  Methods  ({len(self.methods)}): {', '.join(self.methods)}")
        print(f"  Branches ({len(self.branch_names)}): {', '.join(self.branch_names)}")
        n = count_parameters(
            build_backbone("plain", self.spatial_dims, 1, self.cfg.data.model_out_channels, self.cfg.train.dropout)
        )
        print(f"  Params per single-modality branch: {n / 1e6:.2f}M")
        print(f"  Classes: {self.cfg.data.class_names}")
        print("=" * 72)

    def train_branches(self) -> None:
        """Stage 1 -- train only the branches the selected methods actually need."""
        for name in self.branch_names:
            spec = BRANCHES[name]
            ckpt = self.cfg.fold_weights_dir / f"model_{self.cfg.suffix}_{name}.pth"

            model = build_backbone(
                spec.backbone,
                self.spatial_dims,
                spec.in_channels,
                self.cfg.data.model_out_channels,
                self.cfg.train.dropout,
            )

            if self.cfg.resume and ckpt.exists():
                print(f"[{name}] loading existing checkpoint {ckpt}")
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state["model"])
                self.trained[name] = model.cuda().eval()
                continue

            t0 = perf_counter()
            data = self.build_branch_data(spec)
            model, train_losses, val_losses = self.fit_branch(model, data, ckpt, tag=name)
            self.trained[name] = model
            print(f"[{name}] best val loss: {min(val_losses):.4f}  [{perf_counter() - t0:.0f}s]")
            save_convergence_plot(
                train_losses,
                val_losses,
                f"{self.cfg.dim.upper()} — {name}",
                self.cfg.fold_final_dir / f"convergence_{self.cfg.suffix}_{name}.png",
            )

        for model in self.trained.values():
            for p in model.parameters():
                p.requires_grad = False
            model.eval()

    def cache_branch_predictions(self) -> None:
        """Stage 2 -- freeze the branches and cache their val/test outputs."""
        want_logits = needs_logits(self.methods)
        for name, model in self.trained.items():
            spec = BRANCHES[name]
            if self._branch_used_by_meta(name):
                self.val_probs[name], self.val_logits[name] = self.predict_range(
                    model, spec, self.val_h0, self.val_h1, want_logits
                )
            self.test_probs[name], self.test_logits[name] = self.predict_range(
                model, spec, self.test_h0, self.test_h1, want_logits
            )

    def build_method_predictions(self) -> Dict[str, np.ndarray]:
        """Stages 3 and 4 -- fuse cached branch outputs into one prediction per method."""
        preds: Dict[str, np.ndarray] = {}
        for name in self.methods:
            spec = METHODS[name]
            if spec.fusion == "identity":
                preds[name] = self.test_probs[spec.branches[0]]
            elif spec.fusion == "mean":
                preds[name] = np.mean([self.test_probs[b] for b in spec.branches], axis=0)
            elif spec.fusion == "meta":
                ckpt = self.cfg.fold_weights_dir / f"model_{self.cfg.suffix}_{name}.pth"
                meta_model = self.fit_meta(spec.branches, name, ckpt)
                preds[name] = self.apply_meta(meta_model, spec.branches)
            else:
                raise ValueError(f"unknown fusion {spec.fusion!r} for method {name!r}")
        return preds

    def evaluate(self, method_preds: Dict[str, np.ndarray]) -> None:
        """Stage 4 -- metrics against the X-ray-derived ground truth.

        Two levels are reported for every method:

        * **binary clast detection** (``class == "any"``) -- any clast vs. matrix. This is the
          headline number: it asks whether the model found the clast at all.
        * **per clast type** (``class == "dark"`` / ``"medium"`` / ``"bright"``) -- whether the
          clast it found was assigned the right type. Scored by argmax, so the classes are
          mutually exclusive, matching how the softmax output is meant to be read.

        Every reported method is either X-ray-anchored or a joint X-ray+neutron prediction; none
        is a standalone neutron branch, so the X-ray-derived GT is the right reference for all.
        """
        gt_stack, neutron_gt_labels, dead_mask = self.test_targets()
        gt_binary = gt_stack.any(axis=0)
        class_names = self.cfg.data.class_names

        rows: List[dict] = []
        series: Dict[str, np.ndarray] = {}
        class_scores: Dict[str, Dict[str, dict]] = {}
        confusions: Dict[str, np.ndarray] = {}

        print("── Binary clast detection (any clast vs. matrix) ──")
        for name, pred in method_preds.items():
            pred_bin = binarise(pred)
            m = prf1_iou(pred_bin, gt_binary)
            frag = 100 * fragment_rate(pred_bin, dead_mask)
            print(
                f"  {name:18s} Dice={m['dice']:.4f} F1={m['f1']:.4f} "
                f"P={m['precision']:.4f} R={m['recall']:.4f} IoU={m['iou']:.4f} frag={frag:.2f}%"
            )
            rows.append(self._metric_row(name, "any", m, fragmentation=frag))
            series[name] = per_slice_dice(pred_bin, gt_binary)

        print(f"── Per clast type ({', '.join(class_names)}), argmax assignment ──")
        for name, pred in method_preds.items():
            scores = per_class_metrics(pred, gt_stack, class_names, mode="argmax")
            class_scores[name] = scores
            confusions[name] = confusion_matrix(pred, gt_stack)
            summary = "  ".join(f"{cls}={s['dice']:.4f}" for cls, s in scores.items())
            macro = float(np.mean([s["dice"] for s in scores.values()]))
            print(f"  {name:18s} {summary}   macro={macro:.4f}")
            for cls, s in scores.items():
                rows.append(self._metric_row(name, cls, s))
            rows.append(self._metric_row(name, "macro_avg", _macro_average(scores)))

        self._write_metrics(rows)
        self._write_confusions(confusions, class_names)
        save_per_slice_plot(
            self.xs,
            series,
            f"{self.cfg.dim.upper()} — per-slice binary clast-detection Dice",
            self.cfg.fold_final_dir / f"per_slice_dice_methods_{self.cfg.suffix}.png",
        )
        save_per_class_plot(
            class_scores,
            class_names,
            f"{self.cfg.dim.upper()} — Dice per clast type",
            self.cfg.fold_final_dir / f"per_class_dice_methods_{self.cfg.suffix}.png",
        )
        self._save_predictions(method_preds, gt_stack, neutron_gt_labels)
        self._save_run_config()

    def _metric_row(self, method: str, cls: str, m: dict, fragmentation=None) -> dict:
        """One CSV row. ``cls`` is ``"any"``, a clast-type name, or ``"macro_avg"``."""
        return {
            "fold": self.cfg.split.fold_index,
            "k_folds": self.cfg.split.k_folds,
            "method": method,
            "class": cls,
            "gt_source": "xray-GT",
            "dice": m["dice"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "iou": m["iou"],
            "fragmentation_pct": fragmentation if fragmentation is not None else "",
        }

    # -- helpers ----------------------------------------------------------
    def _branch_used_by_meta(self, branch: str) -> bool:
        return any(branch in METHODS[m].branches for m in meta_methods(self.methods))

    def _write_metrics(self, rows: List[dict]) -> None:
        out = self.cfg.fold_final_dir / f"metrics_{self.cfg.suffix}.csv"
        with open(out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {out}")

    def _write_confusions(self, confusions: Dict[str, np.ndarray], class_names: List[str]) -> None:
        """Write one long-format CSV of class-confusion counts across all methods."""
        labels = ["background"] + list(class_names)
        out = self.cfg.fold_final_dir / f"confusion_{self.cfg.suffix}.csv"
        with open(out, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["fold", "method", "gt_class", "pred_class", "voxels"])
            for method, matrix in confusions.items():
                for g, gt_name in enumerate(labels):
                    for p, pred_name in enumerate(labels):
                        writer.writerow(
                            [self.cfg.split.fold_index, method, gt_name, pred_name, int(matrix[g, p])]
                        )
        print(f"[saved] {out}")

    def _save_predictions(self, method_preds, gt_stack, neutron_gt_labels) -> None:
        out = self.cfg.fold_final_dir / f"test_preds_{self.cfg.suffix}.npz"
        np.savez_compressed(
            out,
            **{f"branch_{k}": v.astype(np.float16) for k, v in self.test_probs.items()},
            **{k: v.astype(np.float16) for k, v in method_preds.items()},
            gt=gt_stack.astype(np.uint8),
            neutron_gt_labels=neutron_gt_labels.astype(np.uint8),
            class_names=np.array(self.cfg.data.class_names),
        )
        print(f"[saved] {out}")

    def _save_run_config(self) -> None:
        out = self.cfg.fold_final_dir / f"run_config_{self.cfg.suffix}.json"
        payload = self.cfg.to_dict()
        payload["resolved_methods"] = self.methods
        payload["trained_branches"] = self.branch_names
        payload["splits"] = {
            k: {"train": v.train, "val": list(v.val), "test": list(v.test)} for k, v in self.splits.items()
        }
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[saved] {out}")


def _macro_average(scores: Dict[str, dict]) -> dict:
    """Unweighted mean of each metric across classes -- treats rare classes as equally important."""
    keys = ["dice", "f1", "precision", "recall", "iou"]
    return {k: float(np.mean([s[k] for s in scores.values()])) for k in keys}
