"""3D pipeline — volumetric patches cropped from contiguous slabs.

Each train/val/test range becomes one slab held in memory; ``RandCropByPosNegLabeld`` samples
patches from it, biased towards the ROI. Inference is whole-slab sliding-window on the CPU with
the GPU used only for the forward pass, which keeps peak memory bounded.
"""

from __future__ import annotations

import gc
from pathlib import Path
from time import perf_counter
from typing import Dict, Sequence

import monai
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

from ..data import add_background_channel
from ..methods import BranchSpec
from ..models import build_meta_learner, count_parameters
from ..splits import drop_short_ranges
from ..utils import apply_roi
from .base import BasePipeline
from .pipeline2d import _train_meta_loop


class Pipeline3D(BasePipeline):
    spatial_dims = 3

    @property
    def patch(self):
        return self.cfg.train.patch_size

    def adjust_splits(self) -> None:
        # Carving the val/test holes out of the X-ray range can leave a thin edge sliver -- e.g.
        # 56 slices -- too short in H for a 128-deep patch, which crashes RandCropByPosNegLabeld.
        # 2D has no such constraint. Dropping such a sliver costs well under 3% of X-ray training data.
        self.splits["xray"] = drop_short_ranges(
            self.splits["xray"], self.cfg.train.patch_size[0], label="X-ray"
        )

    # -- stage 1 ----------------------------------------------------------
    def build_branch_data(self, spec: BranchSpec):
        vols = [self.vols.modality(m) for m in spec.modalities]
        roi = self.vols.roi(spec.roi)
        splits = self.splits[spec.splits]

        def slab(h0: int, h1: int) -> dict:
            return {
                "image": np.stack([v[h0:h1] for v in vols], axis=0),
                "label": self.label_stack_bg[:, h0:h1],
                "roi": roi[h0:h1][None].astype(np.float32),
            }

        train_slabs = [slab(h0, h1) for h0, h1 in splits.train]
        val_slab = slab(*splits.val)

        keys = ["image", "label", "roi"]
        train_tf = monai.transforms.Compose(
            [
                monai.transforms.RandCropByPosNegLabeld(
                    keys=keys, label_key="roi", spatial_size=self.patch, pos=2, neg=1, num_samples=1
                ),
                monai.transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                monai.transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                monai.transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            ]
        )
        # each slab is repeated so one "epoch" draws many random patches from it
        return train_slabs * self.cfg.train.repeat, [val_slab], train_tf, monai.transforms.Compose([])

    def fit_branch(self, model, data, checkpoint: Path, tag: str = ""):
        train_files, val_files, train_tf, val_tf = data
        tcfg = self.cfg.train
        model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg.num_epochs, eta_min=1e-7
        )
        loss_fn = monai.losses.DiceLoss(softmax=True)

        train_loader = DataLoader(
            Dataset(data=train_files, transform=train_tf),
            batch_size=1,
            shuffle=True,
            num_workers=tcfg.num_workers,
            persistent_workers=tcfg.num_workers > 0,
            pin_memory=True,
        )
        val_loader = DataLoader(Dataset(data=val_files, transform=val_tf), batch_size=1, shuffle=False)

        train_losses, val_losses, best_val = [], [], float("inf")
        t0 = perf_counter()
        for epoch in range(tcfg.num_epochs):
            model.train()
            epoch_loss, steps = 0.0, 0
            for batch in train_loader:
                inp = batch["image"].cuda(non_blocking=True)
                tgt = batch["label"].cuda(non_blocking=True).float()
                roi = batch["roi"].cuda(non_blocking=True)
                optimizer.zero_grad()
                out = apply_roi(model(inp), roi)
                loss = loss_fn(out, tgt)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach()
                steps += 1
            train_loss = epoch_loss.item() / steps
            train_losses.append(train_loss)
            scheduler.step()

            # validation is a full sliding-window pass over the whole val slab, on CPU tensors
            model.eval()
            val_loss = float("nan")
            with torch.no_grad():
                for batch in val_loader:
                    inp = batch["image"].float()
                    tgt = batch["label"].float()
                    roi = batch["roi"]
                    logits = sliding_window_inference(
                        inp,
                        self.patch,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=tcfg.sw_overlap,
                        sw_device="cuda",
                        device="cpu",
                    )
                    val_loss = loss_fn(apply_roi(logits, roi), tgt).item()
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                    },
                    checkpoint,
                )
            print(
                f"  [{tag}] epoch {epoch + 1:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  [{perf_counter() - t0:.0f}s]"
            )

        state = torch.load(checkpoint)
        model.load_state_dict(state["model"])
        model.eval()
        return model, train_losses, val_losses

    # -- stage 2 ----------------------------------------------------------
    def predict_range(self, model, spec: BranchSpec, h0: int, h1: int, want_logits: bool):
        vols = [self.vols.modality(m) for m in spec.modalities]
        roi = self.vols.roi(spec.roi)
        clip = self.cfg.meta.logit_clip

        inp = torch.from_numpy(np.stack([v[h0:h1] for v in vols], axis=0)[None]).float()
        with torch.no_grad():
            logits = sliding_window_inference(
                inp,
                self.patch,
                sw_batch_size=1,
                predictor=model,
                overlap=self.cfg.train.sw_overlap,
                sw_device="cuda",
                device="cpu",
            )
        roi_crop = roi[h0:h1][None]
        probs = torch.softmax(logits, dim=1)[0].numpy()[1:] * roi_crop
        clamped = torch.clamp(logits, -clip, clip)[0].numpy()[1:] * roi_crop if want_logits else None

        del inp, logits
        torch.cuda.empty_cache()
        gc.collect()
        return probs, clamped

    # -- stage 3 ----------------------------------------------------------
    def _meta_input(self, probs, logits, branch_names: Sequence[str], h0: int, h1: int) -> np.ndarray:
        return np.concatenate(
            [probs[b] for b in branch_names]
            + [logits[b] for b in branch_names]
            + [self.vols.xray[h0:h1][None], self.vols.neutron[h0:h1][None]],
            axis=0,
        )

    def fit_meta(self, branch_names: Sequence[str], tag: str, checkpoint: Path):
        mcfg = self.cfg.meta
        meta_input = self._meta_input(self.val_probs, self.val_logits, branch_names, self.val_h0, self.val_h1)
        gt_val = add_background_channel(self.label_stack[:, self.val_h0 : self.val_h1])
        roi_val = self.vols.roi_xray[self.val_h0 : self.val_h1][None].astype(np.float32)

        keys = ["meta_input", "label", "roi"]
        transform = monai.transforms.Compose(
            [
                monai.transforms.RandCropByPosNegLabeld(
                    keys=keys, label_key="roi", spatial_size=self.patch, pos=2, neg=1, num_samples=1
                ),
                monai.transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                monai.transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                monai.transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            ]
        )
        # one slab, sampled `repeat` times per epoch
        sample = {"meta_input": meta_input, "label": gt_val, "roi": roi_val}
        loader = DataLoader(
            Dataset(data=[sample] * mcfg.repeat, transform=transform), batch_size=1, shuffle=True
        )

        in_channels = meta_input.shape[0]
        model = build_meta_learner(3, in_channels, self.cfg.data.model_out_channels, mcfg.hidden).cuda()
        print(f"{tag} — {count_parameters(model)} trainable parameters, input channels={in_channels}")

        return _train_meta_loop(model, loader, self.cfg, tag, checkpoint)

    def apply_meta(self, meta_model, branch_names: Sequence[str]) -> np.ndarray:
        meta_input = self._meta_input(
            self.test_probs, self.test_logits, branch_names, self.test_h0, self.test_h1
        )
        with torch.no_grad():
            logits = sliding_window_inference(
                torch.from_numpy(meta_input[None]).float(),
                self.patch,
                sw_batch_size=1,
                predictor=meta_model,
                overlap=self.cfg.train.sw_overlap,
                sw_device="cuda",
                device="cpu",
            )
        roi_test = self.vols.roi_xray[self.test_h0 : self.test_h1][None]
        return torch.softmax(logits, dim=1)[0].numpy()[1:] * roi_test

    # -- stage 4 ----------------------------------------------------------
    def test_targets(self):
        gt = self.label_stack[:, self.test_h0 : self.test_h1]
        neutron_labels = self.vols.neutron_seg_labels[self.test_h0 : self.test_h1]
        dead = ~self.vols.neutron_signal_mask[self.test_h0 : self.test_h1]
        return gt, neutron_labels, dead
