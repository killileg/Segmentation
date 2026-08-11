"""2D pipeline — every H slice is an independent training sample.

Slices are zero-padded to a common ``(D, W)`` so they can be batched together; predictions are
cropped back by ROI masking rather than by unpadding, since the padded border is guaranteed
background.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import monai
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from torch.utils.data import WeightedRandomSampler

from ..data import add_background_channel
from ..methods import BranchSpec
from ..models import build_meta_learner, count_parameters
from ..utils import apply_roi, pad_to, padded_image_stack, save_convergence_plot
from .base import BasePipeline


class Pipeline2D(BasePipeline):
    spatial_dims = 2

    @property
    def pad(self) -> Tuple[int, int]:
        return self.cfg.train.pad_size

    # -- stage 1 ----------------------------------------------------------
    def build_branch_data(self, spec: BranchSpec):
        vols = [self.vols.modality(m) for m in spec.modalities]
        roi = self.vols.roi(spec.roi)
        splits = self.splits[spec.splits]

        def build(h0: int, h1: int) -> List[dict]:
            return [
                {
                    "image": np.stack([pad_to(v[i], *self.pad) for v in vols], axis=0),
                    "label": pad_to(self.label_stack_bg[:, i], *self.pad),
                    "roi": pad_to(roi[i][None].astype(np.float32), *self.pad),
                }
                for i in range(h0, h1)
            ]

        train_files = [f for h0, h1 in splits.train for f in build(h0, h1)]
        val_files = build(*splits.val)
        return train_files, val_files

    @staticmethod
    def _transforms(keys=("image", "label", "roi")):
        keys = list(keys)
        train_tf = monai.transforms.Compose(
            [
                monai.transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                monai.transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                monai.transforms.RandRotated(
                    keys=keys,
                    prob=0.2,
                    range_x=0.3,
                    range_y=0.0,
                    range_z=0.0,
                    mode=["bilinear"] + ["nearest"] * (len(keys) - 1),
                ),
            ]
        )
        return train_tf, monai.transforms.Compose([])

    def fit_branch(self, model, data, checkpoint: Path, tag: str = ""):
        train_files, val_files = data
        tcfg = self.cfg.train
        model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg.num_epochs, eta_min=tcfg.eta_min
        )
        loss_fn = monai.losses.DiceLoss(softmax=True)
        train_tf, val_tf = self._transforms()

        # uniform weights today, but the sampler is kept so class-frequency reweighting can be
        # dropped in without changing the loader plumbing
        weights = np.ones(len(train_files), dtype=np.float32)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_files), replacement=True)
        train_loader = DataLoader(
            Dataset(data=train_files, transform=train_tf),
            batch_size=tcfg.batch_size,
            sampler=sampler,
            num_workers=tcfg.num_workers,
            persistent_workers=tcfg.num_workers > 0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            Dataset(data=val_files, transform=val_tf), batch_size=tcfg.batch_size, shuffle=False
        )

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

            model.eval()
            v_loss, v_steps = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    inp = batch["image"].cuda()
                    tgt = batch["label"].cuda().float()
                    roi = batch["roi"].cuda()
                    out = apply_roi(model(inp), roi)
                    v_loss += loss_fn(out, tgt).item()
                    v_steps += 1
            val_loss = v_loss / v_steps
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
        """Per-slice sliding-window inference over ``[h0, h1)``.

        Returns softmax probabilities and, optionally, the raw pre-softmax logits clamped to
        +/-``logit_clip``. Both drop the background channel, so downstream shapes are ``(C, ...)``
        with C == the number of clast classes.
        """
        vols = [self.vols.modality(m) for m in spec.modalities]
        roi = self.vols.roi(spec.roi)
        clip = self.cfg.meta.logit_clip

        out_probs = out_logits = None
        model.eval()
        with torch.no_grad():
            for i in range(h0, h1):
                sl = np.stack([pad_to(v[i], *self.pad) for v in vols], axis=0)
                inp = torch.from_numpy(sl[None]).float()
                logits = sliding_window_inference(
                    inp,
                    self.pad,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=self.cfg.train.sw_overlap,
                    sw_device="cuda",
                    device="cpu",
                )
                probs = torch.softmax(logits, dim=1)[0].numpy()[1:]
                if out_probs is None:
                    C = probs.shape[0]
                    out_probs = np.zeros((C, h1 - h0, *self.pad), dtype=np.float32)
                    if want_logits:
                        out_logits = np.zeros((C, h1 - h0, *self.pad), dtype=np.float32)
                out_probs[:, i - h0] = probs
                if want_logits:
                    out_logits[:, i - h0] = torch.clamp(logits, -clip, clip)[0].numpy()[1:]

        roi_crop = pad_to(roi[h0:h1][None], *self.pad)[0]
        probs = out_probs * roi_crop[None]
        logits = out_logits * roi_crop[None] if want_logits else None
        return probs, logits

    # -- stage 3 ----------------------------------------------------------
    def _meta_input(self, probs: Dict[str, np.ndarray], logits: Dict[str, np.ndarray],
                    branch_names: Sequence[str], h0: int, h1: int) -> np.ndarray:
        return np.concatenate(
            [probs[b] for b in branch_names]
            + [logits[b] for b in branch_names]
            + [
                padded_image_stack(self.vols.xray, h0, h1, self.pad),
                padded_image_stack(self.vols.neutron, h0, h1, self.pad),
            ],
            axis=0,
        )

    def fit_meta(self, branch_names: Sequence[str], tag: str, checkpoint: Path):
        mcfg = self.cfg.meta
        meta_input = self._meta_input(self.val_probs, self.val_logits, branch_names, self.val_h0, self.val_h1)

        # The meta-learner stays X-ray-anchored, so its supervision is the X-ray-derived label
        # stack. The background channel is added *after* padding, so the zero-padded border reads
        # as background=1 rather than background=0.
        gt_val = add_background_channel(pad_to(self.label_stack[:, self.val_h0 : self.val_h1], *self.pad))
        roi_val = pad_to(self.vols.roi_xray[self.val_h0 : self.val_h1][None].astype(np.float32), *self.pad)

        files = [
            {"meta_input": meta_input[:, i], "label": gt_val[:, i], "roi": roi_val[:, i]}
            for i in range(meta_input.shape[1])
        ]
        keys = ["meta_input", "label", "roi"]
        transform = monai.transforms.Compose(
            [
                monai.transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                monai.transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            ]
        )
        loader = DataLoader(Dataset(data=files, transform=transform), batch_size=mcfg.batch_size, shuffle=True)

        in_channels = meta_input.shape[0]
        model = build_meta_learner(2, in_channels, self.cfg.data.model_out_channels, mcfg.hidden).cuda()
        print(f"{tag} — {count_parameters(model)} trainable parameters, input channels={in_channels}")

        return _train_meta_loop(model, loader, self.cfg, tag, checkpoint)

    def apply_meta(self, meta_model, branch_names: Sequence[str]) -> np.ndarray:
        meta_input = self._meta_input(
            self.test_probs, self.test_logits, branch_names, self.test_h0, self.test_h1
        )
        C = self.cfg.data.out_channels
        pred = np.zeros((C, self.test_h1 - self.test_h0, *self.pad), dtype=np.float32)
        with torch.no_grad():
            for i in range(meta_input.shape[1]):
                inp = torch.from_numpy(meta_input[:, i][None]).float().cuda()
                pred[:, i] = torch.softmax(meta_model(inp), dim=1)[0].cpu().numpy()[1:]
        roi_test = pad_to(
            self.vols.roi_xray[self.test_h0 : self.test_h1][None].astype(np.float32), *self.pad
        )[0]
        return pred * roi_test[None]

    # -- stage 4 ----------------------------------------------------------
    def test_targets(self):
        gt = pad_to(self.label_stack[:, self.test_h0 : self.test_h1], *self.pad)
        neutron_labels = pad_to(self.vols.neutron_seg_labels[self.test_h0 : self.test_h1][None], *self.pad)[0]
        dead = ~pad_to(self.vols.neutron_signal_mask[self.test_h0 : self.test_h1][None], *self.pad)[0]
        return gt, neutron_labels, dead


def _train_meta_loop(model, loader, cfg, tag: str, checkpoint: Path):
    """Shared meta-learner training loop (identical in 2D and 3D once the loader is built)."""
    mcfg = cfg.meta
    optimizer = torch.optim.Adam(model.parameters(), lr=mcfg.lr, weight_decay=mcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=mcfg.epochs, eta_min=mcfg.eta_min)
    loss_fn = monai.losses.DiceLoss(softmax=True)

    best, losses = float("inf"), []
    t0 = perf_counter()
    for epoch in range(mcfg.epochs):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch in loader:
            inp = batch["meta_input"].cuda(non_blocking=True).float()
            tgt = batch["label"].cuda(non_blocking=True).float()
            roi = batch["roi"].cuda(non_blocking=True)
            optimizer.zero_grad()
            out = apply_roi(model(inp), roi)
            loss = loss_fn(out, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
            steps += 1
        ep_loss = epoch_loss.item() / steps
        losses.append(ep_loss)
        scheduler.step()
        if ep_loss < best:
            best = ep_loss
            torch.save({"meta_model": model.state_dict()}, checkpoint)
        if (epoch + 1) % 10 == 0:
            print(f"  {tag} epoch {epoch + 1:3d}  loss={ep_loss:.4f}  [{perf_counter() - t0:.0f}s]")

    print(f"Best {tag} loss: {best:.4f}")
    save_convergence_plot(losses, None, tag, cfg.fold_final_dir / f"convergence_{cfg.suffix}_{tag}.png")

    state = torch.load(checkpoint)
    model.load_state_dict(state["meta_model"])
    model.eval()
    return model
