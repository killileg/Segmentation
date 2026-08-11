"""2D backbones and meta-learner.

Every network takes an arbitrary ``in_channels``, which is how the early-fusion branch
(``in_channels=2``, X-ray and neutron stacked) reuses exactly the same class as every
single-modality branch (``in_channels=1``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

from .blocks2d import DynUNetSSFB2D

#: encoder/decoder widths -- shared with the 3D backbones so the two are comparable
DYNUNET_FILTERS = [32, 64, 128, 256]
DYNUNET_KERNELS = [[3, 3]] * 4
DYNUNET_STRIDES = [[1, 1], [2, 2], [2, 2], [2, 2]]
DYNUNET_UPSAMPLE_KERNELS = DYNUNET_STRIDES[1:]

#: attention pool sizes per decoder stage, coarse-to-fine. ``None`` at the finest stage disables
#: the attention path there, where it would be prohibitively expensive.
SSFB_POOL_SIZES = [8, 4, None]


class PlainUNet2D(DynUNet):
    """A stock MONAI ``DynUNet`` with ``spatial_dims=2``, nothing more.

    This class exists only to pin the hyperparameters in one place, so that every branch using
    the "plain" backbone is guaranteed byte-for-byte identical in architecture. That is what
    makes ``single_modality_x`` / ``early_fusion`` / ``late_fusion`` / ``meta_learner`` a clean
    comparison of *fusion strategies* rather than of network capacity.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, dropout: float = 0.2):
        super().__init__(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=DYNUNET_KERNELS,
            strides=DYNUNET_STRIDES,
            upsample_kernel_size=DYNUNET_UPSAMPLE_KERNELS,
            filters=DYNUNET_FILTERS,
            dropout=dropout,
            res_block=True,
        )


class SSFBUNet2D(DynUNetSSFB2D):
    """Identical to :class:`PlainUNet2D` except that the skip connections are SSFB blocks.

    Because every other hyperparameter matches, comparing a method built on this backbone against
    the same method on ``PlainUNet2D`` isolates the architecture's own contribution.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, dropout: float = 0.2):
        super().__init__(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=DYNUNET_KERNELS,
            strides=DYNUNET_STRIDES,
            upsample_kernel_size=DYNUNET_UPSAMPLE_KERNELS,
            filters=DYNUNET_FILTERS,
            dropout=dropout,
            res_block=True,
            ssfb_pool_sizes=SSFB_POOL_SIZES,
        )


class MetaLearner2D(nn.Module):
    """Tiny three-layer convolutional stacking model.

    Input is the concatenation of both branches' softmax probabilities, their clamped raw logits,
    and the two raw modality images. Output is final per-class logits on the same grid.

    Kept deliberately small (a few thousand parameters): its job is to learn *how to weigh* two
    frozen branches, not to re-learn segmentation from scratch on the small validation range.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#: backbone key (as used in the branch registry) -> class
BACKBONES_2D = {"plain": PlainUNet2D, "ssfb": SSFBUNet2D}
