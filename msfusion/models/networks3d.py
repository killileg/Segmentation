"""3D backbones and meta-learner.

Volumetric counterparts to :mod:`msfusion.models.networks2d`, with the same filter widths so the
2D and 3D results are directly comparable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

from .blocks3d import DynUNetSSFB3D

DYNUNET_FILTERS = [32, 64, 128, 256]
DYNUNET_KERNELS = [[3, 3, 3]] * 4
DYNUNET_STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
DYNUNET_UPSAMPLE_KERNELS = DYNUNET_STRIDES[1:]

#: smaller than the 2D pool sizes: a cubic pooled grid has P³ keys, which grows far faster than P²
SSFB_POOL_SIZES = [4, 2, None]


class PlainUNet3D(DynUNet):
    """A stock MONAI ``DynUNet`` with ``spatial_dims=3``, nothing more.

    As in 2D, this exists only to pin the hyperparameters so every "plain" branch is architecturally
    identical, making the fusion-strategy comparison clean.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, dropout: float = 0.2):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=DYNUNET_KERNELS,
            strides=DYNUNET_STRIDES,
            upsample_kernel_size=DYNUNET_UPSAMPLE_KERNELS,
            filters=DYNUNET_FILTERS,
            dropout=dropout,
            res_block=True,
        )


class SSFBUNet3D(DynUNetSSFB3D):
    """Identical to :class:`PlainUNet3D` except that the skip connections are SSFB blocks."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, dropout: float = 0.2):
        super().__init__(
            spatial_dims=3,
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


class MetaLearner3D(nn.Module):
    """Tiny three-layer 3D convolutional stacking model. See :class:`~.networks2d.MetaLearner2D`."""

    def __init__(self, in_channels: int, out_channels: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, hidden), hidden),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, hidden), hidden),
            nn.GELU(),
            nn.Conv3d(hidden, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


BACKBONES_3D = {"plain": PlainUNet3D, "ssfb": SSFBUNet3D}
