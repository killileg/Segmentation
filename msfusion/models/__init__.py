"""Network definitions.

The 2D and 3D implementations are kept fully separate (``*2d`` / ``*3d`` modules). This module
provides the small amount of dispatch the pipelines need, so the branch registry can name a
backbone as a string without either pipeline importing the other's classes.
"""

import torch.nn as nn

from .blocks2d import DynUNetSSFB2D, SSFB2D, SSFBUpBlock2D
from .blocks3d import DynUNetSSFB3D, SSFB3D, SSFBUpBlock3D
from .networks2d import BACKBONES_2D, MetaLearner2D, PlainUNet2D, SSFBUNet2D
from .networks3d import BACKBONES_3D, MetaLearner3D, PlainUNet3D, SSFBUNet3D

BACKBONES = {2: BACKBONES_2D, 3: BACKBONES_3D}
META_LEARNERS = {2: MetaLearner2D, 3: MetaLearner3D}


def build_backbone(kind: str, spatial_dims: int, in_channels: int, out_channels: int, dropout: float):
    """Instantiate a backbone by registry key (``"plain"`` / ``"ssfb"``) and dimensionality."""
    try:
        table = BACKBONES[spatial_dims]
    except KeyError:
        raise KeyError(f"spatial_dims must be 2 or 3, got {spatial_dims}") from None
    if kind not in table:
        raise KeyError(f"unknown backbone {kind!r}; expected one of {sorted(table)}")
    return table[kind](in_channels=in_channels, out_channels=out_channels, dropout=dropout)


def build_meta_learner(spatial_dims: int, in_channels: int, out_channels: int, hidden: int = 16):
    """Instantiate the meta-learner for the given dimensionality."""
    try:
        cls = META_LEARNERS[spatial_dims]
    except KeyError:
        raise KeyError(f"spatial_dims must be 2 or 3, got {spatial_dims}") from None
    return cls(in_channels=in_channels, out_channels=out_channels, hidden=hidden)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "SSFB2D", "SSFBUpBlock2D", "DynUNetSSFB2D",
    "SSFB3D", "SSFBUpBlock3D", "DynUNetSSFB3D",
    "PlainUNet2D", "SSFBUNet2D", "MetaLearner2D",
    "PlainUNet3D", "SSFBUNet3D", "MetaLearner3D",
    "BACKBONES", "BACKBONES_2D", "BACKBONES_3D", "META_LEARNERS",
    "build_backbone", "build_meta_learner", "count_parameters",
]
