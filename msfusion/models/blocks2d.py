"""SSFB — Skip and Spatial Feature Blend, 2D.

A drop-in replacement for the concat-based skip connection in MONAI's ``DynUNet``. The encoder
feature map is first re-weighted channel-wise by a gate conditioned on both the encoder and the
decoder global context, then (optionally) refined by low-rank cross-attention against a spatially
pooled version of itself. A learned scalar ``alpha`` blends the two paths, so the block can fall
back to pure gating if attention doesn't earn its keep.

The 3D counterpart lives in :mod:`msfusion.models.blocks3d`. The two are kept as separate
implementations on purpose: they are read side by side when reasoning about the architecture, and
the pooling / reshape logic is easier to follow written out per dimensionality.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.nets import DynUNet


class SSFB2D(nn.Module):
    """Channel gate + pooled spatial cross-attention on a 2D skip connection.

    Args:
        dec_ch: channels of the (upsampled) decoder feature map.
        enc_ch: channels of the encoder feature map arriving on the skip.
        rank: query/key projection width for the attention path -- kept small on purpose.
        pool_size: side length the encoder map is pooled to before attention. ``None`` disables
            the attention path entirely, leaving only the channel gate (used at the highest
            resolution, where full attention would be prohibitively expensive).
    """

    def __init__(self, dec_ch: int, enc_ch: int, rank: int = 8, pool_size=None):
        super().__init__()
        self.rank = rank
        self.pool_size = pool_size
        groups = min(8, dec_ch)
        hidden = max(4, (dec_ch + enc_ch) // 4)

        self.gate_mlp = nn.Sequential(
            nn.Linear(dec_ch + enc_ch, hidden),
            nn.GELU(),
            nn.Linear(hidden, enc_ch),
            nn.Sigmoid(),
        )
        if pool_size is not None:
            self.q_proj = nn.Conv2d(dec_ch, rank, 1, bias=False)
            self.k_proj = nn.Conv2d(enc_ch, rank, 1, bias=False)
            self.v_proj = nn.Conv2d(enc_ch, enc_ch, 1, bias=False)
            self.out_proj = nn.Conv2d(enc_ch, enc_ch, 1, bias=False)
            self.alpha = nn.Parameter(torch.zeros(1))
        self.fuse = nn.Sequential(
            nn.Conv2d(dec_ch + enc_ch, dec_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, dec_ch),
            nn.GELU(),
        )

    def forward(self, f_dec: torch.Tensor, f_enc: torch.Tensor) -> torch.Tensor:
        B, Cd, H, W = f_dec.shape
        Ce = f_enc.shape[1]

        gap_dec = f_dec.mean(dim=[2, 3])
        gap_enc = f_enc.mean(dim=[2, 3])
        gate = self.gate_mlp(torch.cat([gap_dec, gap_enc], dim=1))
        o_gate = f_enc * gate.view(B, Ce, 1, 1)

        if self.pool_size is not None:
            P = self.pool_size
            Kp = F.adaptive_avg_pool2d(self.k_proj(f_enc), (P, P))
            Vp = F.adaptive_avg_pool2d(self.v_proj(f_enc), (P, P))
            q = self.q_proj(f_dec).flatten(2).permute(0, 2, 1)
            k = Kp.flatten(2)
            v = Vp.flatten(2).permute(0, 2, 1)
            attn = torch.softmax(q @ k / (self.rank ** 0.5), dim=-1)
            o_attn = (attn @ v).permute(0, 2, 1).reshape(B, Ce, H, W)
            o_attn = self.out_proj(o_attn)
            alpha = torch.sigmoid(self.alpha)
            m = alpha * o_attn + (1 - alpha) * o_gate
        else:
            m = o_gate

        return self.fuse(torch.cat([f_dec, m], dim=1))


class SSFBUpBlock2D(nn.Module):
    """Drop-in replacement for MONAI's ``UnetUpBlock``.

    Identical transposed-convolution upsample; the skip connection is merged by :class:`SSFB2D`
    instead of a plain concat followed by a refinement convolution.
    """

    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        upsample_kernel_size,
        norm_name,
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout=None,
        trans_bias=False,
        ssfb_pool_size=None,
        ssfb_rank: int = 8,
    ):
        super().__init__()
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            dropout=dropout,
            bias=trans_bias,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )
        self.ssfb = SSFB2D(dec_ch=out_channels, enc_ch=out_channels, rank=ssfb_rank, pool_size=ssfb_pool_size)

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.ssfb(self.transp_conv(inp), skip)


class DynUNetSSFB2D(DynUNet):
    """``DynUNet`` with every skip connection replaced by an :class:`SSFBUpBlock2D`.

    ``ssfb_pool_sizes`` is ordered coarse-to-fine, matching the decoder's own iteration order.
    """

    def __init__(self, *args, ssfb_pool_sizes: Sequence = None, **kwargs):
        # Set before super().__init__ because DynUNet builds the decoder inside its constructor.
        # object.__setattr__ bypasses nn.Module's attribute machinery, which isn't ready yet.
        object.__setattr__(self, "_ssfb_pool_sizes", ssfb_pool_sizes)
        super().__init__(*args, **kwargs)

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        pool_sizes = self._ssfb_pool_sizes or [None] * len(inp)
        return nn.ModuleList(
            [
                SSFBUpBlock2D(
                    spatial_dims=self.spatial_dims,
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel,
                    stride=stride,
                    upsample_kernel_size=up_kernel,
                    norm_name=self.norm_name,
                    act_name=self.act_name,
                    dropout=self.dropout,
                    trans_bias=self.trans_bias,
                    ssfb_pool_size=pool,
                )
                for in_c, out_c, kernel, stride, up_kernel, pool in zip(
                    inp, out, kernel_size, strides, upsample_kernel_size, pool_sizes
                )
            ]
        )
