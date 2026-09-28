###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################
# Copied from https://github.com/NVIDIA/partialconv/blob/a99cd7cb9f6469c02181d9aa34fe5abd95fb0154/models/partialconv2d.py
# Full license text: LICENSES/partialconv-BSD-3-Clause.txt
# Changes: added __all__, docstrings and type hints; removed unused imports; reformatted with
# ruff.

"""Partial convolution layer (https://arxiv.org/abs/1804.07723)."""

__all__ = ["PartialConv2d"]

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PartialConv2d(nn.Conv2d):
    """2-D convolution conditioned on a mask of valid pixels.

    The convolution sees only the valid pixels, and its output is rescaled by the fraction
    of valid pixels under the kernel; output pixels without any valid input are zero (bias
    included) and invalid in the updated mask.

    Parameters
    ----------
    *args, **kwargs
        Arguments of `torch.nn.Conv2d`, and the keywords below.
    multi_channel : bool, default False
        Use a mask per input channel (shape ``(N, C, H, W)``) instead of ``(N or 1, 1, H, W)``.
    return_mask : bool, default False
        Also return the updated mask from `forward`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:

        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        super().__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
            )
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
        )

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(
        self, input: Tensor, mask_in: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply the partial convolution.

        Parameters
        ----------
        input : torch.Tensor
            Real tensor of shape ``(N, C, H, W)``.
        mask_in : torch.Tensor, optional
            Mask of valid pixels (1 valid, 0 invalid) of the shape described in the class
            docstring. By default, all pixels are valid.

        Returns
        -------
        output : torch.Tensor
            Real tensor of shape ``(N, C_out, H_out, W_out)``.
        update_mask : torch.Tensor
            Updated mask, returned only if ``return_mask`` is True.
        """
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                        ).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super().forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output
