###############################################################################
# Basic Functions
#
# Author: SUNG YUN LEE
#
# Contact: sungyun98@g.postech.edu
###############################################################################

"""Basic functions shared by the phase retrieval algorithms.

Tensors follow the layout ``(N, 1, H, W)``: ``N`` images (random starts or data sets), one
channel, height ``H`` and width ``W``.
"""

__all__ = [
    "MakeSupport",
    "fftshift",
    "ifftshift",
    "amplitude",
    "phase",
    "sqmesh",
    "freqfilter",
    "GaussianSmoothing",
]

import contextlib
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .partialconv2d import PartialConv2d


def _no_tf32() -> contextlib.AbstractContextManager:
    """Run cuDNN convolutions in full float32 precision inside the context.

    PyTorch allows TF32 for convolutions by default, which rounds their inputs to 10-bit
    mantissas on recent GPUs. The convolutions of this package are cheap, so they always use
    full precision; the previous settings are restored on exit.
    """
    c = torch.backends.cudnn
    return c.flags(
        enabled=c.enabled,
        benchmark=c.benchmark,
        benchmark_limit=c.benchmark_limit,
        deterministic=c.deterministic,
        allow_tf32=False,
    )


def MakeSupport(input: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Generate a rectangular or an autocorrelation-based support.

    The autocorrelation support follows the ShrinkWrap reference
    (https://doi.org/10.1103/PhysRevB.68.140101).

    Parameters
    ----------
    input : numpy.ndarray
        Real array of shape ``(H, W)``. For ``type='auto'`` it must be the fftshifted intensity
        (zero frequency at the centre); for ``type='rect'`` only its shape and dtype are used.
    **kwargs
        Keyword arguments listed below; other keywords are ignored.

    Other Parameters
    ----------------
    type : {'rect', 'auto'}
        ``'rect'`` for a rectangle centred at ``(H // 2, W // 2)``, ``'auto'`` for the
        thresholded autocorrelation of the object (inverse Fourier transform of the intensity).
    radius : tuple of int
        Half sizes ``(ri, rj)`` of the rectangle along the two axes (``type='rect'``).
    threshold : float
        Threshold relative to the maximum of the autocorrelation (``type='auto'``).

    Returns
    -------
    numpy.ndarray
        Support of shape ``(H, W)``, with the zero position at the centre: zeros and ones of the
        dtype of ``input`` for ``type='rect'``, bool for ``type='auto'``.

    Raises
    ------
    ValueError
        If ``type`` is neither ``'rect'`` nor ``'auto'``.
    """
    h = input.shape[0]
    w = input.shape[1]
    type = kwargs.pop("type")
    if type == "rect":
        # generate rectangular support
        ri, rj = kwargs.pop("radius")
        support = np.zeros_like(input)
        support[h // 2 - ri : h // 2 + ri, w // 2 - rj : w // 2 + rj] = 1
    elif type == "auto":
        # generate autocorrelation support
        threshold = kwargs.pop("threshold")
        support = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input))))
        support = support > np.amax(support) * threshold
    else:
        raise ValueError

    return support


def fftshift(input: Tensor) -> Tensor:
    """Shift the zero-frequency component to the centre of the last two dimensions.

    Parameters
    ----------
    input : torch.Tensor
        Real or complex tensor of shape ``(N, 1, H, W)``.

    Returns
    -------
    torch.Tensor
        Shifted tensor with the shape and dtype of ``input``.
    """
    return torch.fft.fftshift(input, dim=(2, 3))


def ifftshift(input: Tensor) -> Tensor:
    """Invert `fftshift`, moving the centre back to index ``(0, 0)``.

    Parameters
    ----------
    input : torch.Tensor
        Real or complex tensor of shape ``(N, 1, H, W)``.

    Returns
    -------
    torch.Tensor
        Shifted tensor with the shape and dtype of ``input``.
    """
    return torch.fft.ifftshift(input, dim=(2, 3))


def amplitude(input: Tensor) -> Tensor:
    """Return the amplitude (modulus) of a complex tensor.

    Parameters
    ----------
    input : torch.Tensor
        Complex tensor of shape ``(N, 1, H, W)``.

    Returns
    -------
    torch.Tensor
        Real tensor ``|input|`` of the same shape.
    """
    return torch.abs(input)


def phase(input: Tensor) -> Tensor:
    """Return the phase factor ``exp(i * angle(input))`` of a complex tensor.

    Parameters
    ----------
    input : torch.Tensor
        Complex tensor of shape ``(N, 1, H, W)``.

    Returns
    -------
    torch.Tensor
        Complex tensor of the same shape with unit modulus, and zero where ``input`` is zero.
    """
    r = torch.abs(input)
    r[r == 0] = 1
    return input / r


def sqmesh(height: int, width: int) -> Tensor:
    """Return the squared distance from the centre on an integer grid.

    The origin is at index ``(height // 2, width // 2)``, the zero position used by
    `fftshift`.

    Parameters
    ----------
    height, width : int
        Size of the grid.

    Returns
    -------
    torch.Tensor
        Real tensor of shape ``(1, 1, height, width)`` in the default floating dtype.
    """
    ci = height // 2
    cj = width // 2
    li = torch.linspace(-ci, height - ci - 1, steps=height)
    lj = torch.linspace(-cj, width - cj - 1, steps=width)
    mi, mj = torch.meshgrid(li, lj, indexing="ij")
    m = mi.pow(2) + mj.pow(2)

    return m.view(1, 1, height, width)


def freqfilter(size: int, count: int) -> tuple[float, ...]:
    """Return the frequency filter sequence of the oversampling smoothness (OSS) method.

    The filter coefficient decreases linearly from ``2 * size`` to ``2 * size / count`` in
    ``count`` equal stages. The sequence uses the parameter schedule format of
    `PhaseRetrieval`: ``(ratio_0, value_0, ratio_1, value_1, ...)``, where ``ratio_k`` is the
    fraction of the iterations after which ``value_k`` applies.

    Parameters
    ----------
    size : int
        Size of the data, ``max(H, W)`` (`PhaseRetrieval` passes ``min(H, W)``).
    count : int
        Number of filter stages.

    Returns
    -------
    tuple of float
        Schedule of length ``2 * count``.

    References
    ----------
    .. [1] https://doi.org/10.1107/S0021889813002471
    """
    param = []
    list = torch.linspace(size * 2, size * 2 / count, steps=count)
    for n, alpha in enumerate(list):
        param += [n / count, alpha.item()]

    return tuple(param)


def GaussianSmoothing(input: Tensor, sigma: float, mask: Tensor | None = None) -> Tensor:
    """Smooth with a Gaussian kernel, ignoring masked-out pixels.

    The kernel size is ``2 * ceil(2 * sigma) + 1``, as in the MATLAB function ``imgaussfilt``.
    Borders are padded by reflection, and a partial convolution renormalizes the kernel over
    the valid pixels. The convolution runs on the device and in the dtype of ``input``, without
    TF32.

    Parameters
    ----------
    input : torch.Tensor
        Real tensor of shape ``(N, 1, H, W)``.
    sigma : float
        Standard deviation of the kernel in pixels.
    mask : torch.Tensor, optional
        Real tensor of shape ``(N, 1, H, W)``, 1 for valid and 0 for missing pixels. Missing
        pixels are excluded from the average and set to zero in the output.

    Returns
    -------
    torch.Tensor
        Smoothed real tensor of shape ``(N, 1, H, W)``.
    """
    ksize = 2 * math.ceil(2 * sigma) + 1
    psize = math.ceil(2 * sigma)

    kernel = sqmesh(ksize, ksize).to(device=input.device, dtype=input.dtype)
    kernel = torch.exp(-0.5 * kernel / sigma**2)
    kernel = kernel / kernel.sum()

    gfilter = PartialConv2d(1, 1, ksize, padding=psize, padding_mode="reflect", bias=False).to(
        device=input.device, dtype=input.dtype
    )
    gfilter.weight.data = kernel
    gfilter.weight.requires_grad = False

    with _no_tf32():
        output = gfilter(input, mask_in=mask)
    if mask is not None:
        output = output * mask
    return output
