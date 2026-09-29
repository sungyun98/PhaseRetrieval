###############################################################################
# Additional Evaluation Functions
#
# Author: SUNG YUN LEE
#
# Contact: sungyun98@g.postech.edu
###############################################################################

"""Evaluation of phase retrieval results: alignment, distances, PRTF, PSD and SVD modes.

These functions work on NumPy arrays, except `AlignObject`, which works on PyTorch tensors
(also inside training, with autograd). Real-space results have the layout ``(N, H, W)``;
k-space data are fftshifted (zero frequency at the centre).
"""

__all__ = [
    "SubpixelAlignment",
    "PairwiseDistance",
    "PRTF",
    "PSD",
    "EigenMode",
    "SymmOffset",
    "AlignObject",
]

import itertools

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import svd
from scipy.fft import next_fast_len
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
from torch import Tensor
from tqdm import tqdm

from .func import _no_tf32


def SubpixelAlignment(
    input: np.ndarray,
    error: np.ndarray | None = None,
    ref: np.ndarray | None = None,
    subpixel: int = 1,
    copy: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Align real-space images by phase cross-correlation with subpixel precision.

    Each image, or its 180-degree rotation (twin image) if that matches better, is shifted
    in Fourier space to match the reference. Negative values created by the shift are set to
    zero.

    Parameters
    ----------
    input : numpy.ndarray
        Real array of shape ``(N, H, W)``; not modified unless ``copy`` is False.
    error : numpy.ndarray, optional
        Real array of shape ``(N,)``. If given, the images are first sorted by increasing
        error.
    ref : numpy.ndarray, optional
        Real reference image of shape ``(H, W)``. By default, the first image (after sorting).
    subpixel : int, default 1
        Upsampling factor of the cross-correlation; the precision is ``1 / subpixel`` pixel.
    copy : bool, default True
        If False and ``error`` is None, align ``input`` in place to save memory.

    Returns
    -------
    output : numpy.ndarray
        Aligned images of shape ``(N, H, W)``.
    error : numpy.ndarray
        Sorted errors of shape ``(N,)``; returned only if ``error`` is given.

    References
    ----------
    .. [1] https://doi.org/10.1364/OL.33.000156
    """
    images = input
    if error is not None:
        order = np.argsort(error)
        error = np.asarray(error)[order]
        images = images[order]  # indexing with an array returns a copy
    elif copy:
        images = images.copy()

    # align to ref, or to the first image (which then stays fixed) if ref is None
    start = 0
    if ref is None:
        ref, start = images[0], 1
    for n in tqdm(range(start, images.shape[0]), desc="subpixel alignment"):
        image = images[n]
        twin = np.flip(image)  # 180-degree rotation
        shift, err, _ = phase_cross_correlation(ref, image, upsample_factor=subpixel)
        shift_twin, err_twin, _ = phase_cross_correlation(ref, twin, upsample_factor=subpixel)
        if err_twin < err:
            image, shift = twin, shift_twin
        images[n] = np.fft.ifft2(fourier_shift(np.fft.fft2(image), shift)).real

    # remove negative values due to the shift
    images[images < 0] = 0

    if error is not None:
        return images, error
    return images


def PairwiseDistance(input: np.ndarray) -> np.ndarray:
    """Compute the distance ``sum(|a - b|) / sum(|a + b|)`` between all pairs of images.

    Parameters
    ----------
    input : numpy.ndarray
        Aligned real images of shape ``(N, H, W)``.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(N * (N - 1) // 2,)``, in the order of
        ``itertools.combinations(range(N), 2)``.
    """
    # calculate pairwise distance
    n_max = input.shape[0]
    count = n_max * (n_max - 1) // 2
    dist = np.zeros(count)
    for n, (i, j) in tqdm(
        enumerate(itertools.combinations(range(n_max), 2)), total=count, desc="pairwise distance"
    ):
        dist[n] = np.sum(np.abs(input[i] - input[j])) / np.sum(np.abs(input[i] + input[j]))

    return dist


def PRTF(input: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Compute the phase retrieval transfer function (PRTF).

    The PRTF is the modulus of the mean Fourier transform of the reconstructions divided by
    the measured amplitude.

    Parameters
    ----------
    input : numpy.ndarray
        Aligned real-space reconstructions of shape ``(N, H, W)``.
    ref : numpy.ndarray
        Measured k-space amplitude of shape ``(H, W)``, fftshifted; zeros are treated as 1.
    mask : numpy.ndarray, optional
        Bool array of shape ``(H, W)``, fftshifted: True for missing pixels, which are set to
        zero in the output.

    Returns
    -------
    numpy.ndarray
        Real array of shape ``(H, W)``, fftshifted.

    References
    ----------
    .. [1] https://doi.org/10.1364/JOSAA.23.001179
    """
    mean_amplitude = np.abs(np.mean(np.fft.fftshift(np.fft.fft2(input), axes=(-2, -1)), axis=0))
    prtf = mean_amplitude / np.where(ref == 0, 1, ref)
    if mask is not None:
        prtf[np.asarray(mask, dtype=bool)] = 0

    return prtf


def PSD(input: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Compute the radial average (power spectral density, PSD) of k-space data.

    Ring ``r`` contains the pixels at a distance in ``[r, r + 1)`` from the zero frequency at
    index ``(H // 2, W // 2)``, the centre used by ``numpy.fft.fftshift``.

    Parameters
    ----------
    input : numpy.ndarray
        Real array of shape ``(H, W)``, fftshifted: amplitude, intensity or PRTF.
    mask : numpy.ndarray, optional
        Bool array of shape ``(H, W)``: True for missing pixels, which are ignored.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(min(H, W) // 2,)``; NaN for rings without valid pixels.
    """
    h, w = input.shape
    di, dj = np.meshgrid(np.arange(h) - h // 2, np.arange(w) - w // 2, indexing="ij")
    ring = np.floor(np.sqrt(di**2 + dj**2)).astype(int)
    n_ring = min(h, w) // 2

    valid = ring < n_ring
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)
    total = np.bincount(ring[valid], weights=input[valid], minlength=n_ring)
    count = np.bincount(ring[valid], minlength=n_ring)
    with np.errstate(divide="ignore", invalid="ignore"):
        return total / count


def EigenMode(
    input: np.ndarray, k: int | None = None, lowrank: bool = True
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the eigenmodes of a set of images and their low-rank approximation.

    The images are the columns of a ``(H * W, N)`` matrix whose singular value decomposition
    gives the eigenmodes (left singular vectors). The rank-``l`` approximation of the set is
    the mean over the images of their rank-``l`` approximations, which is the projection of
    the mean image onto the first ``l`` eigenmodes: it keeps the structure common to the
    reconstructions and drops the variations carried by the weaker modes.

    Parameters
    ----------
    input : numpy.ndarray
        Real images of shape ``(N, H, W)``, e.g. aligned reconstructions.
    k : int, optional
        Number of modes to return. By default, all ``M = min(N, H * W)`` modes (and no
        low-rank approximation).
    lowrank : bool, default True
        If True and ``k`` is given, also return the low-rank approximations.

    Returns
    -------
    modes : numpy.ndarray
        Eigenmodes of shape ``(M or k, H, W)``, each normalized to unit norm; their signs are
        arbitrary.
    s : numpy.ndarray
        Singular values in decreasing order, shape ``(M or k,)``.
    approx : numpy.ndarray
        Approximations of rank 1 to ``k``, shape ``(k, H, W)``; returned only if ``k`` is
        given and ``lowrank`` is True.
    """
    h, w = input.shape[1:]
    data = input.reshape(-1, h * w).T  # one image per column
    u, s, vh = svd(data, full_matrices=False)
    modes = u.T.reshape(-1, h, w)
    if k is None:
        return modes, s
    if not lowrank:
        return modes[:k], s[:k]

    # mean of the rank-l approximations U_l S_l V_l^T over the images (columns):
    # sum over the first l modes of u_m * s_m * mean(vh[m, :])
    weights = s[:k] * vh[:k].mean(axis=1)
    approx = np.cumsum(u[:, :k] * weights, axis=1).T.reshape(k, h, w)
    return modes[:k], s[:k], approx


def SymmOffset(input: np.ndarray, device: str | torch.device | None = None) -> np.ndarray:
    """Find the offset of the centre of symmetry of a diffraction pattern from the array centre.

    The diffraction intensity of a real object is centrosymmetric about the zero frequency,
    ``I(q) = I(-q)`` (Friedel's law), so the zero frequency of a measured pattern is its centre
    of symmetry ``c``. Rotating the pattern by 180 degrees about a point ``p`` moves ``c`` to
    ``2 p - c``; the shift ``s = 2 (c - p)`` that registers the rotated pattern with the
    original therefore gives ``c = p + s / 2``. The shift is found with pixel precision by
    masked normalized cross-correlation [1]_, ignoring missing (NaN) pixels, as in
    ``skimage.registration.phase_cross_correlation`` with masks, computed with PyTorch in
    float64 (on the GPU if there is one).

    The offset is measured from index ``(H // 2, W // 2)``, the position of the zero
    frequency after `numpy.fft.fftshift`, for both odd and even sizes: the pattern is centred
    when the offset is zero, and a region centred on the zero frequency is obtained by cropping
    around ``(H // 2 + di, W // 2 + dj)``.

    Parameters
    ----------
    input : numpy.ndarray
        Intensity of shape ``(H, W)``, NaN for missing pixels. The centre of symmetry must lie
        inside the array, and enough of the pattern must overlap with its rotation.
    device : str or torch.device, optional
        Device of the computation; by default the GPU if there is one, else the CPU.

    Returns
    -------
    numpy.ndarray
        Integer offset ``(di, dj)``: the centre of symmetry is at ``(H // 2 + di, W // 2 + dj)``.
        A centre halfway between two pixels (half-integer offset) is rounded toward zero, i.e.
        toward the array centre.

    References
    ----------
    .. [1] D. Padfield, Masked object registration in the Fourier domain, IEEE Trans. Image
       Process. 21, 2706 (2012), https://doi.org/10.1109/TIP.2011.2181402
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = torch.as_tensor(np.ascontiguousarray(input), dtype=torch.float64, device=device)
    rotated = torch.flip(reference, dims=(0, 1))  # rotation about p = ((H - 1) / 2, (W - 1) / 2)
    xcorr = _masked_xcorr(rotated, reference, ~torch.isnan(rotated), ~torch.isnan(reference))
    # average of equal maxima, as in scikit-image
    center = torch.nonzero(xcorr == xcorr.max()).double().mean(dim=0).cpu().numpy()
    shift = np.array(input.shape) - 1 - center
    # c - (H // 2, W // 2) = s / 2 + p - (H // 2, W // 2), where p - (H // 2, W // 2) is -1/2 for
    # even and 0 for odd sizes
    parity = 1 - np.asarray(input.shape) % 2
    return np.trunc((shift - parity) / 2).astype(int)


def _masked_xcorr(
    fixed: Tensor, moving: Tensor, fixed_mask: Tensor, moving_mask: Tensor, overlap_ratio=0.3
) -> Tensor:
    """Masked normalized cross-correlation of two 2-D images ('full' mode), after Padfield.

    A PyTorch port of ``skimage.registration._masked_phase_cross_correlation
    .cross_correlate_masked`` with the same padding to fast FFT sizes; positions where the
    masks overlap by less than ``overlap_ratio`` of the maximum overlap are set to zero.
    """
    eps = torch.finfo(torch.float64).eps
    final_shape = [a + b - 1 for a, b in zip(fixed.shape, moving.shape)]
    fast_shape = [next_fast_len(n) for n in final_shape]

    def fft(x):
        return torch.fft.fftn(x, s=fast_shape)

    def ifft(x):
        return torch.fft.ifftn(x, s=fast_shape).real

    fixed = torch.where(fixed_mask, fixed, 0.0)
    moving = torch.where(moving_mask, moving, 0.0)
    rotated_moving = torch.flip(moving, dims=(0, 1))
    rotated_moving_mask = torch.flip(moving_mask, dims=(0, 1))

    fixed_fft = fft(fixed)
    rotated_moving_fft = fft(rotated_moving)
    fixed_mask_fft = fft(fixed_mask.double())
    rotated_moving_mask_fft = fft(rotated_moving_mask.double())

    overlap = ifft(rotated_moving_mask_fft * fixed_mask_fft).round().clamp(min=eps)
    correlated_fixed = ifft(rotated_moving_mask_fft * fixed_fft)
    correlated_moving = ifft(fixed_mask_fft * rotated_moving_fft)
    numerator = (
        ifft(rotated_moving_fft * fixed_fft) - correlated_fixed * correlated_moving / overlap
    )
    fixed_denom = ifft(rotated_moving_mask_fft * fft(fixed.square()))
    fixed_denom = (fixed_denom - correlated_fixed.square() / overlap).clamp(min=0)
    moving_denom = ifft(fixed_mask_fft * fft(rotated_moving.square()))
    moving_denom = (moving_denom - correlated_moving.square() / overlap).clamp(min=0)
    denom = torch.sqrt(fixed_denom * moving_denom)

    crop = (slice(0, final_shape[0]), slice(0, final_shape[1]))
    numerator, denom, overlap = numerator[crop], denom[crop], overlap[crop]
    tol = 1e3 * eps * denom.abs().max()
    out = torch.where(denom > tol, numerator / torch.where(denom > tol, denom, 1.0), 0.0)
    out = out.clamp(-1, 1)
    return torch.where(overlap < overlap_ratio * overlap.max(), 0.0, out)


def AlignObject(input: Tensor, target: Tensor | None = None) -> Tensor:
    """Align real-space objects by circular shifts.

    Without a target, each object is shifted so that the centroid of its pixels above 1% of
    its maximum is at index ``(H // 2, W // 2)``. With a target, each object, or its
    180-degree rotation (twin image) if that correlates better, is shifted by the
    translation of at most ``max(H, W) // 2`` pixels per axis that maximizes the
    cross-correlation with its target; among equal maxima, the smallest shift is used.

    Parameters
    ----------
    input : torch.Tensor
        Real objects of shape ``(N, 1, H, W)``. Not modified; gradients flow through the
        shifts.
    target : torch.Tensor, optional
        Real targets of shape ``(N or 1, 1, H, W)``. The cross-correlation runs in full
        float32 precision (no TF32).

    Returns
    -------
    torch.Tensor
        Aligned objects of shape ``(N, 1, H, W)``.
    """
    n, _, h, w = input.shape
    aligned = []
    if target is None:
        for obj in input:
            pixels = torch.nonzero(obj > obj.max() * 0.01).to(torch.float32)
            centroid = torch.mean(pixels, dim=0)[-2:].to(torch.int64)
            shift = [h // 2 - int(centroid[0]), w // 2 - int(centroid[1])]
            aligned.append(torch.roll(obj, shift, dims=(-2, -1)))
        return torch.stack(aligned)

    limit = max(h, w) // 2
    rotated = torch.rot90(input, 2, dims=(-2, -1))
    with torch.no_grad(), _no_tf32():
        weight = target.expand(n, 1, h, w).contiguous()
        corr = F.conv2d(input.reshape(1, n, h, w), weight, padding=limit, groups=n)[0]
        corr_rot = F.conv2d(rotated.reshape(1, n, h, w), weight, padding=limit, groups=n)[0]
    for i in range(n):
        use_rot = corr_rot[i].amax() > corr[i].amax()
        c = corr_rot[i] if use_rot else corr[i]
        peaks = torch.nonzero(c == c.amax())
        if len(peaks) == 0:  # no exact maximum (NaN values)
            k = torch.argmax(c)
            peaks = torch.stack([k // c.shape[-1], k % c.shape[-1]])[None]
        shifts = limit - peaks
        shift = shifts[torch.argmin(torch.sum(torch.abs(shifts), dim=-1))]
        aligned.append(
            torch.roll(rotated[i] if use_rot else input[i], shift.tolist(), dims=(-2, -1))
        )
    return torch.stack(aligned)
