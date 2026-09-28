###############################################################################
# Additional Evaluation Functions
#
# Author: SUNG YUN LEE
#
# Contact: sungyun98@g.postech.edu
###############################################################################

"""Evaluation of phase retrieval results: alignment, distances, PRTF, PSD and SVD modes.

These functions work on NumPy arrays. Real-space results have the layout ``(N, H, W)``;
k-space data are fftshifted (zero frequency at the centre).
"""

__all__ = ["SubpixelAlignment", "PairwiseDistance", "PRTF", "PSD", "EigenMode"]

import itertools

import numpy as np
from numpy.linalg import svd
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
from tqdm import tqdm


def SubpixelAlignment(
    input: np.ndarray,
    error: np.ndarray | None = None,
    ref: np.ndarray | None = None,
    subpixel: int = 1,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Align real-space images by phase cross-correlation with subpixel precision.

    Each image, or its 180-degree rotation (twin image) if that matches better, is shifted
    in Fourier space to match the reference. Negative values created by the shift are set to
    zero.

    Parameters
    ----------
    input : numpy.ndarray
        Real array of shape ``(N, H, W)``. If ``error`` is None, it is aligned in place.
    error : numpy.ndarray, optional
        Real array of shape ``(N,)``. If given, the images are first sorted by increasing
        error.
    ref : numpy.ndarray, optional
        Real reference image of shape ``(H, W)``. By default, the first image (after sorting).
    subpixel : int, default 1
        Upsampling factor of the cross-correlation; the precision is ``1 / subpixel`` pixel.

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
    # sort array
    if error is not None:
        order = np.argsort(error)
        error = error[order]
        input = input[order, :, :]

    # align array to ref, or to the first array (which then stays fixed) if ref is None
    start = 0
    if ref is None:
        ref, start = input[0], 1
    for n in tqdm(range(start, input.shape[0]), desc="subpixel alignment"):
        arr = input[n]
        arr_T = np.flip(arr)
        s, err, _ = phase_cross_correlation(ref, arr, upsample_factor=subpixel)
        s_T, err_T, _ = phase_cross_correlation(ref, arr_T, upsample_factor=subpixel)
        if err_T < err:
            input[n, :, :] = np.fft.ifft2(fourier_shift(np.fft.fft2(arr_T), s_T)).real
        else:
            input[n, :, :] = np.fft.ifft2(fourier_shift(np.fft.fft2(arr), s)).real

    # remove negative values due to alignment
    input[input < 0] = 0

    if error is not None:
        return input, error
    else:
        return input


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
        Measured k-space amplitude of shape ``(H, W)``, fftshifted. Modified in place: zeros
        are replaced by 1.
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
    # get Fourier transform of input
    freq = np.fft.fftshift(np.fft.fft2(input))
    freq = np.absolute(np.mean(freq, axis=0))

    # normalization
    ref[ref == 0] = 1
    freq = freq / ref
    if mask is not None:
        freq[mask] = 0

    return freq


def PSD(input: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Compute the radial average (power spectral density, PSD) of k-space data.

    The average over ring ``r`` covers the pixels at distance ``[r, r + 1)`` from the array
    centre ``((H - 1) / 2, (W - 1) / 2)``.

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
    # get distance mesh
    di = input.shape[0]
    dj = input.shape[1]
    li = np.linspace(-di / 2 + 0.5, di / 2 - 0.5, num=di)
    lj = np.linspace(-dj / 2 + 0.5, dj / 2 - 0.5, num=dj)
    mi, mj = np.meshgrid(li, lj, indexing="ij")
    m = np.sqrt(np.power(mi, 2) + np.power(mj, 2))

    # calculate psd
    r_max = min(di, dj) // 2
    psd = np.zeros(r_max)
    for r in tqdm(range(r_max), desc="psd"):
        drop = (m >= r) * (m < r + 1)
        if mask is not None:
            drop = drop * (1 - mask)
        drop = drop > 0
        if np.sum(drop) > 0:
            psd[r] = np.mean(input[drop])
        else:
            psd[r] = np.nan

    return psd


def EigenMode(
    input: np.ndarray, k: int | None = None, lowrank: bool = True
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the eigenmodes of a set of images by singular value decomposition (SVD).

    Parameters
    ----------
    input : numpy.ndarray
        Real images of shape ``(N, H, W)``.
    k : int, optional
        Number of modes to return. By default, all ``M = min(N, H * W)`` modes (and no
        low-rank approximation).
    lowrank : bool, default True
        If True and ``k`` is given, also return the low-rank approximations.

    Returns
    -------
    output : numpy.ndarray
        Eigenmodes (left singular vectors) of shape ``(M or k, H, W)``.
    s : numpy.ndarray
        Singular values in decreasing order, shape ``(M or k,)``.
    approx : numpy.ndarray
        Approximations of rank 1 to ``k`` of the first image ``input[0]``, shape
        ``(k, H, W)``; returned only if ``k`` is given and ``lowrank`` is True.
    """
    h = input.shape[1]
    w = input.shape[2]
    input = input.reshape((-1, h * w)).T
    # calculate singular value decomposition
    u, s, vh = svd(input, full_matrices=False)
    output = u.T.reshape((-1, h, w))
    if k is None:
        return output, s
    else:
        if not lowrank:
            return output[:k], s[:k]
        else:
            # calculate low-rank approximation with order 1 to k
            approx = np.zeros((k, h, w))
            for rank in range(1, k + 1):
                temp = u[:, :rank] @ np.diag(s[:rank]) @ vh[:rank, :]
                temp = temp[:, 0].reshape((h, w))
                approx[rank - 1, :, :] = temp
            return output[:k], s[:k], approx
