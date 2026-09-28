"""Unit tests of SymmOffset and AlignObject."""

import numpy as np
import pytest
import skimage
import torch
import torch.nn.functional as F

from phaseretrieval import AlignObject, SymmOffset

_SKIMAGE_022 = tuple(int(v) for v in skimage.__version__.split(".")[:2]) >= (0, 22)


def _align_obj_v1(output, target, limit=32):
    """CombinedLoss.align_obj of DPR v1.0-legacy (64 x 64 objects only), for comparison."""
    N = output.shape[0]
    xcorr = F.conv2d(output.reshape(1, N, 64, 64), target, padding=limit, groups=N).squeeze(0)
    xcorrT = F.conv2d(
        torch.rot90(output, 2, dims=(-2, -1)).reshape(1, N, 64, 64), target, padding=limit, groups=N
    ).squeeze(0)
    vmax = torch.amax(xcorr, dim=(-2, -1))
    vTmax = torch.amax(xcorrT, dim=(-2, -1))
    for i in range(N):
        trg = vTmax[i] > vmax[i]
        if trg:
            output[i, :, :, :] = torch.rot90(output[i], 2, dims=(-2, -1))
            dpos = limit - torch.nonzero(xcorrT[i] == vTmax[i]).squeeze()
        else:
            dpos = limit - torch.nonzero(xcorr[i] == vmax[i]).squeeze()
        if dpos.numel() == 0:
            pos = torch.argmax(xcorrT[i] if trg else xcorr[i])
            dpos = limit - torch.stack([pos // 65, pos % 65])
        if len(dpos.shape) > 1:
            dpos = dpos[torch.argmin(torch.sum(torch.abs(dpos), dim=-1))]
        output[i, :, :, :] = torch.roll(output[i], dpos.tolist(), dims=(-2, -1))
    return output


def _align_obj_cen_v1(output):
    """align_obj_cen of DPR v1.0-legacy demo.ipynb (64 x 64, one object), for comparison."""
    cen = torch.mean(torch.nonzero(output > output.max() * 0.01).to(torch.float32), dim=0)[-2:]
    return torch.roll(output, (32 - cen.to(torch.int64)).tolist(), dims=(-2, -1))


def _objects(n, h, w, seed=0):
    g = torch.Generator().manual_seed(seed)
    obj = torch.zeros(n, 1, h, w)
    for i in range(n):
        a, b = h // 4 + i % 3, w // 4 + i % 2
        obj[i, 0, a : a + h // 3, b : b + w // 4] = torch.rand(h // 3, w // 4, generator=g)
    return obj


def test_matches_original_for_64x64():
    target = _objects(6, 64, 64)
    shifted = torch.roll(target, (5, -9), dims=(-2, -1))
    shifted[1::2] = torch.rot90(shifted[1::2], 2, dims=(-2, -1))
    shifted += 0.01 * torch.rand(shifted.shape, generator=torch.Generator().manual_seed(3))
    expected = _align_obj_v1(shifted.clone(), target)
    assert torch.equal(AlignObject(shifted, target), expected)
    for i in range(6):
        one = shifted[i : i + 1]
        assert torch.equal(AlignObject(one), _align_obj_cen_v1(one.clone()))


@pytest.mark.parametrize("shape", [(48, 80), (33, 33), (128, 96)])
def test_recovers_shift_and_twin_for_any_size(shape):
    h, w = shape
    target = _objects(4, h, w)
    moved = torch.roll(target, (h // 5, -w // 6), dims=(-2, -1))
    moved[2:] = torch.rot90(moved[2:], 2, dims=(-2, -1))
    before = moved.clone()
    aligned = AlignObject(moved, target)
    assert torch.equal(moved, before)  # input not modified
    for i in range(4):
        corr_max = F.conv2d(aligned[i : i + 1], target[i : i + 1], padding=max(h, w) // 2).amax()
        assert torch.isclose(corr_max, (target[i] ** 2).sum())


def test_centres_each_object_separately():
    obj = _objects(3, 40, 56)
    obj = torch.stack([torch.roll(o, (3 * k, -4 * k), dims=(-2, -1)) for k, o in enumerate(obj)])
    for o in AlignObject(obj):
        pixels = torch.nonzero(o[0] > o.max() * 0.01).float().mean(dim=0)
        assert torch.all(torch.abs(pixels - torch.tensor([20.0, 28.0])) < 1)


def test_gradient_flows_through_alignment():
    target = _objects(2, 32, 32)
    x = torch.roll(target, (3, 2), dims=(-2, -1)).requires_grad_()
    AlignObject(x, target).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def _find_center_v1(input):
    """find_center of DPR demo.ipynb (scikit-image >= 0.22), for comparison."""
    from skimage.registration import phase_cross_correlation

    input_T = np.rot90(input, 2)
    shift, _, _ = phase_cross_correlation(
        input,
        input_T,
        reference_mask=~np.isnan(input),
        moving_mask=~np.isnan(input_T),
        upsample_factor=1,
    )
    return np.trunc(shift / 2).astype(int)


def test_symm_offset():
    rs = np.random.RandomState(0)
    half = rs.rand(41, 51)
    pattern = half + np.rot90(half, 2)  # centrosymmetric about its centre (20, 25)
    assert np.array_equal(SymmOffset(pattern), [0, 0])
    # centre at (26, 25) of a 47 x 59 array, i.e. offset (3, -4) from (47 // 2, 59 // 2)
    pattern = np.pad(pattern, ((6, 0), (0, 8)))
    pattern[-3:, :4] = np.nan
    assert np.array_equal(SymmOffset(pattern), [3, -4])
    if _SKIMAGE_022:  # the DPR function needs scikit-image >= 0.22
        assert np.array_equal(SymmOffset(pattern), _find_center_v1(pattern))  # same for odd sizes


def _symmetric(shape, centre, seed=0):
    """Pattern of the given shape symmetric about centre (NaN where the mirror is outside)."""
    rs = np.random.RandomState(seed)
    base = rs.rand(*shape)
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    mi, mj = np.rint(2 * centre[0] - ii).astype(int), np.rint(2 * centre[1] - jj).astype(int)
    inside = (mi >= 0) & (mi < shape[0]) & (mj >= 0) & (mj < shape[1])
    pattern = np.full(shape, np.nan)
    pattern[inside] = base[inside] + base[mi[inside], mj[inside]]
    return pattern


@pytest.mark.parametrize("shape", [(61, 61), (64, 64), (64, 61)])
@pytest.mark.parametrize("offset", [(-5, 3), (4, -2), (0, 0), (-1, 1)])
def test_symm_offset_odd_and_even_sizes(shape, offset):
    centre = (shape[0] // 2 + offset[0], shape[1] // 2 + offset[1])
    assert np.array_equal(SymmOffset(_symmetric(shape, centre)), offset)
    # halfway between pixels: rounded toward zero
    half = (centre[0] + 0.5, centre[1] - 0.5)
    expected = np.trunc(np.subtract(half, (shape[0] // 2, shape[1] // 2))).astype(int)
    assert np.array_equal(SymmOffset(_symmetric(shape, half)), expected)
