"""Unit tests of phaseretrieval.eval (behaviour not covered by the legacy references)."""

import numpy as np

from phaseretrieval import PRTF, PSD, EigenMode, SubpixelAlignment


def _stack(seed=0, n=4, size=32):
    rs = np.random.RandomState(seed)
    obj = np.zeros((size, size))
    obj[10:20, 12:22] = rs.rand(10, 10)
    return np.stack([np.roll(obj, (k, -k), axis=(0, 1)) for k in range(n)]) + 0.01 * rs.rand(
        n, size, size
    )


def test_subpixel_alignment_keeps_input():
    stack = _stack()
    before = stack.copy()
    aligned = SubpixelAlignment(stack, subpixel=4)
    assert np.array_equal(stack, before)
    in_place = SubpixelAlignment(stack, subpixel=4, copy=False)
    assert in_place is stack and np.array_equal(stack, aligned)


def test_subpixel_alignment_sorts_by_error():
    stack = _stack()
    error = np.array([0.3, 0.1, 0.4, 0.2])
    aligned, err = SubpixelAlignment(stack.copy(), error=error, subpixel=4)
    assert np.array_equal(err, np.sort(error))
    assert np.allclose(aligned[0], np.clip(stack[1], 0, None))  # the reference is not shifted


def test_prtf_keeps_ref():
    stack = _stack()
    ref = np.abs(np.fft.fftshift(np.fft.fft2(stack[0])))
    ref[0, 0] = 0
    before = ref.copy()
    PRTF(stack, ref)
    assert np.array_equal(ref, before)


def _psd_v1(x, mask=None):
    """PSD of v1.0-legacy/v2.0 up to this fix: radii from ((H - 1) / 2, (W - 1) / 2)."""
    h, w = x.shape
    mi, mj = np.meshgrid(
        np.linspace(-h / 2 + 0.5, h / 2 - 0.5, h),
        np.linspace(-w / 2 + 0.5, w / 2 - 0.5, w),
        indexing="ij",
    )
    m = np.sqrt(mi**2 + mj**2)
    out = np.full(min(h, w) // 2, np.nan)
    for r in range(out.size):
        ring = (m >= r) & (m < r + 1) & (True if mask is None else ~mask)
        if ring.any():
            out[r] = x[ring].mean()
    return out


def test_psd_odd_size_unchanged():
    rs = np.random.RandomState(1)
    x, mask = rs.rand(41, 37), rs.rand(41, 37) < 0.1
    assert np.allclose(PSD(x, mask), _psd_v1(x, mask), rtol=1e-12, equal_nan=True)


def test_psd_centred_on_zero_frequency():
    # a radially symmetric function about the fftshift centre has a constant value per ring
    h, w = 64, 48
    di, dj = np.meshgrid(np.arange(h) - h // 2, np.arange(w) - w // 2, indexing="ij")
    r = np.floor(np.hypot(di, dj))
    psd = PSD(r)
    assert psd.shape == (24,)
    assert np.array_equal(psd, np.arange(24.0))
    assert PSD(r, mask=r == 3)[3] != PSD(r, mask=r == 3)[3]  # NaN for an empty ring


def test_eigenmode_lowrank_approximation_of_the_set():
    rs = np.random.RandomState(2)
    stack = _stack(n=6) + 0.05 * rs.randn(6, 32, 32)
    modes, s, approx = EigenMode(stack, k=6)
    assert approx.shape == (6, 32, 32)
    data = stack.reshape(6, -1).T
    u, sv, vh = np.linalg.svd(data, full_matrices=False)
    for rank in (1, 3):
        explicit = (u[:, :rank] @ np.diag(sv[:rank]) @ vh[:rank]).mean(axis=1).reshape(32, 32)
        assert np.allclose(approx[rank - 1], explicit)
    # the full-rank approximation is the mean image
    assert np.allclose(approx[-1], stack.mean(axis=0))
    # rank 1: projection of the mean onto the first mode
    mean = stack.mean(axis=0)
    assert np.allclose(approx[0], modes[0] * np.sum(modes[0] * mean))
