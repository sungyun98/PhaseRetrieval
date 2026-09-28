"""Unit tests of phaseretrieval.eval (behaviour not covered by the legacy references)."""

import numpy as np

from phaseretrieval import PRTF, SubpixelAlignment


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
