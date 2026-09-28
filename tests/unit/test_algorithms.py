"""Unit tests of phaseretrieval.algorithms (behaviour not covered by the legacy references)."""

import numpy as np
import pytest
import torch

from phaseretrieval import PhaseRetrieval
from phaseretrieval.algorithms import ShrinkWrap

SHRINKWRAP = dict(
    shrinkwrap=True, sigma_initial=3, sigma_limit=1.5, ratio_update=0.05, threshold=0.1, interval=5
)


def _data(size=64, seed=0):
    rs = np.random.RandomState(seed)
    obj = np.zeros((size, size))
    obj[24:40, 26:38] = rs.rand(16, 12)
    amplitude = np.abs(np.fft.fft2(obj))
    support = np.zeros((size, size))
    support[20:44, 22:42] = 1
    unknown = np.zeros((size, size))
    unknown[:2, :2] = 1

    def t(x):
        return torch.from_numpy(x.astype(np.float32))[None, None]

    return t(amplitude), t(support), t(unknown)


def _phase(n, size=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    theta = 2 * torch.pi * torch.rand(n, 1, size, size, generator=g)
    return torch.polar(torch.ones_like(theta), theta)


def test_shrinkwrap_rejects_sigma_initial_below_limit():
    with pytest.raises(ValueError):
        ShrinkWrap(0.1, sigma_initial=1, sigma_limit=1.5)


def test_shrinkwrap_fixed_kernel_when_sigma_initial_equals_limit():
    sw = ShrinkWrap(0.1, sigma_initial=1.5, sigma_limit=1.5)
    assert torch.isclose(sw.filter.sum(), torch.tensor(1.0))
    before = sw.filter.clone()
    sw.update()
    assert sw.sigma == 1.5 and torch.equal(sw.filter, before)


@pytest.mark.parametrize("per_sample_support", [False, True])
def test_shrinkwrap_calls_are_independent(per_sample_support):
    amplitude, support, unknown = _data()
    if per_sample_support:
        support = support.repeat(2, 1, 1, 1)
    info = dict(algorithm="HIO", error="R", beta=0.9, beta_type="const", boundary_push=0)
    info.update(SHRINKWRAP)
    iterator = PhaseRetrieval(amplitude, support, unknown, **info)
    phase = _phase(2)
    first, _ = iterator(30, phase, **info)
    sigma_after_first = iterator.shrink.sigma
    second, _ = iterator(30, phase, **info)
    assert sigma_after_first < SHRINKWRAP["sigma_initial"]
    assert torch.equal(first, second)
    assert torch.equal(iterator.initial_support, support)
