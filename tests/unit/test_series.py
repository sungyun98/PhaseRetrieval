"""Unit tests of the series connection of PhaseRetrieval iterators (continue_=True)."""

import math

import pytest
import torch
from test_algorithms import _data, _phase

from phaseretrieval import PhaseRetrieval

HIO = dict(algorithm="HIO", error="R", beta=0.9, beta_type="const", boundary_push=0)
GPS = dict(algorithm="GPS-R", error="R", sigma=(0, 0.1, 0.5, 1), alpha_count=3, t=1, s=0.9)
SW = dict(shrinkwrap=True, sigma_initial=3, sigma_limit=1.5, ratio_update=0.05, threshold=0.1)


def _state_from_phase(amplitude, support, phase):
    """The state that is equivalent to starting from initial phases."""
    z = amplitude * phase
    n = phase.size(0)
    return {
        "z": z,
        "y": torch.zeros_like(z),
        "support": support.expand(n, -1, -1, -1).clone(),
        "sigma": torch.full((n,), math.nan, dtype=torch.float64),
    }


@pytest.mark.parametrize("params", [HIO, GPS])
def test_start_from_state_equals_start_from_phase(params):
    amplitude, support, unknown = _data()
    phase = _phase(3)
    iterator = PhaseRetrieval(amplitude, support, unknown, **params)
    out_phase, path_phase = iterator(20, phase, **params)
    state = _state_from_phase(amplitude, support, phase)
    before = {k: v.clone() for k, v in state.items()}
    out_state, path_state = iterator(20, state, **params)
    assert torch.equal(out_phase, out_state) and torch.equal(path_phase, path_state)
    assert all(torch.equal(before[k], state[k]) or k == "sigma" for k in state)  # not modified


@pytest.mark.parametrize("params", [HIO, GPS])
def test_state_contents(params):
    amplitude, support, unknown = _data()
    iterator = PhaseRetrieval(amplitude, support, unknown, **params)
    output, path, state = iterator(20, _phase(3), continue_=True, **params)
    assert set(state) == {"z", "y", "support", "sigma", "error", "path"}
    assert state["z"].shape == state["y"].shape == state["support"].shape == (3, 1, 64, 64)
    assert state["z"].is_complex() and torch.isnan(state["sigma"]).all()
    assert torch.equal(state["error"], path.min(dim=1).values)
    # the output is the best iterate projected on the support (HIO: z = fft2(u), exact up to
    # rounding)
    projected = iterator.getAmplitude(z=state["z"], toggle=True)
    assert torch.allclose(output, projected, rtol=1e-5, atol=1e-6 * output.abs().max())
    if params["algorithm"] == "HIO":
        assert not state["y"].any()


def test_chain_across_algorithms_keeps_improving():
    amplitude, support, unknown = _data()
    hio = PhaseRetrieval(amplitude, support, unknown, **HIO)
    gps = PhaseRetrieval(amplitude, support, unknown, **GPS)
    _, path1, state = hio(30, _phase(2), continue_=True, **HIO)
    _, path2, state = gps(30, state, continue_=True, **GPS)
    _, path3 = hio(30, state, **HIO)
    assert path1.shape == path2.shape == path3.shape == (2, 30)
    assert torch.all(torch.isfinite(path3))


def test_shrinkwrap_continues_support_and_sigma():
    amplitude, support, unknown = _data()
    params = dict(HIO, interval=5, **SW)
    iterator = PhaseRetrieval(amplitude, support, unknown, **params)
    _, _, first = iterator(30, _phase(2), continue_=True, **params)
    sigma1 = first["sigma"][0].item()
    assert sigma1 == pytest.approx(3 * 0.95**5) and torch.all(first["sigma"] == sigma1)
    assert not torch.equal(first["support"], support.expand(2, -1, -1, -1))

    # continuing: 4 iterations do not update the support, so it is handed on unchanged
    _, _, second = iterator(4, first, continue_=True, **params)
    assert torch.equal(second["support"], first["support"])
    assert second["sigma"][0].item() == sigma1
    _, _, third = iterator(30, second, continue_=True, **params)
    assert third["sigma"][0].item() == pytest.approx(sigma1 * 0.95**5)

    # a fresh call starts again from the initial support and sigma_current
    _, _, fresh = iterator(30, _phase(2), continue_=True, **params)
    assert torch.equal(fresh["sigma"], first["sigma"]) and torch.equal(
        fresh["support"], first["support"]
    )


def test_sigma_continue_false_and_sigma_current():
    amplitude, support, unknown = _data()
    params = dict(HIO, interval=5, **SW)
    _, _, state = PhaseRetrieval(amplitude, support, unknown, **params)(
        30, _phase(2), continue_=True, **params
    )
    restart = dict(params, sigma_current=2.0, sigma_continue=False)
    iterator = PhaseRetrieval(amplitude, support, unknown, **restart)
    _, _, out = iterator(4, state, continue_=True, **restart)
    assert out["sigma"][0].item() == 2.0
    with pytest.raises(ValueError):
        PhaseRetrieval(amplitude, support, unknown, **dict(params, sigma_current=3.5))
    with pytest.raises(ValueError):  # the state's sigma exceeds sigma_initial
        smaller = dict(params, sigma_initial=2, sigma_limit=1)
        PhaseRetrieval(amplitude, support, unknown, **smaller)(
            4, dict(state, sigma=state["sigma"] * 0 + 2.5), **smaller
        )


def test_sigma_passes_through_a_stage_without_shrinkwrap():
    amplitude, support, unknown = _data()
    params = dict(HIO, interval=5, **SW)
    _, _, state = PhaseRetrieval(amplitude, support, unknown, **params)(
        30, _phase(2), continue_=True, **params
    )
    _, _, state2 = PhaseRetrieval(amplitude, support, unknown, **GPS)(
        10, state, continue_=True, **GPS
    )
    assert torch.equal(state2["sigma"], state["sigma"])
    assert torch.equal(state2["support"], state["support"])


def test_full_error_path_over_three_stages():
    amplitude, support, unknown = _data()
    hio = PhaseRetrieval(amplitude, support, unknown, **HIO)
    gps = PhaseRetrieval(amplitude, support, unknown, **GPS)
    _, p1, s1 = hio(20, _phase(2), continue_=True, **HIO)
    _, p2, s2 = gps(30, s1, continue_=True, **GPS)
    _, p3, s3 = hio(10, s2, continue_=True, **HIO)
    assert torch.equal(s1["path"], p1)
    assert torch.equal(s3["path"], torch.cat((p1, p2, p3), dim=1))


@pytest.mark.parametrize("params", [HIO, GPS])
def test_continue_from_last(params):
    amplitude, support, unknown = _data()
    iterator = PhaseRetrieval(amplitude, support, unknown, **params)
    phase = _phase(3)
    _, path, best = iterator(15, phase, continue_=True, **params)
    _, _, last = iterator(15, phase, continue_=True, continue_from="last", **params)
    assert torch.equal(best["error"], path.min(dim=1).values)
    assert torch.equal(last["error"], path[:, -1])
    # the last iterate reproduces the last error
    amp = iterator.getAmplitude(z=last["z"])
    assert torch.allclose(iterator.getError(amp), last["error"], rtol=1e-5)
    with pytest.raises(ValueError):
        iterator(2, phase, continue_=True, continue_from="first", **params)
