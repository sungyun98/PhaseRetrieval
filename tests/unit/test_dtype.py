"""Unit tests: PhaseRetrieval and ReconstructParallel follow the dtype of the input."""

import pytest
import torch
from test_algorithms import _data, _phase

from phaseretrieval import PhaseRetrieval, ReconstructParallel

COMMON = dict(error="R", beta=0.9, beta_type="const", boundary_push=0, sigma=0.5, alpha_count=2)
COMMON.update(t=1, s=0.9, limit=0.25, deep=True)
SW = dict(shrinkwrap=True, sigma_initial=3, sigma_limit=1.5, ratio_update=0.05, threshold=0.1)
ALGORITHMS = ["HIO", "RAAR", "gRAAR", "dRAAR", "GPS-R", "GPS-F", "dpGPS-R", "dpGPS-F"]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _run(params, device, default_dtype):
    amplitude, support, unknown = (x.double().to(device) for x in _data())
    phase = _phase(2).to(torch.complex128).to(device)  # the same phases for both defaults
    previous = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        iterator = PhaseRetrieval(amplitude / 20, support, unknown, **params)
        return iterator(12, phase, continue_out=True, **params)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shrinkwrap", [False, True])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_float64_input_with_float32_default(algorithm, shrinkwrap, device):
    params = dict(COMMON, algorithm=algorithm, **(dict(SW, interval=5) if shrinkwrap else {}))
    output, path, state = _run(params, device, torch.float32)
    assert output.dtype == path.dtype == state["error"].dtype == torch.float64
    assert state["z"].dtype == state["y"].dtype == torch.complex128
    # the same computation as with float64 as the default dtype
    output64, path64, _ = _run(params, device, torch.float64)
    assert torch.equal(output, output64) and torch.equal(path, path64)


def test_reconstruct_parallel_follows_the_input_dtype():
    amplitude, support, unknown = (x.double() for x in _data())
    params = dict(COMMON, algorithm="HIO")
    output, path = ReconstructParallel(
        amplitude, support, unknown, [(5, params)], 3, batch_size=2, devices=["cpu"]
    )
    assert output.dtype == path.dtype == torch.float64
    output, _ = ReconstructParallel(
        amplitude, support, unknown, [(5, params)], 3, batch_size=2, devices=["cpu"], toggle=True
    )
    assert output.dtype == torch.complex128
