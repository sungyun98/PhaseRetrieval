"""Unit tests of several patterns reconstructed together (one reconstruction per pattern)."""

import numpy as np
import pytest
import torch
from test_algorithms import _data, _phase

from phaseretrieval import PhaseRetrieval

PARAMS = {
    "HIO": dict(algorithm="HIO", beta=0.9, beta_type="const", boundary_push=0.1),
    "GPS-R": dict(algorithm="GPS-R", sigma=(0, 0.1, 0.5, 1), alpha_count=3, t=1, s=0.9),
    "dRAAR": dict(
        algorithm="dRAAR", beta=0.9, beta_type="const", boundary_push=0, limit=0.25, deep=True
    ),
    "dpGPS-R": dict(
        algorithm="dpGPS-R", sigma=(0, 0.1, 0.5, 1), alpha_count=3, limit=0.25, deep=True
    ),
    "dpGPS-F": dict(
        algorithm="dpGPS-F", sigma=(0, 0.1, 0.5, 1), alpha_count=3, limit=0.25, deep=True
    ),
}


def _patterns(n=3):
    data = [_data(seed=seed) for seed in range(n)]
    # photon-count-like amplitudes for the preconditioner of dRAAR
    amplitude = torch.cat([d[0] for d in data]) / 20
    support = torch.cat([d[1] for d in data])
    unknown = torch.cat([d[2] for d in data])
    return amplitude, support, unknown


@pytest.mark.parametrize("error", ["R", "NLL"])
@pytest.mark.parametrize("name", list(PARAMS))
def test_batched_patterns_equal_separate_runs(name, error):
    params = dict(PARAMS[name], error=error)
    amplitude, support, unknown = _patterns()
    phase = _phase(3)
    batched = PhaseRetrieval(amplitude, support, unknown, **params)
    out, path = batched(15, phase, **params)
    for i in range(3):
        single = PhaseRetrieval(
            amplitude[i : i + 1], support[i : i + 1], unknown[i : i + 1], **params
        )
        out_i, path_i = single(15, phase[i : i + 1], **params)
        np.testing.assert_allclose(out[i : i + 1].numpy(), out_i.numpy(), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(path[i : i + 1].numpy(), path_i.numpy(), rtol=1e-5)


def test_pattern_count_checks():
    amplitude, support, unknown = _patterns()
    iterator = PhaseRetrieval(amplitude, support, unknown, **PARAMS["HIO"], error="R")
    with pytest.raises(ValueError):
        iterator(3, _phase(2), **PARAMS["HIO"])
