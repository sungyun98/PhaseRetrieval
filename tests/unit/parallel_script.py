"""Run by test_series.py under torchrun: ReconstructParallel in a torchrun job."""

import sys

import torch
from test_algorithms import _data

from phaseretrieval import ReconstructParallel

if __name__ == "__main__":
    HIO = dict(algorithm="HIO", error="R", beta=0.9, beta_type="const", boundary_push=0.1)
    GPS = dict(algorithm="GPS-R", error="R", sigma=(0, 0.1, 0.5, 1), alpha_count=3, t=1, s=0.9)
    amplitude, support, unknown = _data()
    output, path = ReconstructParallel(
        amplitude, support, unknown, [(15, HIO), (15, GPS)], n_seeds=7, batch_size=2
    )
    if output is not None:
        torch.save({"output": output, "path": path}, sys.argv[1])
