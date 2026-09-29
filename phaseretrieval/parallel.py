###############################################################################
# Multi-GPU Phase Retrieval
#
# Author: SUNG YUN LEE
#
# Contact: sungyun98@g.postech.edu
###############################################################################

"""Run many phase retrieval reconstructions on several GPUs, one process per GPU.

This follows the pattern of `torch.nn.parallel.DistributedDataParallel` (one process per GPU,
each working on its own share of the data) rather than `torch.nn.DataParallel` (one process
that splits every batch over the GPUs). DistributedDataParallel itself only applies to models
with trainable parameters, whose gradients it synchronizes; the reconstructions here are
independent, so the processes do not communicate and write their results into shared memory.
"""

__all__ = ["ReconstructParallel"]

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor

from .algorithms import PhaseRetrieval

#: One stage of a series connection: number of iterations and parameters of `PhaseRetrieval`.
Stage = tuple[int, dict[str, Any]]


def _initial_phase(seed: int, batch: int, n: int, h: int, w: int) -> Tensor:
    """Random phase factors of one batch, independent of the number and order of the GPUs."""
    state = np.random.SeedSequence([seed, batch]).generate_state(2, dtype=np.uint32)
    generator = torch.Generator().manual_seed(int(state[0]) << 32 | int(state[1]))
    theta = torch.rand(n, 1, h, w, generator=generator) * 2 * math.pi
    return torch.polar(torch.ones_like(theta), theta)


def _worker(
    rank: int,
    devices: list[torch.device],
    tensors: tuple[Tensor, Tensor, Tensor],
    stages: list[Stage],
    n_seeds: int,
    batch_size: int,
    seed: int,
    toggle: bool,
    output: Tensor,
    path: Tensor,
) -> None:
    """Reconstruct the batches ``rank, rank + len(devices), ...`` on ``devices[rank]``."""
    device = devices[rank]
    if device.type == "cuda":
        torch.cuda.set_device(device)
    input, support, unknown = tensors
    h, w = input.shape[-2:]
    iterators = [
        PhaseRetrieval(input, support, unknown, **params).to(device) for _, params in stages
    ]
    n_batches = math.ceil(n_seeds / batch_size)
    for batch in range(rank, n_batches, len(devices)):
        start = batch * batch_size
        n = min(batch_size, n_seeds - start)
        start_state = _initial_phase(seed, batch, n, h, w).to(device)
        paths = []
        with torch.no_grad():
            for k, (iterator, (iteration, params)) in enumerate(zip(iterators, stages)):
                params = {key: v for key, v in params.items() if key != "continue_"}
                last = k == len(stages) - 1
                result = iterator(
                    iteration, start_state, toggle=toggle and last, continue_=not last, **params
                )
                paths.append(result[1])
                if not last:
                    start_state = result[2]
        output[start : start + n] = result[0].cpu()
        path[start : start + n] = torch.cat(paths, dim=1).cpu()


def ReconstructParallel(
    input: Tensor,
    support: Tensor,
    unknown: Tensor,
    stages: Sequence[Stage],
    n_seeds: int,
    batch_size: int,
    seed: int = 0,
    devices: Sequence[int | str | torch.device] | None = None,
    toggle: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run independent reconstructions from random initial phases, spread over several GPUs.

    The reconstructions are split into batches of ``batch_size``; each device, in a process
    of its own, reconstructs every ``len(devices)``-th batch. Each batch runs through the
    ``stages`` in series (see `PhaseRetrieval`, ``continue_``): every stage starts from the
    state left by the previous one. The random initial phases depend only on ``seed`` and the
    batch index, so the results do not depend on the number of devices.

    With one device (or on the CPU), everything runs in the calling process. With several,
    the processes are started with `torch.multiprocessing` (``spawn``); in a script, call this
    function under ``if __name__ == "__main__":``. The workers print nothing (their output does
    not reach Jupyter notebooks anyway).

    Parameters
    ----------
    input, support, unknown : torch.Tensor
        Arguments of `PhaseRetrieval`, on the CPU: amplitude, support and missing-data mask,
        each of shape ``(1, 1, H, W)``, not fftshifted (the support is a real-space array).
    stages : sequence of (int, dict)
        Number of iterations and parameters (for the constructor and the call of
        `PhaseRetrieval`) of each stage, e.g. ``[(3000, params)]`` for a single stage, or
        ``[(500, hio_params), (1000, gps_params)]``. A stage may set ``continue_from``.
    n_seeds : int
        Number of reconstructions.
    batch_size : int
        Reconstructions per batch on one device.
    seed : int, default 0
        Seed of the random initial phases.
    devices : sequence of int, str or torch.device, optional
        Devices to use; by default all visible GPUs, or the CPU if there is none.
    toggle : bool, default False
        If True, return the k-space result of the last stage without projection on the
        support constraint (see `PhaseRetrieval.forward`).

    Returns
    -------
    output : torch.Tensor
        Best iterates of the last stage, shape ``(n_seeds, 1, H, W)``, on the CPU: real r-space
        objects, or complex k-space iterates if ``toggle`` is True.
    path : torch.Tensor
        Errors after each iteration of all stages, shape ``(n_seeds, total iterations)``.
    """
    if devices is None:
        n_gpu = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(n_gpu)] if n_gpu else ["cpu"]
    devices = [torch.device(f"cuda:{d}" if isinstance(d, int) else d) for d in devices]
    stages = [(int(iteration), dict(params)) for iteration, params in stages]
    h, w = input.shape[-2:]
    dtype = torch.complex64 if toggle else torch.float32
    if torch.get_default_dtype() == torch.float64:
        dtype = torch.complex128 if toggle else torch.float64
    output = torch.zeros(n_seeds, 1, h, w, dtype=dtype)
    path = torch.zeros(n_seeds, sum(iteration for iteration, _ in stages))
    tensors = (input.cpu(), support.cpu(), unknown.cpu())
    args = (devices, tensors, stages, n_seeds, batch_size, seed, toggle)

    if len(devices) == 1:
        _worker(0, *args, output, path)
    else:
        output.share_memory_()
        path.share_memory_()
        mp.spawn(_worker, args=(*args, output, path), nprocs=len(devices), join=True)
    return output, path
