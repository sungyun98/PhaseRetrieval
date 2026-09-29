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
independent, so the processes only exchange their results at the end: through shared memory
on one machine, or through `torch.distributed` when launched with ``torchrun`` on one or more
machines.
"""

__all__ = ["ReconstructParallel", "OptimalBatchSize"]

import math
import os
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

from .algorithms import PhaseRetrieval

#: One stage of a series connection: number of iterations and parameters of `PhaseRetrieval`.
Stage = tuple[int, dict[str, Any]]


def _initial_phase(seed: int, start: int, n: int, h: int, w: int) -> Tensor:
    """Random phase factors of reconstructions ``start`` to ``start + n - 1``.

    Each reconstruction has its own generator, seeded by ``(seed, index)``, so the phases do
    not depend on the batch size or on the number and order of the devices.
    """
    theta = torch.empty(n, 1, h, w)
    for k in range(n):
        state = np.random.SeedSequence([seed, start + k]).generate_state(2, dtype=np.uint32)
        generator = torch.Generator().manual_seed(int(state[0]) << 32 | int(state[1]))
        theta[k] = torch.rand(1, h, w, generator=generator) * 2 * math.pi
    return torch.polar(torch.ones_like(theta), theta)


def OptimalBatchSize(
    height: int,
    width: int,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    l2_cache: int | None = None,
) -> int:
    """Return the batch size that reconstructs fastest on a GPU, from the size of its L2 cache.

    The iterations are limited by memory bandwidth, and they run fastest when the arrays of a
    batch stay in the L2 cache of the GPU. Measured on RTX 6000 Ada GPUs (96 MiB of L2 cache)
    for HIO, GPS-R and dpGPS-F at 256 x 256, 512 x 512 and 1024 x 1024, the fastest batch was
    always the one for which six complex arrays of the batch fill the cache::

        batch = L2 cache / (6 * height * width * bytes per complex element)

    i.e. 32, 8 and 2 reconstructions for these sizes; a batch four times larger took about
    twice as long per reconstruction.

    Parameters
    ----------
    height, width : int
        Size of the patterns.
    device : str or torch.device, optional
        GPU to optimize for; by default the current CUDA device. On the CPU, 8 is returned.
    dtype : torch.dtype, optional
        Complex dtype of the iterates; by default complex64 (complex128 if the default dtype is
        float64).
    l2_cache : int, optional
        L2 cache size in bytes, instead of the one of ``device``.

    Returns
    -------
    int
        Batch size, at least 1.
    """
    if dtype is None:
        dtype = torch.complex128 if torch.get_default_dtype() == torch.float64 else torch.complex64
    if l2_cache is None:
        device = torch.device(device if device is not None else "cuda")
        if device.type != "cuda" or not torch.cuda.is_available():
            return 8
        properties = torch.cuda.get_device_properties(device)
        l2_cache = getattr(properties, "L2_cache_size", None)
        if l2_cache is None:  # PyTorch before 2.4 does not report it
            warnings.warn("the L2 cache size is unknown; assuming 32 MiB", stacklevel=2)
            l2_cache = 32 * 2**20
    element = torch.empty((), dtype=dtype).element_size()
    return max(1, l2_cache // (6 * height * width * element))


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
    # built on the device, so that the preconditioner of dRAAR and dpGPS runs there: on the CPU
    # its result depends on the number of threads, which would make the results depend on how
    # the processes were started
    input, support, unknown = (t.to(device) for t in tensors)
    h, w = input.shape[-2:]
    iterators = [PhaseRetrieval(input, support, unknown, **params) for _, params in stages]
    n_batches = math.ceil(n_seeds / batch_size)
    for batch in range(rank, n_batches, len(devices)):
        start = batch * batch_size
        n = min(batch_size, n_seeds - start)
        start_state = _initial_phase(seed, start, n, h, w).to(device)
        paths = []
        with torch.no_grad():
            for k, (iterator, (iteration, params)) in enumerate(zip(iterators, stages)):
                params = {key: v for key, v in params.items() if key != "continue_out"}
                last = k == len(stages) - 1
                result = iterator(
                    iteration, start_state, toggle=toggle and last, continue_out=not last, **params
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
    batch_size: int | None = None,
    seed: int = 0,
    devices: Sequence[int | str | torch.device] | None = None,
    toggle: bool = False,
) -> tuple[Tensor, Tensor] | tuple[None, None]:
    """Run independent reconstructions from random initial phases, spread over several GPUs.

    The reconstructions are split into batches of ``batch_size``; each process, on a device
    of its own, reconstructs every ``n``-th batch, with ``n`` the number of processes. Each
    batch runs through the ``stages`` in series (see `PhaseRetrieval`, ``continue_out``):
    every stage starts from the state left by the previous one. Each reconstruction has its
    own random initial phase, which depends only on ``seed`` and its index, so the results do
    not depend on the number of devices or on the batch size (on GPUs, the errors in ``path``
    can differ in the last digits between batch sizes, since the order of the sums changes).

    Two ways to run:

    * **One machine** (a notebook or a plain script): with one device, or on the CPU,
      everything runs in the calling process; with several devices, one process per device is
      started with `torch.multiprocessing` (``spawn``), so in a script call this function
      under ``if __name__ == "__main__":``.
    * **One or more machines with torchrun** (``WORLD_SIZE`` is set, or a process group is
      initialized): every process of the job calls this function; each uses the GPU
      ``LOCAL_RANK`` (or the CPU) and a share of the batches, and rank 0 receives all results
      over `torch.distributed` (a Gloo group). A process group is initialized (NCCL on GPUs,
      Gloo on CPUs) and destroyed again if none exists. ``devices`` must then be None.

    Batches run fastest per reconstruction when their arrays fit in the L2 cache of the GPU;
    by default the batch size is the smallest `OptimalBatchSize` of the devices. The workers
    print nothing.

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
    batch_size : int, optional
        Reconstructions per batch on one device; by default the smallest `OptimalBatchSize`
        of the devices used (with torchrun, of all ranks).
    seed : int, default 0
        Seed of the random initial phases.
    devices : sequence of int, str or torch.device, optional
        Devices to use on one machine; by default all visible GPUs, or the CPU if there is
        none. Not used with torchrun.
    toggle : bool, default False
        If True, return the k-space result of the last stage without projection on the
        support constraint (see `PhaseRetrieval.forward`).

    Returns
    -------
    output : torch.Tensor
        Best iterates of the last stage, shape ``(n_seeds, 1, H, W)``, on the CPU: real r-space
        objects, or complex k-space iterates if ``toggle`` is True. None on the ranks other
        than 0 with torchrun.
    path : torch.Tensor
        Errors after each iteration of all stages, shape ``(n_seeds, total iterations)``, on
        the CPU (None on the ranks other than 0 with torchrun).
    """
    if _launched_with_torchrun():
        if devices is not None:
            raise ValueError("devices must be None with torchrun: each rank uses LOCAL_RANK.")
        return _reconstruct_distributed(
            input, support, unknown, stages, n_seeds, batch_size, seed, toggle
        )
    if devices is None:
        n_gpu = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(n_gpu)] if n_gpu else ["cpu"]
    devices = [torch.device(f"cuda:{d}" if isinstance(d, int) else d) for d in devices]
    stages = [(int(iteration), dict(params)) for iteration, params in stages]
    h, w = input.shape[-2:]
    if batch_size is None:
        batch_size = min(OptimalBatchSize(h, w, device) for device in devices)
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


def _launched_with_torchrun() -> bool:
    """Whether this process belongs to a torchrun job (or an initialized process group)."""
    if dist.is_available() and dist.is_initialized():
        return True
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _reconstruct_distributed(
    input: Tensor,
    support: Tensor,
    unknown: Tensor,
    stages: Sequence[Stage],
    n_seeds: int,
    batch_size: int | None,
    seed: int,
    toggle: bool,
) -> tuple[Tensor, Tensor] | tuple[None, None]:
    """ReconstructParallel for the processes of a torchrun job, on one or more machines."""
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    else:
        device = torch.device("cpu")
    cpu_group = dist.new_group(backend="gloo")  # results are gathered from the CPU

    stages = [(int(iteration), dict(params)) for iteration, params in stages]
    h, w = input.shape[-2:]
    if batch_size is None:  # the same for all ranks: the smallest optimum
        size = torch.tensor(OptimalBatchSize(h, w, device))
        dist.all_reduce(size, op=dist.ReduceOp.MIN, group=cpu_group)
        batch_size = int(size)
    dtype = torch.complex64 if toggle else torch.float32
    if torch.get_default_dtype() == torch.float64:
        dtype = torch.complex128 if toggle else torch.float64
    output = torch.zeros(n_seeds, 1, h, w, dtype=dtype)
    path = torch.zeros(n_seeds, sum(iteration for iteration, _ in stages))
    tensors = (input.cpu(), support.cpu(), unknown.cpu())
    # every rank fills its own batches; the rows of the other ranks stay zero
    _worker(
        rank, [device] * world, tensors, stages, n_seeds, batch_size, seed, toggle, output, path
    )

    # summing over the ranks assembles the results on rank 0 (x + 0 = x exactly)
    real_output = torch.view_as_real(output) if output.is_complex() else output
    dist.reduce(real_output, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)
    dist.reduce(path, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)
    dist.destroy_process_group(cpu_group)
    if created:
        dist.destroy_process_group()
    return (output, path) if rank == 0 else (None, None)
