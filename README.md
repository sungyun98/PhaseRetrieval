# Phase Retrieval Module

> **Package renamed:** `PRModule` is now `phaseretrieval`, and `PRModule/phaseretrieval.py` is now
> `phaseretrieval/algorithms.py`. The folders had to be renamed to merge the phase retrieval code
> shared by this repository and [DPR](https://github.com/sungyun98/DPR) into one package. We
> apologize for the inconvenience to existing users: please replace `from PRModule import ...`
> with `from phaseretrieval import ...`. The original code remains available at the tag
> `v1.0-legacy`.

Iterative phase retrieval for coherent diffraction imaging, based on PyTorch: HIO, RAAR, gRAAR,
dRAAR, GPS and dpGPS, with tools to evaluate the reconstructions. The algorithms run many random
starts as one batch on the CPU or a GPU. The package is also used by
[DPR](https://github.com/sungyun98/DPR) to refine its network outputs.

## Installation

Install PyTorch for your platform first (see <https://pytorch.org>), then the package from
GitHub:

```bash
pip install git+https://github.com/sungyun98/PhaseRetrieval.git
```

To run the example notebook and the tests, clone the repository and create the tested
environment instead:

```bash
git clone https://github.com/sungyun98/PhaseRetrieval.git
cd PhaseRetrieval
conda env create -f environment.yml   # environment "phaseretrieval"
# or, in an existing environment:
pip install -r requirements.txt && pip install -e .
```

Tested with Python 3.12, PyTorch 2.14.0 (CUDA 12.6 build), NumPy 2.5.3, SciPy 1.18.1,
scikit-image 0.26.0 and tqdm 4.70.1; the exact versions are in `requirements.txt`. Minimum
versions: Python 3.10, PyTorch 2.1, NumPy 1.26, SciPy 1.11, scikit-image 0.20. The PyTorch build
in `requirements.txt` uses CUDA 12.6 and runs with NVIDIA drivers 525 or newer; replace `cu126`
with `cpu` for a CPU-only installation.

## Usage

`demo.ipynb` walks through a complete reconstruction of `sample_lena.mat`, including the
evaluation (alignment, pairwise distance, PRTF, PSD and eigenmodes), running the
reconstructions on all GPUs with `ReconstructParallel` (one process per GPU). In short, on one
device:

```python
import numpy as np
import torch
from scipy.io import loadmat

from phaseretrieval import PhaseRetrieval, SubpixelAlignment

data = loadmat("sample_lena.mat")
intensity = data["intensity"]  # fftshifted intensity, NaN for missing pixels
missing = np.isnan(intensity)
intensity[missing] = 0


def to_tensor(x):
    return torch.from_numpy(x.astype(np.float32))[None, None]  # (1, 1, H, W)


# k-space inputs are not fftshifted (zero frequency at index (0, 0))
amplitude = to_tensor(np.sqrt(np.fft.ifftshift(intensity)))
unknown = to_tensor(np.fft.ifftshift(missing))
support = to_tensor(data["support"] > 0)

info = {
    "algorithm": "GPS-R",
    "error": "R",
    "sigma": (0, 0.01, 0.4, 0.1, 0.7, 1),  # schedule: (ratio, value, ratio, value, ...)
    "alpha_count": 10,
    "t": 1,
    "s": 0.8,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iterator = PhaseRetrieval(amplitude, support, unknown, **info).to(device)

# 20 independent reconstructions from random initial phases
theta = 2 * torch.pi * torch.rand(20, 1, *amplitude.shape[-2:])
initial_phase = torch.polar(torch.ones_like(theta), theta).to(device)
output, path = iterator(1000, initial_phase, **info)  # (20, 1, H, W), (20, 1000)

# sort by the lowest error and align to the best reconstruction
objects, error = SubpixelAlignment(
    output[:, 0].cpu().numpy(), error=path.min(dim=1).values.cpu().numpy(), subpixel=10
)
```

The sample pattern has about 1.4 x 10<sup>6</sup> photons, so its R-factor stays near 0.53, the
value of the true object itself. The parameters of every algorithm are described in the
docstring of `PhaseRetrieval` (`help(PhaseRetrieval)`), and the same dictionary can be passed to
the constructor and to the call.

Iterators can be connected in series. With `continue_out=True`, a call also returns a state
(best iterates, or the last ones with `continue_from="last"`, support after ShrinkWrap,
ShrinkWrap sigma and the errors of all connected calls in `state["path"]`), from which the
same or another iterator continues. For example, HIO with ShrinkWrap followed by GPS-R on the updated support,
continuing the example above:

```python
hio_params = {
    "algorithm": "HIO",
    "error": "R",
    "beta": 0.9,
    "beta_type": "const",
    "boundary_push": 0,
    "shrinkwrap": True,
    "sigma_initial": 3,
    "sigma_limit": 1.5,
    "ratio_update": 0.01,
    "threshold": 0.1,
    "interval": 50,
}
hio = PhaseRetrieval(amplitude, support, unknown, **hio_params).to(device)
_, _, state = hio(500, initial_phase, continue_out=True, **hio_params)
output, path = iterator(1000, state, **info)  # starts from the HIO results and support
```

A following stage with ShrinkWrap continues from the sigma of the state (`sigma_continue`,
default True) or starts from `sigma_current`.

dRAAR and dpGPS use a preconditioner from a denoising network whose pretrained weights ship with
the package (`phaseretrieval/param_pretrained.pth`). They need the intensity in photon counts,
and the network may perform poorly for conditions different from the trained ones.

## Notations and functions

1. Notations
    - u: r-space complex object (e.g. electron density)
    - z: k-space complex Fourier transform of the oversampled object (e.g. diffraction pattern)
    - y: complex Lagrange multiplier of the dual formulation of the optimization problem

2. Supported algorithms (with the R-factor and the Poisson negative log-likelihood as error
   metrics)
    - Hybrid input-output (HIO) with boundary push
    - Relaxed averaged alternating reflections (RAAR) with boundary push
    - RAAR with the projection on data denoised by Gaussian smoothing or deep learning (gRAAR,
      dRAAR)
    - Generalized proximal smoothing (GPS)
    - Deep preconditioned generalized proximal smoothing (dpGPS)

3. Additional functions
    - Centre of symmetry of a diffraction pattern, and alignment of objects by centroid or by
      cross-correlation with a target (including the twin image)
    - Subpixel alignment by phase cross-correlation
    - Pairwise distance
    - Phase retrieval transfer function (PRTF)
    - Power spectral density (PSD)
    - Eigenmodes and low-rank approximation of a set of reconstructions by singular value
      decomposition (SVD)

The references of each method are given in the docstrings.

## Reproducing the paper results

The tag `v1.0-legacy` is the code used for the paper (Python 3.7, PyTorch 1.6, CUDA 10.2) and
reproduces its results:

```bash
git checkout v1.0-legacy
```

`tests/regression/environment-legacy.yml` describes a CPU environment that runs it. The
regression tests in `tests/regression/` compare the current code with reference outputs of
`v1.0-legacy` (see `tests/regression/README.md`). All algorithms and functions match within
floating-point tolerance, except for three intended changes:

- ShrinkWrap failed in the original code with a TypeError; it now works with a centred Gaussian
  kernel and restarts from the initial support on every call.
- PSD measures radii from the zero frequency `(H // 2, W // 2)` instead of half a pixel off for
  even sizes, which changes PSD and PRTF curves by a few percent.
- The low-rank approximation of EigenMode approximates the whole set of reconstructions instead
  of the first one.

Unit tests of the new behaviour are in `tests/unit/` (`python -m pytest tests/unit`).

## Citation

If you use this code, please cite:

> S. Y. Lee, D. H. Cho, C. Jung, D. Sung, D. Nam, S. Kim, and C. Song, Denoising low-intensity
> diffraction signals using *k*-space deep learning: Applications to phase recovery,
> *Phys. Rev. Research* **3**, 043066 (2021). <https://doi.org/10.1103/PhysRevResearch.3.043066>

```bibtex
@article{lee2021prresearch,
  title   = {Denoising low-intensity diffraction signals using $k$-space deep learning: Applications to phase recovery},
  author  = {Lee, Sung Yun and Cho, Do Hyung and Jung, Chulho and Sung, Daeho and Nam, Daewoong and Kim, Sangsoo and Song, Changyong},
  journal = {Physical Review Research},
  volume  = {3},
  number  = {4},
  pages   = {043066},
  year    = {2021},
  doi     = {10.1103/PhysRevResearch.3.043066}
}
```

## Contact

Sung Yun Lee, sungyun98@g.postech.edu

## License

This code is released under the BSD 2-Clause License (`LICENSE`), except for the third-party
code below, which keeps its original license (full text in `LICENSES/`):

| File | Source | License |
|---|---|---|
| `phaseretrieval/partialconv2d.py` | [NVIDIA/partialconv](https://github.com/NVIDIA/partialconv) `models/partialconv2d.py` | BSD 3-Clause, Copyright (c) 2018 NVIDIA Corporation |
