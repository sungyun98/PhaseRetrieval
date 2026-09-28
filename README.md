# Phase Retrieval Module

> **Package renamed:** `PRModule` is now `phaseretrieval`, and `PRModule/phaseretrieval.py` is now
> `phaseretrieval/algorithms.py`. The folders had to be renamed to merge the phase retrieval code
> shared by this repository and [DPR](https://github.com/sungyun98/DPR) into one package. We
> apologize for the inconvenience to existing users: please replace `from PRModule import ...`
> with `from phaseretrieval import ...`. The original code remains available at the tag
> `v1.0-legacy`.

Phase retrieval module based on PyTorch

## Requirements

Tested with Python 3.12, PyTorch 2.14.0 (CUDA 12.6 build), NumPy 2.5.3, SciPy 1.18.1,
scikit-image 0.26.0 and tqdm 4.70.1; the exact versions are listed in `requirements.txt`.
Minimum versions: Python 3.10, PyTorch 2.1, NumPy 1.26, SciPy 1.11, scikit-image 0.20.

```bash
conda env create -f environment.yml
# or, in an existing environment:
pip install -r requirements.txt && pip install -e .
```

The PyTorch build in `requirements.txt` uses CUDA 12.6 and runs with NVIDIA drivers 525 or
newer; replace `cu126` with `cpu` for a CPU-only installation. The original code for Python 3.7,
PyTorch 1.6 and CUDA 10.2 is available at the tag `v1.0-legacy`.

Multi-GPU calculation supported by torch.nn.DataParallel wrapper

pretrained parameters for phaseretrieval.preconditioner.DenoisingNetwork is required for neural-network-based operations
(it might show poor performance with a case different from the trained condition)

## Notations and Functions

1. Basic Notations
    - u: r-space complex matrix corresponding to object (i.e. electron density)
    - z: k-space complex matrix corresponding to Fourier transform of oversampled object (i.e. diffraction pattern)
    - y: Lagrange multiplier complex matrix for dual formulation of optimization problem

2. Supported Algorithms (with R-factor and Poisson NLL as error metrics)
    - Hybrid input-output (HIO) with boundary push
    - Relaxed averaged alternating reflections (RAAR) with boundary push
    - RAAR with projection operator on denoised constraint by Gaussian smoothing or deep learning (gRAAR, dRAAR)
    - Generalized proximal smoothing (GPS)
    - Deep preconditioned generalized proximal smoothing (dpGPS)

3. Additional Functions
    - Subpixel alignment by phase cross-correlation
    - Pairwise distance
    - Phase retrieval transfer function (PRTF)
    - Power spectral density (PSD)
    - Eigenmode and low-rank approximation by singular value decomposition (SVD)

## Citation
<https://doi.org/10.1103/PhysRevResearch.3.043066>

note that references of each functions are written in docstrings

partial convolution is directly imported from <https://github.com/NVIDIA/partialconv>
