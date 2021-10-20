# Phase Retrieval Module

Phase retrieval module based on Python 3.7.4 and PyTorch 1.6.0 with CUDA 10.2

Multi-GPU calculation supported by torch.nn.DataParallel wrapper

note that references of each functions are written in docstrings

partial convolution is directly imported from https://github.com/NVIDIA/partialconv

pretrained parameters for PRModule.preconditioner.DenoisingNetwork is required for neural-network-based operations
(it might show poor performance with a case different from the trained condition)

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