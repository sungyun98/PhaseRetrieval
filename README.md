# Phase Retrieval Module

Phase retrieval module based on PyTorch 1.6.0

Multi-GPU calculation supported by torch.nn.DataParallel wrapper

note that references of each functions are written in docstring

1. Basic Notations
    - u: r-space complex matrix corresponding to object (i.e. electron density)
    - z: k-space complex matrix corresponding to Fourier transform of oversampled object (i.e. diffraction pattern)
    - y: Lagrange multiplier complex matrix for dual formulation of optimization problem

2. Supported Algorithms
    - Hybrid input-output (HIO) with boundary push
    - Generalized proximal smoothing (GPS)
    - Deep preconditioned generalized proximal smoothing (dpGPS)

3. Additional Functions
    - Subpixel alignment by phase cross-correlation
    - Pairwise distance
    - Phase retrieval transfer function (PRTF)
    - Power spectral density (PSD)
    - Eigenmode and low-rank approximation by singular value decomposition (SVD)