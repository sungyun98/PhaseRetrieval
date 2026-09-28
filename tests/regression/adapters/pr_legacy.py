"""Adapter for the original PRModule API (tag v1.0-legacy, PyTorch 1.6).

The legacy code stores complex numbers as float tensors with a trailing dimension of 2
and uses 5-D tensors N x 1 x H x W x (1 or 2). This adapter exposes the canonical NumPy
interface used by the regression cases:

* real images: ``(N, H, W)`` or ``(H, W)`` float32 arrays
* complex images: ``(N, H, W)`` complex64 arrays

Deviation from the original environment (recorded in the reference metadata): the pretrained
weights were saved on ``cuda:0`` and ``Preconditioner`` calls ``torch.load`` without
``map_location``, which fails on CPU-only PyTorch. ``torch.load`` is therefore wrapped to
default to ``map_location='cpu'``. The algorithm code itself is not modified.
"""
import contextlib
import os
import sys

import numpy as np
import torch

NAME = "pr_legacy"
NOTES = ["torch.load wrapped to default map_location='cpu' (weights saved on cuda:0)",
         "torch.set_num_threads(1) for bit-reproducible CPU results"]


class Adapter:
    name = NAME
    notes = NOTES
    supports_toggle = True

    def __init__(self, code_root):
        self.root = os.path.abspath(code_root)
        sys.path.insert(0, self.root)
        torch.set_num_threads(1)
        _orig_load = torch.load

        def _load(f, map_location=None, **kwargs):
            return _orig_load(f, map_location="cpu" if map_location is None else map_location, **kwargs)

        torch.load = _load
        import PRModule  # noqa: F401  (import after sys.path setup)
        from PRModule import eval as pr_eval
        from PRModule import func
        from PRModule.preconditioner import Preconditioner
        from PRModule.phaseretrieval import PhaseRetrieval

        self.func, self.eval = func, pr_eval
        self.Preconditioner, self.PhaseRetrieval = Preconditioner, PhaseRetrieval

    # ---- conversions -----------------------------------------------------------------
    @staticmethod
    def _r2t(x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        return torch.from_numpy(np.ascontiguousarray(x))[:, None, :, :, None]

    @staticmethod
    def _c2t(z):
        z = np.asarray(z, dtype=np.complex64)
        t = np.stack([z.real, z.imag], axis=-1).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(t))[:, None]

    @staticmethod
    def _t2r(t):
        return t[:, 0, :, :, 0].detach().cpu().numpy()

    @staticmethod
    def _t2c(t):
        t = t[:, 0].detach().cpu().numpy()
        return (t[..., 0] + 1j * t[..., 1]).astype(np.complex64)

    @contextlib.contextmanager
    def _in_root(self):
        old = os.getcwd()
        os.chdir(self.root)  # Preconditioner() uses the cwd-relative default weight path
        try:
            yield
        finally:
            os.chdir(old)

    # ---- func.py -----------------------------------------------------------------------
    def make_support(self, intensity, **kwargs):
        return np.asarray(self.func.MakeSupport(np.array(intensity), **kwargs))

    def fftshift(self, x):
        if np.iscomplexobj(x):
            return self._t2c(self.func.fftshift(self._c2t(x)))
        return self._t2r(self.func.fftshift(self._r2t(x)))

    def ifftshift(self, x):
        if np.iscomplexobj(x):
            return self._t2c(self.func.ifftshift(self._c2t(x)))
        return self._t2r(self.func.ifftshift(self._r2t(x)))

    def amplitude(self, z):
        return self._t2r(self.func.amplitude(self._c2t(z)))

    def phase(self, z):
        return self._t2c(self.func.phase(self._c2t(z)))

    def sqmesh(self, h, w):
        return self.func.sqmesh(h, w)[0, 0, :, :, 0].numpy()

    def freqfilter(self, size, count):
        return list(self.func.freqfilter(size, count))

    def gaussian_smoothing(self, x, sigma, mask=None):
        xt = self._r2t(x)[..., 0]
        mt = None if mask is None else self._r2t(mask)[..., 0]
        with torch.no_grad():
            return self._t2r(self.func.GaussianSmoothing(xt, sigma, mask=mt))

    # ---- preconditioner.py ---------------------------------------------------------------
    def preconditioner(self, amplitude, unknown, limit, deep=True, toggle=False):
        pre = self.Preconditioner(path=os.path.join(self.root, "PRModule", "param_pretrained.pth"))
        out = pre.getKernel(self._r2t(amplitude), self._r2t(unknown), limit=limit, deep=deep, toggle=toggle)
        return self._t2r(out)[0]

    # ---- phaseretrieval.py ---------------------------------------------------------------
    def phase_retrieval(self, amplitude, support, unknown, info, iteration, initial_phase, toggle=False):
        with self._in_root():
            it = self.PhaseRetrieval(self._r2t(amplitude), self._r2t(support), self._r2t(unknown), **dict(info))
        with torch.no_grad():
            out, path = it(iteration, self._c2t(initial_phase), toggle=toggle, **dict(info))
        out = self._t2c(out) if toggle else self._t2r(out)
        return out, path.numpy()

    # ---- eval.py (NumPy; inputs are copied because the functions modify them in place) ------
    def subpixel_alignment(self, x, error=None, ref=None, subpixel=1):
        err = None if error is None else np.array(error)
        return self.eval.SubpixelAlignment(np.array(x), error=err, ref=ref, subpixel=subpixel)

    def pairwise_distance(self, x):
        return self.eval.PairwiseDistance(np.array(x))

    def prtf(self, x, ref, mask=None):
        return self.eval.PRTF(np.array(x), np.array(ref), mask=mask)

    def psd(self, x, mask=None):
        return self.eval.PSD(np.array(x), mask=mask)

    def eigenmode(self, x, k=None, lowrank=True):
        return self.eval.EigenMode(np.array(x), k=k, lowrank=lowrank)
