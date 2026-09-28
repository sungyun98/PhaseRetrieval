"""Adapter for the ``phaseretrieval`` package (native complex tensors, ``torch.fft``).

Tensors are ``N x 1 x H x W``; complex values use complex64 tensors. The canonical NumPy
interface is the same as in ``pr_legacy.py``.
"""
import contextlib
import os
import sys

import numpy as np
import torch

F64 = os.environ.get("REG_FLOAT64") == "1"
FDT, CDT = (np.float64, np.complex128) if F64 else (np.float32, np.complex64)

NAME = "pr_modern"
NOTES = ["torch.set_num_threads(1) for bit-reproducible CPU results",
         "torch.load wrapped to default map_location='cpu' (weights saved on cuda:0)"]


class Adapter:
    name = NAME
    notes = NOTES
    supports_toggle = True

    def __init__(self, code_root):
        self.root = os.path.abspath(code_root)
        sys.path.insert(0, self.root)
        torch.set_num_threads(1)
        if F64:
            torch.set_default_dtype(torch.float64)
        _orig_load = torch.load

        def _load(f, map_location=None, **kwargs):
            return _orig_load(f, map_location="cpu" if map_location is None else map_location, **kwargs)

        torch.load = _load
        from phaseretrieval import eval as pr_eval
        from phaseretrieval import func
        from phaseretrieval.algorithms import PhaseRetrieval
        from phaseretrieval.preconditioner import Preconditioner

        self.func, self.eval = func, pr_eval
        self.Preconditioner, self.PhaseRetrieval = Preconditioner, PhaseRetrieval

    @staticmethod
    def _t(x, dtype=FDT):
        x = np.asarray(x, dtype=dtype)
        if x.ndim == 2:
            x = x[None]
        return torch.from_numpy(np.ascontiguousarray(x))[:, None]

    @staticmethod
    def _n(t):
        return t[:, 0].detach().cpu().numpy()

    def _any(self, x):
        return self._t(x, CDT if np.iscomplexobj(x) else FDT)

    @contextlib.contextmanager
    def _in_root(self):
        old = os.getcwd()
        os.chdir(self.root)
        try:
            yield
        finally:
            os.chdir(old)

    # ---- func.py -----------------------------------------------------------------------
    def make_support(self, intensity, **kwargs):
        return np.asarray(self.func.MakeSupport(np.array(intensity), **kwargs))

    def fftshift(self, x):
        return self._n(self.func.fftshift(self._any(x)))

    def ifftshift(self, x):
        return self._n(self.func.ifftshift(self._any(x)))

    def amplitude(self, z):
        return self._n(self.func.amplitude(self._any(z)))

    def phase(self, z):
        return self._n(self.func.phase(self._any(z)))

    def sqmesh(self, h, w):
        return self.func.sqmesh(h, w)[0, 0].numpy()

    def freqfilter(self, size, count):
        return list(self.func.freqfilter(size, count))

    def gaussian_smoothing(self, x, sigma, mask=None):
        with torch.no_grad():
            return self._n(self.func.GaussianSmoothing(self._t(x), sigma, mask=None if mask is None else self._t(mask)))

    # ---- preconditioner.py ---------------------------------------------------------------
    def preconditioner(self, amplitude, unknown, limit, deep=True, toggle=False):
        pre = self.Preconditioner(path=os.path.join(self.root, "phaseretrieval", "param_pretrained.pth"))
        return self._n(pre.getKernel(self._t(amplitude), self._t(unknown), limit=limit, deep=deep, toggle=toggle))[0]

    # ---- algorithms.py -------------------------------------------------------------------
    def phase_retrieval(self, amplitude, support, unknown, info, iteration, initial_phase, toggle=False):
        with self._in_root():
            it = self.PhaseRetrieval(self._t(amplitude), self._t(support), self._t(unknown), **dict(info))
        with torch.no_grad():
            out, path = it(iteration, self._t(initial_phase, CDT), toggle=toggle, **dict(info))
        return self._n(out), path.numpy()

    # ---- eval.py (NumPy) -----------------------------------------------------------------
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
