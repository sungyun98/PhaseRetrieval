###############################################################################
# Phase Retrieval Algorithms : HIO, RAAR, GPS, gRAAR, dRAAR, dpGPS
#
# Author: SUNG YUN LEE
#
# Contact: sungyun98@g.postech.edu
###############################################################################

"""Iterative phase retrieval algorithms: HIO, RAAR, gRAAR, dRAAR, GPS and dpGPS.

Conventions used throughout the module:

* ``u`` is the real-space (r-space) complex object, ``z`` its Fourier transform (k-space)
  and ``y`` the Lagrange multiplier of the dual problem solved by GPS and dpGPS.
* Tensors have the layout ``(N, 1, H, W)``, where ``N`` is the number of independent
  reconstructions (random initial phases). The measured data have ``N = 1`` and broadcast.
* k-space data are *not* fftshifted: the zero frequency is at index ``(0, 0)``, so that
  ``torch.fft.fft2`` and ``torch.fft.ifft2`` are applied without shifts.
"""

__all__ = ["PhaseRetrieval"]

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.fft import fft2, ifft2

from .func import GaussianSmoothing, fftshift, freqfilter, ifftshift, phase, sqmesh
from .preconditioner import Preconditioner

#: A constant value, or a schedule ``(ratio_0, value_0, ratio_1, value_1, ...)``: ``value_k``
#: applies from iteration ``round(ratio_k * iteration)`` on (``ratio_0`` is normally 0).
Schedule = float | tuple[float, ...] | list[float]


class GaussianFilter(nn.Module):
    """Gaussian low-pass filter of the oversampling smoothness (OSS) method.

    The filter peak is at the centre, index ``(H // 2, W // 2)``; apply `ifftshift` to use it
    on unshifted k-space data.

    Parameters
    ----------
    height, width : int
        Size of the filter.

    References
    ----------
    .. [1] https://doi.org/10.1107/S0021889813002471
    """

    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        mesh = sqmesh(height, width)
        self.register_buffer("mesh", mesh)

    def forward(self, alpha: float) -> Tensor:
        """Return the filter ``exp(-r**2 / (2 * alpha**2))`` normalized to a maximum of 1.

        Parameters
        ----------
        alpha : float
            Width (standard deviation) of the filter in pixels.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``(1, 1, H, W)``.
        """
        filter = torch.exp(-0.5 * self.mesh / alpha**2)
        return filter / filter.max()


class ShrinkWrap(GaussianFilter):
    """ShrinkWrap update of the support constraint.

    The object is smoothed with a normalized Gaussian kernel of size
    ``2 * ceil(2 * sigma_initial) + 1`` (as in the MATLAB function ``imgaussfilt``) and
    thresholded. After each `update`, ``sigma`` is multiplied by ``1 - ratio_update`` until it
    reaches ``sigma_limit``; `reset` restores ``sigma_initial``.

    Parameters
    ----------
    threshold : float
        Threshold of the new support relative to the maximum of the smoothed object.
    sigma_initial : float, default 3
        Initial standard deviation of the kernel in pixels.
    sigma_limit : float, default 1.5
        The kernel is no longer updated once ``sigma`` is not above this value. If it equals
        ``sigma_initial``, the kernel is fixed.
    ratio_update : float, default 0.01
        Relative decrease of ``sigma`` per update.

    Raises
    ------
    ValueError
        If ``sigma_initial`` is smaller than ``sigma_limit``.

    References
    ----------
    .. [1] https://doi.org/10.1103/PhysRevB.68.140101
    """

    def __init__(
        self,
        threshold: float,
        sigma_initial: float = 3,
        sigma_limit: float = 1.5,
        ratio_update: float = 0.01,
    ) -> None:
        if sigma_initial < sigma_limit:
            raise ValueError(
                f"sigma_initial ({sigma_initial}) must not be smaller than sigma_limit "
                f"({sigma_limit})."
            )
        self.sigma_initial = sigma_initial
        self.sigma_limit = sigma_limit
        self.ratio = ratio_update
        self.threshold = threshold

        # the kernel size is set by the initial sigma, the largest one
        size = 2 * math.ceil(2 * sigma_initial) + 1
        self.pad = math.ceil(2 * sigma_initial)
        super().__init__(size, size)
        self.register_buffer("filter", self.mesh * 0)
        self.reset()

    def reset(self) -> None:
        """Restore ``sigma_initial`` and its kernel."""
        self.sigma = self.sigma_initial
        self._compute_filter()

    def _compute_filter(self) -> None:
        self.filter = torch.exp(-0.5 * self.mesh / self.sigma**2)
        self.filter = self.filter / self.filter.sum()

    def update(self, update_sigma: bool = True) -> None:
        """Decrease ``sigma`` and recompute the kernel, while ``sigma > sigma_limit``.

        Parameters
        ----------
        update_sigma : bool, default True
            If False, recompute the kernel for the current ``sigma`` without decreasing it.
        """
        if self.sigma > self.sigma_limit:
            if update_sigma:
                self.sigma = self.sigma * (1 - self.ratio)
            self._compute_filter()

    def forward(self, u: Tensor) -> Tensor:
        """Compute a new support from the smoothed object.

        Parameters
        ----------
        u : torch.Tensor
            Real r-space object of shape ``(N, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Support of shape ``(N, 1, H, W)``: 1 where the smoothed object exceeds
            ``threshold`` times its maximum (per image), 0 elsewhere; dtype float32.
        """
        n = u.size(0)
        u = F.conv2d(
            F.pad(u, pad=(self.pad, self.pad, self.pad, self.pad), mode="reflect"),
            weight=self.filter,
        )
        u_max = u.view(n, -1).max(dim=-1).values.view(n, 1, 1, 1)
        return torch.gt(u, u_max * self.threshold).float()


class PhaseRetrievalUnit(nn.Module):
    """Single iteration of a phase retrieval algorithm.

    Used by `PhaseRetrieval`, which documents the algorithms. The support constraint is the
    non-negative real object inside the support, and all operators use its convex conjugate.

    Parameters
    ----------
    input : torch.Tensor
        Measured k-space amplitude of shape ``(1, 1, H, W)``, not fftshifted.
    support : torch.Tensor
        Real-space support of shape ``(1 or N, 1, H, W)``, 1 inside and 0 outside.
    unknown : torch.Tensor
        Mask of shape ``(1, 1, H, W)``, not fftshifted: 1 where the amplitude is unknown
        (missing data), 0 where it is measured.
    type : str
        Algorithm: ``'HIO'``, ``'RAAR'``, ``'gRAAR'``, ``'dRAAR'``, ``'GPS-R'``, ``'GPS-F'``,
        ``'dpGPS-R'`` or ``'dpGPS-F'``.
    **kwargs
        ``preconditioner`` (required for dRAAR, dpGPS-R and dpGPS-F): preconditioning kernel
        of shape ``(1, 1, H, W)`` from `Preconditioner.getKernel`.
    """

    def __init__(
        self, input: Tensor, support: Tensor, unknown: Tensor, type: str, **kwargs: Any
    ) -> None:
        super().__init__()
        self.register_buffer("magnitude", input)
        self.register_buffer("unknown", unknown)
        self.register_buffer("support", support)
        self.type = type

        # allocate denoised magnitude for projection operator on denoised pattern
        if type in ["gRAAR"]:
            input_dn = GaussianSmoothing(
                fftshift(input).pow(2), sigma=1.5, mask=1 - fftshift(unknown)
            ).sqrt()
            input_dn = ifftshift(input_dn)
            self.register_buffer("magnitude_dn", input_dn)
        if type in ["dRAAR"]:
            kernel = kwargs.pop("preconditioner")
            self.register_buffer("magnitude_dn", input * kernel)

        # allocate Gaussian filter for GPS algorithms
        if type in ["GPS-R", "GPS-F", "dpGPS-R", "dpGPS-F"]:
            self.filter = GaussianFilter(self.magnitude.size(2), self.magnitude.size(3))
        # allocate preconditioner for deep preconditioned algorithms
        if type in ["dpGPS-R", "dpGPS-F"]:
            kernel = kwargs.pop("preconditioner")
            self.register_buffer("kernel", kernel)

    def updateSupport(self, support: Tensor) -> None:
        """Replace the support constraint.

        Parameters
        ----------
        support : torch.Tensor
            Real-space support of shape ``(1 or N, 1, H, W)``.
        """
        self.support = support

    def projS(self, y: Tensor, conj: bool = True) -> Tensor:
        """Project on the support constraint (non-negative real inside the support).

        Parameters
        ----------
        y : torch.Tensor
            Complex r-space tensor of shape ``(N, 1, H, W)``.
        conj : bool, default True
            If True, apply the projection on the convex conjugate of the constraint,
            ``y - P_S(y)``. The real part of ``y`` is then modified in place.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``(N, 1, H, W)`` if ``conj`` is True; otherwise the real
            projection ``P_S(y)`` of the same shape.
        """
        if not conj:
            y = y.real.clamp(min=0) * self.support
        else:
            y.real = y.real - y.real.clamp(min=0) * self.support

        return y

    def projT(self, z: Tensor, denoised: bool = False) -> Tensor:
        """Project on the magnitude constraint in k-space.

        The amplitude of ``z`` is replaced by the measured amplitude where it is known and left
        unchanged where it is unknown.

        Parameters
        ----------
        z : torch.Tensor
            Complex k-space tensor of shape ``(N, 1, H, W)``.
        denoised : bool, default False
            If True, use the denoised amplitude (gRAAR and dRAAR) instead of the measured one.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``(N, 1, H, W)``.
        """
        if denoised:
            return z * self.unknown + self.magnitude_dn * phase(z) * (1 - self.unknown)
        else:
            return z * self.unknown + self.magnitude * phase(z) * (1 - self.unknown)

    def projM(self, u: Tensor) -> Tensor:
        """Project on the magnitude constraint, expressed in r-space.

        Parameters
        ----------
        u : torch.Tensor
            Complex r-space tensor of shape ``(N, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Complex r-space tensor ``ifft2(projT(fft2(u)))`` of shape ``(N, 1, H, W)``.
        """
        return ifft2(self.projT(fft2(u)))

    def reflS(self, u: Tensor) -> Tensor:
        """Reflect on the support constraint, ``2 * P_S(u) - u``.

        Parameters
        ----------
        u : torch.Tensor
            Complex r-space tensor of shape ``(N, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``(N, 1, H, W)``.
        """
        return 2 * self.projS(u, conj=False) - u

    def reflM(self, u: Tensor) -> Tensor:
        """Reflect on the magnitude constraint in r-space, ``2 * P_M(u) - u``.

        Parameters
        ----------
        u : torch.Tensor
            Complex r-space tensor of shape ``(N, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``(N, 1, H, W)``.
        """
        return 2 * self.projM(u) - u

    def proxS(self, y: Tensor, param: float, alpha: float, type: str, conj: bool = True) -> Tensor:
        """Apply the proximal operator of the smoothed support constraint (GPS).

        The operator acts on the convex conjugate of the support constraint, with
        Moreau-Yosida regularization by a Gaussian filter of width ``alpha`` applied in
        k-space (R variant) or in r-space (F variant).

        Parameters
        ----------
        y : torch.Tensor
            Complex r-space Lagrange multiplier of shape ``(N, 1, H, W)``.
        param : float
            Step size of the dual update (``s`` in GPS, ``gamma`` in dpGPS).
        alpha : float
            Frequency filter coefficient from `freqfilter`.
        type : {'R', 'F'}
            Variant. For any other value ``y`` is returned unchanged.
        conj : bool, default True
            Only True (convex conjugate) is supported.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``(N, 1, H, W)``.

        Raises
        ------
        ValueError
            If ``conj`` is False.
        """
        if not conj:
            raise ValueError(
                "Proximal operator on support constraint only supports convex conjugation version."
            )

        if type == "R":
            y = fft2(self.projS(y, True))
            y = y * ifftshift(self.filter(alpha / math.sqrt(param)))
            y = ifft2(y)
        elif type == "F":
            y = self.projS(y, True) * self.filter(2 * math.pi * alpha / math.sqrt(param))

        return y

    def proxT(self, z: Tensor, param: float | Tensor, sigma: float) -> Tensor:
        """Apply the proximal operator of the magnitude constraint (GPS).

        Moreau-Yosida regularization with ``sigma``:
        ``(param * projT(z) + sigma * z) / (param + sigma)``.

        Parameters
        ----------
        z : torch.Tensor
            Complex k-space tensor of shape ``(N, 1, H, W)``.
        param : float or torch.Tensor
            Step size, or a real tensor of shape ``(1, 1, H, W)`` (inverse preconditioner in
            dpGPS).
        sigma : float
            Regularization parameter.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``(N, 1, H, W)``.
        """
        return (param * self.projT(z) + sigma * z) / (param + sigma)

    def forward(self, **kwargs: Any) -> Tensor | tuple[Tensor, Tensor]:
        """Perform one iteration.

        Parameters
        ----------
        **kwargs
            Current iterate and parameters, depending on the algorithm:

            * HIO, RAAR, gRAAR, dRAAR: ``u`` (complex r-space tensor of shape
              ``(N, 1, H, W)``), ``beta`` (float) and ``toggle`` (bool; if True, perform a
              boundary push step instead).
            * GPS-R, GPS-F: ``z``, ``y`` (complex tensors of shape ``(N, 1, H, W)``),
              ``sigma``, ``alpha``, ``t`` and ``s`` (float).
            * dpGPS-R, dpGPS-F: ``z``, ``y``, ``sigma`` and ``alpha``.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            The new ``u`` for HIO, RAAR, gRAAR and dRAAR; the new ``(z, y)`` for GPS and dpGPS.

        Raises
        ------
        ValueError
            If the algorithm is not supported.
        """
        if self.type == "HIO":
            u = kwargs.pop("u")
            beta = kwargs.pop("beta")
            toggle = kwargs.pop("toggle")

            un = self.projM(u)
            # get intersection of support constraint and positivity
            const = self.support * torch.ge(un.real, 0)
            if not toggle:
                # HIO
                un = un * const + (u - beta * un) * (1 - const)
            else:
                # boundary push
                un = un * const + beta * un * (1 - const)

            return un

        elif self.type in ["RAAR", "gRAAR", "dRAAR"]:
            u = kwargs.pop("u")
            beta = kwargs.pop("beta")
            toggle = kwargs.pop("toggle")

            if not toggle:
                if self.type == "RAAR":
                    # RAAR
                    un = 0.5 * beta * (self.reflS(self.reflM(u)) + u) + (1 - beta) * self.projM(u)
                elif self.type in ["gRAAR", "dRAAR"]:
                    # gRAAR or dRAAR
                    z = self.projT(fft2(u), True)
                    un = 0.5 * beta * (self.reflS(self.reflM(u)) + u) + (1 - beta) * ifft2(z)
            else:
                un = self.projM(u)
                # get intersection of support constraint and positivity
                const = self.support * torch.ge(un.real, 0)
                # boundary push
                un = un * const + beta * un * (1 - const)
            return un

        elif self.type in ["GPS-R", "GPS-F"]:
            z = kwargs.pop("z")
            y = kwargs.pop("y")
            sigma = kwargs.pop("sigma")
            alpha = kwargs.pop("alpha")
            t = kwargs.pop("t")
            s = kwargs.pop("s")
            type = "R" if self.type == "GPS-R" else "F"
            # GPS
            zn = z - t * fft2(y)
            zn = self.proxT(zn, t, sigma)
            y = y + s * ifft2(2 * zn - z)
            y = self.proxS(y, s, alpha, type)

            return zn, y

        elif self.type in ["dpGPS-R", "dpGPS-F"]:
            z = kwargs.pop("z")
            y = kwargs.pop("y")
            sigma = kwargs.pop("sigma")
            alpha = kwargs.pop("alpha")
            gamma = (2 * self.kernel.min().pow(2) / self.kernel.max()).item()
            type = "R" if self.type == "dpGPS-R" else "F"
            # dpGPS with sigma condition
            if sigma < 1:
                zn = z - fft2(y) / self.kernel
                zn = self.proxT(zn, 1 / self.kernel, sigma)
            else:
                zn = z - fft2(y)
                zn = self.proxT(zn, 1, sigma)
            y = y + gamma * ifft2(2 * zn - z)
            y = self.proxS(y, gamma, alpha, type)

            return zn, y

        else:
            raise ValueError(f"{self.type} is not supported for phase retrieval.")


class PhaseRetrieval(nn.Module):
    """Phase retrieval iterator.

    Runs ``N`` independent reconstructions (one per initial phase) and keeps, for each, the
    iterate with the lowest error. Whenever a scheduled parameter changes, the iteration
    restarts from the best iterate so far.

    Supported algorithms (``algorithm``):

    ``'HIO'``
        Hybrid input-output [1]_, with an optional final boundary push stage taken from
        guided HIO without the guiding step [2]_.
    ``'RAAR'``
        Relaxed averaged alternating reflections [3]_, with the same boundary push stage.
    ``'gRAAR'``, ``'dRAAR'``
        RAAR whose projection on the magnitude uses data denoised by Gaussian smoothing
        (gRAAR) or by deep learning (dRAAR) [4]_.
    ``'GPS-R'``, ``'GPS-F'``
        Generalized proximal smoothing [5]_: a primal-dual hybrid gradient (PDHG) method with
        Moreau-Yosida regularization of the constraints, smoothed in k-space (R) or r-space (F).
    ``'dpGPS-R'``, ``'dpGPS-F'``
        Deep preconditioned GPS: inexact preconditioned PDHG [6]_ with a preconditioner from a
        denoising network [4]_; the inner iteration is a single proximal gradient step.

    Supported error metrics (``error``):

    ``'R'``
        R-factor ``sum(|a - a0|) / sum(a0)`` over the measured pixels, from the k-space
        amplitude ``a``; better for general purposes.
    ``'NLL'``
        Negative Poisson log-likelihood of the intensity ``a**2`` with the Stirling term,
        averaged over the measured pixels with intensity above 1.

    Parameters
    ----------
    input : torch.Tensor
        Measured k-space amplitude of shape ``(1, 1, H, W)``, not fftshifted. For dRAAR and
        dpGPS it must be scaled to photon counts.
    support : torch.Tensor
        Real-space support of shape ``(1 or N, 1, H, W)``, 1 inside and 0 outside.
    unknown : torch.Tensor
        Mask of shape ``(1, 1, H, W)``, not fftshifted: 1 where the amplitude is unknown
        (missing data), 0 where it is measured.
    algorithm : str
        One of the algorithms listed above.
    error : {'R', 'NLL'}
        Error metric.
    shrinkwrap : bool, default False
        Update the support with `ShrinkWrap` every ``interval`` iterations. Every call of
        `forward` starts again from ``support`` and ``sigma_initial``, so that batches of
        reconstructions are independent.
    **kwargs
        Keyword arguments listed below. Other keywords are ignored, so the same dictionary
        can be passed to the constructor and to `forward`.

    Other Parameters
    ----------------
    beta_type : {'const', 'linear', 'step'}
        HIO, RAAR, gRAAR, dRAAR: how a constant ``beta`` evolves towards ``beta_lim``
        during the iteration (``'step'`` follows the RAAR paper [3]_).
    beta_lim : float
        Final ``beta`` if ``beta_type`` is not ``'const'``.
    limit : float
        dRAAR, dpGPS: the preconditioner is limited to ``[1 - limit, 1 + limit]``.
    deep : bool
        dRAAR, dpGPS: use the deep-learning preconditioner (otherwise a constant kernel on
        the measured photons).
    sigma_initial, sigma_limit, ratio_update, threshold : float
        ShrinkWrap parameters, see `ShrinkWrap`.
    interval : int
        Number of iterations between ShrinkWrap updates.

    References
    ----------
    .. [1] HIO, https://doi.org/10.1364/AO.21.002758
    .. [2] GHIO, https://doi.org/10.1103/PhysRevB.76.064113
    .. [3] RAAR, https://doi.org/10.1088/0266-5611/21/1/004
    .. [4] gRAAR, dRAAR and the dpGPS preconditioner,
       https://doi.org/10.1103/PhysRevResearch.3.043066
    .. [5] GPS, https://doi.org/10.1364/OE.27.002792
    .. [6] Preconditioned PDHG, https://doi.org/10.1007/s10915-020-01371-1
    """

    def __init__(
        self,
        input: Tensor,
        support: Tensor,
        unknown: Tensor,
        algorithm: str,
        error: str,
        shrinkwrap: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.h = input.size(2)
        self.w = input.size(3)
        self.register_buffer("magnitude", input)
        self.register_buffer("unknown", unknown)
        self.register_buffer("support", support)
        self.algorithm = algorithm
        self.error = error
        option = {}
        # get beta control option
        if algorithm in ["HIO", "RAAR", "dRAAR", "gRAAR"]:
            self.beta_type = kwargs.pop("beta_type")
            if self.beta_type != "const":
                self.beta_lim = kwargs.pop("beta_lim")
        # get preconditioner for dRAAR and dpGPS
        if algorithm in ["dRAAR", "dpGPS-R", "dpGPS-F"]:
            denoiser = Preconditioner()
            limit = kwargs.pop("limit")
            deep = kwargs.pop("deep")
            option["preconditioner"] = denoiser.getKernel(input, unknown, limit, deep)
        # initialize phase retrieval iteration unit
        self.block = PhaseRetrievalUnit(input, support, unknown, algorithm, **option)
        # initialize shrinkwrap module
        self.shrinkwrap = shrinkwrap
        if shrinkwrap:
            sigma_initial = kwargs.pop("sigma_initial")
            sigma_limit = kwargs.pop("sigma_limit")
            ratio_update = kwargs.pop("ratio_update")
            threshold = kwargs.pop("threshold")
            self.interval = kwargs.pop("interval")
            self.register_buffer("initial_support", support)
            self.shrink = ShrinkWrap(threshold, sigma_initial, sigma_limit, ratio_update)

    def getParameter(
        self, input: Schedule, iteration: int, name: str = "parameter"
    ) -> tuple[list[int], list[float]]:
        """Convert a parameter schedule to iteration steps and values.

        Parameters
        ----------
        input : float or tuple or list
            Constant value or schedule ``(ratio_0, value_0, ratio_1, value_1, ...)``.
        iteration : int
            Total number of iterations.
        name : str, default 'parameter'
            Parameter name used in the error message.

        Returns
        -------
        step : list of int
            Iterations ``round(ratio_k * iteration)`` at which the value changes.
        values : list of float
            Value from each step on.

        Raises
        ------
        ValueError
            If ``input`` is neither a number nor a tuple or list.
        """
        if isinstance(input, (tuple, list)):
            step = [round(pos * iteration) for pos in input[0::2]]
            plist = input[1::2]
        elif isinstance(input, (int, float)):
            step = [0]
            plist = [input]
        else:
            raise ValueError(f"{input} is invalid value for {name}.")
        return step, plist

    def getAmplitude(self, toggle: bool = False, **kwargs: Tensor) -> Tensor:
        """Project an iterate on the support constraint and return its k-space amplitude.

        Parameters
        ----------
        toggle : bool, default False
            If True, return the projected real-space object instead of the k-space amplitude.
        **kwargs
            Either ``u`` (complex r-space tensor) or ``z`` (complex k-space tensor), of shape
            ``(N, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``(N, 1, H, W)``: ``|fft2(P_S(u))|``, or ``P_S(u)`` if
            ``toggle`` is True.
        """
        if "u" in kwargs:
            u = kwargs.pop("u")
            u = self.block.projS(u, conj=False)
            if toggle:
                return u.real
            else:
                return torch.abs(fft2(u))

        elif "z" in kwargs:
            z = kwargs.pop("z")
            return self.getAmplitude(u=ifft2(z), toggle=toggle)

    def getError(self, a: Tensor) -> Tensor:
        """Compute the error of a retrieved k-space amplitude (see the class docstring).

        Parameters
        ----------
        a : torch.Tensor
            Real k-space amplitude of shape ``(N, 1, H, W)``, not fftshifted.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``(N,)``.

        Raises
        ------
        ValueError
            If the error metric is not supported.
        """
        a0 = self.magnitude
        a = a * (1 - self.unknown)
        if self.error == "R":
            # R-factor
            R = torch.abs(a - a0).sum(dim=(1, 2, 3)) / a0.sum()
            return R
        elif self.error == "NLL":
            # negative Poisson log-likelihood
            i0 = a0.pow(2)
            i = a.pow(2)
            valid = (1 - self.unknown) * (i0 > 1)
            NLL = F.poisson_nll_loss(i, i0, log_input=False, full=True, reduction="none")
            NLL = NLL * valid
            return NLL.sum(dim=(1, 2, 3)) / valid.sum()
        else:
            raise ValueError(f"{self.error} is not supported for error metric.")

    def forward(
        self, iteration: int, initial_phase: Tensor, toggle: bool = False, **kwargs: Any
    ) -> tuple[Tensor, Tensor]:
        """Run the phase retrieval algorithm.

        Parameters
        ----------
        iteration : int
            Number of iterations.
        initial_phase : torch.Tensor
            Complex tensor ``exp(i * theta)`` of shape ``(N, 1, H, W)``; ``theta`` is usually
            drawn uniformly from ``[0, 2 * pi)``. ``N`` sets the number of reconstructions.
        toggle : bool, default False
            If True, return the k-space result without projection on the support constraint.
        **kwargs
            Keyword arguments listed below; other keywords are ignored.

        Other Parameters
        ----------------
        beta : float or tuple or list
            HIO, RAAR, gRAAR, dRAAR: relaxation parameter, constant or `Schedule`.
        boundary_push : float
            HIO, RAAR, gRAAR, dRAAR: fraction of the iterations, at the end, spent in the
            boundary push stage (``beta`` decreasing linearly from 1 to 0).
        sigma : float or tuple or list
            GPS, dpGPS: relaxation of the magnitude constraint, constant or `Schedule`.
        alpha_count : int
            GPS, dpGPS: number of stages of the frequency filter (see `freqfilter`).
        t, s : float or tuple or list
            GPS: step sizes of the proximal operators on the magnitude and support
            constraints, constant or `Schedule`.

        Returns
        -------
        output : torch.Tensor
            Best iterate of each reconstruction, shape ``(N, 1, H, W)``: the real r-space
            object projected on the support constraint, or, if ``toggle`` is True, the complex
            k-space iterate (not fftshifted).
        path : torch.Tensor
            Real tensor of shape ``(N, iteration)``: error after each iteration.

        Raises
        ------
        ValueError
            If the algorithm, error metric, ``beta_type`` or a schedule is not supported, or
            ``iteration`` is less than 1.
        """
        size_batch = initial_phase.size(0)
        device = initial_phase.device
        if self.shrinkwrap:
            # start from the initial support (one per reconstruction) and ShrinkWrap state
            support = self.initial_support
            if support.size(0) == 1:
                support = torch.repeat_interleave(support, repeats=size_batch, dim=0)
            self.support = support
            self.block.updateSupport(support)
            self.shrink.reset()
        # phase retrieval iteration
        var = {}
        u_best = z_best = y_best = None
        error_min = torch.zeros(size_batch, device=device)
        path = torch.zeros(size_batch, iteration, device=device)
        for n in range(iteration):
            # phase retrieval
            if self.algorithm in ["HIO", "RAAR", "gRAAR", "dRAAR"]:
                # initialize
                if n == 0:
                    u_best = ifft2(self.magnitude * initial_phase)
                    beta_step, beta_list = self.getParameter(
                        kwargs.pop("beta"), iteration, name="beta"
                    )
                    bp_step = round((1 - kwargs.pop("boundary_push")) * iteration)
                # update parameter
                refresh = False
                if n < bp_step:
                    var["toggle"] = False
                    if n in beta_step:
                        var["beta"] = beta_list[beta_step.index(n)]
                        refresh = True
                    # beta control during iteration
                    elif not len(beta_step) > 1 and self.beta_type != "const":
                        if self.beta_type == "step":
                            var["beta"] = beta_list[0] + (self.beta_lim - beta_list[0]) * (
                                1 - math.exp(-((n / 7 * 100 / iteration) ** 3))
                            )
                        elif self.beta_type == "linear":
                            var["beta"] = (
                                beta_list[0] + (self.beta_lim - beta_list[0]) / iteration * n
                            )
                        else:
                            raise ValueError(f"{self.beta_type} is not supported for beta control.")
                else:
                    var["toggle"] = True
                    var["beta"] = 1 - (n - bp_step) / (iteration - bp_step)
                    if n == bp_step:
                        refresh = True
                # refresh when parameter updated
                if refresh:
                    var["u"] = u_best.clone().detach()
                # perform single phase retrieval step
                var["u"] = self.block(**var)
                # calculate error
                error = self.getError(self.getAmplitude(u=var["u"]))
                path[:, n] = error
                # update best
                trigger = torch.le(error, error_min if n > 0 else error)
                error_min[trigger] = error[trigger]
                u_best[trigger, :, :, :] = var["u"][trigger, :, :, :]

            elif self.algorithm in ["GPS-R", "GPS-F", "dpGPS-R", "dpGPS-F"]:
                # initialize
                if n == 0:
                    z_best = self.magnitude * initial_phase
                    y_best = torch.zeros_like(initial_phase)
                    sigma_step, sigma_list = self.getParameter(
                        kwargs.pop("sigma"), iteration, name="sigma"
                    )
                    alpha_step, alpha_list = self.getParameter(
                        freqfilter(min(self.h, self.w), kwargs.pop("alpha_count")),
                        iteration,
                        name="alpha",
                    )
                    dp = False
                    if self.algorithm in ["dpGPS-R", "dpGPS-F"]:
                        dp = True
                    else:
                        t_step, t_list = self.getParameter(kwargs.pop("t"), iteration, name="t")
                        s_step, s_list = self.getParameter(kwargs.pop("s"), iteration, name="s")
                # update parameter
                refresh = False
                if n in sigma_step:
                    var["sigma"] = sigma_list[sigma_step.index(n)]
                    refresh = True
                if n in alpha_step:
                    var["alpha"] = alpha_list[alpha_step.index(n)]
                    refresh = True
                if not dp:
                    if n in t_step:
                        var["t"] = t_list[t_step.index(n)]
                        refresh = True
                    if n in s_step:
                        var["s"] = s_list[s_step.index(n)]
                        refresh = True
                # refresh when parameter updated
                if refresh:
                    var["z"] = z_best.clone().detach()
                    var["y"] = y_best.clone().detach()
                # perform single phase retrieval step
                var["z"], var["y"] = self.block(**var)
                # calculate error
                error = self.getError(self.getAmplitude(z=var["z"]))
                path[:, n] = error
                # update best
                trigger = torch.le(error, error_min if n > 0 else error)
                error_min[trigger] = error[trigger]
                z_best[trigger, :, :, :] = var["z"][trigger, :, :, :]
                y_best[trigger, :, :, :] = var["y"][trigger, :, :, :]

            else:
                raise ValueError(f"{self.algorithm} is not supported for phase retrieval.")

            # shrinkwrap
            if self.shrinkwrap:
                if (n + 1) % self.interval == 0 and (n + 1) < iteration:
                    # get object
                    if z_best is not None:
                        obj = self.getAmplitude(z=z_best, toggle=True)
                    else:
                        obj = self.getAmplitude(u=u_best, toggle=True)
                    # update support
                    self.support = self.shrink(obj)
                    self.block.updateSupport(self.support)
                    self.shrink.update()

        # get output
        if z_best is not None:
            if toggle:
                output = z_best
            else:
                output = self.getAmplitude(z=z_best, toggle=True)
        elif u_best is not None:
            if toggle:
                output = fft2(u_best)
            else:
                output = self.getAmplitude(u=u_best, toggle=True)
        else:
            raise ValueError("iteration must be at least 1.")

        return output, path
