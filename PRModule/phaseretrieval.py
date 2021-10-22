###############################################################################
# Phase Retrieval Algorithms : HIO, RAAR, GPS, gRAAR, dRAAR, dpGPS
#
# Author: SUNG YUN LEE
#   
# Contact: sungyun98@postech.ac.kr
###############################################################################

__all__ = ['PhaseRetrieval']

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .func import *
from .preconditioner import *
from .partialconv2d import *

class GaussianFilter(nn.Module):
    """
    get Gaussian kernel in k-space with frequency filter coefficient alpha
    alpha is originated from oversampling smoothness method (OSS)
    reference = https://doi.org/10.1107/S0021889813002471
    """
    def __init__(self, height, width):
        '''
        generate square radius tensor

        args:
            height = int
            width = int
        '''
        super().__init__()
        mesh = sqmesh(height, width)
        mesh = ifftshift(mesh)
        self.register_buffer('mesh', mesh)
    
    def forward(self, alpha):
        '''
        generate Gaussian filter with coefficient alpha

        args:
            alpha = float

        returns:
            output = torch float tensor of size 1 * 1 * H * W * 1
        '''
        
        filter = torch.exp(-0.5 * self.mesh / alpha ** 2)
        return filter / filter.max()

class ShrinkWrap(GaussianFilter):
    """
    ShrinkWrap support constraint update method
    Gaussian filter is based on MATLAB function
    reference = https://doi.org/10.1103/PhysRevB.68.140101
    """
    def __init__(self, threshold, sigma_initial = 3, sigma_limit = 1.5, ratio_update = 0.01):
        '''
        generate initial Gaussian filter
        note that Gaussian filter size is 2*ceil(2*sigma_initial)+1
        sigma update ratio is applied by multiplying 1-ratio to sigma, descending to sigma limit.]

        args:
            threshold = float
            sigma_initial = float (default = 3)
            sigma_limit = float (default = 1.5)
            ratio_update = float (default = 0.01)
        '''

        self.sigma = sigma_initial
        self.sigma_limit = sigma_limit
        self.ratio = ratio_update
        self.threshold = threshold
        
        # calculate initial filter
        size = 2 * math.ceil(2 * self.sigma) + 1
        self.pad = math.ceil(2 * self.sigma)
        super().__init__(size, size)
        self.mesh = self.mesh.squeeze(-1)
        self.register_buffer('filter', self.mesh * 0)
        self.update(False)

    def update(self, update_sigma = True):
        '''
        update sigma and filter

        args:
            update_sigma = bool
        '''

        # update filter
        if self.sigma > self.sigma_limit:
            if update_sigma:
                self.sigma = self.sigma * (1 - self.ratio)
            self.filter = torch.exp(-0.5 * self.mesh / self.sigma ** 2)
            self.filter = self.filter / self.filter.sum()
        
    def forward(self, u):
        '''
        calculate new support constraint by Gaussian filtered r-space data
        threshold is used to generate new suppport constraint

        args:
            u = torch float tensor of size N * 1 * H * W * 1

        returns:
            output = torch float tensor of size N * 1 * H * W * 1
        '''

        n = u.size(0)
        u = F.conv2d(u.squeeze(-1), weight = self.filter, padding = self.pad, padding_mode = 'reflect')
        u_max = u.view(n, -1).max(dim = -1).values.view(n, 1, 1, 1)
        return torch.gt(u, u_max * self.threshold).unsqueeze(-1).float()

class PhaseRetrievalUnit(nn.Module):
    '''
    phase retrieval iteration unit
    detailed informations in PhaseRetrieval class
    note that all operators use convex conjugated support constraint
    '''
    def __init__(self, input, support, unknown, type, **kwargs):
        '''
        prepare iteration unit

        preconditioner is needed for dpGPS

        args:
            input = torch float tensor of size 1 * 1 * H * W * 1
            support = torch float tensor of size (1 or N) * 1 * H * W * 1
            unknown = torch float tensor of size 1 * 1 * H * W * 1
            type = string

        kwargs:
            preconditioner = torch float tensor of size 1 * 1 * H * W * 1
        '''

        super().__init__()
        self.register_buffer('magnitude', input)
        self.register_buffer('unknown', unknown)
        self.register_buffer('support', support)
        self.type = type

        # allocate denoised magnitude for projection operator on denoised pattern
        if type in ['gRAAR']:
            input_dn = GaussianSmoothing(fftshift(input).pow(2).squeeze(-1), sigma = 1.5, mask = 1 - fftshift(unknown).squeeze(-1)).sqrt()
            input_dn = ifftshift(input_dn)
            self.register_buffer('magnitude_dn', input_dn)
        if type in ['dRAAR']:
            kernel = kwargs.pop('preconditioner')
            self.register_buffer('magnitude_dn', input * kernel)

        # allocate Gaussian filter for GPS algorithms
        if type in ['GPS-R', 'GPS-F', 'dpGPS-R', 'dpGPS-F']:
            self.filter = GaussianFilter(self.magnitude.size(2), self.magnitude.size(3))
        # allocate preconditioner for deep preconditioned algorithms
        if type in ['dpGPS-R', 'dpGPS-F']:
            kernel = kwargs.pop('preconditioner')
            self.register_buffer('kernel', kernel)

    def updateSupport(self, support):
        '''
        update support constraint

        args:
            support = torch float tensor of size (1 or N) * 1 * H * W * 1
        '''
        self.support = support

    def projS(self, y, conj = True):
        '''
        projection operator on support constraint

        constraint is non-negative real
        convex conjugation of support constraint supported

        args:
            y = torch float tensor of size N * 1 * H * W * 2

        returns:
            output = torch float tensor of size N * 1 * H * W * 2
        '''

        if not conj:
            y = y.clamp(min = 0) * self.support
            y[:, :, :, :, 1:] = 0
            return y
            
        y_real = y[:, :, :, :, :1]
        y[:, :, :, :, :1] = y_real - y_real.clamp(min = 0) * self.support

        return y

    def projT(self, z, denoised = False):
        '''
        projection operator on magnitude constraint

        constraint is on amplitude of complex tensor

        args:
            z = torch float tensor of size N * 1 * H * W * 2

        returns:
            outputs = torch float tensor of size N * 1 * H * W * 2
        '''
        
        if denoised:
            return z * self.unknown + self.magnitude_dn * phase(z) * (1 - self.unknown)
        else:
            return z * self.unknown + self.magnitude * phase(z) * (1 - self.unknown)

    def projM(self, u):
        '''
        projection operator on magnitude constraint in r-space

        constraint is on amplitude of complex tensor

        args:
            u = torch float tensor of size N * 1 * H * W * 2

        returns:
            outputs = torch float tensor of size N * 1 * H * W * 2
        '''
        
        return torch.ifft(self.projT(torch.fft(u, signal_ndim = 2)), signal_ndim = 2)

    def reflS(self, u):
        '''
        reflection operator on support constraint

        constraint is non-negative real

        args:
            u = torch float tensor of size N * 1 * H * W * 2

        returns:
            outputs = torch float tensor of size N * 1 * H * W * 2
        '''
        
        return 2 * self.projS(u, conj = False) - u
    
    def reflM(self, u):
        '''
        reflection operator on magnitude constraint in r-space

        constraint is on amplitude of complex tensor

        args:
            u = torch float tensor of size N * 1 * H * W * 2

        returns:
            outputs = torch float tensor of size N * 1 * H * W * 2
        '''
        
        return 2 * self.projM(u) - u

    def proxS(self, y, param, alpha, type, conj = True):
        '''
        proximal operator on support constraint

        constraint is non-negative real
        convex conjugation of support constraint is applied
        Moreau-Yosida regularization with alpha is applied (R and F variants)

        args:
            y = torch float tensor of size N * 1 * H * W * 2
            param = float
            alpha = float
            type = string
        
        returns:
            output = torch float tensor of size N * 1 * H * W * 2
        '''

        if not conj:
            raise Exception('Proximal operator on support constraint only supports convex conjugation version.')
        
        if type == 'R':
            y = torch.fft(self.projS(y, True), signal_ndim = 2)
            y = y * self.filter(alpha / math.sqrt(param))
            y = torch.ifft(y, signal_ndim = 2)
        elif type == 'F':
            y = self.projS(y, True) * ifftshift(self.filter(2 * math.pi * alpha / math.sqrt(param)))
        
        return y


    def proxT(self, z, param, sigma):
        '''
        proximal operator on magnitude constraint

        Moreau-Yosida regularization with sigma is applied
        tensor param can be used

        args:
            z = torch float tensor of size N * 1 * H * W * 2
            param = float or torch float tensor of size 1 * 1 * H * W * (1 or 2)
            sigma = float

        returns:
            output = torch float tensor of size N * 1 * H * W * 2
        '''

        return (param * self.projT(z) + sigma * z) / (param + sigma)

    def forward(self, **kwargs):
        '''
        iteration of phase retrieval algorithms

        for [HIO, RAAR, gRAAR, dRAAR], if toggle is True, boundary push is performed
        
        kwargs:
            u = torch float tensor of size N * 1 * H * W * 2 (for HIO, RAAR, gRAAR, dRAAR)
            beta = float (for HIO, RAAR, gRAAR, dRAAR)
            toggle = bool (for HIO, RAAR, gRAAR, dRAAR)
            z = torch float tensor of size N * 1 * H * W * 2 (for GPS, dpGPS)
            y = torch float tensor of size N * 1 * H * W * 2 (for GPS, dpGPS)
            sigma = float (for GPS, dpGPS)
            alpha = float (for GPS, dpGPS)
            t = float (for GPS)
            s = float (for GPS)
            inner_iteration = int (for dpGPS)
        
        returns:
            u = torch float tensor of size N * 1 * H * W * 2 (for HIO, RAAR, gRAAR, dRAAR)
            z = torch float tensor of size N * 1 * H * W * 2 (for GPS, dpGPS)
            y = torch float tensor of size N * 1 * H * W * 2 (for GPS, dpGPS)
        '''
        if self.type == 'HIO':
            u = kwargs.pop('u')
            beta = kwargs.pop('beta')
            toggle = kwargs.pop('toggle')

            un = self.projM(u)
            # get intersection of support constraint and positivity
            const = self.support * torch.ge(un, 0)[:, :, :, :, :1]
            if not toggle:
                # HIO
                un = un * const + (u - beta * un) * (1 - const)
            else:
                # boundary push
                un = un * const + beta * un * (1 - const)
            
            return un

        elif self.type in ['RAAR', 'gRAAR', 'dRAAR']:
            u = kwargs.pop('u')
            beta = kwargs.pop('beta')
            toggle = kwargs.pop('toggle')
            
            if not toggle:
                if self.type == 'RAAR':
                    # RAAR
                    un = 0.5 * beta * (self.reflS(self.reflM(u)) + u) + (1 - beta) * self.projM(u)
                elif self.type in ['gRAAR', 'dRAAR']:
                    # gRAAR or dRAAR
                    z = self.projT(torch.fft(u, signal_ndim = 2), True)
                    un = 0.5 * beta * (self.reflS(self.reflM(u)) + u) + (1 - beta) * torch.ifft(z, signal_ndim = 2)
            else:
                un = self.projM(u)
                # get intersection of support constraint and positivity
                const = self.support * torch.ge(un, 0)[:, :, :, :, :1]
                # boundary push
                un = un * const + beta * un * (1 - const)
            return un

        elif self.type in ['GPS-R', 'GPS-F']:
            z = kwargs.pop('z')
            y = kwargs.pop('y')
            sigma = kwargs.pop('sigma')
            alpha = kwargs.pop('alpha')
            t = kwargs.pop('t')
            s = kwargs.pop('s')
            type = 'R' if self.type == 'GPS-R' else 'F'
            # GPS
            zn = z - t * torch.fft(y, signal_ndim = 2)
            zn = self.proxT(zn, t, sigma)
            y = y + s * torch.ifft(2 * zn - z, signal_ndim = 2)
            y = self.proxS(y, s, alpha, type)
                
            return zn, y

        elif self.type in ['dpGPS-R','dpGPS-F']:
            z = kwargs.pop('z')
            y = kwargs.pop('y')
            sigma = kwargs.pop('sigma')
            alpha = kwargs.pop('alpha')
            gamma = (2 * self.kernel.min().pow(2) / self.kernel.max()).item()
            type = 'R' if self.type == 'dpGPS-R' else 'F'
            # dpGPS with sigma condition
            if sigma < 1:
                zn = z - torch.fft(y, signal_ndim = 2) / self.kernel
                zn = self.proxT(zn, 1 / self.kernel, sigma)
            else:
                zn = z - torch.fft(y, signal_ndim = 2)
                zn = self.proxT(zn, 1, sigma)
            y = y + gamma * torch.ifft(2 * zn - z, signal_ndim = 2)
            y = self.proxS(y, gamma, alpha, type)
                
            return zn, y
            
        else:
            raise ValueError('{} is not supported for phase retrieval.'.format(self.type))

class PhaseRetrieval(nn.Module):
    '''
    phase retrieval iterator

    u is r-space complex tensor, z is k-space complex tensor, and y is Lagrange multiplier
    unfixed parameters can be managed in iteration by giving tuple (0, c0, t1, c1, ...)
    ti's in range (0, 1) indicates ratio of iteration when parameter updates to ci

    support algorithm:
        1. hybrid input-output [HIO]
        HIO with additional boundary push stage originated from guided HIO without guiding
        reference1 (HIO) = https://doi.org/10.1364/AO.21.002758
        reference2 (GHIO) = https://doi.org/10.1103/PhysRevB.76.064113

        2. relaxed averaged alternating reflections [RAAR]
        RAAR with additional boundary push stage refered above
        reference = https://doi.org/10.1088/0266-5611/21/1/004

        2-1. Gaussian RAAR / deep RAAR [gRAAR, dRAAR]
        RAAR with projection operator on denoised data using Gaussian smoothing or deep learning
        reference = https://doi.org/10.1103/PhysRevResearch.3.043066

        3. generalized proximal smoothing [GPS-R, GPS-F]
        primal-dual hybrid gradient (PDHG) method with applying Moreau-Yosida regularization on constraints
        reference = https://doi.org/10.1364/OE.27.002792

        3-1. deep preconditioned GPS [dpGPS-R, dpGPS-F]
        inexact preconditioned GPS with preconditioner based on deep learning
        single-step proximal gradient descent method is applied for inner inexact iteration
        reference (preconditioned PDHG) = https://doi.org/10.1007/s10915-020-01371-1
        reference (preconditioner) = https://doi.org/10.1103/PhysRevResearch.3.043066

    support error metric:
        R-factor [R] and negative Poisson log-likelihood [NLL]
        R performs better for general purpose, and calculate with amplitude of z
        NLL is calculated with square of z as intensity using Stirling approximation
    '''
    def __init__(self, input, support, unknown, algorithm, error, shrinkwrap = False, **kwargs):
        '''
        initialize phase retrieval iterator

        args:
            input = torch float tensor of size N * 1 * H * W * 1
            support = torch float tensor of size N * 1 * H * W * 1
            unknown = torch float tensor of size N * 1 * H * W * 1
            algorithm = string
            error = string
            shrinkwrap = bool (default = False)

        kwargs:
            limit = float (for dpGPS)
            deep = bool (for dpGPS)
            sigma_initial = float (for shrinkwrap)
            sigma_limit = float (for shrinkwrap)
            ratio_update = float (for shrinkwrap)
            threshold = float (for shrinkwrap)
            interval = int (for shrinkwrap)
        '''

        super().__init__()
        self.h = input.size(2)
        self.w = input.size(3)
        self.register_buffer('magnitude', input)
        self.register_buffer('unknown', unknown)
        self.register_buffer('support', support)
        self.algorithm = algorithm
        self.error = error
        option = {}
        # get beta control option
        if algorithm in ['HIO', 'RAAR', 'dRAAR', 'gRAAR']:
            self.beta_type = kwargs.pop('beta_type')
            if self.beta_type != 'const':
                self.beta_lim = kwargs.pop('beta_lim')
        # get preconditioner for dpGPS
        if algorithm in ['dRAAR', 'dpGPS-R', 'dpGPS-F']:
            denoiser = Preconditioner()
            limit = kwargs.pop('limit')
            deep = kwargs.pop('deep')
            option['preconditioner'] = denoiser.getKernel(input, unknown, limit, deep)
        # initialize phase retrieval iteration unit
        self.block = PhaseRetrievalUnit(input, support, unknown, algorithm, **option)
        # initialize shrinkwrap module
        self.shrinkwrap = shrinkwrap
        if shrinkwrap:
            sigma_initial = kwargs.pop('sigma_initial')
            sigma_limit = kwargs.pop('sigma_limit')
            ratio_update = kwargs.pop('ratio_update')
            threshold = kwargs.pop('threshold')
            self.interval = kwargs.pop('interval')
            self.register_buffer('initial_support', support)
            self.shrink = ShrinkWrap(threshold, sigma_initial, sigma_limit, ratio_update)

    def getParameter(self, input, iteration, name = 'parameter'):
        '''
        extract update step and value of parameter

        input should be form of float, tuple or list
        otherwise, raise exception

        args:
            input = any
            iteration = int
            name = string (default = 'parameter')
        
        returns:
            step = list
            list = list
        '''
        
        if isinstance(input, (tuple, list)):
            step = [round(pos * iteration) for pos in input[0::2]]
            plist = input[1::2]
        elif isinstance(input, (int, float)):
            step = [0]
            plist = [input]
        else:
            raise ValueError('{} is invalid value for {}.'.format(input, name))
        return step, plist

    def getAmplitude(self, toggle = False, **kwargs):
        '''
        get k-space amplitude of u or z with projection on support constraint

        if toggle is True, projected r-space data is returned

        kwargs:
            u = torch float tensor of size N * 1 * H * W * 2
            z = torch float tensor of size N * 1 * H * W * 2

        returns:
            output = torch float tensor of size N * 1 * H * W * 1
        '''

        if 'u' in kwargs:
            u = kwargs.pop('u')
            u = self.block.projS(u, conj = False)
            if toggle:
                return u[:, :, :, :, :1]
            u = torch.fft(u, signal_ndim = 2)
            return amplitude(u)
        elif 'z' in kwargs:
            z = kwargs.pop('z')
            z = torch.ifft(z, signal_ndim = 2)
            return self.getAmplitude(u = z, toggle = toggle)

    def getError(self, a):
        '''
        get error of phase retrieved amplitude

        args:
            a = torch float tensor of size N * 1 * H * W * 1
        
        returns:
            output = torch float tensor of size N
        '''

        a0 = self.magnitude
        a = a * (1 - self.unknown)
        if self.error == 'R':
            # R-factor
            R = torch.abs(a - a0).sum(dim = (1, 2, 3, 4)) / a0.sum()
            return R
        elif self.error == 'NLL':
            # negative Poisson log-likelihood
            i0 = a0.pow(2)
            i = a.pow(2)
            valid = (1 - self.unknown) * (i0 > 1)
            NLL = F.poisson_nll_loss(i, i0, log_input = False, full = True, reduction = 'none')
            NLL = NLL * valid
            return NLL.sum(dim = (1, 2, 3, 4)) / valid.sum()
        else:
            raise ValueError('{} is not supported for error metric.'.format(self.error))
    
    def forward(self, iteration, initial_phase, toggle = False, **kwargs):
        '''
        perform phase retrieval alrorithm with given iteration count

        initial phase should be given in complex tensor exp(i*theta)
        theta is recommended to generate from random number in range of [0, 2*pi]
        output is r-space results projected on support constraint
        if toggle is True, output is k-space results without projection

        args:
            iteration = int
            initial_phase = torch float tensor of size N * 1 * H * W * 2
            toggle = bool

        kwargs:
            beta = float or tuple or list (for HIO, RAAR, gRAAR, dRAAR)
            boundary_push = float (for HIO, RAAR, gRAAR, dRAAR)
            sigma = float or tuple or list (for GPS, dpGPS)
            alpha_count = int (for GPS, dpGPS)
            t = float or tuple or list (for GPS)
            s = float or tuple or list (for GPS)

        returns:
            output = torch float tensor of size N * 1 * H * W * (1 or 2)
            path = torch float tensor of size N * iteration
        '''

        size_batch = initial_phase.size(0)
        device = initial_phase.device
        if self.shrinkwrap and self.initial_support.size(0) == 1:
            # allocate support for each data
            self.support = torch.repeat_interleave(self.initial_support, repeats = size_batch, dim = 0)
            self.block.updateSupport(self.support)
        # phase retrieval iteration
        var = {}
        u_best = z_best = y_best = None
        error_min = torch.zeros(size_batch, device = device)
        path = torch.zeros(size_batch, iteration, device = device)
        for n in range(iteration):
            # phase retrieval
            if self.algorithm in ['HIO', 'RAAR', 'gRAAR', 'dRAAR']:
                # initialize
                if n == 0:
                    u_best = torch.ifft(self.magnitude * initial_phase, signal_ndim = 2)
                    beta_step, beta_list = self.getParameter(kwargs.pop('beta'), iteration, name = 'beta')
                    bp_step = round((1 - kwargs.pop('boundary_push')) * iteration)
                # update parameter
                refresh = False
                if n < bp_step:
                    var['toggle'] = False
                    if n in beta_step:
                        var['beta'] = beta_list[beta_step.index(n)]
                        refresh = True
                    # beta control during iteration
                    elif not len(beta_step) > 1 and self.beta_type != 'const':
                        if self.beta_type == 'step':
                            var['beta'] = beta_list[0] + (self.beta_lim - beta_list[0]) * (1 - math.exp(-(n / 7 * 100 / iteration) ** 3))
                        elif self.beta_type == 'linear':
                            var['beta'] = beta_list[0] + (self.beta_lim - beta_list[0]) / iteration * n
                        else:
                            raise ValueError('{} is not supported for beta control.'.format(self.beta_type))
                else:
                    var['toggle'] = True
                    var['beta'] = 1 - (n - bp_step) / (iteration - bp_step)
                    if n == bp_step:
                        refresh = True
                # refresh when parameter updated
                if refresh:
                    var['u'] = u_best.clone().detach()
                    refresh = False
                # perform single phase retrieval step
                var['u'] = self.block(**var)
                # calculate error
                error = self.getError(self.getAmplitude(u = var['u']))
                path[:, n] = error
                # update best
                trigger = torch.le(error, error_min if n > 0 else error)
                error_min[trigger] = error[trigger]
                u_best[trigger, :, :, :, :] = var['u'][trigger, :, :, :, :]

            elif self.algorithm in ['GPS-R', 'GPS-F', 'dpGPS-R', 'dpGPS-F']:
                # initialize
                if n == 0:
                    z_best = self.magnitude * initial_phase
                    y_best = torch.zeros_like(initial_phase)
                    sigma_step, sigma_list = self.getParameter(kwargs.pop('sigma'), iteration, name = 'sigma')
                    alpha_step, alpha_list = self.getParameter(freqfilter(min(self.h, self.w), kwargs.pop('alpha_count')), iteration, name = 'alpha')
                    dp = False
                    if self.algorithm in ['dpGPS-R', 'dpGPS-F']:
                        dp = True
                    else:
                        t_step, t_list = self.getParameter(kwargs.pop('t'), iteration, name = 't')
                        s_step, s_list = self.getParameter(kwargs.pop('s'), iteration, name = 's')
                # update parameter
                refresh = False
                if n in sigma_step:
                    var['sigma'] = sigma_list[sigma_step.index(n)]
                    refresh = True
                if n in alpha_step:
                    var['alpha'] = alpha_list[alpha_step.index(n)]
                    refresh = True
                if not dp:
                    if n in t_step:
                        var['t'] = t_list[t_step.index(n)]
                        refresh = True
                    if n in s_step:
                        var['s'] = s_list[s_step.index(n)]
                        refresh = True
                # refresh when parameter updated
                if refresh:
                    var['z'] = z_best.clone().detach()
                    var['y'] = y_best.clone().detach()
                    refresh = False
                # perform single phase retrieval step
                var['z'], var['y'] = self.block(**var)
                # calculate error
                error = self.getError(self.getAmplitude(z = var['z']))
                path[:, n] = error
                # update best
                trigger = torch.le(error, error_min if n > 0 else error)
                error_min[trigger] = error[trigger]
                z_best[trigger, :, :, :, :] = var['z'][trigger, :, :, :, :]
                y_best[trigger, :, :, :, :] = var['y'][trigger, :, :, :, :]

            else:
                raise ValueError('{} is not supported for phase retrieval.'.format(self.algorithm))
            
            # shrinkwrap
            if self.shrinkwrap:
                if (n + 1) % self.interval == 0 and (n + 1) < iteration:
                    # get object
                    if z_best is not None:
                        obj = self.getAmplitude(z = z_best, toggle = True)
                    elif u_best is not None:
                        obj = self.getAmplitude(u = u_best, toggle = True)
                    else:
                        raise Exception
                    # update support
                    self.support = self.shrink(obj)
                    self.block.updateSupport(self.support)
                    self.shrink.update()

        # get output
        if z_best is not None:
            if toggle:
                output = z_best
            else:
                output = self.getAmplitude(z = z_best, toggle = True)
        elif u_best is not None:
            if toggle:
                output = torch.fft(u_best, signal_ndim = 2)
            else:
                output = self.getAmplitude(u = u_best, toggle = True)
        else:
            raise Exception

        return output, path