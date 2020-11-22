###############################################################################
# Basic Functions
#
# Author: SUNG YUN LEE
#   
# Contact: sungyun98@postech.ac.kr
###############################################################################

__all__ = ['MakeSupport', 'fftshift', 'ifftshift', 'amplitude', 'phase', 'sqmesh', 'freqfilter']

import numpy as np
import torch

def MakeSupport(input, **kwargs):
    '''
    generate rectangular or autocorrelation-based support

    input should be fftshifted intensity data for autocorrelation support
    autocorrelation support is from ShrinkWrap reference

    args:
        input = numpy float ndarray of size H * W

    kwargs:
        type = string
        radius = tuple or list of size 2 (for const)
        threshold = float (for auto)

    returns:
        output = numpy float ndarray of size H * W
    '''

    h = input.shape[0]
    w = input.shape[1]
    type = kwargs.pop('type')
    if type == 'rect':
        # generate rectangular support
        ri, rj = kwargs.pop('radius')
        support = np.zeros_like(input)
        support[h // 2 - ri:h // 2 + ri, w // 2 - rj:w // 2 + rj] = 1
    elif type == 'auto':
        # generate autocorrelation support
        threshold = kwargs.pop('threshold')
        support = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input))))
        support = support > np.amax(support) * threshold
    else:
        raise ValueError

    return support

def fftshift(input):
    '''
    fftshift Fourier transformed r-space data

    args:
        input = torch float tensor of size N * 1 * H * W * (1 for real or 2 for complex)

    returns:
        output = torch float tensor of size N * 1 * H * W * (1 for real or 2 for complex)
    '''

    di = input.size(2) // 2
    dj = input.size(3) // 2
    return input.roll(shifts = (di, dj), dims = (2, 3))

def ifftshift(input):
    '''
    inverse of fftshift

    args:
        input = torch float tensor of size N * 1 * H * W * (1 for real or 2 for complex)

    returns:
        output = torch float tensor of size N * 1 * H * W * (1 for real or 2 for complex)
    '''

    di = input.size(2) // 2
    dj = input.size(3) // 2
    return input.roll(shifts = (-di, -dj), dims = (2, 3))

def amplitude(input):
    '''
    get amplitude of complex tensor

    args:
        input = torch float tensor of size N * 1 * H * W * 2

    returns:
        output = torch float tensor of size N * 1 * H * W * 1
    '''

    return input.pow(2).sum(dim = -1, keepdim = True).sqrt()

def phase(input):
    '''
    get phase of complex tensor

    args:
        input = torch float tensor of size N * 1 * H * W * 2

    returns:
        output = torch float tensor of size N * 1 * H * W * 1
    '''

    r = amplitude(input)
    r[r == 0] = 1
    return input / r

def sqmesh(height, width):
    '''
    make squared radius tensor with origin at center
    integer value given for coordinate

    args:
        hight = integer
        width = integer
    '''
    ci = height // 2
    cj = width // 2
    li = torch.linspace(-ci, height - ci - 1, steps = height)
    lj = torch.linspace(-cj, width - cj - 1, steps = width)
    mi, mj = torch.meshgrid(li, lj)
    m = mi.pow(2) + mj.pow(2)

    return m.view(1, 1, height, width, 1)

def freqfilter(size, count):
    '''
    spatial frequency filter sequence originated from oversampling smoothness method(OSS)
    output is in form of [ratio, value, ]
    reference = https://doi.org/10.1107/S0021889813002471

    size should be max(heigh,width) of data

    args:
        size = integer
        count = integer

    returns:
        output = tuple of size 2*count
    '''

    param = []
    list = torch.linspace(size * 2, size * 2 / count, steps = count)
    for n, alpha in enumerate(list):
        param += [n / count, alpha.item()]

    return tuple(param)