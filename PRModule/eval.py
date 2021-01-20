###############################################################################
# Additional Evaluation Functions
#
# Author: SUNG YUN LEE
#   
# Contact: sungyun98@postech.ac.kr
###############################################################################

__all__ = ['SubpixelAlignment', 'PairwiseDistance', 'PRTF', 'PSD', 'EigenMode']

import itertools
from tqdm import tqdm
import numpy as np
from numpy.linalg import svd
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation

def SubpixelAlignment(input, error = None, ref = None, subpixel = 1):
    '''
    subpixel alignment using phase cross-correlation
    reference = https://doi.org/10.1364/OL.33.000156
    
    input should be r-space data
    automatically sort input with repect to error if error is given
    
    args:
        input = numpy float ndarray of size N * H * W
        error = numpy float ndarray of size N (default = None)
        subpixel = integer (default = 1)
        
    returns:
        output = numpy float array with size N * H * W
        error = numpy float array with size N
    '''
    
    # sort array
    if error is not None:
        order = np.argsort(error)
        error = error[order]
        input = input[order, :, :]
    
    # align array
    if ref is None:
        n_max = input.shape[0]
        for n in tqdm(range(1, n_max), desc = 'subpixel alignment'):
            arr = input[n]
            arr_T = np.flip(arr)
            s, err, _ = phase_cross_correlation(input[0], arr, upsample_factor = subpixel)
            s_T, err_T, _ = phase_cross_correlation(input[0], arr_T, upsample_factor = subpixel)
            if err_T < err:
                input[n, :, :] = np.fft.ifft2(fourier_shift(np.fft.fft2(arr_T), s_T)).real
            else:
                input[n, :, :] = np.fft.ifft2(fourier_shift(np.fft.fft2(arr), s)).real
                
    else:
        n_max = input.shape[0]
        for n in tqdm(range(0, n_max), desc = 'subpixel alignment'):
            arr = input[n]
            arr_T = np.flip(arr)
            s, err, _ = phase_cross_correlation(ref, arr, upsample_factor = subpixel)
            s_T, err_T, _ = phase_cross_correlation(ref, arr_T, upsample_factor = subpixel)
            if err_T < err:
                input[n, :, :] = np.fft.ifft2(fourier_shift(np.fft.fft2(arr_T), s_T)).real
            else:
                input[n, :, :] = np.fft.ifft2(fourier_shift(np.fft.fft2(arr), s)).real
                
    # remove negative values due to alignment
    input[input < 0] = 0
            
    if error is not None:
        return input, error
    else:
        return input

def PairwiseDistance(input):
    '''
    pairwise distance
    
    distance is calculated by sum(|arr1-arr2|)/sum(|arr1+arr2|)
    input should be aligned
    
    args:
        input = numpy float ndarray of size N * H * W
        
    returns:
        output = numpy float ndarray of size N(N-1)/2
    '''
    
    # calculate pairwise distance
    n_max = input.shape[0]
    count = n_max * (n_max - 1) // 2
    dist = np.zeros(count)
    for n, (i, j) in tqdm(enumerate(itertools.combinations(range(n_max), 2)), total = count, desc = 'pairwise distance'):
        dist[n] = np.sum(np.abs(input[i] - input[j])) / np.sum(np.abs(input[i] + input[j]))
    
    return dist

def PRTF(input, ref, mask = None):
    '''
    phase retrieval transfer function (PRTF)
    reference = https://doi.org/10.1364/JOSAA.23.001179
    
    input should be aligned r-space data
    reference should be fftshifted k-space amplitude data for normalization
    ignore masked value if mask is given
    
    args:
        input = numpy float ndarray of size N * H * W
        ref = numpy float ndarray of size H * W
        mask = numpy bool ndarray of size H * W (default = None)
        
    returns:
        output = numpy ndarray with size H * W
    '''
    
    # get Fourier transform of input
    freq = np.fft.fftshift(np.fft.fft2(input))
    freq = np.absolute(np.mean(freq, axis = 0))
    
    # normalization
    ref[ref == 0] = 1
    freq = freq / ref
    if mask is not None:
        freq[mask] = 0
    
    return freq

def PSD(input, mask = None):
    '''
    power spectral density (PSD)
    
    input should be fftshifted k-space amplitude or intensity data
    ignore masked value if mask is given
    
    args:
        input = numpy float ndarray of size H * W
        mask = numpy bool ndarray of size H * W (default = None)
        
    returns:
        output = numpy float ndarray of size max(H,W)/2
    '''
    
    # get distance mesh
    di = input.shape[0]
    dj = input.shape[1]
    li = np.linspace(-di / 2 + 0.5, di / 2 - 0.5, num = di)
    lj = np.linspace(-dj / 2 + 0.5, dj / 2 - 0.5, num = dj)
    mi, mj = np.meshgrid(li, lj, indexing = 'ij')
    m = np.sqrt(np.power(mi, 2) + np.power(mj, 2))
    
    # calculate psd
    r_max = min(di, dj) // 2
    psd = np.zeros(r_max)
    for r in tqdm(range(r_max), desc = 'psd'):
        drop = (m >= r) * (m < r + 1)
        if mask is not None:
            drop = drop * (1 - mask)
        drop = drop > 0
        if np.sum(drop) > 0:
            psd[r] = np.mean(input[drop])
        else:
            psd[r] = np.nan
    
    return psd

def EigenMode(input, k = None, lowrank = True):
    '''
    extract eigenmode of input

    returns low-rank approximation of input if lowrank is True
    k should be given for low-rank approximation

    args:
        input = numpy float ndarray of size N * H * W
        k = integer (default = None)
        lowrank = bool (default = True)

    returns:
        output = numpy float array of size (N or k) * H * W
        s = numpy float array of size (N or k)
        approx = numpy float array of size k * H * W
    '''

    h = input.shape[1]
    w = input.shape[2]
    input = input.reshape((-1, h * w)).T
    # calculate singular value decomposition
    u, s, vh = svd(input, full_matrices = False)
    output = u.T.reshape((-1, h, w))
    if k is None:
        return output, s
    else:
        if not lowrank:
            return output[:k], s[:k]
        else:
            # calculate low-rank approximation with order 1 to k
            approx = np.zeros((k, h, w))
            for l in range(1, k + 1):
                temp = u[:, :l] @ np.diag(s[:l]) @ vh[:l, :]
                temp = temp[:, 0].reshape((h, w))
                approx[l - 1, :, :] = temp
            return output[:k], s[:k], approx