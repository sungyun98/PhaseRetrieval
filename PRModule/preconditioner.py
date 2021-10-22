###############################################################################
# Denoising Network for dRAAR and dpGPS
#
# Author: SUNG YUN LEE
#   
# Contact: sungyun98@postech.ac.kr
###############################################################################

__all__ = ['Preconditioner']

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .func import *
from .partialconv2d import *

class DenoisingNetwork(nn.Module):
    '''
    denoising network for generating preconditioning kernel
    
    partial convolutional U-net from https://arxiv.org/abs/1804.07723v2
    pointwise flip-mixing layer is added for reflecting centrosymmetry
    '''
    def __init__(self, cnum = 16):
        super().__init__()
        
        # Encoder
        self.pconv1 = PartialConv2d(1, cnum, 7, 2, 3, return_mask = True)
        self.bn1 = nn.BatchNorm2d(cnum)
        self.relu = nn.ReLU()
        
        self.pconv2 = PartialConv2d(cnum, cnum * 2, 5, 2, 2, return_mask = True)
        self.bn2 = nn.BatchNorm2d(cnum * 2)
        self.pconv3 = PartialConv2d(cnum * 2, cnum * 4, 5, 2, 2, return_mask = True)
        self.bn3 = nn.BatchNorm2d(cnum * 4)
        self.pconv4 = PartialConv2d(cnum * 4, cnum * 8, 3, 2, 1, return_mask = True)
        self.bn4 = nn.BatchNorm2d(cnum * 8)
        self.pconv5 = PartialConv2d(cnum * 8, cnum * 8, 3, 2, 1, return_mask = True)
        self.bn5 = nn.BatchNorm2d(cnum * 8)
        self.pconv6 = PartialConv2d(cnum * 8, cnum * 8, 3, 2, 1, return_mask = True)
        self.bn6 = nn.BatchNorm2d(cnum * 8)
        self.pconv7 = PartialConv2d(cnum * 8, cnum * 8, 3, 2, 1, return_mask = True)
        self.bn7 = nn.BatchNorm2d(cnum * 8)
        self.pconv8 = PartialConv2d(cnum * 8, cnum * 8, 3, 2, 1, return_mask = False)
        self.bn8 = nn.BatchNorm2d(cnum * 8)
        
        # Pointwise Flip-Mixing Layer
        self.pointwise = PartialConv2d(cnum * 16, cnum * 8, 1, 1, 0, groups = cnum * 8)
        self.bn_pw = nn.BatchNorm2d(cnum * 8)
        
        # Decoder
        self.upsample = nn.Upsample(scale_factor = 2)
        self.pconv9 = PartialConv2d(cnum * (8 + 8), cnum * 8, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(cnum * 8)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        self.pconv10 = PartialConv2d(cnum * (8 + 8), cnum * 8, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(cnum * 8)
        self.pconv11 = PartialConv2d(cnum * (8 + 8), cnum * 8, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(cnum * 8)
        self.pconv12 = PartialConv2d(cnum * (8 + 8), cnum * 8, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(cnum * 8)
        self.pconv13 = PartialConv2d(cnum * (8 + 4), cnum * 4, 3, 1, 1)
        self.bn13 = nn.BatchNorm2d(cnum * 4)
        self.pconv14 = PartialConv2d(cnum * (4 + 2), cnum * 2, 3, 1, 1)
        self.bn14 = nn.BatchNorm2d(cnum * 2)
        self.pconv15 = PartialConv2d(cnum * (2 + 1), cnum, 3, 1, 1)
        self.bn15 = nn.BatchNorm2d(cnum)
        self.pconv16 = PartialConv2d(cnum + 1, 1, 3, 1, 1)

    def forward(self, x0, m0):
        # Encoder
        x1, m = self.pconv1(x0, mask_in = m0)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x2, m = self.pconv2(x1, mask_in = m)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        
        x3, m = self.pconv3(x2, mask_in = m)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        
        x4, m = self.pconv4(x3, mask_in = m)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        
        x5, m = self.pconv5(x4, mask_in = m)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)
        
        x6, m = self.pconv6(x5, mask_in = m)
        x6 = self.bn6(x6)
        x6 = self.relu(x6)
        
        x7, m = self.pconv7(x6, mask_in = m)
        x7 = self.bn7(x7)
        x7 = self.relu(x7)
        
        x = self.pconv8(x7, mask_in = m)
        x = self.bn8(x)
        x = self.relu(x)
        
        # Pointwise Flip-Mixing Layer
        x = torch.cat((x, torch.flip(x, [2, 3])), dim = 2).view(x.size(0), x.size(1) * 2, x.size(2), x.size(3))
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.relu(x)
        
        # Decoder
        x = self.upsample(x)
        x = torch.cat((x, x7), dim = 1)
        x = self.pconv9(x)
        x = self.bn9(x)
        x = self.leakyrelu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x6), dim = 1)
        x = self.pconv10(x)
        x = self.bn10(x)
        x = self.leakyrelu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x5), dim = 1)
        x = self.pconv11(x)
        x = self.bn11(x)
        x = self.leakyrelu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x4), dim = 1)
        x = self.pconv12(x)
        x = self.bn12(x)
        x = self.leakyrelu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x3), dim = 1)
        x = self.pconv13(x)
        x = self.bn13(x)
        x = self.leakyrelu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x2), dim = 1)
        x = self.pconv14(x)
        x = self.bn14(x)
        x = self.leakyrelu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x1), dim = 1)
        x = self.pconv15(x)
        x = self.bn15(x)
        x = self.leakyrelu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x0), dim = 1)
        x = self.pconv16(x)
        
        return x

class Preconditioner():
    '''
    preconditioning kernel for dRAAR and dpGPS
    
    calculate inverted change ratio by denoising neural network in single photon region limited by [1-limit, 1+limit]
    single photon region is defined as FWHM of single photon count assuming normal distribution of sigma 0.5
    if denoising network not cover all single photon region, uncalculated value is set to maximum value 1+limit
    input should be k-space amplitude data, not intensity data
    input value should be scaled to photon count, not detector count
    reference = https://doi.org/10.1103/PhysRevResearch.3.043066
    '''
    def __init__(self, cnum = 16, path = './PRModule/param_pretrained.pth'):
        '''
        load pretrained denoising network
        
        args:
            cnum = integer (default = 0.25)
            path = string (default = './PRModule/pretrained.pth')
        '''
        super().__init__()
        
        self.net = DenoisingNetwork(cnum = cnum).eval()
        self.net.load_state_dict(torch.load(path), strict = True)
    
    def fitSize(self, input, fill, height = 512, width = 512):
        '''
        fit size of input to given heigh and width

        args:
            input = torch float tensor of size 1 * 1 * H * W * 1
            fill = float
            height = integer (default = 512)
            width = integer (default = 512)
        
        returns:
            output = torch float tensor of size 1 * 1 * H * W * 1
        '''

        h = input.size(2)
        w = input.size(3)
        if h != height or w != width:
            if h > height:
                c = height // 2
                input = input[:, :, h // 2 - c:h // 2 + c, :, :]
            elif h < height:
                p2 = (height - h) // 2
                p1 = (height - h) - p2
                input = F.pad(input, pad = (0, 0, 0, 0, p1, p2), value = fill)
            if w > width:
                c = width // 2
                input = input[:, :, :, w // 2 - c:w // 2 + c, :]
            elif w < width:
                p2 = (width - w) // 2
                p1 = (width - w) - p2
                input = F.pad(input, pad = (0, 0, p1, p2, 0, 0), value = fill)

        return input
        
    def getKernel(self, input, mask, limit = 0.25, deep = True, toggle = False):
        '''
        generate preconditioning kernel
        
        limit is change ratio limit for denoised data
        if limit is not positive, change ratio is not limited
        deep is switch for deep learning based kernel
        toggle is for returning denoised data, not preconditioning kernel
        
        args:
            input = torch float tensor of size 1 * 1 * H * W * 1
            mask = torch float tensor of size 1 * 1 * H * W * 1
            limit = float (default = 0.25)
            deep = bool (default = True)
            toggle = bool (default = False)
            
        returns:
            output = torch float tensor of size 1 * 1 * H * W * 1
        '''
        
        input = input * (1 - mask)
        input = fftshift(input)
        mask = fftshift(mask)
        input_raw = input.clone().detach()
        mask_raw = mask.clone().detach()
        
        # fit size of input and mask
        h = input.size(2)
        w = input.size(3)
        input = self.fitSize(input, fill = 0)
        mask = self.fitSize(mask, fill = 1)
        
        # get denoised input
        output = torch.log(input + 1)
        scale = torch.max(output).clamp(min = 1)
        output = output / scale
        with torch.no_grad():
            output = self.net(output.squeeze(-1), 1 - mask.squeeze(-1)).unsqueeze(-1)
        output = output * (1 - mask)
        output = torch.exp(output.clamp(min = 0, max = 2) * scale) - 1
        output = output / 10 # network trained for 100 times intensity, so 10 times for amplitude
        
        # limit change ratio
        if limit > 0:
            cu = output > input * (1 + limit)
            cd = output < input * (1 - limit)
            output[cu] = input[cu] * (1 + limit)
            output[cd] = input[cd] * (1 - limit)
        
        # return denoised input if toggle is True
        if toggle:
            output = self.fitSize(output, fill = -1, height = h, width = w)
            output[output < 0] = input_raw[output < 0]
            return output

        # remove effective zero-photon region
        thr = 0.5 # define threshold for zero-photon region (in photon count)
        bl = math.sqrt(1 - thr)
        output = output * torch.gt(input, bl)
        
        # calculate preconditioning kernel
        input[input == 0] = 1
        kernel = output / input
        kernel = self.fitSize(kernel, fill = 1, height = h, width = w)
        mask_zero = torch.le(input_raw, bl)
        kernel[mask_zero == 1] = 1 - limit
        kernel[mask_raw == 1] = 1

        # return non-deep preconditioning kernel
        if not deep:
            kernel[input_raw > bl] = 1 + limit
            
        # remove zero value by 1 - limit
        kernel[kernel == 0] = 1 - limit
        
        kernel = ifftshift(kernel)

        return kernel
