import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
from math import exp

"""
def Smooth_l1_loss(input, target, reduction='none'):
    # type: (Tensor, Tensor) -> Tensor
    t = torch.abs(input - target)
    ret = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret  
"""


class FFT(torch.nn.Module):
    def __init__(self):
        super(FFT, self).__init__()

    def forward(self, img1, img2):
        zeros=torch.zeros(img1.size()).cuda(img1.device)
        return nn.L1Loss(size_average=True)(torch.fft(torch.stack((img1,zeros),-1),2),torch.fft(torch.stack((img2,zeros),-1),2))#.cuda()

"""
class Smooth_l1_loss(torch.nn.Module):
    def __init__(self):
        super(Smooth_l1_loss, self).__init__()

    def forward(self, input, target, reduction='none'):
        # type: (Tensor, Tensor) -> Tensor
        t = torch.abs(input - target)
        ret = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret 
"""