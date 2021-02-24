# -*- coding: utf-8 -*-
"""
@author: Wei Yi Yiai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import keras
from keras_video import VideoFrameGenerator
# use sub directories names as classes
classes = [i.split(os.path.sep)[1] for i in glob.glob('videos\\*')]
classes.sort()
# some global params
SIZE = (640, 480) #自己設成Dataset的Size
CHANNELS = 3
NBFRAME = 5
BS = 8

glob_pattern='videos\\{classname}\\*.mp4'


# SS convolution
class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        # 手动定义卷积核(weight)，weight矩阵正中间的元素是1，其余为0
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        # nn.Parameter：类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到module里
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        # 根据Share and Separable convolution的定义，复制weights，x的每个通道对应相同的weight
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        # 调用F.conv2d进行卷积操作
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)


# 改进的膨脹卷積
class SmoothDilatedResidualBlock(nn.Module):
    #x代表input，然後依序為kernel_size, dilated_factor, stride, output_size
    def __init__(self, x, kernel_size, dilated_factor, stride, output_size, type, group=1): #group可以試試看>1的數字，聽說效果會比=1好
        super(SmoothDilatedResidualBlock, self).__init__()
        if len(x) == 3:
            input_channel_num = x[0]
        else:
            input_channel_num = x[1]

        output_channel_num = output_size[0]

        # 在膨脹卷積之前先使用SS convolution进行局部信息融合
        self.pre_conv1 = ShareSepConv(kernel_size)
        #膨脹卷積
        #Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode=‘zeros’)
        self.conv1 = nn.Conv2d(input_channel_num, output_channel_num, kernel_size, stride=stride, padding=dilated_factor, dilated_factor=dilated_factor, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(output_channel_num, affine=True)

        self.pre_conv2 = ShareSepConv(kernel_size)
        self.conv2 = nn.Conv2d(input_channel_num, output_channel_num, kernel_size, stride=stride, padding=dilated_factor, dilated_factor=dilated_factor, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(output_channel_num, affine=True)

    #def forward(self, x, type):
        # 残差连接
        if type == 'E': #LeakyReLU使用在encoder
            y =nn.LeakyReLU(self.norm1(self.conv1(self.pre_conv1(x))))
            y = self.norm2(self.conv2(self.pre_conv2(y)))
            return nn.LeakyReLU(x+y)
        elif type == 'D': #relu使用在decoder
            y = nn.ReLU(self.norm1(self.conv1(self.pre_conv1(x))))
            y = self.norm2(self.conv2(self.pre_conv2(y)))
            return nn.ReLU(x+y)


class VDSSNet(nn.Module):
    def __init__(self, x, in_c=15, out_c=3, H, W, only_residual=True):
        super(VDSSNet, self).__init__()
        #x代表input，然後依序為kernel_size, dilated_factor, stride, output_size, type
        #Encoder
        skip1=self.ssconv1 = SmoothDilatedResidualBlock(x, 5, 1, 1, [64, H, W], 'E') #To up_conv3

        # Down_conv1
        output=self.ssconv2 = SmoothDilatedResidualBlock(skip1, 3, 2, 2, [128, H/2, W/2], 'E')
        output=self.ssconv3 = SmoothDilatedResidualBlock(output, 3, 2, 1, [128, H/2, W/2], 'E')
        skip2=self.ssconv4 = SmoothDilatedResidualBlock(output, 3, 2, 1, [128, H/2, W/2], 'E') #To up_conv2

        # Down_conv2
        output=self.ssconv5 = SmoothDilatedResidualBlock(skip2, 3, 2, 2, [256, H/4, W/4], 'E')
        output=self.ssconv6 = SmoothDilatedResidualBlock(output, 3, 2, 1, [256, H/4, W/4], 'E')
        skip3=self.ssconv7 = SmoothDilatedResidualBlock(output, 3, 2, 1, [256, H/4, W/4], 'E') #To up_conv1

        #中間層
        # Down_conv3
        output=self.ssconv8 = SmoothDilatedResidualBlock(skip3, 3, 2, 2, [512, H/8, W/8], 'E')
        output=self.ssconv9 = SmoothDilatedResidualBlock(output, 3, 2, 1, [512, H/8, W/8], 'E')
        output=self.ssconv10 = SmoothDilatedResidualBlock(output, 3, 2, 1, [512, H/8, W/8], 'E') #From semantic-segmentation


        #Decoder
        # Up_conv1  from ssconv7
        #nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
        self.upsampling1 = nn.functional.interpolate(output, size=[output.[0], 256, H/4, W/4], scale_factor=1/2, mode='bilinear', align_corners=None)
        self.norm1 = nn.InstanceNorm2d(output_channel_num, affine=True)
        output = nn.LeakyReLU(self.norm1(self.upsampling1(output)))
        output = nn.ReLU(skip3 + output)
        output=self.ssconv11 = SmoothDilatedResidualBlock(output, 3, 2, 1, [256, H/4, W/4], 'D')
        output=self.ssconv12 = SmoothDilatedResidualBlock(output, 3, 2, 1, [256, H/4, W/4], 'D')


        # Up_conv2  from ssconv4
        self.upsampling2 = nn.functional.interpolate(output, size=[output.[0], 128, H/2, W/2], scale_factor=1/2, mode='bilinear', align_corners=None)
        self.norm2 = nn.InstanceNorm2d(output_channel_num, affine=True)
        output = nn.LeakyReLU(self.norm2(self.upsampling2(output)))
        output = nn.ReLU(skip2 + output)
        output=self.ssconv13 = SmoothDilatedResidualBlock(output, 3, 2, 1, [128, H/2, W/2], 'D')
        output=self.ssconv14 = SmoothDilatedResidualBlock(output, 3, 2, 1, [128, H/2, W/], 'D')


        # Up_conv3  from ssconv1
        self.upsampling3 = nn.functional.interpolate(output, size=[output.[0], 64, H, W], scale_factor=1/2, mode='bilinear', align_corners=None)
        self.norm3 = nn.InstanceNorm2d(output_channel_num, affine=True)
        output = nn.LeakyReLU(self.norm3(self.upsampling3(output)))
        output = nn.ReLU(skip1 + output)
        output=self.ssconv15 = SmoothDilatedResidualBlock(output, 3, 2, 1, [64, H, W], 'D')
        output=self.ssconv16 = SmoothDilatedResidualBlock(output, 3, 2, 1, [64, H, W], 'D')

        return output
