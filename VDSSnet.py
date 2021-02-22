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
    def __init__(self, channel_num, dilation=1, group=1):
        super(SmoothDilatedResidualBlock, self).__init__()
        # 在膨脹卷積之前先使用SS convolution进行局部信息融合
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        #膨脹卷積
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)

        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        # 残差连接
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
        y = self.norm2(self.conv2(self.pre_conv2(y)))
        return F.relu(x+y)


# 残差网络
class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)


class VDSSNet(nn.Module):
    def __init__(self, in_c=4, out_c=3, only_residual=True):
        super(VDSSNet, self).__init__()
        # Encoder：三层卷积，通道数64，卷积核大小3*3，stride=1，padding=1
        self.conv1 = nn.Conv2d(in_c, 64, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True) # Instance Normalization
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)  # stride=2的下采样
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
 
        # 中间层：7层smooth dilated convolution残差块，空洞率r分别为2,2,2,4,4,4,1，通道数64
        self.res1 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res2 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res3 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res4 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res5 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res6 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res7 = ResidualBlock(64, dilation=1)  # 空洞率为1时分离卷积的卷积核为1*1，没有起到信息融合的作用，因此该层退化为一个普通的残差网络

        # Gated Fusion Sub-network：学习低,中,高层特征的权重
        self.gate = nn.Conv2d(64 * 3, 3, 3, 1, 1, bias=True)

        # Decoder：1层反卷积层将feature map上采样到原分辨率+2层卷积层将feature map还原到原图空间
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1) # stride=2的上采样
        self.norm4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(64, affine=True)
        self.deconv1 = nn.Conv2d(64, out_c, 1) # 1*1卷积核进行降维
        self.only_residual = only_residual

    def forward(self, x):
        # Encoder前向传播，使用relu激活
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y1 = F.relu(self.norm3(self.conv3(y))) # 低层级信息

        # 中间层
        y = self.res1(y1)
        y = self.res2(y)
        y = self.res3(y)
        y2 = self.res4(y) # 中层级信息
        y = self.res5(y2)
        y = self.res6(y)
        y3 = self.res7(y) # 高层级信息

         # Gated Fusion Sub-network
        gates = self.gate(torch.cat((y1, y2, y3), dim=1)) # 计算低,中,高层特征的权重
        gated_y = y1 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :] # 对低,中,高层特征加权求和
        y = F.relu(self.norm4(self.deconv3(gated_y)))
        y = F.relu(self.norm5(self.deconv2(y)))
        if self.only_residual: # 去雾
            y = self.deconv1(y)
        else: # 去雨
            y = F.relu(self.deconv1(y))

        return y
