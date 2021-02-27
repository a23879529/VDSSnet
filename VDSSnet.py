# -*- coding: utf-8 -*-
"""
@author: Wei Yi Yiai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import os
import glob
# use sub directories names as classes
#classes = [i.split(os.path.sep)[1] for i in glob.glob('videos\\*')]
#classes.sort()
# some global params
#SIZE = (640, 480) #自己設成Dataset的Size
#CHANNELS = 3
#NBFRAME = 5
#BS = 8

#glob_pattern='videos\\{classname}\\*.mp4'




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
        #print(x)
        inc = x.size(1)
        #print (inc)
        #inc = inc.tolist()
        # 根据Share and Separable convolution的定义，复制weights，x的每个通道对应相同的weight
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        #print(expand_weight)
        # 调用F.conv2d进行卷积操作
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)


# 改进的膨脹卷積
class SmoothDilatedResidualBlock(nn.Module):
    #x代表input，然後依序為kernel_size, dilated_factor, stride, output_size
    def __init__(self, input_channel_num, kernel_size, dilated_factor, stride, output_channel_num, group=1): #group可以試試看>1的數字，聽說效果會比=1好
        super(SmoothDilatedResidualBlock, self).__init__()

        # 在膨脹卷積之前先使用SS convolution进行局部信息融合
        self.pre_conv1 = ShareSepConv(kernel_size)
        #膨脹卷積
        #Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode=‘zeros’)
        self.conv1 = nn.Conv2d(input_channel_num, input_channel_num, kernel_size, stride=stride, padding=dilated_factor, dilation=dilated_factor, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(input_channel_num, affine=True)

        self.pre_conv2 = ShareSepConv(kernel_size)
        self.conv2 = nn.Conv2d(input_channel_num, output_channel_num, kernel_size, stride=stride, padding=dilated_factor, dilation=dilated_factor, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(output_channel_num, affine=True)

    def forward(self, x, H, W, type):
        # 残差连接
        #print (x.size())
        if type == 'E': #LeakyReLU使用在encoder
            y =F.leaky_relu(self.norm1(self.conv1(self.pre_conv1(x))))
            #print(y.size())
            y = self.norm2(self.conv2(self.pre_conv2(y)))
            return y
        elif type == 'D': #relu使用在decoder
            y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
            y = self.norm2(self.conv2(self.pre_conv2(y)))
            return y




class VDSSNet(nn.Module):
    def __init__(self, in_c=3, out_c=3):
        super(VDSSNet, self).__init__()
        #依序為input_channel, kernel_size, dilated_factor, stride, output_channel
        #Encoder
        self.ssconv1 = SmoothDilatedResidualBlock(in_c, 5, 1, 1, 64) #To up_conv3

        # Down_conv1
        self.ssconv2 = SmoothDilatedResidualBlock(64, 3, 2, 2, 128)
        self.ssconv3 = SmoothDilatedResidualBlock(128, 3, 2, 1, 128)
        self.ssconv4 = SmoothDilatedResidualBlock(128, 3, 2, 1, 128) #To up_conv2

        # Down_conv2
        self.ssconv5 = SmoothDilatedResidualBlock(128, 3, 2, 2, 256)
        self.ssconv6 = SmoothDilatedResidualBlock(256, 3, 2, 1, 256)
        self.ssconv7 = SmoothDilatedResidualBlock(256, 3, 2, 1, 256) #To up_conv1

        #中間層
        # Down_conv3
        self.ssconv8 = SmoothDilatedResidualBlock(256, 3, 2, 2, 512)
        self.ssconv9 = SmoothDilatedResidualBlock(512, 3, 2, 1, 512)
        self.ssconv10 = SmoothDilatedResidualBlock(512, 3, 2, 1, 512)
        self.ssconv11 = SmoothDilatedResidualBlock(512, 3, 2, 1, 512) #From semantic-segmentation


        #Decoder
        # Up_conv1  from ssconv7
        #nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
        #self.Up_conv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=2)     
        self.Up_conv1 = nn.Conv2d(input_channel_num, output_channel_num, kernel_size, stride=stride, padding=dilated_factor, dilation=dilated_factor, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(256, affine=True)
        self.ssconv12 = SmoothDilatedResidualBlock(256, 3, 2, 1, 256)
        self.ssconv13 = SmoothDilatedResidualBlock(256, 3, 2, 1, 256)


        # Up_conv2  from ssconv4
        #self.Up_conv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=2)
        self.Up_conv1 = nn.Conv2d(input_channel_num, output_channel_num, kernel_size, stride=stride, padding=dilated_factor, dilation=dilated_factor, groups=group, bias=False)   
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.ssconv14 = SmoothDilatedResidualBlock(128, 3, 2, 1, 128)
        self.ssconv15 = SmoothDilatedResidualBlock(128, 3, 2, 1, 128)


        # Up_conv3  from ssconv1
        #self.Up_conv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=2) 
        self.Up_conv1 = nn.Conv2d(input_channel_num, output_channel_num, kernel_size, stride=stride, padding=dilated_factor, dilation=dilated_factor, groups=group, bias=False) 
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
        self.ssconv16 = SmoothDilatedResidualBlock(64, 3, 2, 1, 64)
        self.ssconv17 = SmoothDilatedResidualBlock(64, 3, 2, 1, 64)
        self.ssconv18 = SmoothDilatedResidualBlock(64, 1, 1, 0, out_c)

    def forward(self, x):
        #Encoder
        skip1 = self.ssconv1(x, x[2], x[3], 'E')

        # Down_conv1
        output = self.ssconv2(skip1, skip1[2], skip1[3], 'E')
        output = self.ssconv3(output, output[2], output[3], 'E')
        skip2 = self.ssconv4(output, output[2], output[3], 'E')

        # Down_conv2
        output = self.ssconv5(skip2, skip2[2], skip2[3], 'E')
        output = self.ssconv6(output, output[2], output[3], 'E')
        skip3 = self.ssconv7(output, output[2], output[3], 'E')

        #中間層
        # Down_conv3
        output = self.ssconv8(skip3, skip3[2], skip3[3], 'E')
        output = self.ssconv9(output, output[2], output[3], 'E')
        output = self.ssconv10(output, output[2], output[3], 'E')
        output = self.ssconv11(output, output[2], output[3], 'E')

        #Decoder
        # Up_conv1  from ssconv7
        print(output.size())
        output = self.upsampling1 = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=None)
        print(output.size())

        output = F.leaky_relu(self.norm1(output))
        print(skip3.size(),output.size())
        output = F.relu(skip3 + output)
        output = self.ssconv12(output, output[2], output[3], 'D')
        output = self.ssconv13(output, output[2], output[3], 'D')

        # Up_conv2  from ssconv4
        output = self.upsampling1 = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=None)
        output = F.leaky_relu(self.norm2(output))
        output = F.relu(skip2 + output)
        output = self.ssconv14(output, output[2], output[3], 'D')
        output = self.ssconv15(output, output[2], output[3], 'D')

        # Up_conv3  from ssconv1
        output = self.upsampling1 = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=None)
        output = F.leaky_relu(self.norm3(output))
        output = F.relu(skip1 + output)
        output = self.ssconv16(output, output[2], output[3], 'D')
        output = self.ssconv17(output, output[2], output[3], 'D')
        output = self.ssconv18(output, output[2], output[3], 'D')

        #output=gaussian_blur2d(output, (3, 3), (1.5, 1.5)) #對T-map進行中值濾波

        #atmosphere = get_atmosphere(output)

        return output
