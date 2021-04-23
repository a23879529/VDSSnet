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
        self.conv1 = nn.Conv2d(input_channel_num, output_channel_num, kernel_size, stride=stride, padding=dilated_factor, dilation=dilated_factor, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(output_channel_num, affine=True)
        # self.norm1 = nn.BatchNorm2d(output_channel_num, affine=True)


        self.pre_conv2 = ShareSepConv(kernel_size)
        self.conv2 = nn.Conv2d(input_channel_num, output_channel_num, kernel_size, stride=stride, padding=dilated_factor, dilation=dilated_factor, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(output_channel_num, affine=True)
        # self.norm2 = nn.BatchNorm2d(output_channel_num, affine=True)


    def forward(self, x, type):
        # 残差连接
        #print (x.size())
        if type == 'E': #LeakyReLU使用在encoder
            y =F.leaky_relu(self.norm1(self.conv1(self.pre_conv1(x))))
            # y =F.leaky_relu(self.norm1(self.conv1(x)))
            #print(y.size())
            # y = self.norm2(self.conv2(self.pre_conv2(y)))
            return y
        elif type == 'D': #relu使用在decoder
            y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
            # y =F.leaky_relu(self.norm2(self.conv2(x)))
            #y = self.norm2(self.conv2(self.pre_conv2(y)))
            return y


class DepthwiseBlock(nn.Module):
    def __init__(self, input_channel_num, kernel_size, dilated_factor, stride, output_channel_num):
        super(DepthwiseBlock, self).__init__()

        self.depthwise_conv = nn.Conv2d(input_channel_num, output_channel_num, kernel_size, stride=stride, padding=dilated_factor, dilation=dilated_factor, groups=input_channel_num)
        self.pointwise_conv = nn.Conv2d(input_channel_num, output_channel_num, 1)

    def forward(self, x, type):
        # 残差连接
        #print (x.size())
        if type == 'E': #LeakyReLU使用在encoder
            y = F.leaky_relu(self.depthwise_conv(x))
            y = torch.sigmoid(self.pointwise_conv(y))
            y = y * x
            #print(y.size())
            #y = self.norm2(self.conv2(self.pre_conv2(y)))
            return y
        elif type == 'D': #relu使用在decoder
            y = F.relu(self.depthwise_conv(x))
            y = torch.sigmoid(self.pointwise_conv(y))
            y = y * x
            #y = self.norm2(self.conv2(self.pre_conv2(y)))
            return y


class VDSSNet(nn.Module):
    def __init__(self, in_c=15, out_c=3):
        super(VDSSNet, self).__init__()
        #Encoder
        self.ssconv1 = nn.Conv2d(in_c, 64, 5, 1, 2) #To up_conv3

        # Down_conv1
        #依序為input_channel, kernel_size, dilated_factor, stride, output_channel
        # self.ssconv2 = nn.Conv2d(64, 128, 3, 2, 2, 2)
        self.ssconv2 = SmoothDilatedResidualBlock(64, 3, 2, 2, 128)
        self.ssconv3 = SmoothDilatedResidualBlock(128, 3, 1, 1, 128)
        self.addictionssconv1 = SmoothDilatedResidualBlock(128, 3, 3, 1, 128)
        self.ssconv4 = SmoothDilatedResidualBlock(128, 3, 5, 1, 128) #To up_conv2
        self.Attention1 = DepthwiseBlock(128, 3, 2, 1, 128)

        # Down_conv2
        self.ssconv5 = SmoothDilatedResidualBlock(128, 3, 2, 2, 256)
        self.ssconv6 = SmoothDilatedResidualBlock(256, 3, 1, 1, 256)
        self.addictionssconv2 = SmoothDilatedResidualBlock(256, 3, 3, 1, 256)
        self.ssconv7 = SmoothDilatedResidualBlock(256, 3, 5, 1, 256) #To up_conv1
        self.Attention2 = DepthwiseBlock(256, 3, 2, 1, 256)

        #中間層
        # Down_conv3
        self.ssconv8 = SmoothDilatedResidualBlock(256, 3, 2, 2, 512)
        self.ssconv9 = SmoothDilatedResidualBlock(512, 3, 1, 1, 512)
        self.ssconv10 = SmoothDilatedResidualBlock(512, 3, 3, 1, 512)
        self.ssconv11 = SmoothDilatedResidualBlock(512, 3, 5, 1, 512) #From semantic-segmentation
        self.Attention3 = DepthwiseBlock(512, 3, 2, 1, 512)


        #Decoder
        # Up_conv1  from ssconv7
        #nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
        self.Up_conv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)     
        #self.Up_conv1 = nn.Conv2d(512, 256, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(256, affine=True)
        self.ssconv12 = SmoothDilatedResidualBlock(256, 3, 2, 1, 256)
        self.ssconv13 = SmoothDilatedResidualBlock(256, 3, 1, 1, 256)
        self.addictionssconv3 = SmoothDilatedResidualBlock(256, 3, 3, 1, 256)
        self.ssconv14 = SmoothDilatedResidualBlock(256, 3, 5, 1, 256)
        self.Attention4 = DepthwiseBlock(256, 3, 2, 1, 256)


        # Up_conv2  from ssconv4
        self.Up_conv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        #self.Up_conv2 = nn.Conv2d(256, 128, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.ssconv15 = SmoothDilatedResidualBlock(128, 3, 2, 1, 128)
        self.ssconv16 = SmoothDilatedResidualBlock(128, 3, 1, 1, 128)
        self.addictionssconv4 = SmoothDilatedResidualBlock(128, 3, 3, 1, 128)
        self.ssconv17 = SmoothDilatedResidualBlock(128, 3, 5, 1, 128)
        self.Attention5 = DepthwiseBlock(128, 3, 2, 1, 128)


        # Up_conv3  from ssconv1
        self.Up_conv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) 
        #self.Up_conv3 = nn.Conv2d(128, 64, 1, bias=False) 
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
        # self.ssconv18 = SmoothDilatedResidualBlock(64, 3, 2, 1, 64)
        # self.ssconv19 = SmoothDilatedResidualBlock(64, 3, 2, 1, 64)
        # self.ssconv20 = SmoothDilatedResidualBlock(64, 3, 2, 1, 64)
        # self.Attention6 = DepthwiseBlock(64, 3, 2, 1, 64)
        self.ssconv21 = nn.Conv2d(64, out_c, 1)

    def forward(self, x):
        #Encoder
        # print("input: ", x.size())
        #skip1 = self.ssconv1(x, x[2], x[3], 'E')
        skip1 = self.ssconv1(x)
        # print("skip1: ", skip1.size())

        # Down_conv1
        # output = F.leaky_relu(self.ssconv2(skip1))
        output = self.ssconv2(skip1, 'E')
        # print("ssconv2: ", output.size())
        output = self.ssconv3(output, 'E')
        #print("ssconv3: ", output.size())
        output = self.addictionssconv1(output, 'E')
        skip2 = self.ssconv4(output, 'E')
        # print("ssconv4: ", skip2.size())
        output = self.Attention1(output, 'E')

        # Down_conv2
        output = self.ssconv5(output, 'E')
        # print("ssconv5: ", output.size())
        output = self.ssconv6(output, 'E')
        # print("ssconv6: ", output.size())
        output = self.addictionssconv2(output, 'E')
        skip3 = self.ssconv7(output, 'E')
        # print("ssconv7: ", skip3.size())
        output = self.Attention2(output, 'E')

        #中間層
        # Down_conv3
        output = self.ssconv8(output, 'E')
        # print("ssconv8: ", output.size())
        output = self.ssconv9(output, 'E')
        #print("ssconv9: ", output.size())
        output = self.ssconv10(output, 'E')
        #print("ssconv10: ", output.size())
        output = self.ssconv11(output, 'E')
        # print("ssconv11: ", output.size())
        output = self.Attention3(output, 'E')

        #Decoder
        # Up_conv1  from ssconv7
        #print(output.size())
        #output = self.upsampling1 = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=None)
        #print(output.size())
        output = self.Up_conv1(output)
        # print("Up_conv1: ", output.size())
        output = F.relu(self.norm1(output))
        # print("skip3 + output: ", skip3.size(),output.size())
        output = F.relu(skip3 + output)
        #print("output: ", output.size())
        output = self.ssconv12(output, 'D')
        #print("ssconv12: ", output.size())
        output = self.ssconv13(output, 'D')
        #print("ssconv13: ", output.size())
        output = self.addictionssconv3(output, 'D')
        output = self.ssconv14(output, 'D')
        #print("ssconv14: ", output.size())
        output = self.Attention4(output, 'D')
        #print("Attention1: ", output.size())

        # Up_conv2  from ssconv4
        #output = self.upsampling1 = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=None)
        output = self.Up_conv2(output)
        #print("Up_conv2: ", output.size())
        output = F.relu(self.norm2(output))
        #print("skip2 + output: ", skip2.size(),output.size())
        output = F.relu(skip2 + output)
        #print("output: ", output.size())
        output = self.ssconv15(output, 'D')
        #print("ssconv15: ", output.size())
        output = self.ssconv16(output, 'D')
        #print("ssconv16: ", output.size())
        output = self.addictionssconv4(output, 'D')
        output = self.ssconv17(output, 'D')
        #print("ssconv17: ", output.size())
        output = self.Attention5(output, 'D')
        # print("Attention2: ", output.size())

        # Up_conv3  from ssconv1
        #output = self.upsampling1 = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=None)
        output = self.Up_conv3(output)
        #print("Up_conv3: ", output.size())
        output = F.relu(self.norm3(output))
        #print("skip1 + output: ", skip1.size(),output.size())
        output = F.relu(skip1 + output)
        #print("output: ", output.size())
        # output = self.ssconv18(output, 'D')
        #print("ssconv18: ", output.size())
        # output = self.ssconv19(output, 'D')
        #print("ssconv19: ", output.size())
        # output = self.ssconv20(output, 'D')
        #print("ssconv20: ", output.size())
        # output = self.Attention6(output, 'D')
        #print("Attention3: ", output.size())


        output = self.ssconv21(output)
        #print("ssconv21: ", output.size())

        return output
