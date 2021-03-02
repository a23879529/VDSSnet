# -*- coding: utf-8 -*-
"""
@author: Wei Yi Yiai
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

from utils import make_dataset, edge_compute

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='VDSSNet')
parser.add_argument('--task', default='dehaze')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--indir', default='examples/')
parser.add_argument('--outdir', default='output')
opt = parser.parse_args()
#assert opt.task in ['dehaze']

'''
没有训练代码，只能猜测作者这两行注释以及接下来的only_residual的意思是训练去雨模型时，
不是针对残差进行学习，最后输出的预测值直接为干净的图像。
对应地，在测试时去雾模型的输出结果是干净图像=雾图+预测值（残差），
而去雨模型的输出结果是干净图像=预测值
'''

## forget to regress the residue for deraining by mistake,
## which should be able to produce better results
#opt.only_residual = opt.task == 'dehaze'  

# 加载模型，指定输入输出路径
#opt.model = 'models/wacv_vdssnet_%s.pth' % opt.task 
opt.model = './snapshots/Epoch29.pth'
opt.use_cuda = opt.gpu_id >= 0
if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)
test_img_paths = make_dataset(opt.indir) # utils.py

# 初始化模型
if opt.network == 'VDSSNet':
    from VDSSNet import VDSSNet
    # 输入通道：3（RGB）；输出通道：3（RGB)
    #dehaze_net = VDSSNet(in_c=3, out_c=3, only_residual=opt.only_residual)
    dehaze_net = VDSSNet()
else:
    print('network structure %s not supported' % opt.network)
    raise ValueError

# GPU or CPU
if opt.use_cuda:
    torch.cuda.set_device(opt.gpu_id)
    dehaze_net.cuda()
else:
    dehaze_net.float()

# 加载参数
dehaze_net.load_state_dict(torch.load(opt.model, map_location='cpu'))
dehaze_net.eval()

# 处理输入
for img_path in test_img_paths:
    img = Image.open(img_path).convert('RGB')
    im_w, im_h = img.size
    if im_w % 4 != 0 or im_h % 4 != 0:
        img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4))) 
    img = np.array(img).astype('float')
    img_data = torch.from_numpy(img.transpose((2, 0, 1))).float() # (坐标x，坐标y，通道)->(通道，坐标x，坐标y)
    edge_data = edge_compute(img_data) # 计算边缘信息
    in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128  # 数据中心化 [0,255]->[-128,127]
    in_data = in_data.cuda() if opt.use_cuda else in_data.float()

    with torch.no_grad():
        pred = dehaze_net(Variable(in_data))

    # round：四舍五入 clamp：大于或小于阈值时被截断(input, min, max, out=None)
    #if opt.only_residual: # 去雾图像=原图+预测值（残差）
    out_img_data = (pred.data[0].cpu().float() + img_data).round().clamp(0, 255)

    # 保存图片
    out_img = Image.fromarray(out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0))
    out_img.save(os.path.join(opt.outdir, os.path.splitext(os.path.basename(img_path))[0] + '_%s.png' % opt.task))
    
#------------------------以下可以從Frame變成Video-----------------------
    
import os
import cv2
import numpy as np
import glob

number = 1
img_array = []
#b = sorted(glob.glob('*.png'))
b = sorted(glob.glob('*.png'), key=os.path.getmtime) #已排序
for filename in b:
    #print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
out = cv2.VideoWriter('Dehazedvideo.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
 
for i in range(len(img_array)):
    print("len=",len(img_array))
    print("name=",img_array[i])
    out.write(img_array[i])
out.release()
print("Done!Done!")



