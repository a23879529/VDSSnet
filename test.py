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

import pytorch_ssim
from utils import make_dataset
from VDSSNet import VDSSNet
from math import log10

parser = argparse.ArgumentParser()
#parser.add_argument('--network', default='VDSSNet')
#parser.add_argument('--task', default='dehaze')
parser.add_argument('--datastes', default='not_real') #test dataset有GT 選"not_real"，沒有GT 選"real"
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--indir', default='test_input/')
parser.add_argument('--outdir', default='test_output/')
config = parser.parse_args()
#assert config.task in ['dehaze']

'''
没有训练代码，只能猜测作者这两行注释以及接下来的only_residual的意思是训练去雨模型时，
不是针对残差进行学习，最后输出的预测值直接为干净的图像。
对应地，在测试时去雾模型的输出结果是干净图像=雾图+预测值（残差），
而去雨模型的输出结果是干净图像=预测值
'''

## forget to regress the residue for deraining by mistake,
## which should be able to produce better results
#config.only_residual = config.task == 'dehaze'  

# 加载模型，指定输入输出路径
#config.model = 'models/wacv_vdssnet_%s.pth' % config.task 
config.model = './snapshots/Epoch29.pth'
config.use_cuda = config.gpu_id >= 0
if not os.path.exists(config.outdir):
    os.makedirs(config.outdir)

if config.dataset == "real":
    test_img_paths = make_dataset(config.indir) # data.py
else:
    test_dataset = HazeDataset(config.val_ori_data_path, config.val_haze_data_path, data_transform)
    test_img_paths = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                             num_workers=config.num_workers, pin_memory=True)

# 初始化模型
# 输入通道：3（RGB）；输出通道：3（RGB)
#dehaze_net = VDSSNet(in_c=3, out_c=3, only_residual=config.only_residual)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dehaze_net = VDSSNet().to(device)

# 加载参数
dehaze_net.load_state_dict(torch.load(config.model, map_location='cpu'))
dehaze_net.eval()

#psnr 與 ssim初始化
total_psnr = 0
total_ssim = 0

criterion = nn.MSELoss(size_average=True).cuda()


# 处理输入
for img_path in test_img_paths:
    img = Image.open(img_path).convert('RGB')
    print(img)
    im_w, im_h = img.size
    #if im_w % 4 != 0 or im_h % 4 != 0:
    #    img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4)))
    
    #等比縮放
    img.thumbnail((240, 320))
    img = np.array(img).astype('float')
    img_data = torch.from_numpy(img.transpose((2, 0, 1))).float() # (坐标x，坐标y，通道)->(通道，坐标x，坐标y)
    #edge_data = edge_compute(img_data) # 计算边缘信息
    #in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128  # 数据中心化 [0,255]->[-128,127]
    #in_data = in_data.cuda() if config.use_cuda else in_data.float()

    #with torch.no_grad():
    #    pred = dehaze_net(Variable(in_data))

    # round：四舍五入 clamp：大于或小于阈值时被截断(input, min, max, out=None)
    #if config.only_residual: # 去雾图像=原图+预测值（残差）
    #out_img_data = (pred.data[0].cpu().float() + img_data).round().clamp(0, 255)

    img_orig = img_orig.cuda()
    img_haze = img_haze.cuda()
    #print (img_haze.shape)
    break
    clean_image = dehaze_net(img_haze)

    torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                         config.sample_output_folder + str(iter_val + 1) + ".jpg")

    # 保存图片
    out_img = Image.fromarray(out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0))
    out_img.save(os.path.join(config.outdir, os.path.splitext(os.path.basename(img_path))[0] + '_%s.png' % config.task))
    
    mse = criterion(clean_image, img_orig)
    psnr = 10 * log10(1 / mse)
    #psnr = pytorch_ssim.ssim(clean_image, img_orig)
    ssim = pytorch_ssim.ssim(clean_image, img_orig)
    ssim = ssim[0]
    total_psnr += psnr
    total_ssim += ssim

#print("Average PSNR = %.3f" % (total_psnr / len(test_img_paths)), "  Averge SSIM = %.3f" % (total_ssim / len(test_img_paths)))
    
#------------------------以下可以從Frame變成Video-----------------------
"""
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
"""


