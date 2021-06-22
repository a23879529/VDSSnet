# -*- coding: utf-8 -*-
"""
@author: Wei Yi Yiai
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
# from torch.autograd import Variable

import pytorch_ssim
import torchvision
from torchvision import transforms
# from utils import make_dataset
# from VDSSnet_3 import VDSSNet
from VDSSnet_3_old import VDSSNet
from math import log10
from input_test3 import test_data
# from input_test5 import train_data

parser = argparse.ArgumentParser()
#parser.add_argument('--network', default='VDSSNet')
#parser.add_argument('--task', default='dehaze')
parser.add_argument('--datastes_type', default='real') #test dataset有GT 選"not_real"，沒有GT 選"real"
parser.add_argument('--input_type', default='video') #test daraset是video 選"video"，單張選"singel_image"
parser.add_argument('--gpu_id', type=int, default=0
    )
parser.add_argument('--gtdir', default='F:\\NYU-2')
# parser.add_argument('--hazedir', default='H:\\NYU_v2_test\\')
# parser.add_argument('--outdir', default='H:\\測試\\conv135_full_IN_EP3\\furniture_store_0002c\\')
# parser.add_argument('--outdir', default='H:\\測試\\kernel123_back3_EP3\\')
# parser.add_argument('--txtoutdir', default='furniture_store_0002c')
# parser.add_argument('--gtdir', default='D:\\論文\\proposed\\VDSSnet\\test_input\\outdoor\\gt')
parser.add_argument('--hazedir', default='C:\\Users\\ian\\Pictures\\FreeVideoToJPGConverter\\')
parser.add_argument('--outdir', default='H:\\測試\\kernel123_back3_EP2_stop\\video\\')
config = parser.parse_args()
#assert config.task in ['dehaze']

'''
没有训练代码，只能猜测作者这两行注释以及接下来的only_residual的意思是训练去雨模型时，
不是针对残差进行学习，最后输出的预测值直接为干净的图像。
对应地，在测试时去雾模型的输出结果是干净图像=雾图+预测值（残差），
而去雨模型的输出结果是干净图像=预测值
'''
# DIRS = ["basement_0001b","bathroom_0006","bathroom_0018","bathroom_0025",
# "bedroom_0058","cafe_0001b","computer_lab_0002","furniture_store_0002c","home_office_0011","indoor_balcony_0001",
# "kitchen_0030a","laundry_room_0001","library_0001a","living_room_0030a","office_0012","office_0020",
# "office_kitchen_0002","printer_room_0001","reception_room_0001a","study_room_0005b"]

# DIRS = ['Crossroad','Driving','Riverside']
DIRS = ['Crossroad']

print(len(DIRS))
TP = 0
TS = 0
tmep = config.outdir

for dirs in DIRS:
    config.outdir = tmep + dirs + "\\"
    config.txtoutdir = dirs
    print(config.outdir, config.txtoutdir)
    ## forget to regress the residue for deraining by mistake,
    ## which should be able to produce better results
    #config.only_residual = config.task == 'dehaze'  

    # 加载模型，指定输入输出路径
    #config.model = 'models/wacv_vdssnet_%s.pth' % config.task 
    # config.model = './snapshots/Epoch29.pth'
    config.use_cuda = config.gpu_id >= 0
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    test_dataset = test_data(config.gtdir, config.hazedir, config.datastes_type, config.input_type, config.txtoutdir) # data.py

    # 初始化模型
    # 输入通道：3（RGB）；输出通道：3（RGB)
    #dehaze_net = VDSSNet(in_c=3, out_c=3, only_residual=config.only_residual)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dehaze_net = VDSSNet().to(device)

    # 加载参数
    checkpoint = torch.load('./snapshots/kernel123_back3_EP2.pth')
    dehaze_net.load_state_dict(checkpoint['model_state_dict'])
    # dehaze_net.load_state_dict(torch.load(config.model, map_location='cpu'))
    dehaze_net.eval()

    #psnr 與 ssim初始化
    total_psnr = 0
    total_ssim = 0

    criterion = nn.MSELoss(reduction='mean').cuda()

    A = np.array(test_dataset)
    itre = A.shape[0]
    # total_count = 0

    for iter_test in range(itre):

        temp1 = torch.tensor([])
        temp2 = torch.tensor([])
        final_gt = torch.tensor([])
        final_haze = torch.tensor([])

        if config.datastes_type == 'not_real':
            if config.input_type == 'video':     #有GT是video
                gt = Image.open(test_dataset[iter_test][0]).convert("RGB")
                img1 = Image.open(test_dataset[iter_test][1]).convert("RGB")
                img2 = Image.open(test_dataset[iter_test][2]).convert("RGB")
                img3 = Image.open(test_dataset[iter_test][3]).convert("RGB")

            elif config.input_type == 'singel_image':    #有GT是單張
                gt = Image.open(test_dataset[iter_test][0]).convert("RGB")
                img1 = Image.open(test_dataset[iter_test][1]).convert("RGB")
                img2 = img1
                img3 = img1

        elif config.datastes_type == 'real':
            if config.input_type == 'video':     #沒GT是video
                img1 = Image.open(test_dataset[iter_test][0]).convert("RGB")
                img2 = Image.open(test_dataset[iter_test][1]).convert("RGB")
                img3 = Image.open(test_dataset[iter_test][2]).convert("RGB")

            elif config.input_type == 'singel_image':    #沒GT是單張
                img1 = Image.open(test_dataset[iter_test][0]).convert("RGB")
                img2 = img1
                img3 = img1
        
        

        my_transforms = transforms.Compose([
        transforms.Resize([240, 320]),
        transforms.ToTensor()
        ])

        if config.datastes_type == 'not_real':
            gt = my_transforms(gt)

        img1 = my_transforms(img1)
        img2 = my_transforms(img2)
        img3 = my_transforms(img3)

        if iter_test == 0:
            if config.datastes_type == 'not_real':
                final_gt = gt
                final_gt = final_gt.unsqueeze(0)

            final_haze = torch.cat((img1, img2, img3), 0)
            final_haze = final_haze.unsqueeze(0)
            # print(final.shape, "0")
        else:
            if config.datastes_type == 'not_real':
                temp1 = gt
                temp1 = temp1.unsqueeze(0)
                final_gt = torch.cat([final_gt, temp1], 0)

            temp2 = torch.cat((img1, img2, img3), 0)
            temp2 = temp2.unsqueeze(0)
            # print(temp1.shape, "temp1")
            final_haze = torch.cat([final_haze, temp2], 0)
            # print(final.shape, "final")
        # count+=1
        # total_count+=1
        # print(final.shape)
        # import matplotlib.pyplot as plt
        if config.datastes_type == 'not_real':
            img_orig = final_gt.cuda()

        img_haze = final_haze.cuda()
        # print (img_haze.shape)
        # print (ss_image.shape)
        # print(img_haze.type())
        # torchvision.utils.save_image(img_haze, config.sample_output_folder + "output.jpg")
        # clean_image = dehaze_net(img_haze, ss_image)
        clean_image = dehaze_net(img_haze)
        
        temp1 = []
        temp2 = []

        if config.datastes_type == 'not_real':
            if config.input_type == 'video':     #有GT是video

                temp1 = torch.stack((img_haze[0][3], img_haze[0][4], img_haze[0][5]), 0)
                temp1 = temp1.unsqueeze(0)
                img_haze = temp1

                # torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                #                                  config.outdir + str(iter_test + 1) + ".jpg")
                torchvision.utils.save_image(clean_image, config.outdir + str(iter_test + 1) + ".jpg")
            elif config.input_type == 'singel_image':    #有GT是單張
                # print(img_haze.shape)

                temp1 = torch.stack((img_haze[0][0], img_haze[0][1], img_haze[0][2]), 0)
                temp1 = temp1.unsqueeze(0)
                # print(temp1.shape)
                img_haze = temp1

                # print (img_haze.shape, clean_image.shape, img_orig.shape)

                torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                                 config.outdir + str(iter_test + 1) + ".jpg")
        elif config.datastes_type == 'real':
            if config.input_type == 'video':     #沒GT是video

                temp1 = torch.stack((img_haze[0][3], img_haze[0][4], img_haze[0][5]), 0)
                temp1 = temp1.unsqueeze(0)
                img_haze = temp1

                # torchvision.utils.save_image(torch.cat((img_haze, clean_image), 0),
                #                                  config.outdir + str(iter_test + 1) + ".jpg")
                torchvision.utils.save_image(clean_image, config.outdir + str(iter_test + 1) + ".jpg")
            elif config.input_type == 'singel_image':    #沒GT是單張
                temp1 = torch.stack((img_haze[0][0], img_haze[0][1], img_haze[0][2]), 0)
                temp1 = temp1.unsqueeze(0)
                img_haze = temp1

                torchvision.utils.save_image(torch.cat((img_haze, clean_image), 0),
                                                 config.outdir + str(iter_test + 1) + ".jpg")

        # 保存图片
        # out_img = Image.fromarray(out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0))
        # out_img.save(os.path.join(config.outdir, os.path.splitext(os.path.basename(img_path))[0] + '_%s.png' % config.task))
        if config.datastes_type == 'not_real':

            mse = criterion(clean_image, img_orig)
            psnr = 10 * log10(1 / mse)
            #psnr = pytorch_ssim.ssim(clean_image, img_orig)
            # print (clean_image.shape, img_orig.shape)
            ssim = pytorch_ssim.ssim(clean_image, img_orig)
            ssim = ssim.item()
            # print("%.3f" % psnr, "%.3f" %ssim)
            total_psnr += psnr
            total_ssim += ssim
    if config.datastes_type == 'not_real':
        print("Average PSNR = %.3f" % (total_psnr / itre), "  Averge SSIM = %.3f" % (total_ssim / itre))
        t = open("H:\\測試\\kernel123_back3_EP2\\新文字文件.txt","a+")
        t.write(config.txtoutdir + "  Average PSNR = %.3f" % (total_psnr / itre) + "  Averge SSIM = %.3f" % (total_ssim / itre) + "\n")
        t.close()
        TP += (total_psnr / itre)
        TS += (total_ssim / itre)

if config.datastes_type == 'not_real':
    print("20 Average PSNR = %.3f" % (TP / 20), "  20 Averge SSIM = %.3f" % (TS / 20))
    t = open("H:\\測試\\kernel123_back3_EP2\\新文字文件.txt","a+")
    t.write("20 Average PSNR = %.3f" % (TP / 20) + "  20 Averge SSIM = %.3f" % (TS / 20) + "\n")
    t.close()

print("Done.")

    # remaining_num -= config.train_batch_size
    
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


