# -*- coding: utf-8 -*-
"""
@author: Wei Yi Yiai
"""

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import logging
import numpy as np
import pytorch_ssim
from PIL import Image
from VDSSnet_3_old import VDSSNet
from torchvision import transforms
from tools import FFT
from math import log10

from input_test3 import train_data


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print("CUDA")
    else:
        print("CPU")

    dehaze_net = VDSSNet().to(device)
    dehaze_net.apply(weights_init)

    criterion = nn.SmoothL1Loss().cuda()
    criterion2 = nn.MSELoss(reduction='mean').cuda()

    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    checkpoint = torch.load('./snapshots/f2.pth')
    dehaze_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # dehaze_net.eval()

    dehaze_net.train()

    logger = logging.getLogger(__name__)  #添加log
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler('./log/output.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('Start training.')

    for epoch in range(config.num_epochs):

        train_dataset, val_dataset = train_data()

        print("train_dataset num: ", len(train_dataset), "  val_dataset num: ", len(val_dataset))
        logger.info("train_dataset num: " + str(len(train_dataset)) + "  val_dataset num: " + str(len(val_dataset)))

        epoch_start = time.time()
        iteration_start = time.time()

        A = np.array(train_dataset)

        if A.shape[0] % config.train_batch_size != 0:
            itre = A.shape[0] // config.train_batch_size + 1
        else:
            itre = A.shape[0] // config.train_batch_size

        remaining_num = A.shape[0]
        total_count = 0


        for iteration in range(itre):


            if remaining_num - config.train_batch_size < 0:
                bz = remaining_num
            else:
                bz = config.train_batch_size

            temp1 = torch.tensor([])
            temp2 = torch.tensor([])
            final_gt = torch.tensor([])
            final_haze = torch.tensor([])



            for batch in range(bz):

                gt = Image.open(train_dataset[total_count][0]).convert("RGB")
                img1 = Image.open(train_dataset[total_count][1]).convert("RGB")
                img2 = Image.open(train_dataset[total_count][2]).convert("RGB")
                img3 = Image.open(train_dataset[total_count][3]).convert("RGB")

                my_transforms = transforms.Compose([
                transforms.Resize([240, 320]),
                transforms.ToTensor()
                ])

                gt = my_transforms(gt)
                img1 = my_transforms(img1)
                img2 = my_transforms(img2)
                img3 = my_transforms(img3)

                if batch == 0:
                    final_gt = gt
                    final_haze = torch.cat((img1, img2, img3), 0)
                    final_gt = final_gt.unsqueeze(0)
                    final_haze = final_haze.unsqueeze(0)
                else:
                    temp1 = gt
                    temp1 = temp1.unsqueeze(0)
                    final_gt = torch.cat([final_gt, temp1], 0)

                    temp2 = torch.cat((img1, img2, img3), 0)
                    temp2 = temp2.unsqueeze(0)
                    final_haze = torch.cat([final_haze, temp2], 0)

                total_count+=1


            img_orig = final_gt.cuda()
            img_haze = final_haze.cuda()

            clean_image = dehaze_net(img_haze)


            loss = 1.0 * criterion(clean_image, img_orig) #L1_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                
                mse = criterion2(clean_image, img_orig)
                psnr = 10 * log10(1 / mse)
                #psnr = pytorch_ssim.ssim(clean_image, img_orig)
                ssim = pytorch_ssim.ssim(clean_image, img_orig)
                ssim = ssim.item()
                #print("PSNR: %.3f" % psnr, "  SSIM: %.3f" % ssim)

                iteration_end = time.time()
                # print("bz%.3f "%aaa)
                # aaa = 0
                #print("Loss at iteration", iteration + 1, ":", loss.item())
                print("Epoch: ", epoch, "  Loss at iteration", iteration + 1, ":%.3f" % loss.item(),"  PSNR: %.3f" % psnr, "  SSIM: %.3f" % ssim, "  運行時間為: %.3f 秒" % (iteration_end - iteration_start))
                logger.info("Epoch: " + str(epoch) + "  Loss at iteration" + str(iteration + 1) + ":%.3f" % loss.item() + "  PSNR: %.3f" % psnr + "  SSIM: %.3f" % ssim + "  運行時間為: %.3f 秒" % (iteration_end - iteration_start))
                iteration_start = time.time()

            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch + 1) + '.pth')
                logger.info("save Epoch" + str(epoch + 1) + '.pth')

                # print(final_gt.shape, final_haze.shape)
            remaining_num -= config.train_batch_size
            # print("----------------  iteration: ", iteration)

        # Validation Stage

        A = np.array(val_dataset)
        # print(A.shape)
        if A.shape[0] % config.val_batch_size != 0:
            itre = A.shape[0] // config.val_batch_size + 1
        else:
            itre = A.shape[0] // config.val_batch_size

        remaining_num = A.shape[0]
        total_count = 0

        for iter_val in range(itre):
            # print("----------------  iteration: ", iteration)

            if remaining_num - config.val_batch_size < 0:
                bz = remaining_num
            else:
                bz = config.val_batch_size

            temp1 = torch.tensor([])
            temp2 = torch.tensor([])
            final_gt = torch.tensor([])
            final_haze = torch.tensor([])

            for batch in range(bz):

                gt = Image.open(val_dataset[total_count][0]).convert("RGB")
                img1 = Image.open(val_dataset[total_count][1]).convert("RGB")
                img2 = Image.open(val_dataset[total_count][2]).convert("RGB")
                img3 = Image.open(val_dataset[total_count][3]).convert("RGB")

                my_transforms = transforms.Compose([
                transforms.Resize([240, 320]),
                transforms.ToTensor()
                ])

                gt = my_transforms(gt)
                img1 = my_transforms(img1)
                img2 = my_transforms(img2)
                img3 = my_transforms(img3)

                if batch == 0:
                    final_gt = gt
                    final_haze = torch.cat((img1, img2, img3), 0)
                    final_gt = final_gt.unsqueeze(0)
                    final_haze = final_haze.unsqueeze(0)
                    # print(final.shape, "0")
                else:
                    temp1 = gt
                    temp1 = temp1.unsqueeze(0)
                    final_gt = torch.cat([final_gt, temp1], 0)

                    temp2 = torch.cat((img1, img2, img3), 0)
                    temp2 = temp2.unsqueeze(0)
                    # print(temp1.shape, "temp1")
                    final_haze = torch.cat([final_haze, temp2], 0)
                    # print(final.shape, "final")
                # count+=1
                total_count+=1

            img_orig = final_gt.cuda()
            img_haze = final_haze.cuda()
            clean_image = dehaze_net(img_haze)

            temp1 = []
            temp2 = []
            for i in range(bz):
                if i == 0:
                    temp1 = torch.stack((img_haze[i][3], img_haze[i][4], img_haze[i][5]), 0)
                    temp1 = temp1.unsqueeze(0)
                else:
                    temp2 = torch.stack((img_haze[i][3], img_haze[i][4], img_haze[i][5]), 0)
                    temp2 = temp2.unsqueeze(0)
                    temp1 = torch.cat([temp1, temp2], 0)

            img_haze = temp1

            mse = criterion2(clean_image, img_orig)
            psnr = 10 * log10(1 / mse)
            #psnr = pytorch_ssim.ssim(clean_image, img_orig)
            }, config.snapshots_folder + "dehazer.pth")

        epoch_end = time.time()
        print("Epoch 運行時間為: %.3f 秒" % (epoch_end - epoch_start))
        logger.info("Epoch 運行時間為: %.3f 秒" % (epoch_end - epoch_start))
            ssim = pytorch_ssim.ssim(clean_image, img_orig)
            ssim = ssim.item()
            #print("PSNR: %.3f" % psnr, "  SSIM: %.3f" % ssim)
            # print("bz%.3f "%aaa)
            # aaa = 0
            #print("Loss at iteration", iteration + 1, ":", loss.item())
            print("PSNR: %.3f" % psnr, "  SSIM: %.3f" % ssim)

            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                         config.sample_output_folder + str(iter_val + 1) + ".jpg", nrow = 4)
            remaining_num -= config.val_batch_size

        # torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")
        torch.save({
            'epoch':epoch,
            'model_state_dict':dehaze_net.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss,

    logger.info("End training.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
	
	# Input Parameters
    parser.add_argument('--ori_data_path', type=str, default='C:\\Users\\ian\\Desktop\\image',  help='Origin image path')
    parser.add_argument('--haze_data_path', type=str, default='C:\\Users\\ian\\Desktop\\data',  help='Haze image path')
    parser.add_argument('--val_ori_data_path', type=str, default='C:\\Users\\ian\\Desktop\\V_GT',  help='Validation origin image path')
    parser.add_argument('--val_haze_data_path', type=str, default='C:\\Users\\ian\\Desktop\\V_h',  help='Validation haze image path')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001) # default 0.0001
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=8) #原本是8
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=50) #原本是10
    parser.add_argument('--snapshot_iter', type=int, default=200) #原本是200
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)