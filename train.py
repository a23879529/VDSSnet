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
import numpy as np
from VDSSnet import VDSSNet
from torchvision import transforms
from data import HazeDataset

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dehaze_net = VDSSNet().to(device)
    dehaze_net.apply(weights_init)

    data_transform = transforms.Compose([
        transforms.Resize([480, 640]),
        transforms.ToTensor()
    ])

    train_dataset = HazeDataset(config.ori_data_path, config.haze_data_path, data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    val_dataset = HazeDataset(config.val_ori_data_path, config.val_haze_data_path, data_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True,
                                             num_workers=config.num_workers, pin_memory=True)

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dehaze_net.train()

    for epoch in range(config.num_epochs):
        print("----------------  epoch: ", epoch)
        for iteration, (img_orig, img_haze) in enumerate(train_loader):

            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()
            #print (img_haze.shape)
            clean_image = dehaze_net(img_haze)

            loss = criterion(clean_image, img_orig)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

        # Validation Stage
        for iter_val, (img_orig, img_haze) in enumerate(val_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)

            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                         config.sample_output_folder + str(iter_val + 1) + ".jpg")

        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
	
	# Input Parameters
    parser.add_argument('--ori_data_path', type=str, default='C:\\Users\\田鼠\\Desktop\\1',  help='Origin image path')
    parser.add_argument('--haze_data_path', type=str, default='C:\\Users\\田鼠\\Desktop\\2',  help='Haze image path')
    parser.add_argument('--val_ori_data_path', type=str, default='C:\\Users\\田鼠\\Desktop\\3',  help='Validation origin image path')
    parser.add_argument('--val_haze_data_path', type=str, default='C:\\Users\\田鼠\\Desktop\\4',  help='Validation haze image path')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001) # default 0.0001
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--train_batch_size', type=int, default=8) #原本是8
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="./snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="./samples/")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)