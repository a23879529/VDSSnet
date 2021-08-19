# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import random

root = "F:\\NYU-2"
target = "H:\\NYU_v2_hazy"

count = 0

for dirs in os.listdir(target):
    for i, j, k in os.walk(root + '\\' + dirs):
        for count in range(len(k)):
            if not os.path.exists(target + '\\' + dirs + '\\haze_5_' + k[count]):
                # print(target + '\\'+ dirs + '\\' + k[count])
                # print(root + '\\' + i)
                img = cv2.imread(root + '\\' + dirs + '\\' + k[count]) # 需要处理的文件夹
                mask_img = cv2.imread(root + '\\' + dirs + '\\' + k[count])  # 需要处理的文件夹
                # print(type(img))
                if type(img) == type(None):
                    os.remove(root + '\\' + dirs + '\\' + k[count])
                else:
                    # 添加雾霾
                    # t : 透视率 0~1
                    # A : 大气光照
                    A = random.uniform(0.8, 1.0)
                    # A = 0.8
                    t = 0.1
                    image = img*t + A*255*(1-t)
                    cv2.imwrite(target + '\\' + dirs + '\\haze_1_' + k[count], image)
                    A = random.uniform(0.8, 1.0)
                    # A = 0.95
                    t = 0.3
                    image = img*t + A*255*(1-t)
                    cv2.imwrite(target + '\\' + dirs + '\\haze_2_' + k[count], image)
                    A = random.uniform(0.8, 1.0)
                    # A = 0.95
                    t = 0.5
                    image = img*t + A*255*(1-t)
                    cv2.imwrite(target + '\\' + dirs + '\\haze_3_' + k[count], image)
                    A = random.uniform(0.8, 1.0)
                    # A = 0.95
                    t = 0.7
                    image = img*t + A*255*(1-t)
                    cv2.imwrite(target + '\\' + dirs + '\\haze_4_' + k[count], image)
                    A = random.uniform(0.8, 1.0)
                    # A = 0.95
                    t = 0.9
                    image = img*t + A*255*(1-t)
                    cv2.imwrite(target + '\\' + dirs + '\\haze_5_' + k[count], image)
    count += 1