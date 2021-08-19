# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import glob

def video(name):
	number = 1
	img_array = []
	#b = sorted(glob.glob('*.png'))
	b = sorted(glob.glob('*.jpg'), key=os.path.getmtime) #已排序
	for filename in b:
	    #print(filename)
	    img = cv2.imread(filename)
	    height, width, layers = img.shape
	    size = (width,height)
	    img_array.append(img)
	 
	out = cv2.VideoWriter(name + '.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
	 
	for i in range(len(img_array)):
	    # print("len=",len(img_array))
	    # print("name=",img_array[i])
	    out.write(img_array[i])
	out.release()
	print("Done!Done!")

video("Driving")
