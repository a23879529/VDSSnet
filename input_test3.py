# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def train_data():

	source_video = "F:\\NYU-2"
	hazy_video = "H:\\NYU_v2_hazy"

	source_image = 'H:\\image'
	hazy_image = 'H:\\data'

	output = []
	fuck = 0

	for dirs in os.listdir(hazy_video):
		files = []
		# number = random.randint(1, 5)		
		for file in os.listdir(source_video + '\\' + dirs):
			name = file.split('.')[0]
			back = '.' + file.split('.')[1]
			files.append([int(name)])
		files.sort()
		# print(len(files))
		# fuck += len(files)
		for number in range(1, 6):
			for i in range(len(files)):
				root = source_video + '\\' + dirs + '\\'
				hazy = hazy_video + '\\' + dirs + '\\' + 'haze_' + str(number) + '_'
				if i == 0:
					output.append([root + str(files[i][0]) + back, hazy + str(files[i][0]) + back, hazy + str(files[i+1][0]) + back, hazy + str(files[i+2][0]) + back])
				elif i == len(files)-1:
					output.append([root + str(files[i][0]) + back, hazy + str(files[i-2][0]) + back, hazy + str(files[i-1][0]) + back, hazy + str(files[i][0]) + back])
				else:
					output.append([root + str(files[i][0]) + back, hazy + str(files[i-1][0]) + back, hazy + str(files[i][0]) + back, hazy + str(files[i+1][0]) + back])
	# print(fuck)
		# print(files)
		# break
	# 	files = []
	# for file in os.listdir(hazy_image):
	# 	# print(file)
	# 	files.append([file])
	# for i in range(len(files)):
	# 	# root = source + '\\'
	# 	# hazy_image = hazy + '\\'
	# 	# print(files[i][0])
	# 	if files[i][0] != 'Thumbs.db':
	# 		name = files[i][0].split('_')[0] + '_' + files[i][0].split('_')[1]
	# 		back = '.' + files[i][0].split('.')[-1]
	# 		output.append([source_image + '\\' + name + back, hazy_image + '\\' + files[i][0], hazy_image + '\\' + files[i][0], hazy_image + '\\' + files[i][0]])
	# print(len(output))
	random.shuffle(output)
	# print(output)
	A = np.array(output)
	t_num = round(A.shape[0]//10*(9.9))



	t_set = A[:t_num][:]
	v_set = A[t_num:][:]
	# print("t_set num: ", len(t_set), "  v_set num: ", len(v_set))
	return t_set, v_set
	

def test_data(source, hazy, datastes_type, input_type, txtoutdir):

	output = []

	if datastes_type == 'not_real':
		if input_type == 'video':
			# for dirs in os.listdir(hazy):
			files = []
			# dirs = 'computer_lab_0002'
			dirs = txtoutdir
			# number = random.randint(1, 5)		
			for file in os.listdir(source + '\\' + dirs):
				name = file.split('.')[0]
				back = '.' + file.split('.')[1]
				files.append([int(name)])
			files.sort()
			# print(back)
			# fuck += len(files)
			# for number in range(1, 6):
			for i in range(len(files)):
				root = source + '\\' + dirs + '\\'
				# hazy_traget = hazy + '\\' + dirs + '\\' + 'haze_' + str(number) + '_'
				hazy_traget = hazy + '\\' + dirs + '\\'
				# print(root , hazy)
				if i == 0:
					output.append([root + str(files[i][0]) + back, hazy_traget + str(files[i][0]) + '_hazed' + back, hazy_traget + str(files[i+1][0]) + '_hazed' + back, hazy_traget + str(files[i+2][0]) + '_hazed' + back])
					# print(hazy_traget + str(files[i][0]) + back)
				elif i == len(files)-1:
					output.append([root + str(files[i][0]) + back, hazy_traget + str(files[i-2][0]) + '_hazed' + back, hazy_traget + str(files[i-1][0]) + '_hazed' + back, hazy_traget + str(files[i][0]) + '_hazed' + back])
				else:
					output.append([root + str(files[i][0]) + back, hazy_traget + str(files[i-1][0]) + '_hazed' + back, hazy_traget + str(files[i][0]) + '_hazed' + back, hazy_traget + str(files[i+1][0]) + '_hazed' + back])
			# print(np.array(output).shape)
			# break

		elif input_type == 'singel_image': #for SOT indoor
			files = []
			# number = random.randint(1, 2)
			for file in os.listdir(hazy):
				# print(file)
				files.append([file])
			for i in range(len(files)):
				# root = source + '\\'
				# hazy_image = hazy + '\\'
				name = files[i][0].split('_')[0]
				# print(name)
				# back = '.' + files[i][0].split('.')[-1]
				output.append([source + '\\' + name + '.png', hazy + '\\' + files[i][0]])
				# print(source + '\\' + name + '.png', hazy + '\\' + files[i][0])
				# print(np.array(output).shape)
			# print(output)

	elif datastes_type == 'real':
		if input_type == 'video':
			# for dirs in os.listdir(hazy):
			files = []
			dirs = txtoutdir
			for file in os.listdir(hazy + '\\' + dirs):
				name = file.split('.')[0]
				back = '.' + file.split('.')[-1]
				files.append([int(name)])
			files.sort()
			# print(files)
			hazy_image = hazy + '\\' + dirs + '\\'
			for i in range(len(files)):
				if i == 0:
					output.append([hazy_image + str(files[i][0]) + back, hazy_image + str(files[i+1][0]) + back, hazy_image + str(files[i+2][0]) + back])
				elif i == len(files)-1:
					output.append([hazy_image + str(files[i-2][0]) + back, hazy_image + str(files[i-1][0]) + back, hazy_image + str(files[i][0]) + back])
				else:
					output.append([hazy_image + str(files[i-1][0]) + back, hazy_image + str(files[i][0]) + back, hazy_image + str(files[i+1][0]) + back])

		elif input_type == 'singel_image': #for SOT outdoor
			for dirs in os.listdir(hazy):
				files = []
				number = random.randint(1, 2)
				for file in os.listdir(hazy + '\\' + dirs):
					files.append([file])
				for i in range(len(files)):
					hazy_image = hazy + '\\' + dirs + '\\' + 'haze_' + str(number) + '_'
					output.append([hazy_image + files[i][0]])

	return output

# train_data()