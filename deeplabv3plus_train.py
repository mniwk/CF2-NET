# coding = utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
from time import time
import os, sys
import numpy as np
import scipy.io as sio
# import random 

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from torchvision import transforms, datasets

# from cenet import CE_Net_
# from RDAU_net import RDAUNet 
from models.deeplabv3_plus import DeepLabV3Plus
from framework import MyFrame
from loss import dice_bce_loss
from metric import dice_coeff
from load_data import MyDataset
# from data import ImageFolder
# from Visualizer import Visualizer

# import Constants 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# recall
def sensitive(gt, pr):
	tp = (pr*gt).sum(1).sum(1).sum(1)
	fn = (gt*(1-pr)).sum(1).sum(1).sum(1)
	score = (tp+0.01)/(tp+fn+0.01) 

	return score.mean()


def positivepv(gt, pr):
	tp = (pr*gt).sum(1).sum(1).sum(1)
	fp = ((1-gt)*pr).sum(1).sum(1).sum(1)
	score = (tp+0.01)/(tp+fp+0.01)

	return score.mean()


def train(train_i=0):
	NAME = 'fold'+str(train_i+1)+'_deeplabv3_plus.th'
	# slover = MyFrame(FSP_Net, dice_bce_loss, 2e-4)
	slover = MyFrame(DeepLabV3Plus, dice_bce_loss, 5e-4)

	batchsize = 4

	txt_train = 'fold'+str(train_i+1)+'_train.csv'
	txt_test = 'fold'+str(train_i+1)+'_test.csv'
	dataset_train = MyDataset(txt_path=txt_train, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
	dataset_test = MyDataset(txt_path=txt_test, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
	train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=2)
	test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=2)

	mylog = open('logs/'+NAME+'.log', 'w')

	total_epoch = 100
	no_optim = 0
	train_epoch_best_loss = 10000
	best_test_score = 0
	for epoch in range(1, total_epoch+1):
		data_loder_iter = iter(train_loader)
		data_loder_test = iter(test_loader)
		train_epoch_loss = 0
		index = 0

		tic = time()

		train_score = 0
		for img, mask in data_loder_iter:
			slover.set_input(img, mask)
			train_loss, pred = slover.optimize()
			train_score += dice_coeff(mask, pred, False)
			train_epoch_loss +=train_loss 
			index +=1

		test_sen = 0
		test_ppv = 0
		test_score = 0
		for img, mask in data_loder_test:
			slover.set_input(img, mask)
			pre_mask, _ = slover.test_batch()
			test_score += dice_coeff(mask, pre_mask, False)
			test_sen += sensitive(mask, pre_mask)
			test_ppv += positivepv(mask, pre_mask)
		test_sen /= len(data_loder_test)
		test_ppv /= len(data_loder_test)
		test_score /= len(data_loder_test)

		if test_score>best_test_score:
			print('1. the dice score up to ', test_score, 'from ', best_test_score, 'saving the model', file=mylog, flush=True)
			print('1. the dice score up to ', test_score, 'from ', best_test_score, 'saving the model')
			best_test_score = test_score
			slover.save('./weights/'+NAME+'.th')

		train_epoch_loss = train_epoch_loss/len(data_loder_iter)
		train_score = train_score/len(data_loder_iter)
		print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', train_epoch_loss.cpu().data.numpy(), 'train_score:', train_score, file=mylog, flush=True)
		print('test_dice_loss: ', test_score, 'test_sen: ', test_sen, 'test_ppv: ', test_ppv, 'best_score is ', best_test_score, file=mylog, flush=True)

		print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', train_epoch_loss.cpu().data.numpy(), 'train_score:', train_score)
		print('test_dice_loss: ', test_score, 'test_sen: ', test_sen, 'test_ppv: ', test_ppv, 'best_score is ', best_test_score, file=mylog)

		if train_epoch_loss >= train_epoch_best_loss:
			no_optim +=1
		else:
			no_optim =0
			train_epoch_best_loss = train_epoch_loss 
	print('Finish!', file=mylog, flush=True)
	print('Finish!')
	mylog.close()


if __name__ == '__main__':
	# for i in range(5):
	print('train for fold'+str(i+1))
	train(train_i=i)
	print('train for fold'+str(i+1)+'finished')