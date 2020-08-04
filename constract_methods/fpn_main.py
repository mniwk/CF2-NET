# coding=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
from time import time
import os, sys
import numpy as np
import scipy.io as sio 

from loss import dice_bce_loss
from metric import dice_coeff

from network import FPN_Net 
from framework import MyFrame
import torch 
from torch.autograd import Variable 
from load_data import MyDataset
from torchvision import transforms, datasets
# from torchsummary import summary

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


# recall
def sensitive(gt, pr):
    tp = (pr*gt).sum(1).sum(1).sum(1)
    fn = (gt*(1-pr)).sum(1).sum(1).sum(1)
    score = (tp+0.01)/(tp+fn+0.01) 

    return score.mean()


def precision(gt, pr):
    tp = (pr*gt).sum(1).sum(1).sum(1)
    fp = ((1-gt)*pr).sum(1).sum(1).sum(1)
    score = (tp+0.01)/(tp+fp+0.01)

    return score.mean()


# recall1
def sensitive1(gt, pr):
    tp = (pr*gt).sum(1).sum(1).sum(1)
    fn = (gt*(1-pr)).sum(1).sum(1).sum(1)
    score = (tp+0.01)/(tp+fn+0.01) 

    return score


def precision1(gt, pr):
    tp = (pr*gt).sum(1).sum(1).sum(1)
    fp = ((1-gt)*pr).sum(1).sum(1).sum(1)
    score = (tp+0.01)/(tp+fp+0.01)

    return score


def accuracy(gt, pr):
    tp = (pr*gt).sum(1).sum(1).sum(1)
    tn = ((1-pr)*(1-gt)).sum(1).sum(1).sum(1)
    fp = (pr*(1-gt)).sum(1).sum(1).sum(1)
    fn = ((1-pr)*gt).sum(1).sum(1).sum(1)
    score = (tp+tn+0.01)/(tp+tn+fp+fn+0.01)

    return score.mean()


def specificity(gt, pr):
    tn = ((1-pr)*(1-gt)).sum(1).sum(1).sum(1)
    fp = (pr*(1-gt)).sum(1).sum(1).sum(1)
    score = (tn+0.01)/(tn+fp+0.01)

    return score.mean()


def f1_score(gt, pr):
    precision = precision1(gt, pr)
    sensitive = sensitive1(gt, pr)
    score = 2*(precision*sensitive)/(precision+sensitive)

    return score.mean()




def pro_in(data_fold):
    x_fold = np.array([data[0].reshape(1, 256, 256) for data in data_fold]).astype('float32')
    x_fold /= 255.
    y1_fold = np.array([data[1].reshape(1, 256, 256) for data in data_fold]).astype('float32')
    y1_fold /= 255.
    y2_fold = np.array([data[2].reshape(1, 256, 256) for data in data_fold]).astype('float32')
    y2_fold /= 255.
    x2_fold = np.array([data[3].reshape(1, 256, 256) for data in data_fold]).astype('float32')
    x2_fold /= 255.

    return (x_fold, y1_fold, y2_fold, x2_fold)


def get_onemat(path):
    list_fold = sio.loadmat(path)
    x_list = list_fold['data']
    sp_list = list_fold['seg_map']
    y1_list = list_fold['gt1']
    y2_list = list_fold['gt2']
    data_fold = list(zip(x_list, y1_list, y2_list, sp_list))
    # random.shuffle(data_fold)

    return data_fold

def data2():
    fold1 = get_onemat('../../mat/fold/data2_fold1.mat')
    fold2 = get_onemat('../../mat/fold/data2_fold2.mat')
    fold3 = get_onemat('../../mat/fold/data2_fold3.mat')
    fold4 = get_onemat('../../mat/fold/data2_fold4.mat')

    return (fold1, fold2, fold3, fold4)

def load_numpy(folds, i):
    test_data = folds[i]
    train_data = []
    for j in range(4):
        if j!=i:
            train_data = train_data+folds[j]
    print(len(test_data), len(train_data))
    # random.shuffle(test_data)
    (x_test, y1_test, _, _) = pro_in(test_data)
    # random.shuffle(train_data)
    (x_train, y1_train, _, _) = pro_in(train_data)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y1_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y1_test)

    return (x_train, y_train), (x_test, y_test)


def fpn_Net_Train(train_i=0):
    NAME = 'fold'+str(train_i+1)+'3fpn-Net'
    model = FPN_Net(1, 1).cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # model = FPN_Net(1, 1)
    # print(model)
    folds = data2()
    batch_size = 4
    (x_train, y_train), (x_test, y_test) = load_numpy(folds, train_i)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()
    no_optim = 0
    lr = 2e-4
    total_epoch=300
    train_epoch_best_loss=10000
    best_test_score = 0
    decay_factor = 1.5
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_com = dice_bce_loss()

    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        train_score = 0
        for img, mask in data_loader_iter:
            img = Variable(img.cuda(), volatile=False)
            mask = Variable(mask.cuda(), volatile=False)
            optimizer.zero_grad()
            pre = model.forward(img)
            loss = loss_com(mask, pre)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss
            train_score_b = dice_coeff(mask, pre, False)
            train_score += train_score_b*batch_size
        train_score /= x_train.size(0)
        train_epoch_loss /= len(data_loader_iter)

        test_img = Variable(x_test.cuda(), volatile=False)
        test_mask = Variable(y_test.cuda(), volatile=False)
        pre_test = model.forward(test_img)
        loss_test = loss_com(test_mask, pre_test)
        test_score = dice_coeff(test_mask, pre_test, False)

        if test_score > best_test_score:
            print('1. the dice score up to ', test_score, 'from ', best_test_score, 'saving the model', file=mylog, flush=True)
            print('1. the dice score up to ', test_score, 'from ', best_test_score, 'saving the model')
            best_test_score = test_score
            solver.save('./weights/' + NAME + '.th')

        print('********', file=mylog, flush=True)
        print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', train_epoch_loss, 'train_score:', train_score, file=mylog, flush=True)
        print('test_loss:', loss_test, 'test_dice_score: ', test_score, 'best_score is ', best_test_score, file=mylog, flush=True)

        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', train_epoch_loss, 'train_score:', train_score)
        print('test_loss:', loss_test, 'test_dice_score: ', test_score, 'best_score is ', best_test_score)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss

        if no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            if solver.old_lr > 5e-5:
                model.load_state_dict(torch.load('./weights/' + NAME + '.th'))
                lr /= decay_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

    print('Finish!', file=mylog, flush=True)
    print('Finish!')
    mylog.close()



def train(train_i=0):
	NAME = 'D2F5_fold'+str(train_i+1)+'_FPN.th'  
	print(NAME)

	batchsize = 4

	txt_train = 'N5fold'+str(train_i+1)+'_train.csv'
	txt_test = 'N5fold'+str(train_i+1)+'_test.csv'
	dataset_train = MyDataset(root='/home/wangke/ultrasound_data2/', txt_path=txt_train, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
	dataset_test = MyDataset(root='/home/wangke/ultrasound_data2/', txt_path=txt_test, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
	train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=2)
	test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=2, drop_last=True)

	mylog = open('logs/'+NAME+'.log', 'w')


	# model = FPN_Net(1, 1)
	# summary(model)

	slover = MyFrame(FPN_Net, dice_bce_loss, 2e-4)

	total_epoch = 100
	no_optim = 0
	train_epoch_best_loss = 10000
	best_test_score = 0
	for epoch in range(1, total_epoch+1):
		data_loader_iter = iter(train_loader)
		data_loader_test = iter(test_loader)
		train_epoch_loss = 0
		index = 0

		tic = time()

		train_score = 0
		for img, mask in data_loader_iter:
			slover.set_input(img, mask)
			train_loss, pred = slover.optimize()
			train_score += dice_coeff(mask, pred.cpu().data, False)
			train_epoch_loss +=train_loss 
			index +=1

		test_sen = 0
		test_ppv = 0
		test_score = 0
		test_acc = 0
		test_spe = 0
		test_f1s = 0
		for img, mask in data_loader_test:
			slover.set_input(img, mask)
			pre_mask, _ = slover.test_batch()
			test_score += dice_coeff(mask, pre_mask, False)
			test_sen += sensitive(mask, pre_mask)
			test_ppv += precision(mask, pre_mask)
			test_acc += accuracy(mask, pre_mask)
			test_spe += specificity(mask, pre_mask)
			test_f1s += f1_score(mask, pre_mask)

		test_sen /= len(data_loader_test)
		test_ppv /= len(data_loader_test)
		test_score /= len(data_loader_test)
		test_acc /= len(data_loader_test)
		test_spe /= len(data_loader_test)
		test_f1s /= len(data_loader_test)

		if test_score>best_test_score:
			print('1. the dice score up to ', test_score, 'from ', best_test_score, 'saving the model', file=mylog, flush=True)
			print('1. the dice score up to ', test_score, 'from ', best_test_score, 'saving the model')
			best_test_score = test_score
			slover.save('./weights/'+NAME+'.th')

		train_epoch_loss = train_epoch_loss/len(data_loader_iter)
		train_score = train_score/len(data_loader_iter)
		print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', train_epoch_loss.cpu().data.numpy(), 'train_score:', train_score, file=mylog, flush=True)
		print('test_dice_loss: ', test_score, 'test_sen: ', test_sen.numpy(), 'test_ppv: ', test_ppv.numpy(), 'test_acc: ', test_acc.numpy(), 'test_spe: ', test_spe.numpy(), 'test_f1s: ', test_f1s.numpy(), 'best_score is ', best_test_score, file=mylog, flush=True)
		
		print('********')
		print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', train_epoch_loss.cpu().data.numpy(), 'train_score:', train_score)
		print('test_dice_loss: ', test_score, 'test_sen: ', test_sen.numpy(), 'test_ppv: ', test_ppv.numpy(), 'test_acc: ', test_acc.numpy(), 'test_spe: ', test_spe.numpy(), 'test_f1s: ', test_f1s.numpy(), 'best_score is ', best_test_score)

		if train_epoch_loss >= train_epoch_best_loss:
			no_optim +=1
		else:
			no_optim =0
			train_epoch_best_loss = train_epoch_loss

	print('Finish!', file=mylog, flush=True)
	print('Finish!')
	mylog.close()






if __name__ == '__main__':
    # for i in range(4):
    i = 1
    print('training for fold'+str(i+1))
    train(train_i=i)
    print('training for fold'+str(i+1)+'finished')


