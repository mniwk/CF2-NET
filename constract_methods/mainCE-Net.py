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

from cenet import CE_Net_
from framework import MyFrame
from loss import dice_bce_loss
from metric import dice_coeff
from load_data import MyDataset
# from data import ImageFolder
# from Visualizer import Visualizer

import Constants 


# Please specify the ID of graphics cards that you want to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



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



def CE_Net_Train(train_i=0):

    NAME = 'fold'+str(i+1)+'_6CE-Net' + Constants.ROOT.split('/')[-1]

    solver = MyFrame(CE_Net_, dice_bce_loss, 2e-4)
    batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD #4

    # For different 2D medical image segmentation tasks, please specify the dataset which you use
    # for examples: you could specify "dataset = 'DRIVE' " for retinal vessel detection.

    txt_train = 'fold'+str(train_i+1)+'_train.csv'
    txt_test = 'fold'+str(train_i+1)+'_test.csv'
    dataset_train = MyDataset(txt_path=txt_train, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    dataset_test = MyDataset(txt_path=txt_test, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset, batchsize=batchsize, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset, batchsize=batchsize, shuffle=False, num_workers=2)

    # start the logging files
    mylog = open('logs/' + NAME + '.log', 'w')

    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH         # 300
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS         # 10000
    best_test_score = 0
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(train_loader)
        data_loader_test = iter(test_loader)
        train_epoch_loss = 0
        index = 0

        tic = time()

        # train
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss
            index = index + 1

        # test
        test_sen = 0
        test_ppv =0
        test_score = 0
        for img, mask in data_loader_test:
            solver.set_input(img, mask)
            pre_mask, _ = solver.test_batch()
            test_score += dice_coeff(y_test, pre_mask, False)
            test_sen += sensitive(y_test, pre_mask)
            # test_sen = test_sen.cpu().data.numpy()
            test_ppv += positivepv(y_test, pre_mask)
    # test_ppv = test_ppv.cpu().data.numpy()
        print(test_sen/len(data_loader_test), test_ppv/len(data_loader_test), test_score/len(data_loader_test))
        # solver.set_input(x_test, y_test)
        # pre_mask, _ = solver.test_batch()
        # test_score = dice_coeff(y_test, pre_mask, False)
        # test_sen = sensitive(y_test, pre_mask)
        # test_sen = test_sen.cpu().data.numpy()
        # test_ppv = positivepv(y_test, pre_mask)
        # test_ppv = test_ppv.cpu().data.numpy()
        # print('111111111111111111111',type(test_score))

        # # show the original images, predication and ground truth on the visdom.
        # show_image = (img + 1.6) / 3.2 * 255.
        # viz.img(name='images', img_=show_image[0, :, :, :])
        # viz.img(name='labels', img_=mask[0, :, :, :])
        # viz.img(name='prediction', img_=pred[0, :, :, :])
        
        if test_score > best_test_score:
            print('1. the dice score up to ', test_score, 'from ', best_test_score, 'saving the model')
            best_test_score = test_score
            solver.save('./weights/' + NAME + '.th')

        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        # print(mylog, '********')
        print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', train_epoch_loss.cpu().data.numpy(), file=mylog, flush=True)
        print('test_dice_loss: ', test_score, 'test_sen: ', test_sen, 'test_ppv: ', test_ppv, 'best_score is ', best_test_score, file=mylog, flush=True)
        
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', train_epoch_loss.cpu().data.numpy())
        print('test_dice_score: ', test_score, 'test_sen: ', test_sen, 'test_ppv: ', test_ppv, 'best_score is ', best_test_score)
        # print('train_loss:', train_epoch_loss)
        # print('SHAPE:', Constants.Image_size)
        
        

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            # solver.save('./weights/' + NAME + '.th')
        # if no_optim > Constants.NUM_EARLY_STOP:
        #     print(mylog, 'early stop at %d epoch' % epoch)
        #     print('early stop at %d epoch' % epoch)
        #     break
        if no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            if solver.old_lr > 5e-4:
                solver.load('./weights/' + NAME + '.th')
                solver.update_lr(1.5, factor=True, mylog=mylog)

    print('Finish!', file=mylog, flush=True)
    print('Finish!')
    mylog.close()


if __name__ == '__main__':
    for i in range(5):
        print('training for fold'+str(i+1))
        CE_Net_Train(train_i=i)
        print('training for fold'+str(i+1)+'finished')