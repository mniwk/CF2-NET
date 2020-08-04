# coding=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
from time import time
import os, sys
import numpy as np
import scipy.io as sio 

from losses import calc_loss
from metric import dice_coeff
from Models import NestedUNet, AttU_Net
from Models import U_Net
import matplotlib.pyplot as plt 

import torch 
from torch.autograd import Variable 
from load_data import MyDataset 
from framework import MyFrame

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
    num = list_fold['num']
    data_fold = list(zip(x_list, y1_list, y2_list, sp_list, num))
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


def Net_Train(train_i=0):
    # NAME = 'fold'+str(train_i+1)+'_25ATT-UNet'
    NAME = 'fold'+str(train_i+1)+'_25NEST-UNet'
    mylog = open('logs/' + NAME + '.log', 'w')
    print(NAME)
    print(NAME, file=mylog, flush=True)
    # model = AttU_Net(img_ch=1, output_ch=1).cuda()
    model = NestedUNet(in_ch=1, out_ch=1).cuda()
    # print(model)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # model = FPN_Net(1, 1)
    # print(model)
    folds = data2()
    test_data=folds[train_i]
    batch_size1 = 4
    batch_size = 2
    (x_train, y_train), (x_test, y_test) = load_numpy(folds, train_i)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size1,
        shuffle=True,
        num_workers=4)

    # data_test = torch.utils.data.TensorDataset(x_test, y_test)
    # loader_test = torch.utils.data.DataLoader(
    #     data_test,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=4)
  
    no_optim = 0
    lr = 2e-4
    total_epoch=250
    train_epoch_best_loss=10000
    best_test_score = 0
    decay_factor = 1.5
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    for epoch in range(1, total_epoch + 1):
        tic = time()
        data_loader_iter = iter(data_loader)
        # data_test_iter = iter(loader_test)
        train_epoch_loss = 0
        train_score = 0
        test_epoch_loss = 0
        test_score = 0
        test_sen = 0
        test_ppv = 0

        for img, mask in data_loader_iter:
            img = Variable(img.cuda(), volatile=False)
            mask = Variable(mask.cuda(), volatile=False)
            optimizer.zero_grad()
            pre = model.forward(img)
            loss = calc_loss(pre, mask)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.data
            train_score_b = dice_coeff(mask, pre, False)
            train_score += train_score_b.data*batch_size1
        train_score /= x_train.size(0)
        train_score = train_score.cpu().data.numpy()
        train_epoch_loss /= len(data_loader_iter)
        train_epoch_loss = train_epoch_loss.cpu().data.numpy()
        # print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', train_epoch_loss, 'train_score:', train_score)
        
        with torch.no_grad():
            # for img, mask in data_test_iter:
            img = Variable(x_test.cuda(), volatile=True)
            mask = Variable(y_test.cuda(), volatile=True)
            pre = model.forward(img)
            test_epoch_loss = calc_loss(pre, mask)
            # test_epoch_loss += loss.data
            test_score = dice_coeff(mask, pre, False)
            # test_score += test_score_b.data*batch_size
            pre[pre>0.5]=1
            pre[pre<=0.5]=0
            test_sen = sensitive(mask, pre)
            # test_sen += test_sen_b.data*batch_size
            test_ppv = positivepv(mask, pre)
            # test_ppv += test_ppv_b.data*batch_size

            
        # test_score /= x_test.size(0)
        test_score = test_score.cpu().data.numpy()
        # test_sen /= x_test.size(0)
        test_sen = test_sen.cpu().data.numpy()
        # test_ppv /= x_test.size(0)
        test_ppv = test_ppv.cpu().data.numpy()
        # test_epoch_loss /= len(data_test_iter)
        test_epoch_loss = test_epoch_loss.cpu().data.numpy()

        # x_test = Variable(x_test.cuda(), volatile=True)
        # pre_test = model.forward(x_test).cpu().data
        # loss_test = calc_loss(y_test, pre_test)
        # # loss_test = loss_test.cpu().data.numpy()
        # test_score = dice_coeff(y_test, pre_test, False)
        # # test_score = test_score.cpu().data.numpy()
        print('********', file=mylog, flush=True)
        print('epoch:', epoch, train_i, ' time:', int(time() - tic), 'train_loss:', train_epoch_loss, 'train_score:', train_score, end='  ', file=mylog, flush=True)
        print('test_loss:', test_epoch_loss, 'test_dice_score: ', test_score, 'test_sen: ', test_sen, 'test_ppv: ', test_ppv, 'best_score is ', best_test_score, file=mylog, flush=True)

        print('********')
        print('epoch:', epoch, train_i, ' time:', int(time() - tic), 'train_loss:', train_epoch_loss, 'train_score:', train_score,  end='  ')
        print('test_loss:', test_epoch_loss, 'test_dice_score: ', test_score, 'test_sen: ', test_sen, 'test_ppv: ', test_ppv, 'best_score is ', best_test_score)

        if test_score > best_test_score:
            print('1. the dice score up to ', test_score, 'from ', best_test_score, 'saving the model', file=mylog, flush=True)
            print('1. the dice score up to ', test_score, 'from ', best_test_score, 'saving the model')
            best_test_score = test_score
            torch.save(model, './weights/' + NAME + '.pkl')
            if best_test_score>0.75:
                with torch.no_grad():
                    for test in test_data:
                        img = test[0].reshape(1, 1, 256, 256).astype('float32')
                        img = torch.from_numpy(img)
                        img = Variable(img.cuda())
                        pre = model.forward(img).cpu().data.numpy()
                        plt.imsave('../../model/fold/png_constract/'+test[-1]+'_fold_z5_nest.png', pre[0,0,:,:], cmap='gray')


        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss

        if no_optim > 10:
            if lr < 1e-7:
                break
            if lr > 1e-5:
                # model.load_state_dict(torch.load('./weights/' + NAME + '.th'))
                lr /= decay_factor
                print ('update learning rate: %f -> %f' % (lr*decay_factor, lr), file=mylog, flush=True)
                print ('update learning rate: %f -> %f' % (lr*decay_factor, lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

    print('Finish!', file=mylog, flush=True)
    print('Finish!')
    mylog.close()



def Net2_Train(train_i=0):
    # NAME = 'fold'+str(train_i+1)+'_1ATT_New-UNet'
    NAME = 'fold'+str(train_i+1)+'_1NEST_New-UNet'
    mylog = open('logs/' + NAME + '.log', 'w')
    print(NAME)
    print(NAME, file=mylog, flush=True)

    txt_train = 'fold'+str(train_i+1)+'_train.csv'
    txt_test = 'fold'+str(train_i+1)+'_test.csv'
    dataset_train = MyDataset(txt_path=txt_train, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    dataset_test = MyDataset(txt_path=txt_test, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=2)

    slover = MyFrame(NestedUNet, dice_bce_loss, 2e-4)
    batch_size = 4
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
            test_score += dice_coeff(y_test, pre_mask, False)
            test_sen += sensitive(y_test, pre_mask)
            test_ppv += positivepv(y_test, pre_mask)
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

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim +=1
        else:
            no_optim =0
            train_epoch_best_loss = train_epoch_loss 
    print('Finish!', file=mylog, flush=True)
    print('Finish!')
    mylog.close()

def Net3_Train(train_i=0):
    NAME = 'fold'+str(train_i+1)+'_1UNet'
    mylog = open('logs/' + NAME + '.log', 'w')
    print(NAME)
    print(NAME, file=mylog, flush=True)

    txt_train = 'fold'+str(train_i+1)+'_train.csv'
    txt_test = 'fold'+str(train_i+1)+'_test.csv'
    dataset_train = MyDataset(txt_path=txt_train, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    dataset_test = MyDataset(txt_path=txt_test, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=2)

    slover = MyFrame(NestedUNet, dice_bce_loss, 2e-4)
    batch_size = 4
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
            test_score += dice_coeff(y_test, pre_mask, False)
            test_sen += sensitive(y_test, pre_mask)
            test_ppv += positivepv(y_test, pre_mask)
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

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim +=1
        else:
            no_optim =0
            train_epoch_best_loss = train_epoch_loss 
    print('Finish!', file=mylog, flush=True)
    print('Finish!')
    mylog.close()

if __name__ == '__main__':
    # for i in range(1,2):
    i = 0
    print('training for fold'+str(i+1))
    Net3_Train(i)
    print('training for fold'+str(i+1)+'finished')