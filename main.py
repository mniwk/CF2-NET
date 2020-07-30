# coding=utf-8

import os, sys
import numpy as np
from convs import build_res_unet, build_level_net, build_merge_unet, build_res2_unet, build_level2_net
import pandas as pd 
import cv2 
import keras
from loss_history import LossHistory
from loss_history2 import LossHistory2
from loss_history3 import LossHistory3
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from loss_metric.losses import bce_jaccard_loss, bce_dice_loss, edge_loss, bce0_dice_loss
from loss_metric.metrics import iou_score, dice_score, get_iou_score, ofuse_pixel_error
from keras.models import model_from_json 
from convs import stages, stages_none
from keras.layers import Input, Lambda, Average, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator 
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.utils import np_utils
from keras.utils import multi_gpu_model 
import tensorflow as tf 
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
sess = tf.Session(config=config)
set_session(sess)

from keras.models import load_model


import random


def img_image_generator(path_img, path_lab, batch_size, data_list, isshuffle):
	while True:
            # '../ultrasound_data1/train_list.csv'
            file_list = pd.read_csv(data_list, sep=',',usecols=[1]).values.tolist()
            file_list = [i[0] for i in file_list]
            if isshuffle:random.shuffle(file_list)
            cnt = 0
            X = []
            Y1 = []
            for file_i in file_list:
                  # print((path_img+file_i))
                  x = cv2.imread(path_img+file_i, cv2.IMREAD_GRAYSCALE)
                  # print(x)
                  x = x.astype('float32')
                  x /= 255.
                  y = cv2.imread(path_lab+file_i, cv2.IMREAD_GRAYSCALE)
                  y = y.astype('float32')
                  y /= 255.
                  X.append(x.reshape(256, 256, 1))
                  Y1.append(y.reshape(256, 256, 1))
                  cnt += 1
                  if cnt == batch_size:
                        cnt = 0
                        yield (np.array(X), [np.array(Y1), np.array(Y1)])
                        X = []
                        Y1 = []


def img2_image_generator(path_img, path_lab, batch_size, data_list, isshuffle):
	while True:
		# '../ultrasound_data1/train_list.csv'
		file_list = pd.read_csv(data_list, sep=',',usecols=[1]).values.tolist()
		file_list = [i[0] for i in file_list]
		if isshuffle:random.shuffle(file_list)
		cnt = 0
		X = []
		Y = []
		for file_i in file_list:
			x = cv2.imread(path_img+'/'+file_i, cv2.IMREAD_GRAYSCALE)
			x = x.astype('float32')
			x /= 255.
			y = cv2.imread(path_lab+'/'+file_i, cv2.IMREAD_GRAYSCALE)
			y = y.astype('float32')
			y /= 255.
			X.append(x.reshape(256, 256, 1))
			Y.append(y.reshape(256, 256, 1))
			cnt += 1
			if cnt == batch_size:
				cnt = 0
				yield (np.array(X), [np.array(Y)])
				X = []
				Y = []


def img3_image_generator(path_img, path_slic, path_lab, path_edge, batch_size, data_list, isshuffle):
      while True:
            # '../ultrasound_data1/train_list.csv'
            file_list = pd.read_csv(data_list, sep=',',usecols=[1]).values.tolist()
            file_list = [i[0] for i in file_list]
            if isshuffle:random.shuffle(file_list)
            cnt = 0
            X = []
            X1 = []
            Y = []
            Y1 = []
            for file_i in file_list:
                  x = cv2.imread(path_img+'/'+file_i, cv2.IMREAD_GRAYSCALE)
                  x = x.astype('float32')
                  x /= 255.
                  x1 = cv2.imread(path_slic+'/'+file_i, cv2.IMREAD_GRAYSCALE)
                  x1 = x1.astype('float32')
                  x1 /= 255.
                  y = cv2.imread(path_lab+'/'+file_i, cv2.IMREAD_GRAYSCALE)
                  y = y.astype('float32')
                  y /= 255.
                  y1 = cv2.imread(path_edge+'/'+file_i, cv2.IMREAD_GRAYSCALE)
                  y1 = y1.astype('float32')
                  y1 /= 255.
                  X.append(x.reshape(256, 256, 1))
                  X1.append(x1.reshape(256, 256, 1))
                  Y.append(y.reshape(256, 256, 1))
                  Y1.append(y1.reshape(256, 256, 1))
                  cnt += 1
                  if cnt == batch_size:
                        cnt = 0
                        yield ([np.array(X), np.array(X1)], [np.array(Y), np.array(Y), np.array(Y1)])
                        X = []
                        Y = []
                        X1 = []
                        Y1 = []


def img4_image_generator(path_img, path_lab, path_edge, batch_size, data_list, isshuffle):
      while True:
            # '../ultrasound_data1/train_list.csv'
            file_list = pd.read_csv(data_list, sep=',',usecols=[1]).values.tolist()
            file_list = [i[0] for i in file_list]
            if isshuffle:random.shuffle(file_list)
            cnt = 0
            X = []
            Y = []
            Y1 = []
            for file_i in file_list:
                  x = cv2.imread(path_img+'/'+file_i, cv2.IMREAD_GRAYSCALE)
                  x = x.astype('float32')
                  x /= 255.
                  y = cv2.imread(path_lab+'/'+file_i, cv2.IMREAD_GRAYSCALE)
                  y = y.astype('float32')
                  y /= 255.
                  y1 = cv2.imread(path_edge+'/'+file_i, cv2.IMREAD_GRAYSCALE)
                  y1 = y1.astype('float32')
                  y1 /= 255.
                  X.append(x.reshape(256, 256, 1))
                  Y.append(y.reshape(256, 256, 1))
                  Y1.append(y1.reshape(256, 256, 1))
                  cnt += 1
                  if cnt == batch_size:
                        cnt = 0
                        yield (np.array(X), [np.array(Y), np.array(Y), np.array(Y1)])
                        X = []
                        Y = []
                        Y1 = []


class ParallelModelCheckpoint(ModelCheckpoint):
      def __init__(self, model, filepath, monitor='val_out_final_score', verbose=0,\
            save_best_only=False, save_weights_only=False, mode='auto', period=1):
            self.single_model = model 
            super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)
      
      def set_model(self, model):
            super(ParallelModelCheckpoint, self).set_model(self.single_model)




def FSP2_net():
      input1_1 = Input(shape=(256, 256, 1))
      input1_2 = Input(shape=(256, 256, 1))
      input1 = concatenate([input1_1, input1_2], axis=-1, name='f_cc1')
      output1 = build_res_unet(input1)
      model1 = Model(inputs=[input1_1, input1_2], outputs=output1)
      stage_conv, stage_mid, stage_deconv = stages(model1)
      out_final, out_edge = build_level_net(stage_conv, stage_mid, stage_deconv)
      model = Model(inputs=[input1_1, input1_2], outputs=[output1, out_final, out_edge])
      model_parallel = multi_gpu_model(model, gpus=2)
      opetimizer = Adam(lr=0.0006, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=True)
      model_parallel.compile(optimizer=opetimizer, loss={'output1':bce_dice_loss, 'out_final':bce_dice_loss, 'out_edge':edge_loss},\
		 loss_weights={'output1':1, 'out_final':1, 'out_edge':0.1}, metrics={'output1':dice_score, \
		 'out_final':dice_score})

      order_name = 'Data2_later/D2F2CASE_5SLIC'
      i = 2
      batch_size = 4
      model_checkpoint = ParallelModelCheckpoint(model=model, filepath='model/'+order_name+'_fold'+str(i+1)+'_beste.h5', monitor='val_out_final_score',verbose=1, save_best_only=True, mode='max')
      histories = LossHistory2()
      callbacks_list = [histories, model_checkpoint]

      generator_train = img3_image_generator('../ultrasound_data2/process_pic/', '../ultrasound_data2/process_slic/', '../ultrasound_data2/process_lab/',\
             '../ultrasound_data2/process_edge3/', batch_size, '../ultrasound_data2/N5fold'+str(i+1)+'_train.csv', True)
      train_list = pd.read_csv('../ultrasound_data2/N5fold'+str(i+1)+'_train.csv', sep=',',usecols=[1]).values.tolist()
      train_list = [i[0] for i in train_list]
      lens_train = len(train_list)

      generator_test = img3_image_generator('../ultrasound_data2/process_pic/', '../ultrasound_data2/process_slic/', '../ultrasound_data2/process_lab/',\
             '../ultrasound_data2/process_edge3/', batch_size, '../ultrasound_data2/N5fold'+str(i+1)+'_test.csv', False)
      test_list = pd.read_csv('../ultrasound_data2/N5fold'+str(i+1)+'_test.csv', sep=',',usecols=[1]).values.tolist()
      test_list = [i[0] for i in test_list]
      lens_test = len(test_list)
      print(lens_train, lens_test)
      model_json = model.to_json()
      open('model/'+order_name+'_fold'+str(i+1)+'seconde.json', 'w').write(model_json)
      model_parallel.fit_generator(generator_train, steps_per_epoch=lens_train//batch_size, epochs=100, callbacks=callbacks_list, validation_data=generator_test, validation_steps=lens_test//batch_size)

      histories.loss_plot('epoch', order_name+'_fold'+str(i+1)+'.png', order_name+'_fold'+str(i+1))
      model.load_weights('model/'+order_name+'_fold'+str(i+1)+'_beste.h5')
      model.save_weights('model/'+order_name+'_fold'+str(i+1)+'_beste.h5', overwrite=True)



if __name__ == '__main__':
      FSP2_net()