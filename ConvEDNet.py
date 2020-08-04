# -*- coding=utf-8 -*-


from keras.layers import Conv2DTranspose, Conv2D, UpSampling2D, BatchNormalization, Activation, Add, concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Multiply, Dense, Reshape, Dropout, Average, Maximum, LeakyReLU
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda, GlobalAveragePooling2D, Multiply, Subtract, LocallyConnected2D, Conv2DTranspose
from keras.utils.vis_utils import plot_model 


def one_layer1(x, kernel_num, kernel_size, name):
	x = Conv2D(filters=kernel_num, kernel_size=kernel_size, padding='same', name=name+'_cv1')(x)
	x = BatchNormalization(name=name+'_bn1')(x)
	x = LeakyReLU(alpha=0.2, name=name+'_ac1')(x)
	
	x = Conv2D(filters=kernel_num, kernel_size=kernel_size, padding='same', name=name+'_cv2')(x)
	x = BatchNormalization(name=name+'_bn2')(x)
	x = LeakyReLU(alpha=0.2, name=name+'_ac2')(x)

	return x

def one_layer2(x, kernel_num, kernel_size, name):
	x = Conv2D(filters=kernel_num, kernel_size=kernel_size, padding='same', name=name+'_cv1')(x)
	x = BatchNormalization(name=name+'_bn1')(x)
	x = LeakyReLU(alpha=0.2, name=name+'_ac1')(x)
	
	x = Conv2D(filters=kernel_num, kernel_size=kernel_size, padding='same', name=name+'_cv2')(x)
	x = BatchNormalization(name=name+'_bn2')(x)
	x = LeakyReLU(alpha=0.2, name=name+'_ac2')(x)
	
	x = Conv2D(filters=kernel_num, kernel_size=1, padding='same', name=name+'_cv3')(x)
	x = BatchNormalization(name=name+'_bn3')(x)
	x = LeakyReLU(alpha=0.2, name=name+'_ac3')(x)

	return x

def encoder(input1):
	# VGG
	x_1 = one_layer1(input1, 64, 3, 'en_lay1')
	x = MaxPooling2D((2,2))(x_1)
	x_2 = one_layer1(x, 128, 3, 'en_lay2')
	x = MaxPooling2D((2,2))(x_2)
	x_3 = one_layer2(x, 256, 3, 'en_lay3')
	x = MaxPooling2D((2,2))(x_3)
	x_4 = one_layer2(x, 512, 3, 'en_lay4')
	x = MaxPooling2D((2,2))(x_4)
	x_5 = one_layer2(x, 512, 3, 'en_lay5')
	# 6、7
	x = MaxPooling2D((2,2))(x_5)
	x = Conv2D(filters=1024, kernel_size=5, padding='same', name='en_lay61_cv1')(x)
	x = BatchNormalization(name='en_lay61_bn1')(x)
	x = LeakyReLU(alpha=0.2, name='en_lay61_ac1')(x)
	x = MaxPooling2D((2,2))(x)
	x = Conv2D(filters=1024, kernel_size=3, padding='same', name='en_lay62_cv1')(x)
	x = BatchNormalization(name='en_lay62_bn1')(x)
	x = LeakyReLU(alpha=0.2, name='en_lay62_ac1')(x)

	# x = GlobalAveragePooling2D(name='global_pool')(x)
	# x = Dense(4096)(x)
	# # x = Conv2D(filters=1024, kernel_size=5, padding='same', name='en_lay7_cv1')(x)
	# x = BatchNormalization(name='en_lay7_bn1')(x)
	# x = LeakyReLU(alpha=0.2, name='en_lay7_ac1')(x)
	# x = Reshape((1, 1, 4096))(x)
	
	return x, (x_1, x_2, x_3, x_4, x_5)



def decoder(x):
	# x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
	x = Conv2D(filters=1024, kernel_size=3, padding='same', name='de_lay61_cv1')(x)
	x = BatchNormalization(name='de_lay61_bn1')(x)
	x = LeakyReLU(alpha=0.2, name='de_lay61_ac1')(x)
	x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
	x = Conv2D(filters=1024, kernel_size=5, padding='same', name='de_lay62_cv1')(x)
	x = BatchNormalization(name='de_lay62_bn1')(x)
	x = LeakyReLU(alpha=0.2, name='de_lay62_ac1')(x)

	x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
	x = one_layer2(x, 512, 3, 'de_lay5')
	x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
	x = one_layer2(x, 512, 3, 'de_lay4')
	x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
	x = one_layer2(x, 256, 3, 'de_lay3')
	x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
	x = one_layer2(x, 128, 3, 'de_lay2')
	x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
	x = one_layer2(x, 64, 3, 'de_lay1')

	x = Conv2D(1, 1, activation='sigmoid', name='out_final')(x)

	return x


def boundary_layer(x, filters, kernel_size, strides, name):

	x = Conv2D(filters=filters, kernel_size=1, padding='same', name='nest_cn'+name)(x)
	x = BatchNormalization(name='nest_bn1'+name)(x)
	x = LeakyReLU(alpha=0.2, name='nest_ac1'+name)(x)
	x = Conv2DTranspose(filters=filters, kernel_size=1, strides=1, name='nest_dcn'+name)(x)
	x = BatchNormalization(name='nest_bn2'+name)(x)
	x = LeakyReLU(alpha=0.2, name='nest_ac2'+name)(x)

	return x

def boundary_net(x_mid):
	filters = 32
	x_1 = boundary_layer(x_mid[0], filters, 1, 1, '1')
	x_2 = boundary_layer(x_mid[1], filters, 4, 2, '2')
	x_3 = boundary_layer(x_mid[2], filters, 8, 4, '3')
	x_4 = boundary_layer(x_mid[3], filters, 16, 8, '4')
	x_5 = boundary_layer(x_mid[4], filters, 32, 16, '5')

	x_5 = UpSampling2D(size=(16,16), data_format='channels_last', interpolation='bilinear', name='up_x5')(x_5)
	x_4 = UpSampling2D(size=(8,8), data_format='channels_last', interpolation='bilinear', name='up_x4')(x_4)
	x_3 = UpSampling2D(size=(4,4), data_format='channels_last', interpolation='bilinear', name='up_x3')(x_3)
	x_2 = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='up_x2')(x_2)
	nest_feature = concatenate([x_1, x_2, x_3, x_4, x_5], axis=-1, name='nest_cc')
	out_nest = Conv2D(1, 1, activation='sigmoid', name='out_edge')(nest_feature)

	return out_nest



def create_model():
	input1 = Input(shape=(256,256,1))
	x, x_mid = encoder(input1)
	out_nest = boundary_net(x_mid)
	out_final = decoder(x)
	model = Model(inputs=input1, outputs=[out_final, out_nest])

	return model 