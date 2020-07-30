# coding=utf-8


from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Activation, Add, concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Multiply, Dense, Reshape, Dropout, Average, Maximum
import keras.backend as K
from keras.layers import Lambda, GlobalAveragePooling2D, Multiply, Subtract, LocallyConnected2D

# 需修改
def sse(x, filters, name):
	xo = Conv2D(filters=filters//4, kernel_size=1, kernel_initializer='he_normal', name=name+'_as1')(x)
	xo = BatchNormalization(name=name+'_asbn1')(xo)
	xo = Activation('relu')(xo)
	xo = Conv2D(filters=1, kernel_size=1, kernel_initializer='he_normal', name=name+'_as2')(x)
	xo = BatchNormalization(name=name+'_asbn2')(xo)
	xo = Activation('sigmoid')(xo)
	xo = Multiply()([xo, x])
	xo = Add()([xo, x])
	return xo

def cse(x, filters, name):
	se_shape = (1, 1, filters) if K.image_data_format() == 'channels_last' else (filters, 1, 1)
	xo = GlobalAveragePooling2D()(x)
	xo = Reshape(se_shape)(xo)
	xo = Conv2D(filters, 1, kernel_initializer='he_normal', name=name+'ac1')(xo)
	xo = BatchNormalization(name=name+'_acbn2')(xo)
	xo = Activation('sigmoid')(xo)
	xo = Multiply()([xo, x])
	xo = Add()([xo, x])
	return xo


import tensorflow as tf 

def reshape1(x, chose):
	batch_size, height, width, C = x.get_shape()
	batch_size, height, width, C = batch_size, int(height), int(width), int(C)
	x = K.reshape(x, (-1, height*width, C))
	if chose==0:
		x = tf.transpose(x, [0,2,1])
		return x
	else:
		return x

def reshape2(x, height):
	batch_size, hw, C = x.get_shape()
	batch_size, hw, C = batch_size, int(hw), int(C)
	return K.reshape(x, (-1, height, hw//height, C))

def mydot(x):
	x1, x2 = x
	return K.batch_dot(x1, x2)

def mytranspose(x):
	return tf.transpose(x, [0,2,1])

# 修改后的新注意力机制函数
# 第一次训练得到：结果并不好，尝试是因为batch_size的原因，batch_size最好设置为4，但是由于本修改的函数太耗内存，只能改成128*128的
def Position_attention(x):
	batch_size, height, width, C = x.get_shape()
	batch_size, height, width, C = batch_size, int(height), int(width), int(C)
	# print(batch_size, height, width, C//8)
	proj_query = Conv2D(C//8, 1, kernel_initializer='he_normal')(x)
	proj_query = Lambda(reshape1, output_shape=(C//8, height*width), arguments={'chose':0})(proj_query)
	proj_key = Conv2D(C//8, 1, kernel_initializer='he_normal')(x)
	proj_key = Lambda(reshape1, output_shape=(height*width, C//8), arguments={'chose':1})(proj_key)
	energy = Lambda(mydot, output_shape=(height*width, height*width))([proj_key, proj_query])
	attention = Activation(activation='softmax')(energy)
	attention = Lambda(mytranspose, output_shape=(height*width, height*width))(attention)
	proj_value = Conv2D(C, 1, kernel_initializer='he_normal')(x)
	proj_value = Lambda(reshape1, output_shape=(height*width, C), arguments={'chose':1})(proj_value)
	out = Lambda(mydot, output_shape=(height*width, C))([attention, proj_value])
	out = Lambda(reshape2, output_shape=(height, width, C), arguments={'height':height})(out)
	out = Add()([out, x])
	print(out.shape)

	return out 


def Channel_attention(x):
	batch_size, height, width, C = x.shape
	batch_size, height, width, C = batch_size, int(height), int(width), int(C)
	proj_query = Lambda(reshape1, output_shape=(C, height*width), arguments={'chose':0})(x)
	# print(proj_query.shape)
	proj_key = Lambda(reshape1, output_shape=(height*width, C), arguments={'chose':1})(x)
	energy = Lambda(mydot, output_shape=(C, C))([proj_query, proj_key])
	attention = Activation(activation='softmax')(energy)
	proj_value = Lambda(reshape1, output_shape=(height*width, C), arguments={'chose':1})(x)

	out = Lambda(mydot, output_shape=(height*width, C))([proj_value, attention])
	out = Lambda(reshape2, output_shape=(height, width, C), arguments={'height':height})(out)
	out = Add()([out, x])
	print(out.shape)

	return out





def res_block(x, nb_filters, name):
	res_path = Conv2D(filters=nb_filters[0], kernel_size=3, padding='same', strides=1, kernel_initializer = 'he_normal')(x)
	res_path = BatchNormalization()(res_path)
	res_path_1 = Activation(activation='relu', name=name+'_1')(res_path)
	res_path = Conv2D(filters=nb_filters[1], kernel_size=3, padding='same', strides=1, kernel_initializer = 'he_normal')(res_path_1)
	res_path = BatchNormalization()(res_path)
	res_path_2 = Activation(activation='relu', name=name+'_2')(res_path)
	conv = Conv2D(filters=nb_filters[1], kernel_size=3, padding='same', strides=1, kernel_initializer = 'he_normal')(res_path_2)
	res_path = BatchNormalization()(conv)
	res_path = Activation(activation='relu', name=name+'_3')(res_path)

	# onestage = get_stage((res_path_1, res_path_2, res_path), int(nb_filters[0]/2), name)

	return res_path, res_path

# x---(x1, x2)
def get_stage2(x, num, name):
	main_path_1 = Conv2D(filters=num, kernel_size=1, padding='same', kernel_initializer = 'he_normal')(x[0])
	main_path_1 = BatchNormalization()(main_path_1)

	main_path_2 = Conv2D(filters=num, kernel_size=1, padding='same', kernel_initializer = 'he_normal')(x[1])
	main_path_2 = BatchNormalization()(main_path_2)

	main_path = Add()([main_path_1, main_path_2])
	main_path = Activation(activation='relu', name=name)(main_path)

	return main_path

def res_block2(x, nb_filters, name):
	res_path = Conv2D(filters=nb_filters[0], kernel_size=3, padding='same', strides=1, kernel_initializer = 'he_normal', name=name+'_1')(x)
	# res_path = BatchNormalization()(res_path)
	# res_path_1 = Activation(activation='relu')(res_path)
	# res_path = Conv2D(filters=nb_filters[1], kernel_size=3, padding='same', strides=1, kernel_initializer = 'he_normal')(res_path_1)
	res_path = BatchNormalization(name=name+'_bn')(res_path)
	res_path = Activation(activation='relu')(res_path)

	# onestage = get_stage2((res_path_1, res_path), nb_filters[0], name)

	return res_path, res_path



def encoder(x):
	# ker_nums = [45, 125, 245, 405]
	ker_nums = [64, 128, 256, 512]
	# first branching to decoder
	main_path, conv1 = res_block(x, [ker_nums[0], ker_nums[0]], name='conv1')
	main_path = MaxPooling2D(pool_size=(2, 2))(main_path)

	main_path, conv2 = res_block(main_path, [ker_nums[1], ker_nums[1]], name='conv2')
	main_path = MaxPooling2D(pool_size=(2, 2))(main_path)

	main_path, conv3 = res_block(main_path, [ker_nums[2], ker_nums[2]], name='conv3')
	main_path = MaxPooling2D(pool_size=(2, 2))(main_path)

	main_path, conv4 = res_block(main_path, [ker_nums[3], ker_nums[3]], name='conv4')
	main_path = MaxPooling2D(pool_size=(2, 2))(main_path)


	return (conv1, conv2, conv3, conv4), main_path



def decoder(x, from_encoder):
	ker_nums = [64, 128, 256, 512]
	main_path = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear')(x)
	main_path = concatenate([main_path, from_encoder[3]], axis=3)
	# from_encoder3 = Conv2D(filters=ker_nums[3]*2, kernel_size=(1, 1), kernel_initializer = 'he_normal')(from_encoder[3])
	# main_path = Add()([main_path, from_encoder3])
	# main_path = BatchNormalization()(main_path)
	# main_path = Activation(activation='relu')(main_path)
	deconv1, _ = res_block(main_path, [ker_nums[3], ker_nums[3]], name='deconv1')

	main_path = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear')(deconv1)
	main_path = concatenate([main_path, from_encoder[2]], axis=3)
	# from_encoder2 = Conv2D(filters=ker_nums[2]*2, kernel_size=(1, 1), kernel_initializer = 'he_normal')(from_encoder[2])
	# main_path = Add()([main_path, from_encoder2])
	# main_path = BatchNormalization()(main_path)
	# main_path = Activation(activation='relu')(main_path)
	deconv2, _ = res_block(main_path, [ker_nums[2], ker_nums[2]], name='deconv2')

	main_path = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear')(deconv2)
	main_path = concatenate([main_path, from_encoder[1]], axis=3)
	# from_encoder1 = Conv2D(filters=ker_nums[1]*2, kernel_size=(1, 1), kernel_initializer = 'he_normal')(from_encoder[1])
	# main_path = Add()([main_path, from_encoder1])
	# main_path = BatchNormalization()(main_path)
	# main_path = Activation(activation='relu')(main_path)
	deconv3, _ = res_block(main_path, [ker_nums[1], ker_nums[1]], name='deconv3')

	main_path = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear')(deconv3)
	main_path = concatenate([main_path, from_encoder[0]], axis=3)
	# from_encoder0 = Conv2D(filters=ker_nums[0]*2, kernel_size=(1, 1), kernel_initializer = 'he_normal')(from_encoder[0])
	# main_path = Add()([main_path, from_encoder0])
	# main_path = BatchNormalization()(main_path)
	# main_path = Activation(activation='relu')(main_path)
	main_path, _ = res_block(main_path, [ker_nums[0], ker_nums[0]], name='deconv4')

	return main_path, (deconv1, deconv2, deconv3, main_path)


def build_res_unet(inputs):

	(conv1, conv2, conv3, conv4), main_path = encoder(inputs)
	# main_path = Dropout(0.5)(main_path)
	main_path, _ = res_block(main_path, [1024, 1024], name='mid')
	# main_path = Dropout(0.5)(main_path)
	main_path, _ = decoder(main_path, from_encoder=[conv1, conv2, conv3, conv4])

	# main_path = Conv2D(filters=32, kernel_size=1, padding='same', kernel_initializer = 'he_normal')(main_path)
	# main_path = BatchNormalization()(main_path)
	# main_path = Activation(activation='relu')(main_path)
	# main_pathw1 = sse(main_path, name='first')
	# main_pathw2 = cse(main_path, 64, name='first')
	# main_path = Add()([main_pathw1, main_pathw2])
	output1 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid', name='output1', kernel_initializer = 'he_normal')(main_path)

	return output1

def build_res2_unet(inputs):

	(conv1, conv2, conv3, conv4), main_path = encoder(inputs)
	main_path, _ = res_block(main_path, [1024, 1024], name='mid')
	main_path, (deconv1, deconv2, deconv3, main_path) = decoder(main_path, from_encoder=[conv1, conv2, conv3, conv4])
	     
	return main_path, (deconv1, deconv2, deconv3, main_path)


def build_merge_unet(inputs):
	(conv1, conv2, conv3, conv4), main_path = encoder(inputs)
	# main_path = Dropout(0.5)(main_path)
	main_path, _ = res_block(main_path, [1024, 1024], name='mid')
	main_path, (deconv1, deconv2, deconv3, deconv4) = decoder(main_path, from_encoder=[conv1, conv2, conv3, conv4])
	deconv1 = Conv2D(48, 1, kernel_initializer = 'he_normal', name='mer_decon1')(deconv1)
	deconv1 = BatchNormalization(name='mer_decon1_bn')(deconv1)
	deconv1 = Activation('relu')(deconv1)

	deconv2 = Conv2D(48, 1, kernel_initializer = 'he_normal', name='mer_decon2')(deconv2)
	deconv2 = BatchNormalization(name='mer_decon2_bn')(deconv2)
	deconv2 = Activation('relu')(deconv2)

	deconv3 = Conv2D(48, 1, kernel_initializer = 'he_normal', name='mer_decon3')(deconv3)
	deconv3 = BatchNormalization(name='mer_decon3_bn')(deconv3)
	deconv3 = Activation('relu')(deconv3)

	deconv4 = Conv2D(48, 1, kernel_initializer = 'he_normal', name='mer_decon4')(deconv4)
	deconv4 = BatchNormalization(name='mer_decon4_bn')(deconv4)
	deconv4 = Activation('relu')(deconv4)

	deconv1 = UpSampling2D(size=(8,8), data_format='channels_last', interpolation='bilinear', name='decon_map1')(deconv1)
	deconv2 = UpSampling2D(size=(4,4), data_format='channels_last', interpolation='bilinear', name='decon_map2')(deconv2)
	deconv3 = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='decon_map3')(deconv3)
	output1 = concatenate([deconv1, deconv2, deconv3, deconv4], axis=-1, name='output_cc')
	output1 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid', name='output1', kernel_initializer = 'he_normal')(output1)

	return output1



def tiny_unet(x, kernel_num, name):
	convt1, stage_convt1 = res_block2(x, [kernel_num, kernel_num], name='stage_conv_tiny'+name)
	main_path = MaxPooling2D(pool_size=(2, 2), name='stage_mp'+name)(convt1)

	main_path, stage_convt2 = res_block2(main_path, [kernel_num*2, kernel_num*2], name='stage_M_tiny'+name)
	main_path = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='tiny_up'+name)(main_path)

	main_path = concatenate([main_path, convt1], axis=3, name='tiny_cc'+name)
	main_path, stage_decont1 = res_block2(main_path, [kernel_num, kernel_num], name='stage_deconv_tiny'+name)

	main_path2 = Conv2D(filters=48, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same', name='stage'+name)(main_path)
	main_path2 = BatchNormalization(name='bn1'+name)(main_path2)
	main_path2 = Activation(activation='relu')(main_path2)

	# main_show = Conv2D(1, 1, activation='sigmoid', name='tiny_show_'+name, kernel_initializer='he_normal')(main_path)

	# main_path2 = concatenate([main_path2, main_show], name='tiny_cc_'+name)

	return (stage_convt1, stage_convt2, stage_decont1), main_path, main_path2


# x--(conv4, 5, deconv1)
def CFF_block(x, kernel_num, name):
	convtrans = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name=name+'up1')(x[1])
	convtrans = Conv2D(kernel_num, 3, dilation_rate=2, kernel_initializer='he_normal', padding='same', name=name+'_1')(convtrans)
	convtrans = BatchNormalization(name=name+'_bn1')(convtrans)
	conca1 = Conv2D(kernel_num//2, 1, kernel_initializer='he_normal', name=name+'_2')(x[0])
	conca1 = BatchNormalization(name=name+'_bn2')(conca1)
	conca2 = Conv2D(kernel_num//2, 1, kernel_initializer='he_normal', name=name+'_3')(x[2])
	conca2 = BatchNormalization(name=name+'_bn3')(conca2)
	conca = concatenate([conca1, conca2], axis=-1, name=name+'_cc')
	out = Add()([convtrans, conca])
	out = Activation(activation='relu', name=name)(out)

	return out

def edge_feature(x, num, name):
	x1 = Conv2D(num, 1, kernel_initializer='he_normal', name=name+'_1', padding='same')(x)
	x1_1 = BatchNormalization(name=name+'_bn1')(x1)
	x1 = Activation(activation='relu')(x1_1)
	x1 = Conv2D(num, 3, kernel_initializer='he_normal', dilation_rate=3, name=name+'_2', padding='same')(x1)
	x1 = BatchNormalization(name=name+'_bn2')(x1)
	x1_2 = Activation(activation='relu')(x1)
	# x1 = Add(name=name+'_add1')([x1_1, x1_2])
	# x1 = concatenate([x1_1, x1], axis=-1, name=name+'_cc1')
	# x1 = BatchNormalization(name=name+'_bn2')(x1)
	x1 = Activation(activation='relu')(x1)
	x1 = Conv2D(num//8, 1, kernel_initializer='he_normal', name=name+'_3', padding='same')(x1_2)
	x1 = BatchNormalization(name=name+'_bn3')(x1)
	x1 = Activation(activation='relu')(x1)

	x2 = Conv2D(16, 1, kernel_initializer='he_normal', name=name+'_4', padding='same')(x1_2)
	x2 = BatchNormalization(name=name+'_bn4')(x2)
	x2 = Activation(activation='relu')(x2)

	return x1, x2 


# x---(x1, x2, x3)
def get_stage(x, num, name, isneed):
	main_path_1 = Conv2D(filters=num, kernel_size=1, padding='same', kernel_initializer = 'he_normal', name=name+'_1')(x[0])
	main_path_1 = BatchNormalization(name=name+'bn1')(main_path_1)
	main_path_1a = Conv2D(filters=num, kernel_size=3, dilation_rate=2, padding='same', kernel_initializer = 'he_normal', name=name+'_2')(x[0])
	main_path_1a = BatchNormalization(name=name+'bn1a')(main_path_1a)

	main_path_2 = Conv2D(filters=num, kernel_size=1, padding='same', kernel_initializer = 'he_normal', name=name+'_3')(x[1])
	main_path_2 = BatchNormalization(name=name+'bn2')(main_path_2)
	main_path_2a = Conv2D(filters=num, kernel_size=3, dilation_rate=4, padding='same', kernel_initializer = 'he_normal', name=name+'_4')(x[1])
	main_path_2a = BatchNormalization(name=name+'bn2a')(main_path_2a)

	main_path_3 = Conv2D(filters=num, kernel_size=1, padding='same', kernel_initializer = 'he_normal', name=name+'_5')(x[2])
	main_path_3 = BatchNormalization(name=name+'bn3')(main_path_3)
	main_path_3a = Conv2D(filters=num, kernel_size=3, dilation_rate=6, padding='same', kernel_initializer = 'he_normal', name=name+'_6')(x[2])
	main_path_3a = BatchNormalization(name=name+'bn3a')(main_path_3a)   

	if isneed:
		main_path = Add()([main_path_1a, main_path_2a, main_path_3a])
		# main_path = concatenate([main_path_1a, main_path_2a, main_path_3a], axis=-1, name=name+'_cc111')  
	else:
		main_path = Add()([main_path_1, main_path_2, main_path_3])

	main_path = Activation(activation='relu', name=name)(main_path)

	return main_path

def stage_one(model, name1, num, name2, isneed=True):
	x1 = model.get_layer(name1+'_1').output
	x2 = model.get_layer(name1+'_2').output
	x3 = model.get_layer(name1+'_3').output
	stage = get_stage((x1, x2, x3), num, name2, isneed)

	return stage

def stages(model):
	# conv部分
	conv1 = stage_one(model, 'conv1', 32, 'stage_conv1', True)
	conv2 = stage_one(model, 'conv2', 64, 'stage_conv2', True)
	conv3 = stage_one(model, 'conv3', 128, 'stage_conv3', True)
	conv4 = stage_one(model, 'conv4', 256, 'stage_conv4', True)
	mid = stage_one(model, 'mid', 512, 'stage_mid', True)
	deconv1 = stage_one(model, 'deconv1', 256, 'stage_deconv1', True)
	deconv2 = stage_one(model, 'deconv2', 128, 'stage_deconv2', True)
	deconv3 = stage_one(model, 'deconv3', 64, 'stage_deconv3', True)
	deconv4 = stage_one(model, 'deconv4', 32, 'stage_deconv4', True)

	return (conv1, conv2, conv3, conv4), mid, (deconv1, deconv2, deconv3, deconv4)


def stages_none(model):
	# conv部分
	conv1 = stage_one(model, 'conv1', 32, 'stage_conv1', False)
	conv2 = stage_one(model, 'conv2', 64, 'stage_conv2', False)
	conv3 = stage_one(model, 'conv3', 128, 'stage_conv3', False)
	conv4 = stage_one(model, 'conv4', 256, 'stage_conv4', False)
	mid = stage_one(model, 'mid', 512, 'stage_mid', False)
	deconv1 = stage_one(model, 'deconv1', 256, 'stage_deconv1', False)
	deconv2 = stage_one(model, 'deconv2', 128, 'stage_deconv2', False)
	deconv3 = stage_one(model, 'deconv3', 64, 'stage_deconv3', False)
	deconv4 = stage_one(model, 'deconv4', 32, 'stage_deconv4', False)

	return (conv1, conv2, conv3, conv4), mid, (deconv1, deconv2, deconv3, deconv4)


def sp_onestage(x, kernel_num, kernel_num2, num_n, name):
	# x1 = Conv2D(kernel_num, 1, kernel_initializer='he_normal', name=name+'_'+str(num_n)+'11', padding='same')(x)
	# x1 = BatchNormalization(name=name+'_bn'+str(num_n)+'11')(x1)
	# x1 = Activation(activation='relu')(x1)
	
	x1_1 = Conv2D(kernel_num2, 1, kernel_initializer='he_normal', name=name+'_'+str(num_n)+'3', padding='same')(x)
	x1_1 = BatchNormalization(name=name+'_bn'+str(num_n)+'3')(x1_1)
	x1_1 = Activation(activation='relu')(x1_1)

	return x1_1, x1_1

def sp_net(x, name):
	kernel_nums = [32, 64, 128, 256]
	x1_1, x1 = sp_onestage(x, kernel_nums[0], 4, 1, name)
	x1 = MaxPooling2D(pool_size=(2, 2), name=name+'mp_1')(x1)
	x1_2, x1 = sp_onestage(x1, kernel_nums[1], 8, 2, name)
	x1 = MaxPooling2D(pool_size=(2, 2), name=name+'mp_2')(x1)
	x1_3, x1 = sp_onestage(x1, kernel_nums[2], 16, 3, name)
	x1 = MaxPooling2D(pool_size=(2, 2), name=name+'mp_3')(x1)
	x1_4, x1 = sp_onestage(x1, kernel_nums[3], 32, 4, name)

	return (x1_1, x1_2, x1_3, x1_4)

from keras.layers import Input 
from keras.models import Model 


def build_level_net(stage_conv, stage_mid, stage_deconv):

	# (sp_1, sp_2, sp_3, sp_4)= sp_net(input2, 'sp_net')

	outCFF1 = CFF_block((stage_conv[3], stage_mid, stage_deconv[0]), 256, 'level1_CFF1')
	# outCFF11 = concatenate([outCFF1, sp_4], axis=3, name='cc1_1')
	edge1, edge11 = edge_feature(outCFF1, 256, name='edge1')
	outCFF1 = concatenate([outCFF1, edge1], axis=3, name='cc1')
	_, out_feature1, out_feature11 = tiny_unet(outCFF1, 256, '_1')
	# out_feature11 = sse(out_feature11, 256, 'at_min_1')
	# out_feature11 = concatenate([out_feature11, edge1], axis=3, name='cco1')

	# stage_mid = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='cff_map10')(stage_mid)
	# outCFF1 = concatenate([stage_mid, outCFF1], axis=-1, name='cff_11')
	outCFF2 = CFF_block((stage_conv[2], out_feature1, stage_deconv[1]), 128, 'level1_CFF2')
	# outCFF21 = concatenate([outCFF2, sp_3], axis=3, name='cc2_1')
	edge2, edge21 = edge_feature(outCFF2, 128, name='edge2')
	outCFF2 = concatenate([outCFF2, edge2], axis=3, name='cc2')
	_, out_feature2, out_feature21 = tiny_unet(outCFF2, 128, '_2')
	# out_feature21 = sse(out_feature21, 128, 'at_min_2')
	# out_feature21 = concatenate([out_feature21, edge2], axis=3, name='cco2')

	# stage_mid = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='cff_map11')(stage_mid)
	# outCFF2 = concatenate([stage_mid, outCFF2], axis=-1, name='cff_21')
	outCFF3 = CFF_block((stage_conv[1], out_feature2, stage_deconv[2]), 64, 'level1_CFF3')
	# outCFF31 = concatenate([outCFF3, sp_2], axis=3, name='cc3_1')
	edge3, edge31 = edge_feature(outCFF3, 64, name='edge3')
	outCFF3 = concatenate([outCFF3, edge3], axis=3, name='cc3')
	_, out_feature3, out_feature31 = tiny_unet(outCFF3, 64, '_3')
	# out_feature31 = sse(out_feature31, 64, 'at_min_3')
	# out_feature31 = concatenate([out_feature31, edge3], axis=3, name='cco3')

	# stage_mid = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='cff_map12')(stage_mid)
	# outCFF3 = concatenate([stage_mid, outCFF3], axis=-1, name='cff_31')
	outCFF4 = CFF_block((stage_conv[0], out_feature3, stage_deconv[3]), 32, 'level1_CFF4')
	# outCFF41 = concatenate([outCFF4, sp_1], axis=3, name='cc4_1')
	edge4, edge41 = edge_feature(outCFF4, 32, name='edge4')
	outCFF4 = concatenate([outCFF4, edge4], axis=3, name='cc4')
	_, out_feature4, out_feature41 = tiny_unet(outCFF4, 32, '_4')
	# out_feature41 = sse(out_feature41, 32, 'at_min_4')
	# out_feature41 = concatenate([out_feature41, edge4], axis=3, name='cco4')

	# 边缘
	edge11 = UpSampling2D(size=(8,8), data_format='channels_last', interpolation='bilinear', name='edge1')(edge11)
	edge21 = UpSampling2D(size=(4,4), data_format='channels_last', interpolation='bilinear', name='edge2')(edge21)
	edge31 = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='edge3')(edge31)
	edge_fuse = concatenate([edge11, edge21, edge31, edge41], axis=3, name='cc5')
	# edge_fusew1 = sse(edge_fuse, name='edge')
	# edge_fuse = cse(edge_fuse, 128, name='edge')
	# edge_fuse = Add()([edge_fusew1, edge_fusew2])

	out_edge = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', name='out_edge', kernel_initializer='he_normal')(edge_fuse)
	# 平均融合
	# 上采样:
	# 试一下只有只要
	out_feature11 = UpSampling2D(size=(8,8), data_format='channels_last', interpolation='bilinear', name='out_map1')(out_feature11)
	out_feature21 = UpSampling2D(size=(4,4), data_format='channels_last', interpolation='bilinear', name='out_map2')(out_feature21)
	out_feature31 = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='out_map3')(out_feature31)
	out_feature = concatenate([out_feature11, out_feature21, out_feature31, out_feature41], axis=3, name='cc6')
	out_final = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', name='out_final', kernel_initializer='he_normal')(out_feature)
	# 2. 仅仅第4个feature出结果。
	# out_final = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', name='out_final', kernel_initializer='he_normal')(out_feature4)

	# 3. 将得到的4张概率图平均融合
	# out_final = Average(name='out_final')([out_feature11, out_feature21, out_feature31, out_feature41])

	return out_final, out_edge

def one_conv(x, kernel_num, name):
	x1 = Conv2D(kernel_num, 1, kernel_initializer='he_normal', name=name+'_cn1')(x)
	x1 = BatchNormalization(name=name+'_bn1')(x1)
	x1 = Activation(activation='relu')(x1)

	return x1


def build_level2_net(stage_conv, stage_mid, stage_deconv, main_path):

	outCFF1 = CFF_block((stage_conv[3], stage_mid, stage_deconv[0]), 256, 'level1_CFF1')
	edge1, edge11 = edge_feature(outCFF1, 256, name='edge1')
	outCFF1 = concatenate([outCFF1, edge1], axis=3, name='cc1')
	_, out_feature1, _ = tiny_unet(outCFF1, 256, '_1')


	outCFF2 = CFF_block((stage_conv[2], out_feature1, stage_deconv[1]), 128, 'level1_CFF2')
	edge2, edge21 = edge_feature(outCFF2, 128, name='edge2')
	outCFF2 = concatenate([outCFF2, edge2], axis=3, name='cc2')
	_, out_feature2, _ = tiny_unet(outCFF2, 128, '_2')


	outCFF3 = CFF_block((stage_conv[1], out_feature2, stage_deconv[2]), 64, 'level1_CFF3')
	edge3, edge31 = edge_feature(outCFF3, 64, name='edge3')
	outCFF3 = concatenate([outCFF3, edge3], axis=3, name='cc3')
	_, out_feature3, _ = tiny_unet(outCFF3, 64, '_3')


	outCFF4 = CFF_block((stage_conv[0], out_feature3, stage_deconv[3]), 32, 'level1_CFF4')
	edge4, edge41 = edge_feature(outCFF4, 32, name='edge4')
	outCFF4 = concatenate([outCFF4, edge4], axis=3, name='cc4')
	_, out_feature4, _ = tiny_unet(outCFF4, 32, '_4')


	# 边缘
	edge11 = UpSampling2D(size=(8,8), data_format='channels_last', interpolation='bilinear', name='edge1')(edge11)
	edge21 = UpSampling2D(size=(4,4), data_format='channels_last', interpolation='bilinear', name='edge2')(edge21)
	edge31 = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='edge3')(edge31)
	edge_fuse = concatenate([edge11, edge21, edge31, edge41], axis=3, name='cc5')

	out_edge = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', name='out_edge', kernel_initializer='he_normal')(edge_fuse)
	
	# # 1. 仅仅最后一层与U-Net结果进行融合
	# out_final = concatenate([out_feature4, main_path], axis=3, name='cc6')
	# out_final = Conv2D(48, 1, kernel_initializer='he_normal', name='f_cn1', padding='same')(out_final)
	# out_final = BatchNormalization(name='f_bn1')(out_final)
	# out_final = Activation(activation='relu')(out_final)

	# 2. 将4个level-size的特征进行融合
	out_feature1 = concatenate([out_feature1, main_path[0]], axis=3, name='cc7')
	out_feature1 = one_conv(out_feature1, 48, 'final1')
	out_feature1 = UpSampling2D(size=(8,8), data_format='channels_last', interpolation='bilinear', name='out_map1')(out_feature1)
	out_feature2 = concatenate([out_feature2, main_path[1]], axis=3, name='cc8')
	out_feature2 = one_conv(out_feature2, 48, 'final2')
	out_feature2 = UpSampling2D(size=(4,4), data_format='channels_last', interpolation='bilinear', name='out_map2')(out_feature2)
	out_feature3 = concatenate([out_feature3, main_path[2]], axis=3, name='cc9')
	out_feature3 = one_conv(out_feature3, 48, 'final3')
	out_feature3 = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='out_map3')(out_feature3)
	out_feature4 = concatenate([out_feature4, main_path[3]], axis=3, name='cc10')

	out_feature = concatenate([out_feature1, out_feature2, out_feature3, out_feature4], axis=3, name='cc11')

	out_final = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', name='out_final', kernel_initializer='he_normal')(out_feature)
	
	return out_final, out_edge
