# coding=utf-8

import keras.backend as K

def disypre1_loss(y_true, y_pred):
	error = -K.log(y_pred)
	loss = K.sum(error, axis=[1, 2])
	loss = K.mean(loss, axis=0)
	return loss

def disypre2_loss(y_true, y_pred):
	error = -K.log(1-y_pred)
	loss = K.sum(error, axis=[1, 2])
	loss = K.mean(loss, axis=0)
	return loss


def disyreal1_loss(y_true, y_pred):
	error = -K.log(y_pred)
	loss = K.sum(error, axis=[1, 2])
	loss = K.mean(loss, axis=0)
	return loss
