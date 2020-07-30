# coding=utf-8

import keras.backend as K
from .compare_metric import disypre1_loss, disypre2_loss, disyreal1_loss 

def disypre1(y_true, y_pred):

	return disypre1_loss(y_true, y_pred)


def disypre2(y_true, y_pred):

	return disypre2_loss(y_true, y_pred)


def disyreal1(y_true, y_pred):

	return disyreal1_loss(y_true, y_pred)