import os

import numpy as np

import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

import random

from utils_colab import patchMaker,cmMaker,binningMaker,resultMaker


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import concatenate, UpSampling2D, BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard

import time

import tensorflow_addons as tfa

STABILITY_FACTOR = 0.00001

def loss_ce(y_true, y_pred,binning=4,thresh_t=0.1,changeLoss=True, y_channel=4):
    n = tf.keras.backend.shape(y_true)[-1]
    if changeLoss:
        y_true, y_pred = binningMaker(y_true, y_pred, binning, thresh_t, y_channel)
    return tf.reduce_mean(tf.keras.backend.binary_crossentropy(y_true,y_pred[...,:n]))

def loss_dice (y_true, y_pred,binning=4,thresh_t=0.25,changeLoss=True, y_channel=4):
    if changeLoss:
        y_true, y_pred = binningMaker(y_true, y_pred, binning, thresh_t, y_channel)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))+STABILITY_FACTOR

    return 1 - tf.reduce_mean( (numerator ) / (denominator) )

def loss_dice_continuous (y_true, y_pred,binning=4,thresh_t=0.25,changeLoss=True, y_channel=4):
    if changeLoss:
        y_true, y_pred = binningMaker(y_true, y_pred, binning, thresh_t, y_channel)
    AB = tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    A = tf.reduce_sum(y_true, axis=(1,2,3))+STABILITY_FACTOR
    B = tf.reduce_sum(y_pred, axis=(1,2,3))+STABILITY_FACTOR
    denum = tf.reduce_sum(y_true * tf.sign(y_pred), axis=(1,2,3))+STABILITY_FACTOR
    c = AB/denum

    return 1 - tf.reduce_mean( (2 * AB) / (c*A + B) )

def loss_jaccard (y_true, y_pred,binning=4,thresh_t=0.25,changeLoss=True, y_channel=4):
    if changeLoss:
        y_true, y_pred = binningMaker(y_true, y_pred, binning, thresh_t, y_channel)
    numerator = tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred - y_true * y_pred, axis=(1,2,3))+STABILITY_FACTOR

    return 1 - tf.reduce_mean( (numerator ) / (denominator) )

def loss_focalc (y_true, y_pred,gamma=2,binning=4,thresh_t=0.5,changeLoss=False,alpha=0.25, y_channel=4):
    if changeLoss:
        y_true, y_pred = binningMaker(y_true, y_pred, binning, thresh_t, y_channel)
    #y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    #logits = tf.math.log(y_pred / (1 - y_pred))

    #loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
    # or reduce_sum and/or axis=-1
    #return tf.reduce_mean(loss)
    return tfa.losses.SigmoidFocalCrossEntropy()(y_true, y_pred)

def accuracy(y_true, y_pred, binning=4,changeLoss=True, thresh_t=0.1, thresh_p=0.1, y_channel=4):
    n = tf.keras.backend.shape(y_true)[-1]
    if changeLoss:
        y_true,y_pred=binningMaker(y_true,y_pred,binning,thresh_t,y_channel)

    y_true_=tf.keras.backend.flatten(y_true)
    y_true_= tf.keras.backend.cast(tf.math.greater(y_true_, thresh_t), tf.float64)

    y_pred_=tf.keras.backend.flatten(y_pred[...,:n])
    y_pred_=tf.keras.backend.cast(tf.math.greater(y_pred_,thresh_p),tf.float64)

    cmPre=tf.zeros([2,2])
    cmPre=tf.math.confusion_matrix(y_true_,y_pred_,num_classes=2)
    cmPre = tf.cast(cmPre,tf.float64)
    return (cmPre[1][1] + cmPre[0][0]) / (cmPre[0][0] + cmPre[0][1] + cmPre[1][0] +cmPre[1][1]+STABILITY_FACTOR)


def precision(y_true, y_pred, binning=4,changeLoss=True, thresh_t=0.1, thresh_p=0.1,channel = None, y_channel=4):
    n = tf.keras.backend.shape(y_true)[-1]
    if changeLoss:
        y_true,y_pred=binningMaker(y_true,y_pred,binning,thresh_t, y_channel)

    if channel != None:
        y_true_=tf.keras.backend.flatten(y_true[...,channel])
    else:
        y_true_=tf.keras.backend.flatten(y_true)
    y_true_= tf.keras.backend.cast(tf.math.greater(y_true_, thresh_t), tf.float64)

    if channel != None:
        y_pred_=tf.keras.backend.flatten(y_pred[...,channel])
    else:
        y_pred_=tf.keras.backend.flatten(y_pred[...,:n])
    y_pred_=tf.keras.backend.cast(tf.math.greater(y_pred_,thresh_p),tf.float64)

    cmPre=tf.zeros([2,2])
    cmPre=tf.math.confusion_matrix(y_true_,y_pred_,num_classes=2)
    cmPre = tf.cast(cmPre,tf.float64)
    return cmPre[1][1] / (cmPre[0][1] + cmPre[1][1]+STABILITY_FACTOR)

def recall(y_true, y_pred, binning=4,changeLoss=True, thresh_t=0.1, thresh_p=0.1,channel = None, y_channel=4):
    n = tf.keras.backend.shape(y_true)[-1]
    if changeLoss:
        y_true,y_pred=binningMaker(y_true,y_pred,binning,thresh_t, y_channel)

    if channel != None:
        y_true_=tf.keras.backend.flatten(y_true[...,channel])
    else:
        y_true_=tf.keras.backend.flatten(y_true)
    y_true_= tf.keras.backend.cast(tf.math.greater(y_true_, thresh_t), tf.float64)

    if channel != None:
        y_pred_=tf.keras.backend.flatten(y_pred[...,channel])
    else:
        y_pred_=tf.keras.backend.flatten(y_pred[...,:n])
    y_pred_=tf.keras.backend.cast(tf.math.greater(y_pred_,thresh_p),tf.float64)

    cmPre=tf.zeros([2,2])
    cmPre=tf.math.confusion_matrix(y_true_,y_pred_,num_classes=2)
    cmPre = tf.cast(cmPre,tf.float64)
    return cmPre[1][1] / (cmPre[1][0] + cmPre[1][1]+STABILITY_FACTOR)

def f1score(y_true, y_pred,binning=4,changeLoss=True,thresh_t=0.1,thresh_p=0.1,channel = None, y_channel=4):
    n = tf.keras.backend.shape(y_true)[-1]
    if changeLoss:
        y_true,y_pred=binningMaker(y_true,y_pred,binning,thresh_t,y_channel)

    if channel != None:
        y_true_=tf.keras.backend.flatten(y_true[...,channel])
    else:
        y_true_=tf.keras.backend.flatten(y_true)
    y_true_= tf.keras.backend.cast(tf.math.greater(y_true_, thresh_t), tf.float64)

    if channel != None:
        y_pred_=tf.keras.backend.flatten(y_pred[...,channel])
    else:
        y_pred_=tf.keras.backend.flatten(y_pred[...,:n])
    y_pred_=tf.keras.backend.cast(tf.math.greater(y_pred_,thresh_p),tf.float64)

    cm=tf.zeros([2,2])
    cm=tf.math.confusion_matrix(y_true_,y_pred_,num_classes=2)
    cm = tf.cast(cm,tf.float64)
    precision=cm[1][1] / (cm[0][1] + cm[1][1]+STABILITY_FACTOR)
    recall=cm[1][1] / (cm[1][0] + cm[1][1]+STABILITY_FACTOR)


    return 2*(precision*recall)/(precision+recall+STABILITY_FACTOR)

def csi(y_true, y_pred,binning=4,changeLoss=True,thresh_t=0.1,thresh_p=0.1,channel = None, y_channel=4):
    n = tf.keras.backend.shape(y_true)[-1]
    if changeLoss:
        y_true,y_pred=binningMaker(y_true,y_pred,binning,thresh_t,y_channel)

    if channel != None:
        y_true_=tf.keras.backend.flatten(y_true[...,channel])
    else:
        y_true_=tf.keras.backend.flatten(y_true)
    y_true_= tf.keras.backend.cast(tf.math.greater(y_true_, thresh_t), tf.float64)

    if channel != None:
        y_pred_=tf.keras.backend.flatten(y_pred[...,channel])
    else:
        y_pred_=tf.keras.backend.flatten(y_pred[...,:n])
    y_pred_=tf.keras.backend.cast(tf.math.greater(y_pred_,thresh_p),tf.float64)

    cm=tf.zeros([2,2])
    cm=tf.math.confusion_matrix(y_true_,y_pred_,num_classes=2)
    cm = tf.cast(cm,tf.float64)
    return cm[1][1] / (cm[1][0] + cm[1][1]+cm[0][1]+STABILITY_FACTOR)

def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b



def f1score_with_channel(y_true, y_pred, channel = None, thresh_t=0.1, thresh_p=0.1,changeLoss=True):
    if channel is None:
        return f1score(y_true,y_pred, thresh_t = thresh_t, thresh_p = thresh_p,changeLoss=changeLoss)
    else:
        return f1score(y_true,y_pred,channel = channel, thresh_t = thresh_t, thresh_p = thresh_p,changeLoss=changeLoss)

def precision_with_channel(y_true, y_pred, channel = None, thresh_t=0.1, thresh_p=0.1,changeLoss=True):
    if channel is None:
        return precision(y_true,y_pred,thresh_t = thresh_t, thresh_p = thresh_p,changeLoss=changeLoss)
    else:
        return precision(y_true,y_pred,channel = channel, thresh_t = thresh_t, thresh_p = thresh_p,changeLoss=changeLoss)

def recall_with_channel(y_true, y_pred, channel = None, thresh_t=0.1, thresh_p=0.1,changeLoss=True):
    if channel is None:
        return recall(y_true,y_pred, thresh_t = thresh_t, thresh_p = thresh_p,changeLoss=changeLoss)
    else:
        return recall(y_true,y_pred,channel = channel, thresh_t = thresh_t, thresh_p = thresh_p,changeLoss=changeLoss)

def csi_with_channel(y_true, y_pred, channel = None, thresh_t=0.1, thresh_p=0.1,changeLoss=True):
    if channel is None:
        return csi(y_true,y_pred, thresh_t = thresh_t, thresh_p = thresh_p,changeLoss=changeLoss)
    else:
        return csi(y_true,y_pred,channel = channel, thresh_t = thresh_t, thresh_p = thresh_p,changeLoss=changeLoss)



def loss_1(y_true,y_pred,y_channel):
    return loss_dice(y_true,y_pred,changeLoss=True,y_channel=y_channel) + 5 * loss_ce(y_true,y_pred,changeLoss=True,y_channel=y_channel)

def loss_2(y_true,y_pred,y_channel):
    return loss_jaccard(y_true,y_pred,changeLoss=True,y_channel=y_channel) + 5 * loss_ce(y_true,y_pred,changeLoss=True,y_channel=y_channel)

def loss_3(y_true,y_pred,y_channel):
    return loss_dice_continuous(y_true,y_pred,changeLoss=False,y_channel=y_channel) + 5 * loss_ce(y_true,y_pred,changeLoss=False,y_channel=y_channel)