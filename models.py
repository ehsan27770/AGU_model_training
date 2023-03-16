import os

import numpy as np

import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

import random

from utils_colab import patchMaker,cmMaker,binningMaker,resultMaker


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
#from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.python.keras.layers import concatenate, UpSampling2D, BatchNormalization,Dropout
from tensorflow.keras.layers import Add, Average, Lambda, Conv2D, Conv3D, MaxPooling2D, concatenate, add, UpSampling2D, BatchNormalization, Dropout, ReLU, LeakyReLU, ConvLSTM2D, Attention, Conv2DTranspose#, MultiHeadAttention
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard

import time

# %%
def downsample(filters, size, apply_batchnorm=True, apply_dropout = False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.7))

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result



def Generator(patchSize,depth):
  inputs = tf.keras.layers.Input(shape=[patchSize,patchSize,depth])


  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='sigmoid') # (bs, 256, 256, 1)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def Generator_plus():
  inputs = tf.keras.layers.Input(shape=[patchSize,patchSize,depth])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4, apply_dropout=True), # (bs, 64, 64, 128)
    downsample(256, 4, apply_dropout=True), # (bs, 32, 32, 256)
    downsample(512, 4, apply_dropout=True), # (bs, 16, 16, 512)
    downsample(512, 4, apply_dropout=True), # (bs, 8, 8, 512)
    downsample(512, 4, apply_dropout=True), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 16, 16, 1024)
    upsample(256, 4, apply_dropout=True), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='sigmoid') # (bs, 256, 256, 1)
  x = inputs

  skips = []
  downlayers = [[[]for i in range(len(down_stack)-1)]for j in range(len(down_stack)-1)]
  for i,down in enumerate(down_stack[:-1]):
     downlayers[i][0] = down(x)
     x = downlayers[i][0]
     for j in range(1,i+1):
         n = np.shape(downlayers[i][j-1])[1]
         y = upsample(n,4,apply_dropout=True)(downlayers[i][j-1])
         print(np.shape(y))
         downlayers[i][j] = tf.keras.layers.Concatenate()([y,downlayers[i-1][j-1]])

  skips = downlayers[-1][:]

  x = down_stack[-1](x)

  print(len(skips))
  print(len(up_stack))
  skips = reversed(skips[:-1])
  for up, skip in zip(up_stack, skips):
    x = up(x)
    print("x")
    print(np.shape(x))
    print("skip")
    print(np.shape(skip))
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x)


def Generator_2():
  input_shape = (patchSize, patchSize, depth)
  inputs = Input(shape=input_shape)
  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  drop5 = Dropout(0.5)(conv5)

  up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
  merge6 = concatenate([drop4,up6], axis = 3)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  outputs = Conv2D(1, 1, activation = 'sigmoid')(conv9)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def Persistence(width = 256, height = 256, depth = 8, multi = True):
    input_shape = (width, height, depth)

    inputs = Input(shape = input_shape)
    x = inputs[:,:,:,-1:]
    if multi == True:
        outputs = concatenate([x,x,x,x], axis = 3)
    else:
        outputs = x
    model = Model(inputs = inputs, outputs = outputs)
    return model

def Lightning_Prediction(width = 256, height = 256, depth = 12, residual = True):
    def basic(x, filters, residual = True):
        res = Conv2D(filters=filters,kernel_size=(3,3),padding='same',data_format='channels_last',kernel_initializer=he_normal())(x)
        res = BatchNormalization()(res)
        res = LeakyReLU()(res)
        res = Conv2D(filters=filters,kernel_size=(3,3),padding='same',data_format='channels_last',kernel_initializer=he_normal())(res)
        if residual == True:
            y = Conv2D(filters=filters,kernel_size=(1,1),padding='same',data_format='channels_last',kernel_initializer=he_normal())(x)
            res = tf.keras.layers.add([res, y])
        return res

    def down(x, filters, residual = True):
        res = BatchNormalization()(x)
        res = LeakyReLU()(res)
        res = MaxPooling2D(pool_size=(2,2),strides=2)(res)
        res = BatchNormalization()(res)
        res = LeakyReLU()(res)
        res = Conv2D(filters=filters,kernel_size=(3,3),padding='same',data_format='channels_last',kernel_initializer=he_normal())(res)
        if residual == True:
            y = Conv2D(filters=filters,kernel_size=(1,1),strides=(2,2),padding='same',data_format='channels_last',kernel_initializer=he_normal())(x)
            res = tf.keras.layers.add([res, y])
        return res

    def up(x, filters, residual = True):
        y = UpSampling2D(size=(2,2),data_format='channels_last',interpolation='nearest')(x)
        res = BatchNormalization()(y)
        res = LeakyReLU()(res)
        res = Conv2D(filters=filters,kernel_size=(3,3),padding='same',data_format='channels_last',kernel_initializer=he_normal())(res)
        res = BatchNormalization()(res)
        res = LeakyReLU()(res)
        res = Conv2D(filters=filters,kernel_size=(3,3),padding='same',data_format='channels_last',kernel_initializer=he_normal())(res)
        if residual == True:
            z = Conv2D(filters=filters,kernel_size=(1,1),padding='same',data_format='channels_last',kernel_initializer=he_normal())(y)
            res = tf.keras.layers.add([res, z])
        return res


    input_shape = (width, height, depth)

    inputs = Input(shape = input_shape) # 256,256,6

    skips = []
    basic_filter_sizes = [16,512,1]
    down_filter_size = [32,64,128,256,512,1024]
    up_filter_size = [256,128,64,32,16,8]

    #basic_filter_sizes = [32,1024,1]
    #down_filter_size = [64,128,256,512,1024,2048]
    #up_filter_size = [512,256,128,64,32,16]

    x = basic(inputs, filters = basic_filter_sizes[0], residual=residual) #256,256,16

    for filter_size in down_filter_size:
        x = down(x, filters = filter_size, residual = residual)
        skips.append(x)

    x = basic(x, filters = basic_filter_sizes[1], residual = residual)

    for filter_size in up_filter_size:
        x = tf.keras.layers.Concatenate(axis=-1)([x,skips.pop()])
        x = up(x, filters = filter_size, residual = residual)

    x = basic(x, filters = basic_filter_sizes[2], residual = residual)

    x = Activation('sigmoid')(x)

    model = Model(inputs = inputs, outputs = x)
    return model

def Lightning_Prediction_ConvLSTM(width = 256, height = 256, depth = 12):
    input_shape = (None,width, height, depth)

    inputs = Input(shape = input_shape)
    x = ConvLSTM2D(filters=16,kernel_size=(3,3),padding='same',return_sequences=True,strides=(1,1))(inputs)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=16,kernel_size=(3,3),padding='same',return_sequences=True,strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=16,kernel_size=(3,3),padding='same',return_sequences=True,strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=1,kernel_size=(3,3),padding='same',strides=(1,1),activation="sigmoid")(x)
    #x = Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='valid')(x)

    model = Model(inputs = inputs, outputs = x)
    return model

def Lightning_Prediction_ConvLSTM_baseline(width = 256, height = 256, depth = 12):
    input_shape = (None,width, height, depth)

    inputs = Input(shape = input_shape)
    x = ConvLSTM2D(filters=32,kernel_size=(3,3),padding='same',strides=(1,1),return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=32,kernel_size=(3,3),padding='same',strides=(1,1),return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=1,kernel_size=(3,3),padding='same',strides=(1,1),activation="sigmoid")(x)

    model = Model(inputs = inputs, outputs = x)
    return model

def Lightning_Prediction_CARE(width = 256, height = 256, depth = 11):
    input_shape = (width, height, depth)
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    outputs = Conv2D(1, 1, activation = 'sigmoid', kernel_initializer = 'he_normal')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def Lightning_Prediction_CARE_No_Dropout(width = 256, height = 256, depth = 11):
    input_shape = (width, height, depth)
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    outputs = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def Lightning_Prediction_Residual_CARE(width = 256, height = 256, depth = 11):
    input_shape = (width, height, depth)
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1)(conv9)
    conv10 = Add()([conv10, Lambda(lambda x:x[:,:,:,depth-2:depth-1])(inputs)])
    outputs = Activation('sigmoid')(conv10)
    # outputs = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    #outputs = conv10

    model = Model(inputs=inputs, outputs=outputs)

    return model

def Lightning_Prediction_CARE_Multi_Output(width = 256, height = 256, depth = 11):
    input_shape = (width, height, depth)
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    outputs = Conv2D(3, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv9)


    model = Model(inputs=inputs, outputs=outputs)

    return model


def Lightning_Prediction_CARE_Multi_Output_variable(width = 256, height = 256, depth = 11, multi = True, layers = [64, 128, 256, 512, 1024], skip_connection=True):
    input_shape = (width, height, depth)
    inputs = Input(shape=input_shape)
    temp = inputs

    if skip_connection == True:
        to_concat = []

    for layer in layers[:-1]:
        conv1 = Conv2D(layer, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(temp)
        conv2 = Conv2D(layer, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        if skip_connection == True:
            to_concat.append(conv2)
        #drop = Dropout(0.5)(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv2)
        temp = pool

    conv_down = Conv2D(layers[-1], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(temp)
    temp = Conv2D(layers[-1], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_down)
    #drop5 = Dropout(0.5)(conv5)

    for layer in layers[-2::-1]:
        up = UpSampling2D(size = (2,2))(temp)
        #conv1 = Conv2D(layer, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up)
        if skip_connection == True:
            up = concatenate([to_concat.pop(),up], axis = 3)
        conv2 = Conv2D(layer, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up)
        conv3 = Conv2D(layer, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        temp = conv3
    if multi == True:
        #outputs = Conv2D(3, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(temp)
        outputs = Conv2D(4, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(temp)
    else:
        outputs = Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(temp)

    model = Model(inputs=inputs, outputs=outputs)

    return model






def Lightning_Prediction_ResUnet(width = 256, height = 256, depth = 8, multi = True, layers = [64, 128, 256, 512, 1024]):
    input_shape = (width, height, depth)
    inputs = Input(shape=input_shape)
    temp = inputs

    to_concat = []

    # encoder
    for i,layer in enumerate(layers[:-1]):
        if i == 0:
            shortcut = Conv2D(layer, 1, kernel_initializer = 'he_normal')(temp)
        else:
            shortcut = Conv2D(layer, 1, strides = 2, kernel_initializer = 'he_normal')(temp)
        shortcut = BatchNormalization()(shortcut)

        if i == 0:
            temp = Conv2D(layer, 3, padding = 'same', kernel_initializer = 'he_normal')(temp)
        else:
            temp = BatchNormalization()(temp)
            temp = ReLU()(temp)
            temp = Conv2D(layer, 3, strides=2, padding = 'same', kernel_initializer = 'he_normal')(temp)
        temp = BatchNormalization()(temp)
        temp = ReLU()(temp)
        temp = Conv2D(layer, 3, padding = 'same', kernel_initializer = 'he_normal')(temp)


        temp = add([shortcut, temp])
        to_concat.append(temp)


    # bridge
    layer = layers[-1]

    shortcut = Conv2D(layer, 1, strides = 2, kernel_initializer = 'he_normal')(temp)
    shortcut = BatchNormalization()(shortcut)

    temp = BatchNormalization()(temp)
    temp = ReLU()(temp)
    temp = Conv2D(layer, 3, strides=2, padding = 'same', kernel_initializer = 'he_normal')(temp)
    temp = BatchNormalization()(temp)
    temp = ReLU()(temp)
    temp = Conv2D(layer, 3, padding = 'same', kernel_initializer = 'he_normal')(temp)
    temp = add([shortcut, temp])


    # decoder
    for layer in layers[-2::-1]:
        temp = UpSampling2D(size = (2,2))(temp)
        temp = concatenate([to_concat.pop(),temp], axis = 3)

        shortcut = Conv2D(layer, 1, kernel_initializer = 'he_normal')(temp)
        shortcut = BatchNormalization()(shortcut)

        temp = BatchNormalization()(temp)
        temp = ReLU()(temp)
        temp = Conv2D(layer, 3, padding = 'same', kernel_initializer = 'he_normal')(temp)
        temp = BatchNormalization()(temp)
        temp = ReLU()(temp)
        temp = Conv2D(layer, 3, padding = 'same', kernel_initializer = 'he_normal')(temp)
        temp = add([shortcut, temp])


    if multi == True:
        outputs = Conv2D(4, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(temp)
    else:
        outputs = Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(temp)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def Lightning_Prediction_ResUnet_TransposedConv_noBatchNorm(width = 256, height = 256, depth = 8, multi = True, layers = [64, 128, 256, 512, 1024]):
    input_shape = (width, height, depth)
    inputs = Input(shape=input_shape)
    temp = inputs

    to_concat = []

    # encoder
    for i,layer in enumerate(layers[:-1]):
        if i == 0:
            shortcut = Conv2D(layer, 1, kernel_initializer = 'he_normal')(temp)
        else:
            shortcut = Conv2D(layer, 1, strides = 2, kernel_initializer = 'he_normal')(temp)
        shortcut = BatchNormalization()(shortcut)

        if i == 0:
            temp = Conv2D(layer, 3, padding = 'same', kernel_initializer = 'he_normal')(temp)
        else:
            temp = BatchNormalization()(temp)
            temp = ReLU()(temp)
            temp = Conv2D(layer, 3, strides=2, padding = 'same', kernel_initializer = 'he_normal')(temp)
        temp = BatchNormalization()(temp)
        temp = ReLU()(temp)
        temp = Conv2D(layer, 3, padding = 'same', kernel_initializer = 'he_normal')(temp)


        temp = add([shortcut, temp])
        to_concat.append(temp)


    # bridge
    layer = layers[-1]

    shortcut = Conv2D(layer, 1, strides = 2, kernel_initializer = 'he_normal')(temp)
    shortcut = BatchNormalization()(shortcut)

    temp = BatchNormalization()(temp)
    temp = ReLU()(temp)
    temp = Conv2D(layer, 3, strides=2, padding = 'same', kernel_initializer = 'he_normal')(temp)
    temp = BatchNormalization()(temp)
    temp = ReLU()(temp)
    temp = Conv2D(layer, 3, padding = 'same', kernel_initializer = 'he_normal')(temp)
    temp = add([shortcut, temp])


    # decoder

    for layer in layers[-2::-1]:
        temp = Conv2DTranspose(layer,3,strides=(2, 2),padding="same",output_padding=None,activation=None,kernel_initializer="glorot_uniform")(temp)
        #temp = UpSampling2D(size = (2,2))(temp)
        temp = concatenate([to_concat.pop(),temp], axis = 3)

        shortcut = Conv2D(layer, 1, kernel_initializer = 'he_normal')(temp)
        shortcut = BatchNormalization()(shortcut)

        temp = BatchNormalization()(temp)
        temp = ReLU()(temp)
        temp = Conv2D(layer, 3, padding = 'same', kernel_initializer = 'he_normal')(temp)
        temp = BatchNormalization()(temp)
        temp = ReLU()(temp)
        temp = Conv2D(layer, 3, padding = 'same', kernel_initializer = 'he_normal')(temp)
        temp = add([shortcut, temp])


    if multi == True:
        outputs = Conv2D(4, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(temp)
    else:
        outputs = Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(temp)

    model = Model(inputs=inputs, outputs=outputs)

    return model
