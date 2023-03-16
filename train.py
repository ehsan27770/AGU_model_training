import os
import resource

import numpy as np
import matplotlib.pyplot as plt
import pickle

import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

import random

from utils_colab import cmMaker, binningMaker, resultMaker

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import concatenate, UpSampling2D, BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from tensorflow.keras.metrics import Metric
from tensorflow.keras.losses import Loss

from tensorflow.keras.callbacks import TensorBoard

import time

from sklearn.model_selection import train_test_split
import datetime

from DataGenerator import PatchGenerator, PatchGenerator_multioutput
from loses import loss_ce, loss_dice, loss_focalc, f1score, precision,recall
from models import Generator, Lightning_Prediction, Lightning_Prediction_ConvLSTM, Lightning_Prediction_ConvLSTM_baseline, Lightning_Prediction_CARE, Lightning_Prediction_Residual_CARE, Lightning_Prediction_CARE_Multi_Output
from callbacks import MemoryCheck

from train_settings import configuration_generator

#only for macbook m1
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
#from tensorflow.python.compiler.mlcompute import mlcompute
#mlcompute.set_mlc_device(device_name='gpu')





def train():
    configurations = configuration_generator()

    for config in configurations:
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")
        print(config["description"])
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")
        #fixed
        path = config["path"]


        #variables
        patch_size = config["patch_size"]
        inputInd = config["inputInd"]
        batch_size = config["batch_size"]
        numPatches = config["numPatches"]
        contex = config["contex"]
        region = config["region"]
        bound = config["bound"]
        keep_float = config["keep_float"]
        include_position = config["include_position"]
        include_time = config["include_time"]
        time_dimension = config["time_dimension"]
        epochs = config["epochs"]

        model = config["model"]
        model.summary()

        #patch_train = PatchGenerator(path+'val/',time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=11,flashThresh=10)
        #patch_val   = PatchGenerator(path+'val/'  ,time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=11,flashThresh=1)
        #patch_test  = PatchGenerator(path+'val/' ,time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=11)

        patch_train = PatchGenerator_multioutput(path+'train/',time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=8,flashThresh=10,do_sum=False)
        patch_val   = PatchGenerator_multioutput(path+'val/'  ,time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=8,flashThresh=1,do_sum=False)
        patch_test  = PatchGenerator_multioutput(path+'test/' ,time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=8,do_sum=False)

        try:
            model.load_weights('./training_checkpoints/'+config["name"]+'.hdf5')
        except:
            pass

        ## Define the Optimizers and Checkpoint-saver
        optimizer = tf.keras.optimizers.Adam(1e-4)
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='/home/semansou/code/long_range_prediction/outputs/checkpoints/'+config["name"]+'.hdf5',monitor='val_f1score', verbose=1, save_best_only=True,save_weights_only=False, mode='max', save_freq='epoch')
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1score', factor=0.5, patience=3, verbose=2, mode='max', cooldown=1)
        memorycheck = MemoryCheck()
        tensorboard = TensorBoard(log_dir='/home/semansou/code/long_range_prediction/outputs/tensorboard/'+config["name"], histogram_freq=0, write_graph=False, write_images=True)#,write_grads=True)

        #model.summary()

        loss = config["loss"]

        model.compile(optimizer = optimizer, loss = loss, metrics = [loss_ce, loss_dice, loss_focalc, f1score, precision, recall])
        H = model.fit(patch_train, epochs=15, validation_data = patch_val, verbose = 1, callbacks = [checkpointer, lr_scheduler, memorycheck, tensorboard])

        with open('/home/semansou/code/long_range_prediction/outputs/history/'+config["name"], 'wb') as file_pi:
            pickle.dump(H.history, file_pi)



# %%

if __name__ == '__main__':
    train()

    # patch = PatchGenerator_multioutput('data/val/',time_path=None,position_path=None,include_position=False,include_time=False,time_dimension=False,contex=False,region='CUS',bound=[500,1000,500,1000],batch_size=16,numPatches=4,patch_size=128,inputInd=[2,3,4,5,6,7,8,9],keep_float=True,n_channels=11,flashThresh=10)
    #
    # model = Lightning_Prediction_CARE_Multi_Output(width=128,height=128,depth=11)
    #
    # x,y_true = patch[0]
    # y_pred = model(x)
    # y_true.shape
    # y_true.dtype
    # y_pred.dtype
    # y_pred.numpy().astype('double').dtype
    # binningMaker(y_true,y_pred)
    #
    # n = tf.keras.backend.shape(y_true)[-1]
    # binning = 4
    # filters = np.ones((binning, binning, n, 1),dtype='float32') / binning / binning
    #
    # #y_true = tf.keras.layers.DepthwiseConv2D(y_true,filters,[1,1,1,1],padding='VALID')
    # y_true = tf.nn.depthwise_conv2d(y_true,filters,[1,1,1,1],padding='VALID')
    # y_pred = tf.nn.depthwise_conv2d(y_pred,filters,[1,1,1,1],padding='VALID')
    #
    #
    # filters = np.ones((binning, binning, n, 1),dtype='float32') / binning / binning
    #
    # y_true = tf.nn.depthwise_conv2d(y_true, filters, [1,1,1,1], padding='VALID')
    # y_pred = tf.nn.depthwise_conv2d(y_pred, filters, [1,1,1,1], padding='VALID')
    # y_true = tf.keras.backend.cast(tf.math.greater(y_true, .1), tf.float32)
    # m = n.numpy()
    model = Lightning_Prediction_CARE_Multi_Output()
    model = Lightning_Prediction_ConvLSTM()
    model.summary()
    keras.utils.plot_model(model,show_shapes=False)
