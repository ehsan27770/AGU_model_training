import os
import sys
import resource
import argparse
import yaml

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
from loses import loss_ce, loss_dice, loss_jaccard, loss_focalc, f1score, precision, recall, accuracy, f1score_15min, f1score_30min, f1score_45min, f1score_60min, loss_1, loss_2
from models import  Lightning_Prediction_CARE_Multi_Output, Lightning_Prediction_CARE_Multi_Output_variable, Lightning_Prediction_ResUnet, Persistence
from callbacks import MemoryCheck

from train_settings import configuration_generator
import functools
#only for macbook m1
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
#from tensorflow.python.compiler.mlcompute import mlcompute
#mlcompute.set_mlc_device(device_name='gpu')


# weights and Biases:
#import wandb
#from wandb.keras import WandbCallback




def visualize(config):
    print("--------------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------------")
    print(config["description"])
    print("--------------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------------")
    ####################################################################### Wandb
    #wandb.init(project='long_range_prediction',config=config)
    ####################################################################### fixed
    path = config["path"]
    seed = config["seed"]
    ####################################################################### variables
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
    ####################################################################### model
    height = config["height"]
    width = config["width"]
    depth = config["depth"]
    layers = config["layers"]
    multi = config["multi"]
    output_index = config["output_index"]
    if multi == True:
        assert output_index == None
    else:
        assert output_index != None
    use_amplitude = config["use_amplitude"]
    do_sum = config["do_sum"]
    skip_connection = config["skip_connection"]
    if config["model"] == 0:
        model = Persistence(width = width, height = height, depth = depth, multi = multi)
    elif config["model"] == 1:
        model = Lightning_Prediction_CARE_Multi_Output_variable(width = width, height = height, depth = depth, multi = multi, layers = layers, skip_connection = skip_connection)
    elif config["model"] == 2:
        model = Lightning_Prediction_ResUnet(width = width, height = height, depth = depth, multi = multi, layers = layers)
    else:
        pass

    model.summary()
    ####################################################################### weights
    if config["model"] != 0:
        model.load_weights('/home/semansou/code/long_range_prediction/outputs/checkpoints/'+config["name"]+'.hdf5')
    ####################################################################### data generator
    if config["generator"] == 1:
        patch_train = PatchGenerator(path+'val/',time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=11,flashThresh=10,seed=seed)
        patch_val   = PatchGenerator(path+'val/'  ,time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=11,flashThresh=1,seed=seed)
        patch_test  = PatchGenerator(path+'val/' ,time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=11,seed=seed)
    elif config["generator"] == 2:
        patch_train = PatchGenerator_multioutput(path+'train/',time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=8,flashThresh=config["flashThresh"],p_ratio=config["p_ratio"],do_sum=do_sum,use_amplitude=use_amplitude,output_index=output_index,seed=seed)
        patch_val   = PatchGenerator_multioutput(path+'val/'  ,time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=8,flashThresh=1,do_sum=do_sum,use_amplitude=use_amplitude,output_index=output_index,seed=seed,coordinates_for_drawing=True)
        patch_test  = PatchGenerator_multioutput(path+'test/' ,time_path=None,position_path=None,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=8,do_sum=do_sum,use_amplitude=use_amplitude,output_index=output_index,seed=seed, coordinates_for_drawing=True)
    else:
        pass
    ####################################################################### loss
    if output_index == None:
        y_channel = 4
    else:
        y_channel = 1
    if config["loss"] == 1:
        loss = functools.partial(loss_1,y_channel=y_channel)
        functools.update_wrapper(loss,loss_1)
        #loss = lambda y_true,y_pred : loss_1(y_true,y_pred,y_channel=y_channel)
    elif config["loss"] == 2:
        loss = functools.partial(loss_2,y_channel=y_channel)
        functools.update_wrapper(loss,loss_2)
        #loss = lambda y_true,y_pred : loss_2(y_true,y_pred,y_channel=y_channel)
    else:
        pass
    ####################################################################### rest
    def wrapped_partial(func,*args,**kwargs):
        partial_func = functools.partial(func,*args,**kwargs)
        functools.update_wrapper(partial_func,func)
        return partial_func
    ####################################################################### Define the Optimizers and Checkpoint-saver
    optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='/home/semansou/code/long_range_prediction/outputs/checkpoints/' + config["name"] + '.hdf5',monitor='val_f1score', verbose=1, save_best_only=True,save_weights_only=False, mode='max', save_freq='epoch')
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1score', factor=0.5, patience=3, verbose=2, mode='max', cooldown=1)
    memorycheck = MemoryCheck()
    tensorboard = TensorBoard(log_dir='/home/semansou/code/long_range_prediction/outputs/tensorboard/' + config["name"], histogram_freq=0, write_graph=False, write_images=True)#,write_grads=True)
    #wandb_logger = WandbCallback()
    ####################################################################### compile
    if multi == True:
        model.compile(optimizer = optimizer, loss = loss, metrics = [loss_ce, loss_dice, loss_jaccard, loss_focalc, accuracy, precision, recall, f1score, f1score_15min, f1score_30min, f1score_45min, f1score_60min])
    else:
        model.compile(optimizer = optimizer, loss = loss, metrics = [wrapped_partial(loss_ce,y_channel=y_channel), wrapped_partial(loss_dice,y_channel=y_channel), wrapped_partial(loss_jaccard,y_channel=y_channel), wrapped_partial(loss_focalc,y_channel=y_channel), wrapped_partial(accuracy,y_channel=y_channel), wrapped_partial(precision,y_channel=y_channel), wrapped_partial(recall,y_channel=y_channel), wrapped_partial(f1score,y_channel=y_channel)])




    ####################################################################### train (fit)
    #H = model.fit(patch_train, epochs=epochs, validation_data = patch_val, verbose = 1, callbacks = [checkpointer, lr_scheduler, memorycheck, tensorboard, wandb_logger])
    ####################################################################### save results
    #with open('/home/semansou/code/long_range_prediction/outputs/history/' + config["name"], 'wb') as file_pi:
    #    pickle.dump(H.history, file_pi)
    ####################################################################### finish weights and Biases logger
    #wandb.finish()

    ####################################################################### Image generator
    X, Y, x_cord, y_cord = patch_test[0]
    Pred = model.predict(X)
    output = {'pred':Pred,'X':X, 'Y':Y, 'x_cord':x_cord, 'y_cord':y_cord,'patch_size':patch_size}
    with open('/home/semansou/code/long_range_prediction/outputs/sample_images/'+config["name"]+'.pkl','wb') as file:
        pickle.dump(output,file)
# %%

if __name__ == '__main__':
    parser = argparse.ArgumentParser("model trainer")
    parser.add_argument("--config", help = "address of .yml file with config")
    #parser.add_argument("--output", help = "output folder address normally starts like dataSequenceCNN_XXXX_multi_output/")
    args = parser.parse_args()
    address = args.config
    #dir_output = args.output

    with open(address,'r') as f:
        config = yaml.load(f,Loader=yaml.SafeLoader)

    visualize(config)
