import os
import resource

import numpy as np
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

import random

from utils_colab import cmMaker,binningMaker,resultMaker


#for CARE model
#from csbdeep.models import Config, CARE
#for CARE model


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import concatenate, UpSampling2D, BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard

import time

from sklearn.model_selection import train_test_split
import datetime





from DataGenerator import FullPatchGenerator, ContextualPatchGenerator, FromRawGenerator, PatchGenerator
from loses import loss_ce,loss_dice,loss_f1score,loss_focalc,loss_precision,loss_recall
from models import Generator,Lightning_Prediction,Lightning_Prediction_ConvLSTM, Lightning_Prediction_ConvLSTM_baseline, Lightning_Prediction_CARE, Lightning_Prediction_Residual_CARE, Lightning_Prediction_CARE_No_Dropout
from callbacks import MemoryCheck

def train():


    batch_size = 16
    numPatches = 32
    patch_size = 256

    configurations = []

    #fixed the order of sequence. most recent data in X is input_ind = 10
    #configurations.append({"name":"EGU_2018_15min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":16, "patch_size":128, "inputInd":[3,4,5,6,7,8,9,10], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"../data/", "description":"CARE model 2018 lead 15min"})
    #configurations.append({"name":"EGU_2018_30min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":16, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"../data/", "description":"CARE model 2018 lead 30min"})
    #configurations.append({"name":"EGU_2018_45min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":16, "patch_size":128, "inputInd":[1,2,3,4,5,6,7,8], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"../data/", "description":"CARE model 2018 lead 45min"})
    configurations.append({"name":"EGU_2018_60min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":16, "patch_size":128, "inputInd":[0,1,2,3,4,5,6,7], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"../data/", "description":"CARE model 2018 lead 60min"})


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
        #code_path = '/home/emclab_epfl/code/'
        code_path = './Lat-Lon/'
        #data_path = '/home/emclab_epfl/data/'



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

        patch_train = PatchGenerator(path+'val/',time_path=path+'val/',position_path=code_path,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=11,flashThresh=10)
        patch_val   = PatchGenerator(path+'val/'  ,time_path=path+'val/',position_path=code_path,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=11,flashThresh=1)
        #patch_test  = PatchGenerator(path+'test/' ,time_path=path+'test/',position_path=code_path,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=11)

        #print(os.listdir(path+'train/'))
        a,b = patch_train[0]
        print(a.shape,b.shape)

        model = config["model"]
        try:
            model.load_weights('./training_checkpoints/'+config["name"])
        except:
            pass

        print(patch_train.xx.shape)
        #print(len(patch_train),len(patch_val),len(patch_test))

        ## Define the Optimizers and Checkpoint-saver
        optimizer = tf.keras.optimizers.Adam(1e-4)
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./training_checkpoints/'+config["name"],monitor='val_loss_f1score', verbose=1, save_best_only=True,save_weights_only=False, mode='max', save_freq='epoch')
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss_f1score', factor=0.5, patience=3, verbose=2, mode='max', cooldown=1)
        memorycheck = MemoryCheck()
        tensorboard = TensorBoard(log_dir='models/'+config["name"], histogram_freq=0, write_graph=False, write_images=True,write_grads=True)

        model.summary()
        ## compile and train
        #model.compile(optimizer = optimizer,loss=lambda x,y :loss_dice(x,y,changeLoss=True) + 10* loss_ce(x,y,changeLoss=True),metrics=[loss_ce,loss_dice,loss_f1score,loss_precision,loss_recall])#loss_focalc
        model.compile(optimizer = optimizer,loss=lambda x,y :loss_dice(x,y,changeLoss=False),metrics=[loss_ce,loss_dice,loss_f1score,loss_precision,loss_recall])#loss_focalc


        H = model.fit(patch_train,epochs=epochs,validation_data=patch_val, verbose=1 ,callbacks=[checkpointer,lr_scheduler,memorycheck,tensorboard])


# %%
path = "../data/"
code_path = './Lat-Lon/'
patch_train = PatchGenerator(path+'val/',time_path=path+'val/',position_path=code_path,include_position=False,include_time=False,time_dimension=False,contex=False,region="CUS",bound=[500,1000,500,1000],batch_size=16,numPatches=16,patch_size=128,inputInd=[0,1,2,3,4,5,6,7,8],keep_float=True,n_channels=11,flashThresh=10)


a,b = patch_train[0]
print(a.shape,b.shape)

plt.imshow(b[1,:,:,0])
plt.imshow(a[1,:,:,0])
plt.imshow(a[1,:,:,8])
# %%



if __name__ == '__main__':
    train()
