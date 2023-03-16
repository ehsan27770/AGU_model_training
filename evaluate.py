import os

import numpy as np
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

import random

import time

from sklearn.model_selection import train_test_split
import datetime




from DataGenerator import FullPatchGenerator, ContextualPatchGenerator, FromRawGenerator, PatchGenerator
from loses import loss_ce,loss_dice,loss_f1score,loss_focalc,loss_precision,loss_recall
from models import Generator,Lightning_Prediction,Lightning_Prediction_ConvLSTM, Lightning_Prediction_ConvLSTM_baseline, Lightning_Prediction_CARE
from callbacks import MemoryCheck

def evaluate():
    patch_size = 256
    configurations = []

    #for EGU conferance
    configurations.append({"name":"EGU_2018_15min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[1,2,3,4,5,6,7,8], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 15min"})
    configurations.append({"name":"EGU_2018_30min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 30min"})
    configurations.append({"name":"EGU_2018_45min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[3,4,5,6,7,8,9,10], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 45min"})

    file = open("result.txt","a+")

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
        code_path = '/home/emclab_epfl/code/'
        data_path = '/home/emclab_epfl/data/'



        #variables
        patch_size = config["patch_size"]
        inputInd = config["inputInd"]
        batch_size = config["batch_size"]
        numPatches = config["numPatches"]
        contex = config["contex"]
        region = config["region"]
        bound = config["bound"]
        include_position = config["include_position"]
        include_time = config["include_time"]
        time_dimension = config["time_dimension"]
        epochs = config["epochs"]

        patch_train = PatchGenerator(path+'train/',time_path=path+'train/',position_path=code_path,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,n_channels=11,flashThresh=10)
        patch_val   = PatchGenerator(path+'val/'  ,time_path=path+'val/',position_path=code_path,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,n_channels=11,flashThresh=1)
        patch_test  = PatchGenerator(path+'test/' ,time_path=path+'test/',position_path=code_path,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,n_channels=11)

        model = config["model"]
        model.load_weights('./training_checkpoints/'+config["name"]+'.hdf5')

        ## Define the Optimizers and Checkpoint-saver
        optimizer = tf.keras.optimizers.Adam(1e-4)

        ## compile and train
        model.compile(optimizer = optimizer,loss=lambda x,y :loss_dice(x,y,changeLoss=True) + 10* loss_ce(x,y,changeLoss=True),metrics=[loss_ce,loss_dice,loss_f1score,loss_precision,loss_recall])#loss_focalc

        n = 5
        result = np.empty((n, 6))
        for i in range(n):
            H = model.evaluate(patch_train, verbose=1)
            result[i,] = np.array(H)

        file.write("%s" % config["description"])
        for item in np.mean(result,axis=0).tolist():
            file.write(",%s" % item)
        file.write("\n")
    file.close()

if __name__ == '__main__':
    evaluate()
