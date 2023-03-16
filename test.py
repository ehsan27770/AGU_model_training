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

import visualkeras




from DataGenerator import FullPatchGenerator, ContextualPatchGenerator, FromRawGenerator, PatchGenerator
from loses import loss_ce,loss_dice,loss_f1score,loss_focalc,loss_precision,loss_recall
from models import Generator,Lightning_Prediction,Lightning_Prediction_ConvLSTM, Lightning_Prediction_ConvLSTM_baseline, Lightning_Prediction_CARE, Lightning_Prediction_Residual_CARE, Lightning_Prediction_CARE_No_Dropout
from callbacks import MemoryCheck

def train():


    batch_size = 16
    numPatches = 32
    patch_size = 256

    configurations = []

    #CARE model for Brazil

    configurations.append({"checkpoint":"for_visualization.hdf5", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":30, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 ce"})

    for config in configurations:

        model = config["model"]
        model.save('./training_checkpoints/'+config["checkpoint"])
        return model






if __name__ == '__main__':
    model = train()
    visualkeras.layered_view(model).show()
