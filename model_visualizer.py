import os
import sys
import resource


import numpy as np
import matplotlib.pyplot as plt


import keras
import tensorflow as tf

from models import  Lightning_Prediction_CARE_Multi_Output, Lightning_Prediction_CARE_Multi_Output_variable, Lightning_Prediction_ResUnet, Persistence

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Activation
#from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.python.keras.layers import concatenate, UpSampling2D, BatchNormalization,Dropout
from tensorflow.keras.layers import Add, Average, Lambda, Conv2D, Conv3D, MaxPooling2D, add, Concatenate, concatenate, UpSampling2D, BatchNormalization, Dropout, ReLU, LeakyReLU, ConvLSTM2D, Attention#, MultiHeadAttention
from tensorflow.keras.initializers import he_normal

import visualkeras

from PIL import ImageFont
from collections import defaultdict

# %%
config = {
"name":"multi_output_func_small_wide_5_loss_1",
"model":1,
"width" :128,
"height":128,
"depth":8,
"multi":True,
"layers":[16,32,64,128,256],
"loss":1,
"generator":2,
"include_position":False,
"include_time":False,
"time_dimension":False,
"contex":False,
"epochs":30,
"batch_size":16,
"numPatches":128,
"patch_size":128,
"inputInd":[2,3,4,5,6,7,8,9],
"region":"CUS",
"bound":[500,1000,500,1000],
"keep_float":True,
"load_weight":False,
"skip_connection":True,
"use_amplitude":False,
"do_sum":False,
"path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
"description":"CARE model multioutput with Dice loss 2019"}

#%%
####################################################################### fixed
path = config["path"]
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
use_amplitude = config["use_amplitude"]
do_sum = config["do_sum"]
skip_connection = config["skip_connection"]
# %%
color_map = defaultdict(dict)
#color_map[Conv2D]['fill'] = 'deeppink'
color_map[Concatenate]['fill'] = 'gray'
color_map[Dropout]['fill'] = 'green'
color_map[MaxPooling2D]['fill'] = 'orange'
#color_map[UpSampling2D]['fill'] = 'pink'
color_map[Add]['fill'] = 'red'
# %%
layers = [16, 32, 64, 128]
width = 256
height = 256
depth = 8
skip_connection =False
model = Lightning_Prediction_CARE_Multi_Output_variable(width = width, height = height, depth = depth, multi = multi, layers = layers, skip_connection = skip_connection)
# %%
visualkeras.layered_view(model,legend=True,color_map=color_map,draw_volume=True,scale_xy=2,min_xy=0,scale_z=0.05,font=ImageFont.truetype("/System/Library/Fonts/Geneva.ttf", 30),type_ignore=[],to_file='autoencoder.png')
# %%
tf.keras.utils.plot_model(model,show_shapes=True,show_dtype=False,show_layer_names=False,rankdir='TB',to_file='autoencoder_keras.png')
# %%
skip_connection =True
model = Lightning_Prediction_CARE_Multi_Output_variable(width = width, height = height, depth = depth, multi = multi, layers = layers, skip_connection = skip_connection)
# %%


visualkeras.layered_view(model,legend=True,color_map=color_map,draw_volume=True,scale_xy=2,min_xy=0,scale_z=0.1,font=ImageFont.truetype("/System/Library/Fonts/Geneva.ttf", 20),type_ignore=[],to_file='unet.png')

# %%
tf.keras.utils.plot_model(model,show_shapes=True,show_dtype=False,show_layer_names=False,rankdir='TB',to_file='unet_keras.png')


# %%
model = Lightning_Prediction_ResUnet(width = 256, height = 256, depth = 8, multi = True, layers = [16, 32, 64, 128])

# %%
visualkeras.layered_view(model,legend=True,color_map=color_map,draw_volume=True,scale_xy=2,min_xy=0,scale_z=0.05,font=ImageFont.truetype("/System/Library/Fonts/Geneva.ttf", 30),type_ignore=[ReLU],to_file='resunet.png')
# %%
tf.keras.utils.plot_model(model,show_shapes=True,show_dtype=False,show_layer_names=False,rankdir='TB',to_file='resunet_keras.png')
# %%
import functools
def func(a,b,c):
    return a+2*b-c
newfunc = functools.partial(func,c=0,name='test')
newfunc.__name__ = 'test'
newfunc.__name__
# %%
model = Persistence(multi=True)
tf.keras.utils.plot_model(model)
visualkeras.layered_view(model,legend=True)
model.summary()
