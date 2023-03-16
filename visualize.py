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




from DataGenerator import PatchGenerator, PatchGenerator_multioutput
from loses import loss_ce, loss_dice, loss_jaccard, loss_focalc, f1score, precision, recall, accuracy, f1score_15min, f1score_30min, f1score_45min, f1score_60min, loss_1, loss_2
from models import Lightning_Prediction_CARE_Multi_Output, Lightning_Prediction_CARE_Multi_Output_variable, Lightning_Prediction_ResUnet, Persistence
from callbacks import MemoryCheck

#from extra import draw_images
import pickle

def visualize():
    patch_size = 256
    configurations = []


    #CARE model for Brazil
    # configurations.append({"checkpoint":"27.hdf5", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":30, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 ce"})
    # configurations.append({"checkpoint":"28.hdf5", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":30, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 dice"})
    # configurations.append({"checkpoint":"32.hdf5", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 ce+dice"})
    # configurations.append({"checkpoint":"31.hdf5", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"CARE model 2019 ce+dice"})

    #EGU
    configurations.append({"name":"EGU_2018_15min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[1,2,3,4,5,6,7,8], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 15min"})
    configurations.append({"name":"EGU_2018_30min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 30min"})
    configurations.append({"name":"EGU_2018_45min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[3,4,5,6,7,8,9,10], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 45min"})






    #fixed
    path = "/home/emclab_epfl/data/dataSequenceCNN_2018/"
    code_path = '/home/emclab_epfl/code/'
    data_path = '/home/emclab_epfl/data/'
    patch_size = 128
    inputInd = [1,2,3,4,5,6,7,8,9,10]
    region = "CUS"
    bound = [500,1000,500,1000]
    batch_size = 16
    numPatches = 16

    #patch_train = PatchGenerator(path+'train/',time_path=path+'train/',position_path=code_path,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,n_channels=11,flashThresh=10)
    #patch_val   = PatchGenerator(path+'val/'  ,time_path=path+'val/',position_path=code_path,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,n_channels=11,flashThresh=1)
    patch_test  = PatchGenerator(path+'train/' ,time_path=path+'train/',position_path=code_path,include_position=True,include_time=True,time_dimension=False,contex=False,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,n_channels=11,flashThresh=10,p_ratio=0,keep_float=True)
    X,Y = patch_test[0]
    pred = []
    name = []

    for index,config in enumerate(configurations):
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")
        print(config["description"])
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")

        #variables

        contex = config["contex"]
        include_position = config["include_position"]
        include_time = config["include_time"]
        time_dimension = config["time_dimension"]


        model = config["model"]
        model.load_weights('./training_checkpoints/'+config["name"]+'.hdf5')

        ## Define the Optimizers and Checkpoint-saver
        optimizer = tf.keras.optimizers.Adam(1e-4)

        ## compile and train
        model.compile(optimizer = optimizer,loss=lambda x,y :loss_dice(x,y,changeLoss=True) + 10* loss_ce(x,y,changeLoss=True),metrics=[loss_ce,loss_dice,loss_f1score,loss_precision,loss_recall])#loss_focalc


        lon = X[:,:,:,-3]
        lat = X[:,:,:,-2]
        x = X.astype('uint16')
        Y = Y.astype('uint16')

        if include_time == False and include_position == False:
            x = np.delete(x,[10,11,12],axis=3)
        elif include_time == False and include_position == True:
            x = np.delete(x,[12],axis=3)
        elif include_time == True and include_position == False:
            x = np.delete(x,[10,11],axis=3)

        # if include_position == True:
        #     if contex == True:
        #         pass
        #     elif contex == False:
        #         x = np.delete(x,[1,3,5,7,9,11],axis=3)
        # elif include_position == False:
        #     x = np.delete(x,[12,13],axis=3)
        #     if contex == True:
        #         pass
        #     elif contex == False:
        #         x = np.delete(x,[1,3,5,7,9,11],axis=3)

        # if time_dimension == True:
        #     if contex == False:
        #         if full == False:
        #             #temp = np.empty((numPatches,6,256,256,1))
        #             #x = np.moveaxis(np.expand_dims(x,axis=4),-1,1)
        #             x = np.reshape(x,(numPatches,6,256,256,1))
        #         elif full == True:
        #             #temp = np.empty((numPatches,6,256,256,3))
        #             x = np.concatenate([np.reshape(x,(numPatches,6,256,256,1)),np.repeat(np.expand_dims(x[:,:,:,6:8],axis=1),6,axis=1)],axis=4)
        #     elif contex == True:
        #         if full == False:
        #             #temp = np.empty((numPatches,6,256,256,2))
        #             x = np.reshape(x,(numPatches,6,256,256,2))
        #         elif full == True:
        #             temp = np.empty((numPatches,6,256,256,4))
        #             x = np.concatenate([np.reshape(x[:,:,:,0:12],(numPatches,6,256,256,2)),np.repeat(np.expand_dims(x[:,:,:,12:14],axis=1),6,axis=1)],axis=4)

        if index == 0:
            x_in = np.delete(x,[8,9],axis=3)
        elif index == 1:
            x_in = np.delete(x,[0,9],axis=3)
        elif index == 2 :
            x_in = np.delete(x,[0,1],axis=3)

        pred.append(model.predict(x_in))
        name.append(config["description"])

    output = {'pred':pred,'X':X, 'Y':Y}
    with open('EGU.pkl','wb') as file:
        pickle.dump(output,file)

    # print(pred[0].dtype,X.dtype)
    #
    # for i in range(numPatches):
    #     display_list = [ Y[i,:,:,0].astype("float32") ] + [ X[i,:,:,-4].astype("float32") ] + [pred[j][i,:,:,0] for j in range(len(pred))]
    #     f1_score = [tf.keras.backend.get_value(loss_f1score(np.expand_dims(display_list[j],(0,3)),np.expand_dims(display_list[0],(0,3)))) for j in range(len(display_list))]
    #     #title = ['Persistence', 'Generator', 'Generator+p+c' ,'google','google+p+c', 'ConvLSTM', 'ConvLSTM+p+c', 'Ground Truth']
    #     title = ['Ground Truth', 'Persistence', 'CARE 2018 ce', 'CARE 2018 dice', 'CARE 2018 ce+dice', 'CARE 2019 ce+dice']
    #     title_final = [zip1+'\n F1={:.4f}'.format(zip2) for (zip1,zip2) in list(zip(title,f1_score))]
    #     draw_images(display_list,title=title_final,name=str(i))



if __name__ == '__main__':
    visualize()
