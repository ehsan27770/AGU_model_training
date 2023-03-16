import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from DataGenerator import PatchGenerator, PatchGenerator_multioutput
from loses import loss_ce,loss_dice,loss_focalc,f1score,precision,recall
from models import Generator,Lightning_Prediction,Lightning_Prediction_ConvLSTM, Lightning_Prediction_ConvLSTM_baseline, Lightning_Prediction_CARE, Lightning_Prediction_Residual_CARE, Lightning_Prediction_CARE_Multi_Output, Lightning_Prediction_CARE_Multi_Output_variable

def configuration_generator():
    configurations = []
    #whole continent + 2019 data
    #configurations.append({"name":"1.hdf5", "model":Generator(patch_size,depth=6),           "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Generator model without contex and position"})
    #configurations.append({"name":"2.hdf5", "model":Generator(patch_size,depth= 8),          "include_position":True,  "include_time":False, "time_dimension":False, "contex":False, "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Generator model with position without contex"})#not done
    #configurations.append({"name":"3.hdf5", "model":Generator(patch_size,depth=12),          "include_position":False, "include_time":False, "time_dimension":False, "contex":True,  "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Generator model with contex without position"})#not done
    #configurations.append({"name":"4.hdf5", "model":Generator(patch_size,depth=14),          "include_position":True,  "include_time":False, "time_dimension":False, "contex":True,  "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Generator model with position and contex"})
    #configurations.append({"name":"5.hdf5", "model":Lightning_Prediction(depth=6),           "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction without contex and position"})
    #configurations.append({"name":"6.hdf5", "model":Lightning_Prediction(depth= 8),          "include_position":True,  "include_time":False, "time_dimension":False, "contex":False, "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction with position without contex"})#not done
    #configurations.append({"name":"7.hdf5", "model":Lightning_Prediction(depth=12),          "include_position":False, "include_time":False, "time_dimension":False, "contex":True,  "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction with contex without position"})#not done
    #configurations.append({"name":"8.hdf5", "model":Lightning_Prediction(depth=14),          "include_position":True,  "include_time":False, "time_dimension":False, "contex":True,  "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction with position and contex"})
    #configurations.append({"name":"9.hdf5", "model":Lightning_Prediction_ConvLSTM(depth=1),  "include_position":False, "include_time":False, "time_dimension":True,  "contex":False, "epochs":10, "batch_size":4,  "numPatches":8, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction_ConvLSTM without position and contex"})
    #configurations.append({"name":"10.hdf5", "model":Lightning_Prediction_ConvLSTM(depth=3), "include_position":True,  "include_time":False, "time_dimension":True,  "contex":False, "epochs":10, "batch_size":4,  "numPatches":8, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction_ConvLSTM with position without contex"})#not done
    #configurations.append({"name":"11.hdf5", "model":Lightning_Prediction_ConvLSTM(depth=2), "include_position":False, "include_time":False, "time_dimension":True,  "contex":True,  "epochs":10, "batch_size":4,  "numPatches":8, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction_ConvLSTM with contex without position"})#not done
    #configurations.append({"name":"12.hdf5", "model":Lightning_Prediction_ConvLSTM(depth=4), "include_position":True,  "include_time":False, "time_dimension":True,  "contex":True,  "epochs":10, "batch_size":4,  "numPatches":8, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction_ConvLSTM with position and contex"})

    #Brazil + 2019 data
    #configurations.append({"name":"13.hdf5", "model":Generator(patch_size,depth=6),          "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Generator model without contex and position Brazil"})
    #configurations.append({"name":"14.hdf5", "model":Generator(patch_size,depth= 8),         "include_position":True,  "include_time":False, "time_dimension":False, "contex":False, "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Generator model with position without contex Brazil"})#not done
    #configurations.append({"name":"15.hdf5", "model":Generator(patch_size,depth=12),         "include_position":False, "include_time":False, "time_dimension":False, "contex":True,  "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Generator model with contex without position Brazil"})#not done
    #configurations.append({"name":"16.hdf5", "model":Generator(patch_size,depth=14),         "include_position":True,  "include_time":False, "time_dimension":False, "contex":True,  "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Generator model with position and contex"})
    #configurations.append({"name":"17.hdf5", "model":Lightning_Prediction(depth=6),          "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction without contex and position Brazil"})
    #configurations.append({"name":"18.hdf5", "model":Lightning_Prediction(depth= 8),         "include_position":True,  "include_time":False, "time_dimension":False, "contex":False, "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction with position without contex Brazil"})#not done
    #configurations.append({"name":"19.hdf5", "model":Lightning_Prediction(depth=12),         "include_position":False, "include_time":False, "time_dimension":False, "contex":True,  "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction with contex without position Brazil"})#not done
    #configurations.append({"name":"20.hdf5", "model":Lightning_Prediction(depth=14),         "include_position":True,  "include_time":False, "time_dimension":False, "contex":True,  "epochs":10, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction with position and contex Brazil"})
    #configurations.append({"name":"21.hdf5", "model":Lightning_Prediction_ConvLSTM(depth=1), "include_position":False, "include_time":False, "time_dimension":True,  "contex":False, "epochs":10, "batch_size":4,  "numPatches":8, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction_ConvLSTM without position and contex Brazil"})
    #configurations.append({"name":"22.hdf5", "model":Lightning_Prediction_ConvLSTM(depth=3), "include_position":True,  "include_time":False, "time_dimension":True,  "contex":False, "epochs":10, "batch_size":4,  "numPatches":8, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction_ConvLSTM with position without contex Brazil"})#not done
    #configurations.append({"name":"23.hdf5", "model":Lightning_Prediction_ConvLSTM(depth=2), "include_position":False, "include_time":False, "time_dimension":True,  "contex":True,  "epochs":10, "batch_size":4,  "numPatches":8, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction_ConvLSTM with contex without position Brazil"})#not done
    #configurations.append({"name":"24.hdf5", "model":Lightning_Prediction_ConvLSTM(depth=4), "include_position":True,  "include_time":False, "time_dimension":True,  "contex":True,  "epochs":10, "batch_size":4,  "numPatches":8, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"Lightning_Prediction_ConvLSTM with position and contex Brazil"})

    #Brazil + 2018 data
    #configurations.append({"name":"25.hdf5", "model":Generator(patch_size,depth=6),          "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":30, "batch_size":16, "numPatches":32, "patch_size":256, "inputInd":[2,3,4,5,6,7], "region":"CUS", "bound":[411,923,444,956], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"Generator model without contex and position Brazil"})

    #CARE model for Brazil
    #configurations.append({"name":"26", "model":Lightning_Prediction_CARE(width=128,height=128,depth=11), "include_position":True, "include_time":True, "time_dimension":False, "contex":False, "epochs":30, "batch_size":16, "numPatches":32, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model"})
    #configurations.append({"name":"27", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":30, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 ce"})
    #configurations.append({"name":"28", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":30, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 dice"})
    #configurations.append({"name":"29", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":30, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model"})
    #configurations.append({"name":"30", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"CARE model"})
    #configurations.append({"name":"31", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"CARE model 2019 ce+dice"})
    #configurations.append({"name":"32", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 ce+dice"})

    #CARE model with Residual connection
    #configurations.append({"name":"33.hdf5", "model":Lightning_Prediction_Residual_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":32, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"Residual CARE model 2018 ce+dice"})
    #configurations.append({"name":"34.hdf5", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 ce+dice"})

    #for EGU conferance
    #configurations.append({"name":"EGU_2018_15min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[1,2,3,4,5,6,7,8], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 15min"})
    #configurations.append({"name":"EGU_2018_30min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 30min"})
    #configurations.append({"name":"EGU_2018_45min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[3,4,5,6,7,8,9,10], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 45min"})
    #configurations.append({"name":"EGU_2019_15min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[1,2,3,4,5,6,7,8], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"CARE model 2019 lead 15min"})
    #configurations.append({"name":"EGU_2019_30min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"CARE model 2019 lead 30min"})
    #configurations.append({"name":"EGU_2019_45min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[3,4,5,6,7,8,9,10], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"CARE model 2019 lead 45min"})
    #configurations.append({"name":"test", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":30, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 ce"})

    #fixed the order of sequence. most recent data in X is input_ind = 10
    #configurations.append({"name":"EGU_2018_15min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[3,4,5,6,7,8,9,10], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 15min"})
    #configurations.append({"name":"EGU_2018_30min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 30min"})
    #configurations.append({"name":"EGU_2018_45min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[1,2,3,4,5,6,7,8], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 lead 45min"})
    #configurations.append({"name":"EGU_2019_15min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[1,2,3,4,5,6,7,8], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"CARE model 2019 lead 15min"})
    #configurations.append({"name":"EGU_2019_30min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"CARE model 2019 lead 30min"})
    #configurations.append({"name":"EGU_2019_45min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[3,4,5,6,7,8,9,10], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019/", "description":"CARE model 2019 lead 45min"})
    #configurations.append({"name":"test", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":30, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 ce"})

    #multioutput

    #configurations.append({"name":"EGU_2019_multi_output", "model":Lightning_Prediction_CARE_Multi_Output(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/home/emclab_epfl/data/dataSequenceCNN_2019_multi_output/", "description":"CARE model 2019 multioutput"})


    #test of macbook m1
    #configurations.append({"name":"EGU_2018_15min", "model":Lightning_Prediction_CARE(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[3,4,5,6,7,8,9,10], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/Users/ehsan/Documents/EPFL/FRL/codes/long_range_prediction/data/", "description":"CARE model 2018 lead 15min"})

    #first run on EPFL IZAR
    #configurations.append({"name":"EGU_2019_multi_output", "model":Lightning_Prediction_CARE_Multi_Output(width=128,height=128,depth=8), "include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":True, "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2018_multi_output/", "description":"CARE model 2019 multioutput"})

    ########################################################################################
    # added loss and data generator to configurations
    # configurations.append({
    # "name":"multi_output_CE",
    # "model":Lightning_Prediction_CARE_Multi_Output(width=128,height=128,depth=8),
    # "loss":lambda y_true,y_pred :loss_ce(y_true,y_pred,changeLoss=True),
    # "generator":None,
    # "include_position":False,
    # "include_time":False,
    # "time_dimension":False,
    # "contex":False,
    # "epochs":20,
    # "batch_size":16,
    # "numPatches":128,
    # "patch_size":128,
    # "inputInd":[2,3,4,5,6,7,8,9],
    # "region":"CUS",
    # "bound":[500,1000,500,1000],
    # "keep_float":True,
    # "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    # "description":"CARE model multioutput with CrossEntropy loss 2019"})
########################################################################### 13.07.2021  ###########################################################################
    # configurations.append({
    # "name":"multi_output_DICE",
    # "model":Lightning_Prediction_CARE_Multi_Output(width=128,height=128,depth=8),
    # "loss":lambda y_true,y_pred :loss_dice(y_true,y_pred,changeLoss=True),
    # "generator":None,
    # "include_position":False,
    # "include_time":False,
    # "time_dimension":False,
    # "contex":False,
    # "epochs":20,
    # "batch_size":16,
    # "numPatches":128,
    # "patch_size":128,
    # "inputInd":[2,3,4,5,6,7,8,9],
    # "region":"CUS",
    # "bound":[500,1000,500,1000],
    # "keep_float":True,
    # "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    # "description":"CARE model multioutput with Dice loss 2019"})
    #
    # configurations.append({
    # "name":"multi_output_CE_AND_DICE",
    # "model":Lightning_Prediction_CARE_Multi_Output(width=128,height=128,depth=8),
    # "loss":lambda y_true,y_pred :loss_dice(y_true,y_pred,changeLoss=True) + 10 * loss_ce(y_true,y_pred,changeLoss=True),
    # "generator":None,
    # "include_position":False,
    # "include_time":False,
    # "time_dimension":False,
    # "contex":False,
    # "epochs":20,
    # "batch_size":16,
    # "numPatches":128,
    # "patch_size":128,
    # "inputInd":[2,3,4,5,6,7,8,9],
    # "region":"CUS",
    # "bound":[500,1000,500,1000],
    # "keep_float":True,
    # "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    # "description":"CARE model multioutput with CrossEntropy and Dice loss 2019"})
    #
    # configurations.append({
    # "name":"multi_output_FOCAL",
    # "model":Lightning_Prediction_CARE_Multi_Output(width=128,height=128,depth=8),
    # "loss":lambda y_true,y_pred :loss_focalc(y_true,y_pred,changeLoss=True),
    # "generator":None,
    # "include_position":False,
    # "include_time":False,
    # "time_dimension":False,
    # "contex":False,
    # "epochs":20,
    # "batch_size":16,
    # "numPatches":128,
    # "patch_size":128,
    # "inputInd":[2,3,4,5,6,7,8,9],
    # "region":"CUS",
    # "bound":[500,1000,500,1000],
    # "keep_float":True,
    # "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    # "description":"CARE model multioutput with Focal loss 2019"})
    #
    # configurations.append({
    # "name":"multi_output_FOCAL_AND_DICE",
    # "model":Lightning_Prediction_CARE_Multi_Output(width=128,height=128,depth=8),
    # "loss":lambda y_true,y_pred :loss_dice(y_true,y_pred,changeLoss=True) + 10 * loss_focalc(y_true,y_pred,changeLoss=True),
    # "generator":None,
    # "include_position":False,
    # "include_time":False,
    # "time_dimension":False,
    # "contex":False,
    # "epochs":20,
    # "batch_size":16,
    # "numPatches":128,
    # "patch_size":128,
    # "inputInd":[2,3,4,5,6,7,8,9],
    # "region":"CUS",
    # "bound":[500,1000,500,1000],
    # "keep_float":True,
    # "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    # "description":"CARE model multioutput with Focal and Dice loss 2019"})
########################################################################### 27.07.2021  ###########################################################################
    # def func(y_true,y_pred):
    #     return loss_dice(y_true,y_pred,changeLoss=True) + 10 * loss_ce(y_true,y_pred,changeLoss=True)
    #
    # configurations.append({
    # "name":"multi_output_func_small",
    # "model":Lightning_Prediction_CARE_Multi_Output_variable(width=128,height=128,depth=8,multi = True, layers = [16, 32, 64, 128, 256]),
    # "loss":func,
    # "generator":None,
    # "include_position":False,
    # "include_time":False,
    # "time_dimension":False,
    # "contex":False,
    # "epochs":20,
    # "batch_size":16,
    # "numPatches":128,
    # "patch_size":128,
    # "inputInd":[2,3,4,5,6,7,8,9],
    # "region":"CUS",
    # "bound":[500,1000,500,1000],
    # "keep_float":True,
    # "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    # "description":"CARE model multioutput with Dice loss 2019"})
    #
    # configurations.append({
    # "name":"multi_output_func_medium",
    # "model":Lightning_Prediction_CARE_Multi_Output_variable(width=128,height=128,depth=8,multi = True, layers = [32, 64, 128, 256, 512]),
    # "loss":func,
    # "generator":None,
    # "include_position":False,
    # "include_time":False,
    # "time_dimension":False,
    # "contex":False,
    # "epochs":20,
    # "batch_size":16,
    # "numPatches":128,
    # "patch_size":128,
    # "inputInd":[2,3,4,5,6,7,8,9],
    # "region":"CUS",
    # "bound":[500,1000,500,1000],
    # "keep_float":True,
    # "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    # "description":"CARE model multioutput with Dice loss 2019"})
    #
    # configurations.append({
    # "name":"multi_output_func_large",
    # "model":Lightning_Prediction_CARE_Multi_Output_variable(width=128,height=128,depth=8,multi = True, layers = [64, 128, 256, 512, 1024]),
    # "loss":func,
    # "generator":None,
    # "include_position":False,
    # "include_time":False,
    # "time_dimension":False,
    # "contex":False,
    # "epochs":20,
    # "batch_size":16,
    # "numPatches":128,
    # "patch_size":128,
    # "inputInd":[2,3,4,5,6,7,8,9],
    # "region":"CUS",
    # "bound":[500,1000,500,1000],
    # "keep_float":True,
    # "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    # "description":"CARE model multioutput with Dice loss 2019"})

########################################################################### 28.07.2021  ###########################################################################
    def func(y_true,y_pred):
        return loss_dice(y_true,y_pred,changeLoss=True) + 10 * loss_ce(y_true,y_pred,changeLoss=True)

    configurations.append({
    "name":"multi_output_func_small_wide_10",
    "model":Lightning_Prediction_CARE_Multi_Output_variable(width=128,height=128,depth=8,multi = True, layers = [16, 32, 64, 128, 256]),
    "loss":func,
    "generator":None,
    "include_position":False,
    "include_time":False,
    "time_dimension":False,
    "contex":False,
    "epochs":20,
    "batch_size":16,
    "numPatches":128,
    "patch_size":128,
    "inputInd":[2,3,4,5,6,7,8,9],
    "region":"CUS",
    "bound":[500,1000,500,1000],
    "keep_float":True,
    "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    "description":"CARE model multioutput with Dice loss 2019"})

    configurations.append({
    "name":"multi_output_func_small_narrow_10",
    "model":Lightning_Prediction_CARE_Multi_Output_variable(width=128,height=128,depth=8,multi = True, layers = [16, 24, 40, 72, 136]),
    "loss":func,
    "generator":None,
    "include_position":False,
    "include_time":False,
    "time_dimension":False,
    "contex":False,
    "epochs":20,
    "batch_size":16,
    "numPatches":128,
    "patch_size":128,
    "inputInd":[2,3,4,5,6,7,8,9],
    "region":"CUS",
    "bound":[500,1000,500,1000],
    "keep_float":True,
    "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    "description":"CARE model multioutput with Dice loss 2019"})

    def func(y_true,y_pred):
        return loss_dice(y_true,y_pred,changeLoss=True) + 5 * loss_ce(y_true,y_pred,changeLoss=True)

    configurations.append({
    "name":"multi_output_func_small_wide_5",
    "model":Lightning_Prediction_CARE_Multi_Output_variable(width=128,height=128,depth=8,multi = True, layers = [16, 32, 64, 128, 256]),
    "loss":func,
    "generator":None,
    "include_position":False,
    "include_time":False,
    "time_dimension":False,
    "contex":False,
    "epochs":20,
    "batch_size":16,
    "numPatches":128,
    "patch_size":128,
    "inputInd":[2,3,4,5,6,7,8,9],
    "region":"CUS",
    "bound":[500,1000,500,1000],
    "keep_float":True,
    "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    "description":"CARE model multioutput with Dice loss 2019"})

    configurations.append({
    "name":"multi_output_func_small_narrow_5",
    "model":Lightning_Prediction_CARE_Multi_Output_variable(width=128,height=128,depth=8,multi = True, layers = [16, 24, 40, 72, 136]),
    "loss":func,
    "generator":None,
    "include_position":False,
    "include_time":False,
    "time_dimension":False,
    "contex":False,
    "epochs":20,
    "batch_size":16,
    "numPatches":128,
    "patch_size":128,
    "inputInd":[2,3,4,5,6,7,8,9],
    "region":"CUS",
    "bound":[500,1000,500,1000],
    "keep_float":True,
    "path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
    "description":"CARE model multioutput with Dice loss 2019"})

    return configurations

# %%
