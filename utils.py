from __future__ import print_function, unicode_literals, absolute_import, division
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="1";

import numpy as np
import matplotlib.pyplot as plt
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches
import numpy as np
import matplotlib.pyplot as plt
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import random
from tqdm import tnrange
import os, fnmatch

def patchMaker(X,y,inputInd,nmlx,nmly,flashThresh, modePatch, patchSize,numPatches,region,normalizer,x_mean,x_std,x_max):
    X_patch=[]
    y_patch=[]
    validInd = []
    k=0
    if region == 'US':
        X = X[:, 90:318, 55:676, :]
        y = y[:, 90:318, 55:676, :]
    elif region == 'BRZ':
        X = X[:, 485:850, 500:900, :]
        y = y[:, 485:850, 500:900, :]

    if modePatch=='rnd':
        for _ in tnrange(5000000, desc='Train Patch'):
            ind0 = int(np.random.randint(X.shape[0]))
            startX = int(np.random.randint(X.shape[1]-patchSize))
            startY = int(np.random.randint(X.shape[2]-patchSize))
            y_patch_temp = y[ind0:ind0+1,startX:startX+patchSize,startY:startY+patchSize,0:1]
            if np.sum(np.sign(y_patch_temp))>flashThresh:
                X_patch.append(X[ind0:ind0+1,startX:startX+patchSize,startY:startY+patchSize,:])
                y_patch.append(y[ind0:ind0+1,startX:startX+patchSize,startY:startY+patchSize,0:1])
                k+=1
            if k>numPatches:
                break
    elif modePatch=='normal':
        for startY in range(0, X.shape[2]-patchSize, patchSize):
            for startX in range(0, X.shape[1]-patchSize, patchSize):
                #print(startY, startY+patchSize, startX, startX+patchSize)
                X_patch.append(X[:,startX:startX+patchSize,startY:startY+patchSize,:])
                y_patch.append(y[:,startX:startX+patchSize,startY:startY+patchSize,0:1])
    X_patch=np.concatenate(X_patch,axis=0)
    y_patch=np.concatenate(y_patch,axis=0)
    if modePatch=='normal':
        for i in range(y_patch.shape[0]):
            if np.sum(np.sign(y_patch[i, :, :, 0])) > flashThresh:
                validInd.append(i)

        if len(validInd)>numPatches:
            #random.shuffle(validInd)
            #validInd=np.concatenate(validInd,axis=0)
            validInd=validInd[0:numPatches-1]
        X_patch = X_patch[validInd, ...]
        y_patch = y_patch[validInd, ...]

    X_patch = X_patch.astype('float32')
    y_patch = y_patch.astype('float32')

    if normalizer == 'standardize':
        for i in range(11):
            X_patch[:, :, :, i] = (X_patch[:, :, :, i] - x_mean[i]) / x_std[i]
        print('Normalization is applied successfully')
    elif normalizer=='normalize':
        for i in range(11):
            X_patch[:, :, :, i] = X_patch[:, :, :, i] / x_max[i]
        print('Normalization is applied successfully')
    elif normalizer == 'fix':
        for i in range(11):
            if nmlx[i] > 0:
                X_patch[:, :, :, i] /= nmlx[i]
        print('Normalization is applied successfully')

    if X_patch.shape[-1]>22:
        X_patch[:, :, :, 22] /= 1086
        X_patch[:, :, :, 23] /= 1086
    y_patch /= nmly
    X_patch=X_patch[...,inputInd]
    return X_patch,y_patch


def modelMaker(inputInd,y_train,batchSize,
                            numPatches,trainEpochs,
                            unetNfirst,unetKernSize,
                            unetNdepth,trainLoss,
                            tr_lr,train_reduce_lr_0,modeConfig,
                            leadTime,region,alpha,alphaF,beta,
                            gamma,binning,thresh_t,thresh_p,changeLoss,
                            batch_norm,activation,dropout,binning_layer):
    config = Config('SYXC',
                    # X_train.shape[-1],
                    len(inputInd),
                    y_train.shape[-1],
                    train_batch_size=batchSize,
                    train_steps_per_epoch=int(numPatches / batchSize),
                    unet_residual=False,
                    train_epochs=trainEpochs,
                    unet_n_first=unetNfirst,
                    unet_kern_size=unetKernSize,
                    unet_n_depth=unetNdepth,
                    train_loss=trainLoss,
                    train_reduce_lr=train_reduce_lr_0,
                    train_learning_rate=tr_lr,
                    alpha=alpha,
                    alphaF=alphaF,
                    beta=beta,
                    gamma=gamma,
                    binning=binning,
                    thresh_t=thresh_t,
                    thresh_p=thresh_p,
                    changeLoss=changeLoss,
                    batch_norm=batch_norm,
                    activation=activation,
                    dropout=dropout,
                    binning_layer=binning_layer
                    )

    if config.train_loss in ['ce', 'wce','dice','focal', 'focala','focalc']:
        config.unet_last_activation = 'sigmoid'

    print(config)
    vars(config)

    if modeConfig == 'universal':
        model = CARE(config, 'preLight_Nfirst%d_Depth%d_Lead%d_Input%d' % (config.unet_n_first, config.unet_n_depth, leadTime,len(inputInd)),
                     basedir='models')
    elif modeConfig == 'specified':
        model = CARE(config, 'preLight_Nfirst%d_Depth%d_%s_Lead%d_Reg%s' % (
        config.unet_n_first, config.unet_n_depth, config.train_loss, leadTime, region), basedir='models')



    return model, config

def visualizer(X_patch,y_patch,model,mode,pMax,thresh,filterMode,idx):
    numVis=5
    ind=np.random.permutation(X_patch.shape[0])
    ind=ind[:numVis]

    X_vis=[]
    y_vis=[]
    X_vis=X_patch[ind,:,:,:]
    y_vis=y_patch[ind,:,:,:]

    y_pred = np.zeros_like(y_vis)

    #for i in range(X_vis.shape[0]):
    #    y_pred[i] = model.predict(X_vis[i,:,:,inputInd],'YXC',normalizer=None)

    y_pred = model.keras_model.predict(X_vis[:,:,:,:])
    if filterMode:
        y_pred=(y_pred>thresh)*1
    plt.figure(figsize=(15,10))
    plot_some(np.sign(X_vis[:,:,:,idx]), y_vis[:,:,:,0],y_pred[:,:,:,0],pmax=pMax)
    plt.suptitle('5 example %s patches\n'      
                 'top row: input (source),  '          
                 'middle row: target (ground truth),  '
                 'bottom row: predicted from source'%mode);

def cmMaker(X_patch,y_patch,model,mode,thresh):
    y_true=y_patch.flatten()
    y_pred=model.keras_model.predict(X_patch).flatten()
    y_pred=(y_pred>thresh)*1
    cm=confusion_matrix(y_true,y_pred)
    print('Confusion Matrix for %s: \n'%mode, cm)
    print('--------------------')
    print('Recall    : % {}'.format(round(100 * (cm[1][1] / (cm[1][0] + cm[1][1])))))
    print('Precision : % {}'.format(round(100 * (cm[1][1] / (cm[0][1] + cm[1][1])))))
    #print('POD : % {}'.format(round(100 * (cm[1][1] / (cm[1][0] + cm[1][1])))))
    #print('FAR : % {}'.format(100 - round(100 * (cm[1][1] / (cm[0][1] + cm[1][1])))))
    #print('CSI : % {}'.format(round(100 * (cm[1][1] / (cm[1][1] + cm[0][1] + cm[1][0])))))
    #print('HSS : % {}'.format(round(100 * hssMaker(cm[1][1], cm[0][1], cm[1][0], cm[0][0]))))
    print('---------------------------------------------------------')



def getData (dir_Input,dir_Target,leadTime,numSamples,valRatio,includeLatLon):

    #numSamples = 250
    #dir_Input = '/Volumes/LaCie/dataInput_processed_2019/'
    #dir_Target = '/Volumes/LaCie/dataTarget_processed_2019/'
    gridSize = 1086
    inputInd = np.arange(0, 22, 1)  # [2,4,5,7,8,9,10]
    targetInd = [0]
    numChannels = 22

    listFiles=fnmatch.filter(os.listdir(dir_Target), 'dataTarget_ABI*')


    listFiles_train=listFiles[:round(len(listFiles)*(1-valRatio))]
    listFiles_val = listFiles[round(len(listFiles) * (1 - valRatio)):]
    random.shuffle(listFiles_train)
    random.shuffle(listFiles_val)
    #listFiles=sorted(listFiles)

    X_train,y_train=dataReader(listFiles_train,dir_Input,dir_Target,round((1-valRatio)*numSamples),gridSize,numChannels,leadTime)
    X_train,y_train=dataPreProcessor(X_train,y_train,includeLatLon,gridSize,inputInd,targetInd)

    X_val, y_val = dataReader(listFiles_val,dir_Input,dir_Target,round((valRatio) * numSamples), gridSize, numChannels,leadTime)
    X_val, y_val = dataPreProcessor(X_val, y_val, includeLatLon, gridSize, inputInd, targetInd)

    return X_train,y_train,X_val,y_val


def dataReader(listFiles,dir_Input,dir_Target,numSamples,gridSize,numChannels,leadTime):
    X = np.zeros((numSamples, numChannels, gridSize, gridSize), np.uint16)
    y = np.zeros((numSamples, 4, gridSize, gridSize), np.uint16)
    index=0
    for fileName in listFiles:
        if index<numSamples:
            t_index=fileName.split("_")[2]
            #print("index=" + str(index + 1) + "/" + str(numSamples)+', time slice: '+str(t_index.split(".")[0]))
            try:
                X[index,:,:,:]=np.load(dir_Input+"dataInput_ABI_"+str(int(t_index.split(".")[0])-leadTime*60)+".npy")
                y[index,:,:,:]=np.load(dir_Target+"dataTarget_ABI_"+t_index)
                index = index + 1
            except:
                pass

    return X,y

def dataPreProcessor (X,y,includeLatLon,gridSize,inputInd,targetInd):

    X = X[:, inputInd, :, :]
    y = y[:, targetInd, :, :]


    if includeLatLon:
        lat = np.linspace(0, gridSize, gridSize, dtype='uint16')
        lon = np.linspace(0, gridSize, gridSize, dtype='uint16')
        latInd, lonInd = np.meshgrid(lat, lon)
        latInd = np.tile(latInd, (X.shape[0], 1, 1, 1))
        lonInd = np.tile(lonInd, (X.shape[0], 1, 1, 1))
        X = np.concatenate((X, latInd, lonInd), axis=1)

        latInd = None
        lonInd = None

    X = np.moveaxis(X, 1, -1)
    y = np.moveaxis(y, 1, -1)
    return X,y


def binningMaker(y_true, y_pred, binning=4, thresh_t=0.1):
    filters = np.ones((binning, binning, 1, 1)) / binning / binning

    y_true = tf.nn.conv2d(
        y_true,
        filters,
        binning,
        padding='VALID')
    y_pred = tf.nn.conv2d(
        y_pred,
        filters,
        binning,
        padding='VALID')
    return y_true, y_pred


def resultMaker(y_true, y_pred, binning=4, thresh_t=0.1, thresh_p=0.1):
    sess = tf.Session()
    # y_true=np.sign(X_patch_val[:,:,:,5:6])
    # y_pred=model.keras_model.predict(X_patch_val)
    y_true_, y_pred_ = binningMaker(y_true, y_pred, binning=binning)

    y_true_ = tf.keras.backend.flatten(y_true_)
    y_true_ = tf.keras.backend.cast(tf.math.greater(y_true_, thresh_t), tf.float32)

    y_pred_ = tf.keras.backend.flatten(y_pred_)
    y_pred_ = tf.keras.backend.cast(tf.math.greater(y_pred_, thresh_p), tf.float32)

    cmTemp = tf.zeros([2, 2])
    cmTemp = tf.math.confusion_matrix(y_true_, y_pred_, num_classes=2)
    cm = sess.run(cmTemp)
    per = (cm[1][1] / (cm[0][1] + cm[1][1]))
    rec = (cm[1][1] / (cm[1][0] + cm[1][1]))

    print('Confusion Matrix : \n', cm)
    print('--------------------')
    print('Precision : % {}'.format(round(100 * per)))
    print('Recall    : % {}'.format(round(100 * rec)))
    print('F1-score    : % {}'.format(round(100 * 2 * per * rec / (per + rec))))
    return cm
