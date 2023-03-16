from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/'My Drive'/preLight/Codes/utils_colab.py /usr/local/lib/python3.6/dist-packages

#from __future__ import print_function, unicode_literals, absolute_import, division
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

#from tifffile import imread


#from tifffile import imread

#from tensorflow import keras
import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

import random
#from tqdm import tnrange
from utils_colab import patchMaker,cmMaker,binningMaker,resultMaker
%load_ext autoreload
%autoreload 2
#from sunpy.image.resample import resample

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import concatenate, UpSampling2D, BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard

import time
from IPython import display


leadTime = 0 #leadtime in minutes
gridSize=1086

#X=np.load('Data/dataX/XN_2018_45.npy').astype('uint16')
#X=np.load('Data/dataX/XN_2018_%d.npy'%leadTime).astype('uint16')
y=np.load('/content/drive/My Drive/preLight/Data/yN_2019_5_%d.npy'%leadTime).astype('uint16')

print(y.shape)

sum=np.sum(y[:,0,:,:],axis=0)
plt.imshow(np.sign(sum))

#y=y[:,0:1,:,:]+y[:,1:2,:,:]+y[:,2:3,:,:]+y[:,3:4,:,:]
#y=y[:,0:1,:,:]
Y = np.sum(y,axis=1,keepdims=True)
Y.shape

splited = np.split(Y,100,axis=0)
start = 100
for i,item in enumerate(splited):
  np.save('/content/drive/My Drive/data_FRL/Y/{:03d}.npy'.format(i+start),item.squeeze())

leadTime = 0 #leadtime in minutes
gridSize=1086
z=np.load('/content/drive/My Drive/preLight/Data/zN_2019_5_%d.npy'%leadTime).astype('uint16')

z.shape

ind=2
z_1hour=z[:,0+ind:1+ind,:,:]+z[:,1+ind:2+ind,:,:]+z[:,2+ind:3+ind,:,:]+z[:,3+ind:4+ind,:,:]
#z_1hour = np.sum(z[:,ind:ind+4,:,:],axis = 1,keepdims=True)
z=np.concatenate((z,z_1hour),axis=1)
z=np.sign(z).astype('float32')

z_max = z.max(axis=(2, 3), keepdims=True)
z /=z_max

#X=np.concatenate((X,z),axis=1)
X=z
#print(X.shape,y.shape)
print(X.shape)
z=None

splited = np.split(X,100,axis=0)
start = 100
for i,item in enumerate(splited):
  np.save('/content/drive/My Drive/data_FRL/X/{:03d}.npy'.format(i+start),item.squeeze())


n=5
idx=np.random.randint(0,X.shape[0])
fig=plt.figure(figsize=(25,15))
for i in range(n-1):
    fig.add_subplot(1,n,i+1)
    plt.imshow(X[idx,i+2,:,:])
fig.add_subplot(1,n,n)
plt.imshow(np.sign(y[idx,0,:,:]))





















!cp /content/drive/'My Drive'/preLight/Codes/utils_colab.py /usr/local/lib/python3.6/dist-packages

!cp -a /content/drive/'My Drive'/data_FRL .
