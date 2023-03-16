import os
import re
import numpy as np

import keras
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence
from scipy.io import loadmat

import random
random.seed(0)

class PatchGenerator(Sequence):
    'Generates Patches for Keras'
    def __init__(self, path, time_path=None, position_path=None, include_position=True, include_time = True, time_dimension=True, contex=True, batch_size=32, numPatches=500, dim=(1086,1086), patch_size=256, inputInd=[2,3,4,5,6,7], region = '', bound = [411,923,444,956], n_channels=12, shuffle=True, flashThresh=-1, p_ratio=0.2, seed=None, keep_float = False, *args):
        'Initialization'
        super(PatchGenerator,self).__init__()

        self.names = sorted([i.split(".")[0].split("_")[1] for i in os.listdir(path) if re.match(r'x_',i)])
        self.batch_size = batch_size
        self.numPatches = numPatches
        self.path = path
        self.time_path = time_path
        self.position_path = position_path
        self.dim = dim
        self.patch_size = patch_size
        self.inputInd = inputInd
        self.region = region
        self.bound = bound
        self.n_channels = n_channels
        self.shuffle = shuffle
        if self.time_path:
            self.time = np.load(time_path+'t.npy',allow_pickle=True)
            self.time = np.mod(self.time,3600*24)//3600
        else:
            self.time = None
        # self.xx = np.load(position_path+'x.npy',allow_pickle=True)
        # self.yy = np.load(position_path+'y.npy',allow_pickle=True)

        #lat = np.linspace(0, dim[0], dim[1],dtype='uint16')
        #lon = np.linspace(0, dim[0], dim[1],dtype='uint16')
        #latInd, lonInd = np.meshgrid(lat, lon)
        #self.xx = lonInd
        #self.yy = latInd
        if self.position_path:
            self.xx = loadmat(position_path+'gridLonMat.mat')['gridLon']
            self.yy = loadmat(position_path+'gridLatMat.mat')['gridLat']
            self.position_crop(self.region)
        else:
            self.xx = None
            self.yy = None

        self.include_position = include_position
        self.include_time = include_time
        self.time_dimension = time_dimension
        self.contex = contex
        self.flashThresh = flashThresh
        self.p_ratio = p_ratio
        self.seed = seed
        np.random.seed(seed=seed)

        #self.max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid')
        self.keep_float = keep_float

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.names) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data patches
        X_patch, Y_patch = self.__data_generation(indexes)


        return X_patch, Y_patch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Find list of IDs
        names_temp = [self.names[k] for k in indexes]

        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim),dtype='float32')
        Y = np.empty((self.batch_size, 1, *self.dim),dtype='float32')
        #Y = np.empty((self.batch_size, 1, *self.dim), dtype=int)

        # Generate data
        for i, name in enumerate(names_temp):
            # Input
            X[i,] = np.sign(np.load(self.path + 'x_' + name + '.npy'))

            # Target
            Y[i,] = np.sign(np.sum(np.load(self.path + 'y_' + name + '.npy'),axis=0))

        # Time
        if self.time:
            T = self.time[indexes]
        else:
            T = None


        ''' means channel last input '''
        X = np.moveaxis(X,1,-1)
        Y = np.moveaxis(Y,1,-1)

        X = X[:,:,:,self.inputInd]

        #the area of interest: 'US', 'BRZ'
        #X,Y = region_selection(X, Y, region = self.region)

        X_patch, Y_patch = self.patchMaker(X=X,Y=Y,T=T,flashThresh=self.flashThresh, patchSize=self.patch_size,numPatches=self.numPatches,region=self.region)
        del X
        del Y
        return X_patch, Y_patch

    def region_selection(self,X,Y,region = ''):
        if region == 'US':
            X = X[:, 90:318, 55:676, :]
            Y = Y[:, 90:318, 55:676, :]
        elif region == 'BRZ':
            X = X[:, 485:850, 500:900, :]
            Y = Y[:, 485:850, 500:900, :]
        elif region == 'CUS':
            X = X[:, self.bound[0]:self.bound[1], self.bound[2]:self.bound[3], :]
            Y = Y[:, self.bound[0]:self.bound[1], self.bound[2]:self.bound[3], :]
        return X,Y

    def patching_by_mode(self,X,Y,T,patchSize,numPatches,flashThresh):
        X_patch = []
        Y_patch = []
        validInd = []
        k = 0
        time_steps = X.shape[3]
        for _ in range(5000000):
            ind0 = int(np.random.randint(X.shape[0]))
            if self.contex == True:
                startX = int(np.random.randint(X.shape[1]-2*patchSize+1))
                startY = int(np.random.randint(X.shape[2]-2*patchSize+1))
                Y_patch_temp = Y[ind0:ind0+1,startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize,0:1]
            else:
                startX = int(np.random.randint(X.shape[1]-patchSize+1))
                startY = int(np.random.randint(X.shape[2]-patchSize+1))
                Y_patch_temp = Y[ind0:ind0+1,startX:startX+patchSize,startY:startY+patchSize,0:1]
            if random.random()<self.p_ratio or np.sum(np.sign(Y_patch_temp))>flashThresh:
                Y_patch.append(Y_patch_temp)
                #X_patch_core = X[ind0:ind0+1,startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize,:].astype('float64')
                #X_patch_contex = tf.keras.backend.eval(self.max_pool_2d(X[ind0:ind0+1,startX:startX+2*patchSize,startY:startY+2*patchSize,:].astype('float64')))
                if self.contex == True:
                    X_patch_core = X[ind0:ind0+1,startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize,:]
                else:
                    X_patch_core = X[ind0:ind0+1,startX:startX+patchSize,startY:startY+patchSize,:]
                if self.time_dimension == True:
                    if self.contex == True:
                        #X_patch_contex = tf.keras.backend.eval(self.max_pool_2d(X[ind0:ind0+1,startX:startX+2*patchSize,startY:startY+2*patchSize,:]))
                        X_patch_contex = X[ind0:ind0+1,startX:startX+2*patchSize:2,startY:startY+2*patchSize:2,:]
                        X_patch_temp = np.empty((1,X_patch_core.shape[3],patchSize,patchSize,2))
                        X_patch_temp[0,:,:,:,0] = np.moveaxis(X_patch_core,-1,1)
                        X_patch_temp[0,:,:,:,1] = np.moveaxis(X_patch_contex,-1,1)
                    else:
                        X_patch_temp = np.empty((1,X_patch_core.shape[3],patchSize,patchSize,1))
                        X_patch_temp[0,:,:,:,0] = np.moveaxis(X_patch_core,-1,1)
                    if self.include_position == True:
                        x_cord = 1+np.expand_dims(self.xx[startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize],(0,1,4))
                        y_cord = 1+np.expand_dims(self.yy[startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize],(0,1,4))
                        X_patch_temp = np.concatenate([X_patch_temp,np.repeat(x_cord,time_steps,axis=1),np.repeat(y_cord,time_steps,axis=1)],4)
                    if self.include_time == True:
                        t_cord = np.tile(np.expand_dims(np.array([T[ind0]]),(1,2,3,4)),(time_steps,patchSize,patchSize,1))
                        X_patch_temp = np.concatenate([X_patch_temp,t_cord],4)
                else:
                    if self.contex == True:
                        #X_patch_contex = tf.keras.backend.eval(self.max_pool_2d(X[ind0:ind0+1,startX:startX+2*patchSize,startY:startY+2*patchSize,:]))
                        X_patch_contex = X[ind0:ind0+1,startX:startX+2*patchSize:2,startY:startY+2*patchSize:2,:]
                        X_patch_temp = np.empty((1,patchSize,patchSize,X_patch_core.shape[3]+X_patch_contex.shape[3]))
                        X_patch_temp[0,:,:,0::2] = X_patch_core
                        X_patch_temp[0,:,:,1::2] = X_patch_contex
                    else:
                        X_patch_temp = np.empty((1,patchSize,patchSize,X_patch_core.shape[3]))
                        X_patch_temp[0,:,:,0::1] = X_patch_core
                    if self.include_position == True:
                        x_cord = 1+np.expand_dims(self.xx[startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize],(0,3))
                        y_cord = 1+np.expand_dims(self.yy[startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize],(0,3))
                        X_patch_temp = np.concatenate([X_patch_temp,x_cord,y_cord],3)
                    if self.include_time == True:
                        t_cord = np.tile(np.expand_dims(np.array([T[ind0]]),(1,2,3)),(patchSize,patchSize,1))
                        X_patch_temp = np.concatenate([X_patch_temp,t_cord],3)
                X_patch.append(X_patch_temp)
                k+=1
            if k==numPatches:
                break

        X_patch = np.concatenate(X_patch,axis=0)
        Y_patch = np.concatenate(Y_patch,axis=0)

        #X_patch = X_patch.astype('float64')
        #Y_patch = Y_patch.astype('float64')
        #X_patch = X_patch.astype('uint16')
        #Y_patch = Y_patch.astype('uint16')

        return X_patch,Y_patch

    def normilize_by_mode(self,X_patch, Y_patch):
        #### important : should not be used since it convert time and dimension as well into signs
        X_patch = np.sign(X_patch).astype('uint16')
        Y_patch = np.sign(Y_patch).astype('uint16')
        return X_patch,Y_patch

    def patchMaker(self,X,Y,T,flashThresh, patchSize,numPatches,region):

        X,Y = self.region_selection(X, Y, region = region)

        X_patch,Y_patch = self.patching_by_mode(X,Y,T,patchSize,numPatches,flashThresh)

        if self.keep_float == False:
            X_patch,Y_patch = self.normilize_by_mode(X_patch, Y_patch)

        return X_patch,Y_patch

    def position_crop(self,region):
        if region == 'US':
            self.xx = self.xx[90:318, 55:676]
            self.yy = self.yy[90:318, 55:676]
        elif region == 'BRZ':
            self.xx = self.xx[485:850, 500:900]
            self.yy = self.yy[485:850, 500:900]
        elif region == 'CUS':
            self.xx = self.xx[self.bound[0]:self.bound[1], self.bound[2]:self.bound[3]]
            self.yy = self.yy[self.bound[0]:self.bound[1], self.bound[2]:self.bound[3]]


class PatchGenerator_multioutput(Sequence):
    'Generates Patches for Keras'
    def __init__(self, path, time_path=None, position_path=None, include_position=True, include_time = True, time_dimension=True, contex=True, batch_size=32, numPatches=500, dim=(1086,1086), patch_size=256, inputInd=[2,3,4,5,6,7], region = '', bound = [411,923,444,956], n_channels=12, shuffle=True, flashThresh=-1, p_ratio=0.2, seed=None, keep_float = False, do_sum=False, use_amplitude=True, output_index=None, coordinates_for_drawing = False, *args):
        'Initialization'
        super(PatchGenerator_multioutput,self).__init__()

        self.names = sorted([i.split(".")[0].split("_")[1] for i in os.listdir(path) if re.match(r'x_',i)])
        self.batch_size = batch_size
        self.numPatches = numPatches
        self.path = path
        self.time_path = time_path
        self.position_path = position_path
        self.dim = dim
        self.patch_size = patch_size
        self.inputInd = inputInd
        self.region = region
        self.bound = bound
        self.n_channels = n_channels
        self.shuffle = shuffle
        if self.time_path:
            self.time = np.load(time_path+'t.npy',allow_pickle=True)
            self.time = np.mod(self.time,3600*24)//3600
        else:
            self.time = None
        # self.xx = np.load(position_path+'x.npy',allow_pickle=True)
        # self.yy = np.load(position_path+'y.npy',allow_pickle=True)

        #lat = np.linspace(0, dim[0], dim[1],dtype='uint16')
        #lon = np.linspace(0, dim[0], dim[1],dtype='uint16')
        #latInd, lonInd = np.meshgrid(lat, lon)
        #self.xx = lonInd
        #self.yy = latInd
        if self.position_path:
            self.xx = loadmat(position_path+'gridLonMat.mat')['gridLon']
            self.yy = loadmat(position_path+'gridLatMat.mat')['gridLat']
            self.position_crop(self.region)
        else:
            self.xx = None
            self.yy = None

        self.include_position = include_position
        self.include_time = include_time
        self.time_dimension = time_dimension
        self.contex = contex
        self.flashThresh = flashThresh
        self.p_ratio = p_ratio
        self.seed = seed
        np.random.seed(seed=seed)

        #self.max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid')
        self.keep_float = keep_float

        self.do_sum = do_sum
        self.use_amplitude = use_amplitude
        self.output_index = output_index
        self.coordinates_for_drawing = coordinates_for_drawing

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.names) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data patches
        #X_patch, Y_patch = self.__data_generation(indexes)


        #return X_patch, Y_patch
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Find list of IDs
        names_temp = [self.names[k] for k in indexes]

        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim),dtype='float32')
        if self.output_index == None:
            y_channel = 4
        else:
            y_channel = 1
        Y = np.empty((self.batch_size, y_channel, *self.dim),dtype='float32')
        #Y = np.empty((self.batch_size, 1, *self.dim), dtype=int)

        # Generate data
        for i, name in enumerate(names_temp):
            # Input
            if self.use_amplitude == True:
                X[i,] = np.load(self.path + 'x_' + name + '.npy')
            else:
                X[i,] = np.sign(np.load(self.path + 'x_' + name + '.npy'))
            # Target
            y_temp = np.load(self.path + 'y_' + name + '.npy')
            #if self.do_sum == True:
            #    Y[i,0,] = np.sign(np.sum(y_temp[0:4],axis=0))
            #    Y[i,1,] = np.sign(np.sum(y_temp[1:5],axis=0))
            #    Y[i,2,] = np.sign(np.sum(y_temp[2:6],axis=0))
            #else:
            #    Y[i,0,] = np.sign(y_temp[0:1])
            #    Y[i,1,] = np.sign(y_temp[1:2])
            #    Y[i,2,] = np.sign(y_temp[2:3])
            if self.output_index == None:
                Y[i,] = np.sign(y_temp[0:4])
            else:
                Y[i,] = np.sign(y_temp[self.output_index:self.output_index+1])

        # Time
        if self.time:
            T = self.time[indexes]
        else:
            T = None


        ''' means channel last input '''
        X = np.moveaxis(X,1,-1)
        Y = np.moveaxis(Y,1,-1)

        #X = X[:,:,:,self.inputInd]

        #the area of interest: 'US', 'BRZ'
        #X,Y = region_selection(X, Y, region = self.region)

        #X_patch, Y_patch = self.patchMaker(X=X,Y=Y,T=T,flashThresh=self.flashThresh, patchSize=self.patch_size,numPatches=self.numPatches,region=self.region)
        #del X
        #del Y
        #return X_patch, Y_patch
        return self.patchMaker(X=X,Y=Y,T=T,flashThresh=self.flashThresh, patchSize=self.patch_size,numPatches=self.numPatches,region=self.region)

    def region_selection(self,X,Y,region = ''):
        if region == 'US':
            X = X[:, 90:318, 55:676, :]
            Y = Y[:, 90:318, 55:676, :]
        elif region == 'BRZ':
            X = X[:, 485:850, 500:900, :]
            Y = Y[:, 485:850, 500:900, :]
        elif region == 'CUS':
            X = X[:, self.bound[0]:self.bound[1], self.bound[2]:self.bound[3], :]
            Y = Y[:, self.bound[0]:self.bound[1], self.bound[2]:self.bound[3], :]
        return X,Y

    def patching_by_mode(self,X,Y,T,patchSize,numPatches,flashThresh):
        X_patch = []
        Y_patch = []
        Coordinates_indexes_x = []
        Coordinates_indexes_y = []
        validInd = []
        k = 0
        time_steps = X.shape[3]
        for _ in range(5000000):
            ind0 = int(np.random.randint(X.shape[0]))
            if self.contex == True:
                startX = int(np.random.randint(X.shape[1]-2*patchSize+1))
                startY = int(np.random.randint(X.shape[2]-2*patchSize+1))
                #Y_patch_temp = Y[ind0:ind0+1,startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize,0:3]
                Y_patch_temp = Y[ind0:ind0+1,startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize,:]
            else:
                startX = int(np.random.randint(X.shape[1]-patchSize+1))
                startY = int(np.random.randint(X.shape[2]-patchSize+1))
                #Y_patch_temp = Y[ind0:ind0+1,startX:startX+patchSize,startY:startY+patchSize,0:3]
                Y_patch_temp = Y[ind0:ind0+1,startX:startX+patchSize,startY:startY+patchSize,:]
            if random.random()<self.p_ratio or np.sum(np.sign(Y_patch_temp))>flashThresh:
            #if random.random()<self.p_ratio or np.sum(np.sign(Y_patch_temp))<flashThresh:#inverted condition, for worst case only, use top line instead
                Y_patch.append(Y_patch_temp)
                Coordinates_indexes_x.append(self.bound[0]+startX)
                Coordinates_indexes_y.append(self.bound[2]+startY)
                #X_patch_core = X[ind0:ind0+1,startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize,:].astype('float64')
                #X_patch_contex = tf.keras.backend.eval(self.max_pool_2d(X[ind0:ind0+1,startX:startX+2*patchSize,startY:startY+2*patchSize,:].astype('float64')))
                if self.contex == True:
                    X_patch_core = X[ind0:ind0+1,startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize,:]
                else:
                    X_patch_core = X[ind0:ind0+1,startX:startX+patchSize,startY:startY+patchSize,:]
                if self.time_dimension == True:
                    if self.contex == True:
                        #X_patch_contex = tf.keras.backend.eval(self.max_pool_2d(X[ind0:ind0+1,startX:startX+2*patchSize,startY:startY+2*patchSize,:]))
                        X_patch_contex = X[ind0:ind0+1,startX:startX+2*patchSize:2,startY:startY+2*patchSize:2,:]
                        X_patch_temp = np.empty((1,X_patch_core.shape[3],patchSize,patchSize,2))
                        X_patch_temp[0,:,:,:,0] = np.moveaxis(X_patch_core,-1,1)
                        X_patch_temp[0,:,:,:,1] = np.moveaxis(X_patch_contex,-1,1)
                    else:
                        X_patch_temp = np.empty((1,X_patch_core.shape[3],patchSize,patchSize,1))
                        X_patch_temp[0,:,:,:,0] = np.moveaxis(X_patch_core,-1,1)
                    if self.include_position == True:
                        x_cord = 1+np.expand_dims(self.xx[startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize],(0,1,4))
                        y_cord = 1+np.expand_dims(self.yy[startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize],(0,1,4))
                        X_patch_temp = np.concatenate([X_patch_temp,np.repeat(x_cord,time_steps,axis=1),np.repeat(y_cord,time_steps,axis=1)],4)
                    if self.include_time == True:
                        t_cord = np.tile(np.expand_dims(np.array([T[ind0]]),(1,2,3,4)),(time_steps,patchSize,patchSize,1))
                        X_patch_temp = np.concatenate([X_patch_temp,t_cord],4)
                else:
                    if self.contex == True:
                        #X_patch_contex = tf.keras.backend.eval(self.max_pool_2d(X[ind0:ind0+1,startX:startX+2*patchSize,startY:startY+2*patchSize,:]))
                        X_patch_contex = X[ind0:ind0+1,startX:startX+2*patchSize:2,startY:startY+2*patchSize:2,:]
                        X_patch_temp = np.empty((1,patchSize,patchSize,X_patch_core.shape[3]+X_patch_contex.shape[3]))
                        X_patch_temp[0,:,:,0::2] = X_patch_core
                        X_patch_temp[0,:,:,1::2] = X_patch_contex
                    else:
                        X_patch_temp = np.empty((1,patchSize,patchSize,X_patch_core.shape[3]))
                        X_patch_temp[0,:,:,0::1] = X_patch_core
                    if self.include_position == True:
                        x_cord = 1+np.expand_dims(self.xx[startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize],(0,3))
                        y_cord = 1+np.expand_dims(self.yy[startX+patchSize//2:startX+patchSize//2+patchSize,startY+patchSize//2:startY+patchSize//2+patchSize],(0,3))
                        X_patch_temp = np.concatenate([X_patch_temp,x_cord,y_cord],3)
                    if self.include_time == True:
                        t_cord = np.tile(np.expand_dims(np.array([T[ind0]]),(1,2,3)),(patchSize,patchSize,1))
                        X_patch_temp = np.concatenate([X_patch_temp,t_cord],3)
                X_patch.append(X_patch_temp)
                k+=1
            if k==numPatches:
                break

        X_patch = np.concatenate(X_patch,axis=0)
        Y_patch = np.concatenate(Y_patch,axis=0)

        Coordinates_indexes_x = np.array(Coordinates_indexes_x)
        Coordinates_indexes_y = np.array(Coordinates_indexes_y)

        #X_patch = X_patch.astype('float64')
        #Y_patch = Y_patch.astype('float64')
        #X_patch = X_patch.astype('uint16')
        #Y_patch = Y_patch.astype('uint16')
        if self.coordinates_for_drawing == True:
            return X_patch, Y_patch, Coordinates_indexes_x, Coordinates_indexes_y
        else:
            return X_patch,Y_patch

    def normilize_by_mode(self,X_patch, Y_patch):
        #### important : should not be used since it convert time and dimension as well into signs
        X_patch = np.sign(X_patch).astype('uint16')
        Y_patch = np.sign(Y_patch).astype('uint16')
        return X_patch,Y_patch

    def patchMaker(self,X,Y,T,flashThresh, patchSize,numPatches,region):

        X,Y = self.region_selection(X, Y, region = region)

        #X_patch,Y_patch = self.patching_by_mode(X,Y,T,patchSize,numPatches,flashThresh)

        #if self.keep_float == False:
        #    X_patch,Y_patch = self.normilize_by_mode(X_patch, Y_patch)

        #return X_patch,Y_patch
        return self.patching_by_mode(X,Y,T,patchSize,numPatches,flashThresh)

    def position_crop(self,region):
        if region == 'US':
            self.xx = self.xx[90:318, 55:676]
            self.yy = self.yy[90:318, 55:676]
        elif region == 'BRZ':
            self.xx = self.xx[485:850, 500:900]
            self.yy = self.yy[485:850, 500:900]
        elif region == 'CUS':
            self.xx = self.xx[self.bound[0]:self.bound[1], self.bound[2]:self.bound[3]]
            self.yy = self.yy[self.bound[0]:self.bound[1], self.bound[2]:self.bound[3]]
