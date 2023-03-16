# %%
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import numpy
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy
from loses import f1score, precision, recall

from utils_colab import binningMaker

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


# %%
Latitude = loadmat('gridLatMat.mat')['gridLat']
Longitude = loadmat('gridLonMat.mat')['gridLon']


# %% for candidacy
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.Geostationary(central_longitude=-75.2))# GEOS-16 Wikipedia
ax.coastlines()
ax.gridlines()
#ax.background_img(resolution='low')
ax.background_img(name = 'natural-earth-1', resolution='large')
#ax.background_img(name='ne_shaded',resolution='low')
#ax.pcolormesh(Longitude,Latitude,Longitude+Latitude,alpha=0.5,cmap='viridis',shading='gouraud')#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'
X = Longitude[500:900,500:950]
Y = Latitude[500:900,500:950]
#X = Longitude
#Y = Latitude
boundary = np.zeros_like(X)
boundary[:,:] = np.nan
boundary[0:5,:] = 0
boundary[-5:-1,:] = 0
boundary[:,0:5] = 0
boundary[:,-5:-1] = 0

#ax.contourf(X,Y,X+Y, 100,alpha=0.5,transform=ccrs.PlateCarree())
ax.contourf(X,Y,boundary, 100,alpha=1,transform=ccrs.PlateCarree(),cmap='autumn')


plt.axis('off')
plt.show()
#fig.savefig('saved_images/roi.png',dpi=300)

# %%

counter = np.zeros([240,11])
for i in range(239):
    x = np.load(f'data/val/x_{i:04d}.npy')
    for j in range(11):
        sum = np.sum(x[j,])
        counter[i,j] = sum


file_num, channel = np.unravel_index(counter.argmax(),counter.shape)

to_show = np.load(f'data/val/x_{file_num:04d}.npy')[channel]

fig = plt.figure(figsize=(10,10))
mode = ccrs.Geostationary(central_longitude=-75.2)

ax = plt.axes(projection=mode)
ax.coastlines()
# ax.background_img(name='ne_shaded',resolution='low')
ax.background_img(name = 'natural-earth-1', resolution='large')
#ax.pcolormesh(Longitude,Latitude,Longitude+Latitude,alpha=0.5,cmap='viridis',shading='gouraud')#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'
temp = to_show.copy().astype('float')
temp[temp>0.01] = 1
temp[temp<0.01] = np.nan
t = 200
b = -200
l = 150
r = -150
ax.pcolormesh(Longitude[t:b,l:r],Latitude[t:b,l:r],temp[t:b,l:r],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'

#ax.pcolormesh(Longitude[150:-200,200:-200],Latitude[150:-200,200:-200],temp[150:-200,200:-200],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'
t = 70
b = 200
l = 300
r = -300
ax.pcolormesh(Longitude[t:b,l:r],Latitude[t:b,l:r],temp[t:b,l:r],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'
t = -200
b = -70
l = 300
r = -300
ax.pcolormesh(Longitude[t:b,l:r],Latitude[t:b,l:r],temp[t:b,l:r],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'

# t = 110
# b = 200
# l = 230
# r = 300
# ax.pcolormesh(Longitude[t:b,l:r],Latitude[t:b,l:r],temp[t:b,l:r],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'
# t = 30
# b = 70
# l = 400
# r = -400
# ax.pcolormesh(Longitude[t:b,l:r],Latitude[t:b,l:r],temp[t:b,l:r],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'


plt.axis('off')
plt.show()
#fig.savefig('saved_images/dataset.png',dpi=300)



# %%
# %%
better = [7, 12]
best = [15,18,22,24,26,31]
index = 26
with open('sample_images/model_final_Persistence.pkl','rb') as file:
    base = pickle.load(file)
#fig, ax = plt.subplots(11,4,figsize=(40,110),subplot_kw={'projection':ccrs.Geostationary(central_longitude=-75.2),'extent':[-85, 5, 5, -57]})
#fig, ax = plt.subplots(11,4,figsize=(20,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-85, 5, 5, -57]})
fig, ax = plt.subplots(3,4,figsize=(20,15),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-75, -25, 5, -35]})
Y = base['Y']
X = base['X']

for j,leadtime in enumerate(['-105min','-90min','-75min','-60min']):
    to_show = X[index][:,:,j].copy()
    to_show[to_show < 0.1] = 'nan'
    to_show[to_show > 0.1] = 255
    startX = base['x_cord'][index]
    startY = base['y_cord'][index]
    patch_size = base['patch_size']
    lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
    lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]
    ax[0][j].axis('off')
    ax[0][j].coastlines()
    ax[0][j].background_img(resolution='low')
    #ax[0][j].background_img(name = 'natural-earth-1', resolution='large')
    #ax[0][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='viridis',shading='gouraud',transform=ccrs.PlateCarree())
    ax[0][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='rainbow',vmin=0,vmax=1,shading='gouraud',transform=ccrs.PlateCarree())
    ax[0][j].set_title(f'{leadtime} (X)',fontsize=20)

for j,leadtime in enumerate(['-45min','-30min','-15min','0min']):
    to_show = X[index][:,:,j+4].copy()
    to_show[to_show < 0.1] = 'nan'
    to_show[to_show > 0.1] = 255
    startX = base['x_cord'][index]
    startY = base['y_cord'][index]
    patch_size = base['patch_size']
    lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
    lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]
    ax[1][j].axis('off')
    ax[1][j].coastlines()
    ax[1][j].background_img(resolution='low')
    #ax[1][j].background_img(name = 'natural-earth-1', resolution='large')
    #ax[1][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='viridis',shading='gouraud',transform=ccrs.PlateCarree())
    ax[1][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='rainbow',vmin=0,vmax=1,shading='gouraud',transform=ccrs.PlateCarree())
    ax[1][j].set_title(f'{leadtime} (X)',fontsize=20)

for j,leadtime in enumerate(['+15min','+30min','+45min','+60min']):
    to_show = Y[index][:,:,j].copy()
    to_show[to_show < 0.1] = 'nan'
    to_show[to_show > 0.1] = 255
    startX = base['x_cord'][index]
    startY = base['y_cord'][index]
    patch_size = base['patch_size']
    lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
    lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]
    ax[2][j].axis('off')
    ax[2][j].coastlines()
    ax[2][j].background_img(resolution='low')
    #ax[2][j].background_img(name = 'natural-earth-1', resolution='large')
    #ax[2][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='viridis',shading='gouraud',transform=ccrs.PlateCarree())
    ax[2][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='rainbow',vmin=0,vmax=1,shading='gouraud',transform=ccrs.PlateCarree())
    ax[2][j].set_title(f'{leadtime} (Y)',fontsize=20)


fig.tight_layout()
plt.show()
# fig.savefig('saved_images/in-out.png',dpi=300)



# %% for candidacy
import matplotlib.cm as cm
cmap = 'rainbow'
best = [15,18,22,24,26,31]
best = [26]
for ind,index in enumerate(best):
#index = 31
    with open('sample_images/model_final_Persistence.pkl','rb') as file:
        base = pickle.load(file)
    #fig, ax = plt.subplots(11,4,figsize=(40,110),subplot_kw={'projection':ccrs.Geostationary(central_longitude=-75.2),'extent':[-85, 5, 5, -57]})
    #fig, ax = plt.subplots(11,4,figsize=(20,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-85, 5, 5, -57]})
    fig, ax = plt.subplots(5,4,figsize=(26,32),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-75, -25, 5, -35]})



    Y = base['Y']
    for j,leadtime in enumerate(['15min','30min','45min','60min']):
        to_show = Y[index][:,:,j].copy()
        to_show[to_show < 0.1] = 'nan'
        to_show[to_show > 0.1] = 255
        startX = base['x_cord'][index]
        startY = base['y_cord'][index]
        patch_size = base['patch_size']
        lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
        lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]
        ax[0][j].axis('off')
        ax[0][j].coastlines()
        ax[0][j].background_img(resolution='low')
        #ax[0][j].background_img(name = 'natural-earth-1', resolution='large')
        ax[0][j].pcolormesh(lon,lat,to_show,alpha=0.9,cmap=cmap,vmin=0,vmax=1,shading='gouraud',transform=ccrs.PlateCarree())
        ax[0][j].set_title(f'Ground-truth {leadtime}',fontsize=20)

    models = ['Persistence','autoencoder_small','Unet_small','ResUnet_small']#,'Unet_small','Unet_medium','Unet_large','ResUnet_small','ResUnet_medium','ResUnet_large']
    for i,name in enumerate(models):
        with open(f'sample_images/model_final_{name}.pkl','rb') as file:
            dic = pickle.load(file)
            pred = dic['pred']
            for j,leadtime in enumerate(['15min','30min','45min','60min']):
                ax[i+1][j].axis('off')
                ax[i+1][j].coastlines()
                ax[i+1][j].background_img(resolution='low')
                startX = dic['x_cord'][index]
                startY = dic['y_cord'][index]
                patch_size = dic['patch_size']
                lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
                lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]

                to_show = np.zeros_like(pred[index][:,:,j])
                n = 10
                step = 1. / n
                for thresh in np.linspace(0.05,0.95,n):
                    to_show[pred[index][:,:,j] > thresh] += step

                to_show[to_show == 0] = 'nan'
                ax[i+1][j].pcolormesh(lon,lat,to_show,alpha=0.9,cmap=cmap,vmin=0,vmax=1,shading='gouraud',transform=ccrs.PlateCarree())
                f1 = f1score(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                p = precision(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                r = recall(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                ax[i+1][j].set_title(f'{name} {leadtime}\n f1={f1:.2f}, precision={p:.2f}, recall={r:.2f}',fontsize=15)

    fig.subplots_adjust(right=0.8 ,wspace=0.06 ,hspace=-.3)
    cbar_ax = fig.add_axes([0.82, 0.18, 0.05, 0.65])
    #cbar_ax = fig.add_axes([1.0, 0.016, 0.05, 0.965])
    fig.colorbar(cm.ScalarMappable(cmap=cmap),cax=cbar_ax)

    #fig.tight_layout()
    plt.show()
    #fig.savefig(f'saved_images/sample_{ind+1}.png',dpi=300,bbox_inches='tight')

# %% for paper diffrential
import matplotlib.cm as cm
cmap = 'rainbow'
best = [15,18,22,24,26,31]
best = [26]
for ind,index in enumerate(best):
#index = 31
    with open('sample_images/model_final_Persistence.pkl','rb') as file:
        base = pickle.load(file)
    #fig, ax = plt.subplots(11,4,figsize=(40,110),subplot_kw={'projection':ccrs.Geostationary(central_longitude=-75.2),'extent':[-85, 5, 5, -57]})
    #fig, ax = plt.subplots(11,4,figsize=(20,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-85, 5, 5, -57]})
    fig, ax = plt.subplots(5,4,figsize=(26,32),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-75, -25, 5, -35]})



    Y = base['Y']
    for j,leadtime in enumerate(['15min','30min','45min','60min']):
        to_show = Y[index][:,:,j].copy()
        to_show[to_show < 0.1] = 'nan'
        to_show[to_show > 0.1] = 255
        startX = base['x_cord'][index]
        startY = base['y_cord'][index]
        patch_size = base['patch_size']
        lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
        lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]
        #ax[0][j].axis('off')
        #ax[0][j].coastlines()
        #ax[0][j].background_img(resolution='low')
        #ax[0][j].background_img(name = 'natural-earth-1', resolution='large')
        ax[0][j].pcolormesh(lon,lat,to_show,alpha=0.9,cmap=cmap,vmin=0,vmax=1,shading='gouraud',transform=ccrs.PlateCarree())
        ax[0][j].set_title(f'Ground-truth {leadtime}',fontsize=20)

    models = ['Persistence','autoencoder_small','Unet_small','ResUnet_small']#,'Unet_small','Unet_medium','Unet_large','ResUnet_small','ResUnet_medium','ResUnet_large']
    for i,name in enumerate(models):
        with open(f'sample_images/model_final_{name}.pkl','rb') as file:
            dic = pickle.load(file)
            pred = dic['pred']
            for j,leadtime in enumerate(['15min','30min','45min','60min']):
                #ax[i+1][j].axis('off')
                #ax[i+1][j].coastlines()
                #ax[i+1][j].background_img(resolution='low')
                startX = dic['x_cord'][index]
                startY = dic['y_cord'][index]
                patch_size = dic['patch_size']
                lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
                lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]

                to_show = pred[index][:,:,j]
                to_show = np.zeros_like(pred[index][:,:,j])
                n = 10
                step = 1. / n
                for thresh in np.linspace(0.05,0.95,n):
                    to_show[pred[index][:,:,j] > thresh] += step

                to_show[to_show == 0] = 'nan'
                ax[i+1][j].pcolormesh(lon,lat,to_show,alpha=0.9,cmap=cmap,vmin=0,vmax=1,shading='gouraud',transform=ccrs.PlateCarree())
                f1 = f1score(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                p = precision(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                r = recall(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                ax[i+1][j].set_title(f'{name} {leadtime}\n f1={f1:.2f}, precision={p:.2f}, recall={r:.2f}',fontsize=15)

    fig.subplots_adjust(right=0.8 ,wspace=0.06 ,hspace=-.3)
    cbar_ax = fig.add_axes([0.82, 0.18, 0.05, 0.65])
    #cbar_ax = fig.add_axes([1.0, 0.016, 0.05, 0.965])
    fig.colorbar(cm.ScalarMappable(cmap=cmap),cax=cbar_ax)

    #fig.tight_layout()
    plt.show()
    #fig.savefig(f'saved_images/sample_{ind+1}.png',dpi=300,bbox_inches='tight')


# %% for candidacy

counter = np.zeros([240,11])
for i in range(239):
    x = np.load(f'data/val/x_{i:04d}.npy')
    for j in range(11):
        sum = np.sum(x[j,])
        counter[i,j] = sum


file_num, channel = np.unravel_index(counter.argmax(),counter.shape)

to_show = np.load(f'data/val/x_{file_num:04d}.npy')[channel]

fig = plt.figure(figsize=(10,10))
mode = ccrs.Geostationary(central_longitude=-75.2)

ax = plt.axes(projection=mode)
ax.coastlines()
#ax.background_img(name='ne_shaded',resolution='low')
ax.background_img(name = 'natural-earth-1', resolution='large')
#ax.pcolormesh(Longitude,Latitude,Longitude+Latitude,alpha=0.5,cmap='viridis',shading='gouraud')#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'
temp = to_show.copy().astype('float')
temp[temp>0.01] = 1
temp[temp<0.01] = np.nan
t = 200
b = -200
l = 150
r = -150
ax.pcolormesh(Longitude[t:b,l:r],Latitude[t:b,l:r],temp[t:b,l:r],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'

#ax.pcolormesh(Longitude[150:-200,200:-200],Latitude[150:-200,200:-200],temp[150:-200,200:-200],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'
t = 70
b = 200
l = 300
r = -300
ax.pcolormesh(Longitude[t:b,l:r],Latitude[t:b,l:r],temp[t:b,l:r],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'
t = -200
b = -70
l = 300
r = -300
ax.pcolormesh(Longitude[t:b,l:r],Latitude[t:b,l:r],temp[t:b,l:r],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'

# t = 110
# b = 200
# l = 230
# r = 300
# ax.pcolormesh(Longitude[t:b,l:r],Latitude[t:b,l:r],temp[t:b,l:r],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'
# t = 30
# b = 70
# l = 400
# r = -400
# ax.pcolormesh(Longitude[t:b,l:r],Latitude[t:b,l:r],temp[t:b,l:r],alpha=0.5,cmap='autumn',shading='gouraud',transform=ccrs.PlateCarree())#,color=plt.get_cmap('viridis'), shading='gouraud', shading = 'nearest' shading = 'auto'
X = Longitude[500:900,500:950]
Y = Latitude[500:900,500:950]
#X = Longitude
#Y = Latitude
boundary = np.zeros_like(X)
boundary[:,:] = np.nan
boundary[0:5,:] = 0
boundary[-5:-1,:] = 0
boundary[:,0:5] = 0
boundary[:,-5:-1] = 0

#ax.contourf(X,Y,X+Y, 100,alpha=0.5,transform=ccrs.PlateCarree())
ax.contourf(X,Y,boundary, 100,alpha=1,transform=ccrs.PlateCarree(),cmap='autumn')

plt.axis('off')
plt.show()
#fig.savefig('saved_images/dataset_merge.png',dpi=300)




# best index finder
with open('sample_images/model_final_Persistence.pkl','rb') as file:
    base = pickle.load(file)

for index in range(32):
    startX = dic['x_cord'][index]
    startY = dic['y_cord'][index]
    patch_size = dic['patch_size']
    lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
    lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]

    to_show = pred[index][:,:,1]

    to_show[to_show == 0] = 'nan'

    plt.figure()
    plt.title(f'{index}')
    plt.imshow(to_show)
    #plt.pcolormesh(lon,lat,to_show,alpha=0.9,cmap=cmap,vmin=0,vmax=1,shading='gouraud',transform=ccrs.PlateCarree())

best = [0,7,9,11,14]



# %% for paper NHESS
import matplotlib.cm as cm
cmap = 'rainbow'
#best = [15,18,22,24,26,31]
#best = [26]
best = [0,7,9,11,14][1:]

for ind,index in enumerate(best):
    with open('sample_images/model_final_Persistence.pkl','rb') as file:
        base = pickle.load(file)
    #fig, ax = plt.subplots(11,4,figsize=(40,110),subplot_kw={'projection':ccrs.Geostationary(central_longitude=-75.2),'extent':[-85, 5, 5, -57]})
    #fig, ax = plt.subplots(11,4,figsize=(20,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-85, 5, 5, -57]})
    fig, ax = plt.subplots(3,4,figsize=(26,24),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-80, -25, 5, -40]})
    #fig.suptitle(f'index={index}')


    Y = base['Y']


    models = ['autoencoder_small','Unet_small','ResUnet_small']#,'Unet_small','Unet_medium','Unet_large','ResUnet_small','ResUnet_medium','ResUnet_large']
    for i,name in enumerate(models):
        with open(f'sample_images/model_final_{name}.pkl','rb') as file:
            dic = pickle.load(file)
            pred = dic['pred']
            for j,leadtime in enumerate(['15min','30min','45min','60min']):
                ax[i][j].axis('off')
                ax[i][j].coastlines()
                ax[i][j].background_img(resolution='low')
                startX = dic['x_cord'][index]
                startY = dic['y_cord'][index]
                patch_size = dic['patch_size']
                lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
                lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]

                to_show = np.zeros_like(pred[index][:,:,j])

                # this visualization
                # n = 10
                # step = 1. / n
                # for thresh in np.linspace(0.05,0.95,n):
                #     to_show[pred[index][:,:,j] > thresh] += step

                # or this visualization
                thresh = 0.1
                to_show[pred[index][:,:,j]>thresh] = 1

                to_show[to_show == 0] = 'nan'

                ax[i][j].pcolormesh(lon,lat,to_show,alpha=0.9,cmap=cmap,vmin=0,vmax=1,shading='gouraud',transform=ccrs.PlateCarree())
                f1 = f1score(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                p = precision(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                r = recall(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                ax[i][j].set_title(f'{name} {leadtime}\n f1={f1:.2f}, precision={p:.2f}, recall={r:.2f}',fontsize=15)

                #draw region

                boundary = np.zeros_like(lon)
                boundary[:,:] = np.nan
                boundary[0:5,:] = 0
                boundary[-5:-1,:] = 0
                boundary[:,0:5] = 0
                boundary[:,-5:-1] = 0

                #ax.contourf(X,Y,X+Y, 100,alpha=0.5,transform=ccrs.PlateCarree())
                #ax[i][j].contourf(lon,lat,boundary, 100,alpha=1,transform=ccrs.PlateCarree(),cmap='autumn')
                ax[i][j].contourf(lon,lat,boundary, 100,alpha=1,transform=ccrs.PlateCarree(),colors='black')

    fig.subplots_adjust(right=0.8 ,wspace=0.06 ,hspace=-.3)
    #cbar_ax = fig.add_axes([0.82, 0.18, 0.05, 0.65])
    #cbar_ax = fig.add_axes([1.0, 0.016, 0.05, 0.965])
    #fig.colorbar(cm.ScalarMappable(cmap=cmap),cax=cbar_ax)

    #fig.tight_layout()
    plt.show()

    #fig.savefig(f'saved_images/sample_{ind+1}.png',dpi=300,bbox_inches='tight')


# %% for paper NHESS
import matplotlib.cm as cm
cmap = 'rainbow'
#best = [15,18,22,24,26,31]
#best = [26]
best = [0,7,9,11,14][1:]

for ind,index in enumerate(best):
    with open('sample_images/model_final_Persistence.pkl','rb') as file:
        base = pickle.load(file)
    #fig, ax = plt.subplots(11,4,figsize=(40,110),subplot_kw={'projection':ccrs.Geostationary(central_longitude=-75.2),'extent':[-85, 5, 5, -57]})
    #fig, ax = plt.subplots(11,4,figsize=(20,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-85, 5, 5, -57]})
    fig, ax = plt.subplots(4,4,figsize=(26,26),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-80, -25, 5, -40]})
    #fig.suptitle(f'index={index}')


    Y = base['Y']


    models = ['Persistence','autoencoder_small','Unet_small','ResUnet_small']#,'Unet_small','Unet_medium','Unet_large','ResUnet_small','ResUnet_medium','ResUnet_large']
    titles = ['Persistence','Auto Encoder','Unet','ResUnet']
    for i,(name,title) in enumerate(zip(models,titles)):
        with open(f'sample_images/model_final_{name}.pkl','rb') as file:
            dic = pickle.load(file)
            pred = dic['pred']
            for j,leadtime in enumerate(['15min','30min','45min','60min']):
                ax[i][j].axis('off')
                ax[i][j].coastlines()
                ax[i][j].background_img(resolution='low')
                startX = dic['x_cord'][index]
                startY = dic['y_cord'][index]
                patch_size = dic['patch_size']
                lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
                lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]

                y_true = Y[index:index+1,:,:,j:j+1]
                y_pred = pred[index:index+1,:,:,j:j+1]
                #print(y_true.shape,y_pred.shape,Y.shape,Y[index].shape,Y[index:index+1].shape)

                # new part
                y_true, y_pred = binningMaker(y_true, y_pred, binning=4, thresh_t=0.1, y_channel=1,padding = "SAME")
                y_true = np.squeeze(y_true)
                y_pred = np.squeeze(y_pred)
                # end of new part

                # alternative



                # end of alternative

                to_show = np.zeros_like(y_pred)

                thresh = 0.1
                thresholded = (y_pred > thresh).astype(np.float)

                mask_tp = np.logical_and(thresholded , y_true)
                mask_fn = np.logical_and(np.logical_not(thresholded) , y_true)
                mask_fp = np.logical_and(thresholded , np.logical_not(y_true))

                to_show[mask_tp] = 0.6
                to_show[mask_fp] = 0.2
                to_show[mask_fn] = 0.8

                cmap = (mpl.colors.ListedColormap([ 'yellow', 'green', 'red' ])) #.with_extremes(over='red', under='blue'))
                bounds = [0, 0.4, 0.8, 1.1]
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

                to_show[to_show < 0.1] = 'nan'

                ax[i][j].pcolormesh(lon,lat,to_show,alpha=0.9,cmap=cmap,vmin=0,vmax=1,shading='gouraud',transform=ccrs.PlateCarree())
                #f1 = f1score(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                #p = precision(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                #r = recall(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
                y_true = np.expand_dims(y_true,[0,3])
                y_pred = np.expand_dims(y_pred,[0,3])
                f1 = f1score(y_true,y_pred,y_channel=1)
                p = precision(y_true,y_pred,y_channel=1)
                r = recall(y_true,y_pred,y_channel=1)
                ax[i][j].set_title(f'{title} {leadtime}\n f1={f1:.2f}, precision={p:.2f}, recall={r:.2f}',fontsize=15)

                #draw region

                boundary = np.zeros_like(lon)
                boundary[:,:] = np.nan
                boundary[0:5,:] = 0
                boundary[-5:-1,:] = 0
                boundary[:,0:5] = 0
                boundary[:,-5:-1] = 0

                #ax.contourf(X,Y,X+Y, 100,alpha=0.5,transform=ccrs.PlateCarree())
                #ax[i][j].contourf(lon,lat,boundary, 100,alpha=1,transform=ccrs.PlateCarree(),cmap='autumn')
                ax[i][j].contourf(lon,lat,boundary, 100,alpha=1,transform=ccrs.PlateCarree(),colors='black')

    fig.subplots_adjust(right=0.8 ,wspace=0.06 ,hspace=-.3)



    #cbar_ax = fig.add_axes([0.82, 0.18, 0.05, 0.65])
    #cbar_ax = fig.add_axes([1.0, 0.016, 0.05, 0.965])
    #fig.colorbar(cm.ScalarMappable(cmap=cmap),cax=cbar_ax)

    plt.show()

    #fig.savefig(f'saved_images/sample_{ind+1}_different_color.png',dpi=300,bbox_inches='tight')

# %%
