# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import numpy
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy

from loses import f1score, precision, recall
# %%
# downloader = cartopy.io.shapereader.NEShpDownloader(url_template='https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/raster/NE1_HR_LC_SR_W_DR.zip',target_path_template=None,pre_downloaded_path_template='')
# #shapereader.NEShpDownloader(url_template='https://naturalearth.s3.amazonaws.com/{resolution}_{category}/ne_{resolution}_{name}.zip', target_path_template=None, pre_downloaded_path_template='')
# #cartopy.io.Downloader()
# #cartopy.io.RasterSource()
# test = downloader.default_downloader()
# cartopy.io.shapereader.natural_earth(resolution='110m', category='physical', name='coastline')
# cartopy.io.shapereader.natural_earth(resolution='10m', category='raster', name='natural_earth')
# %%
Latitude = loadmat('gridLatMat.mat')['gridLat']
Longitude = loadmat('gridLonMat.mat')['gridLon']
# %%

best = [7, 12, 15,18,22,24,26,31]




# %%

# def draw_images_cartopy_ensemble(display_list,lon_list,lat_list,f1_list=None,title=None,name=None):
#     fig = plt.figure(figsize=(60,20))
#     ax = plt.axes(projection=ccrs.PlateCarree(),extent=[-82, 5, 5, -57])
#     ax.coastlines()
#     ax.background_img(resolution='low')
#     for i in range(len(display_list)):
#
#         try:
#             plt.title(title[i])
#         except:
#             pass
#
#         display_list[i] = display_list[i].astype('float')
#         display_list[i][display_list[i] < 0.1] = 'nan'
#         display_list[i][display_list[i] > 0.1] = 1
#         ax.pcolormesh(lon_list[i],lat_list[i],display_list[i],alpha=0.5)
#         plt.axis('off')
#
#         # Create a Rectangle patch
#         rect = patches.Rectangle((np.nanmin(lon_list[i]), np.nanmin(lat_list[i])), np.nanmax(lon_list[i])-np.nanmin(lon_list[i]), np.nanmax(lat_list[i])-np.nanmin(lat_list[i]), linewidth=1, edgecolor='r', facecolor='none')
#         #Add the patch to the Axes
#         ax.add_patch(rect)
#
#         #, bbox=dict, color=str, family=str, fontsize=int)
#         pos_x = np.nanmin(lon_list[i]) + (np.nanmax(lon_list[i])-np.nanmin(lon_list[i]))/2 -2
#         pos_y = np.nanmin(lat_list[i]) + (np.nanmax(lat_list[i])-np.nanmin(lat_list[i]))
#         ax.text(pos_x,pos_y, f"f1 = {f1_list[i]:.3f}", fontsize=20)
#
#
#     plt.show()
#     #if name:
#         #fig.savefig("./figs/"+str(name)+'.png',transparent=True)


# %%
better = [7, 12, 15,18,22,24,26,31]
best = [15, 18]
#index = 18

# %%
index = 31
with open('sample_images/model_final_Persistence.pkl','rb') as file:
    base = pickle.load(file)
#fig, ax = plt.subplots(11,4,figsize=(40,110),subplot_kw={'projection':ccrs.Geostationary(central_longitude=-75.2),'extent':[-85, 5, 5, -57]})
#fig, ax = plt.subplots(11,4,figsize=(20,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-85, 5, 5, -57]})
fig, ax = plt.subplots(5,4,figsize=(17,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-75, -25, 5, -35]})
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
    ax[0][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='viridis',shading='gouraud',transform=ccrs.PlateCarree())
    ax[0][j].set_title(f'Ground-truth {leadtime}',fontsize=20)

models = ['Persistence','autoencoder_small','Unet_small','ResUnet_small']#,'Unet_small','Unet_medium','Unet_large','ResUnet_small','ResUnet_medium','ResUnet_large']
for i,name in enumerate(models):
    with open(f'sample_images/model_final_{name}.pkl','rb') as file:
        dic = pickle.load(file)
        pred = dic['pred']
        for j,leadtime in enumerate(['15min','30min','45min','60min']):
            to_show = pred[index][:,:,j].copy()
            to_show[to_show < 0.1] = 'nan'
            to_show[to_show > 0.1] = 255
            startX = dic['x_cord'][index]
            startY = dic['y_cord'][index]
            patch_size = dic['patch_size']
            lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
            lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]
            ax[i+1][j].axis('off')
            ax[i+1][j].coastlines()
            ax[i+1][j].background_img(resolution='low')
            ax[i+1][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='viridis',shading='gouraud',transform=ccrs.PlateCarree())
            f1 = f1score(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            p = precision(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            r = recall(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            ax[i+1][j].set_title(f'{name} {leadtime}\n f1={f1:.2f}, precision={p:.2f}, recall={r:.2f}',fontsize=15)

fig.tight_layout()
plt.show()
fig.savefig('saved_images/sample_6.png',dpi=300)


# %%
with open('sample_images/model_final_Persistence.pkl','rb') as file:
    base = pickle.load(file)
#fig, ax = plt.subplots(11,4,figsize=(40,110),subplot_kw={'projection':ccrs.Geostationary(central_longitude=-75.2),'extent':[-85, 5, 5, -57]})
#fig, ax = plt.subplots(11,4,figsize=(20,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-85, 5, 5, -57]})
fig, ax = plt.subplots(5,4,figsize=(17,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-75, -25, 5, -35]})
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
    ax[0][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='viridis',shading='gouraud',transform=ccrs.PlateCarree())
    ax[0][j].set_title(f'Ground-truth {leadtime}',fontsize=20)

models = ['Persistence','autoencoder_small','autoencoder_medium','autoencoder_large']#,'Unet_small','Unet_medium','Unet_large','ResUnet_small','ResUnet_medium','ResUnet_large']
for i,name in enumerate(models):
    with open(f'sample_images/model_final_{name}.pkl','rb') as file:
        dic = pickle.load(file)
        pred = dic['pred']
        for j,leadtime in enumerate(['15min','30min','45min','60min']):
            to_show = pred[index][:,:,j].copy()
            to_show[to_show < 0.1] = 'nan'
            to_show[to_show > 0.1] = 255
            startX = dic['x_cord'][index]
            startY = dic['y_cord'][index]
            patch_size = dic['patch_size']
            lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
            lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]
            ax[i+1][j].axis('off')
            ax[i+1][j].coastlines()
            ax[i+1][j].background_img(resolution='low')
            ax[i+1][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='viridis',shading='gouraud',transform=ccrs.PlateCarree())
            f1 = f1score(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            p = precision(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            r = recall(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            ax[i+1][j].set_title(f'{name} {leadtime}\n f1={f1:.2f}, precision={p:.2f}, recall={r:.2f}',fontsize=15)

fig.tight_layout()
plt.show()
fig.savefig('saved_images/autoencoder_sample_2.png',dpi=300)

# %%
with open('sample_images/model_final_Persistence.pkl','rb') as file:
    base = pickle.load(file)
#fig, ax = plt.subplots(11,4,figsize=(40,110),subplot_kw={'projection':ccrs.Geostationary(central_longitude=-75.2),'extent':[-85, 5, 5, -57]})
#fig, ax = plt.subplots(11,4,figsize=(20,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-85, 5, 5, -57]})
fig, ax = plt.subplots(5,4,figsize=(17,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-75, -25, 5, -35]})
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
    ax[0][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='viridis',shading='gouraud',transform=ccrs.PlateCarree())
    ax[0][j].set_title(f'Ground-truth {leadtime}',fontsize=20)

models = ['Persistence','Unet_small','Unet_medium','Unet_large']
for i,name in enumerate(models):
    with open(f'sample_images/model_final_{name}.pkl','rb') as file:
        dic = pickle.load(file)
        pred = dic['pred']
        for j,leadtime in enumerate(['15min','30min','45min','60min']):
            to_show = pred[index][:,:,j].copy()
            to_show[to_show < 0.1] = 'nan'
            to_show[to_show > 0.1] = 255
            startX = dic['x_cord'][index]
            startY = dic['y_cord'][index]
            patch_size = dic['patch_size']
            lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
            lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]
            ax[i+1][j].axis('off')
            ax[i+1][j].coastlines()
            ax[i+1][j].background_img(resolution='low')
            ax[i+1][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='viridis',shading='gouraud',transform=ccrs.PlateCarree())
            f1 = f1score(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            p = precision(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            r = recall(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            ax[i+1][j].set_title(f'{name} {leadtime}\n f1={f1:.2f}, precision={p:.2f}, recall={r:.2f}',fontsize=15)

fig.tight_layout()
plt.show()
fig.savefig('saved_images/unet_sample_2.png',dpi=300)

# %%
with open('sample_images/model_final_Persistence.pkl','rb') as file:
    base = pickle.load(file)
#fig, ax = plt.subplots(11,4,figsize=(40,110),subplot_kw={'projection':ccrs.Geostationary(central_longitude=-75.2),'extent':[-85, 5, 5, -57]})
#fig, ax = plt.subplots(11,4,figsize=(20,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-85, 5, 5, -57]})
fig, ax = plt.subplots(5,4,figsize=(17,20),subplot_kw={'projection':ccrs.PlateCarree(),'extent':[-75, -25, 5, -35]})
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
    ax[0][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='viridis',shading='gouraud',transform=ccrs.PlateCarree())
    ax[0][j].set_title(f'Ground-truth {leadtime}',fontsize=20)

models = ['Persistence','ResUnet_small','ResUnet_medium','ResUnet_large']
for i,name in enumerate(models):
    with open(f'sample_images/model_final_{name}.pkl','rb') as file:
        dic = pickle.load(file)
        pred = dic['pred']
        for j,leadtime in enumerate(['15min','30min','45min','60min']):
            to_show = pred[index][:,:,j].copy()
            to_show[to_show < 0.1] = 'nan'
            to_show[to_show > 0.1] = 255
            startX = dic['x_cord'][index]
            startY = dic['y_cord'][index]
            patch_size = dic['patch_size']
            lat = Latitude[startX:startX+patch_size,startY:startY+patch_size]
            lon = Longitude[startX:startX+patch_size,startY:startY+patch_size]
            ax[i+1][j].axis('off')
            ax[i+1][j].coastlines()
            ax[i+1][j].background_img(resolution='low')
            ax[i+1][j].pcolormesh(lon,lat,to_show,alpha=0.5,cmap='viridis',shading='gouraud',transform=ccrs.PlateCarree())
            f1 = f1score(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            p = precision(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            r = recall(pred[index:index+1,:,:,j:j+1],Y[index:index+1,:,:,j:j+1],y_channel=1)
            ax[i+1][j].set_title(f'{name} {leadtime}\n f1={f1:.2f}, precision={p:.2f}, recall={r:.2f}',fontsize=15)

fig.tight_layout()
plt.show()
fig.savefig('saved_images/resunet_sample_2.png',dpi=300)
