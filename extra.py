import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as text
import cartopy.crs as ccrs

class Timer(object):
    '''
    Timer class to time operations
    Use "with" functionality of python
    Example:
    with Timer('test'):
        foo()
    '''
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


def downSampler(a,binning,thresh,mode='train'):
    sess=tf.Session()
    #a_temp=np.zeros((a.shape[0],int(a.shape[1]/binning),
                            #int(a.shape[2]/binning),a.shape[3]))
    #a_temp = tf.convert_to_tensor(a_temp)
    for i in range(a.shape[3]):
        filters = np.ones((binning, binning, 1, 1)) / binning / binning

        temp = tf.nn.conv2d(
            a[:,:,:,i:i+1],
            filters,
            binning,
            padding='VALID')
        if mode=='train':
            if i>=2:
                temp=tf.keras.backend.cast(tf.math.greater(temp, thresh), tf.float16)
        elif mode=='test':
            temp=tf.keras.backend.cast(tf.math.greater(temp, thresh), tf.float16)

        if i==0:
            a_temp=temp
        else:
            a_temp=tf.concat([a_temp,temp], -1)
            #print(temp.shape)
    return sess.run(a_temp)


def draw_images(display_list,title=None,name=None):
    fig = plt.figure(figsize=(20,20))
    for i in range(len(display_list)):
        plt.subplot(3, len(display_list)/3, i+1)
        try:
            plt.title(title[i])
        except:
            pass
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()
    if name:
        fig.savefig("./figs/"+str(name)+'.png')


def draw_images_cartopy(display_list,lon,lat,title=None,name=None):
    fig = plt.figure(figsize=(60,20))
    for i in range(len(display_list)):
        #ax = plt.subplot(3, len(display_list)/3, i+1,projection=ccrs.PlateCarree(),extent=[-147, -3, 50, -50])
        #ax = plt.subplot(2, len(display_list)/2, i+1,projection=ccrs.PlateCarree(),extent=[-147, -3, 50, -50])
        #ax = plt.subplot(2, len(display_list)/2, i+1,projection=ccrs.PlateCarree(),extent=[-147, -3, 50, -50])
        ax = plt.subplot(1, len(display_list)/1, i+1,projection=ccrs.PlateCarree(),extent=[-82, 5, 5, -57])
        ax.coastlines()
        #ax.set_xlim([np.nanmin(lon),np.nanmax(lon)])
        #ax.set_ylim([np.nanmin(lat),np.nanmax(lat)])
        ax.background_img(resolution='low')
        #ax.natural_earth_shp(alpha=0.5)

        try:
            plt.title(title[i])
        except:
            pass
        #plt.imshow(display_list[i])
        display_list[i] = display_list[i].astype('float')
        display_list[i][display_list[i] < 0.1] = 'nan'
        display_list[i][display_list[i] > 0.1] = 1
        ax.pcolormesh(lon,lat,display_list[i],alpha=0.5)
        plt.axis('off')

        # Create a Rectangle patch
        rect = patches.Rectangle((np.nanmin(lon), np.nanmin(lat)), np.nanmax(lon)-np.nanmin(lon), np.nanmax(lat)-np.nanmin(lat), linewidth=1, edgecolor='r', facecolor='none')
        #Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
    if name:
        fig.savefig("./figs/"+str(name)+'.png',transparent=True)


def draw_images_cartopy_ensemble(display_list,lon_list,lat_list,f1_list=None,title=None,name=None):
    fig = plt.figure(figsize=(60,20))
    ax = plt.axes(projection=ccrs.PlateCarree(),extent=[-82, 5, 5, -57])
    ax.coastlines()
    ax.background_img(resolution='low')
    for i in range(len(display_list)):

        try:
            plt.title(title[i])
        except:
            pass

        display_list[i] = display_list[i].astype('float')
        display_list[i][display_list[i] < 0.1] = 'nan'
        display_list[i][display_list[i] > 0.1] = 1
        ax.pcolormesh(lon_list[i],lat_list[i],display_list[i],alpha=0.5)
        plt.axis('off')

        # Create a Rectangle patch
        rect = patches.Rectangle((np.nanmin(lon_list[i]), np.nanmin(lat_list[i])), np.nanmax(lon_list[i])-np.nanmin(lon_list[i]), np.nanmax(lat_list[i])-np.nanmin(lat_list[i]), linewidth=1, edgecolor='r', facecolor='none')
        #Add the patch to the Axes
        ax.add_patch(rect)

        #, bbox=dict, color=str, family=str, fontsize=int)
        pos_x = np.nanmin(lon_list[i]) + (np.nanmax(lon_list[i])-np.nanmin(lon_list[i]))/2 -2
        pos_y = np.nanmin(lat_list[i]) + (np.nanmax(lat_list[i])-np.nanmin(lat_list[i]))
        ax.text(pos_x,pos_y, f"f1 = {f1_list[i]:.3f}", fontsize=20)


    plt.show()
    if name:
        fig.savefig("./figs/"+str(name)+'.png',transparent=True)


#
#test_data = test_data.astype('float')
#test_data[test_data == 0] = 'nan'

#lim1=200
#lim2=900
#ax.pcolormesh(lon[lim1:lim2,lim1:lim2],lat[lim1:lim2,lim1:lim2],test_data[lim1:lim2,lim1:lim2],alpha=0.5)
#ax.pcolor(lon,lat,test_data,alpha=0.5)
