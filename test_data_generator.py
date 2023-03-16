import keras
import tensorflow as tf

from DataGenerator import PatchGenerator

def test():


    batch_size = 16
    numPatches = 32
    patch_size = 256

    config = {"include_position":False, "include_time":False, "time_dimension":False, "contex":False, "epochs":20, "batch_size":16, "numPatches":128, "patch_size":128, "inputInd":[2,3,4,5,6,7,8,9], "region":"CUS", "bound":[500,1000,500,1000], "keep_float":False, "path":"/home/emclab_epfl/data/dataSequenceCNN_2018/", "description":"CARE model 2018 ce+dice"}

    #fixed
    path = config["path"]
    code_path = '/home/emclab_epfl/code/'
    data_path = '/home/emclab_epfl/data/'



    #variables
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

    patch_train = PatchGenerator(path+'train/',time_path=path+'train/',position_path=code_path,include_position=include_position,include_time=include_time,time_dimension=time_dimension,contex=contex,region=region,bound=bound,batch_size=batch_size,numPatches=numPatches,patch_size=patch_size,inputInd=inputInd,keep_float=keep_float,n_channels=11,flashThresh=10)
    a,b = patch_train[0]
    print(a.shape,b.shape)

if __name__ == '__main__':
    test()
