import yaml
import pickle

root = '/home/semansou/code/long_range_prediction/'

base = {
"name":"multi_output_func_small_wide_5_loss_1",
"model":1,
"width" :128,
"height":128,
"depth":8,
"multi":True,
"output_index":None,
"layers":[16,32,64,128,256],
"loss":1,
"generator":2,
"p_ratio":0.2,
"flashThresh":10,
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
"seed":0,
"path":"/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output/",
"description":"CARE model multioutput with Dice loss 2019",
"tags":None,
"changeLoss":True}

#Lightning_Prediction_CARE_Multi_Output_variable(width=128,height=128,depth=8,multi = True, layers = [16, 32, 64, 128, 256]),
with open(root + 'base.yml','w') as f:
    yaml.dump(base,f,default_flow_style=False,Dumper=yaml.SafeDumper)


with open(root + 'base.yml','r') as f:
    temp = yaml.load(f,Loader=yaml.SafeLoader)

# %%

###############################################################################


base_layers = [
[8 ,16, 32, 64, 128, 256],
[8, 12, 20, 36, 68, 132],
[8, 10, 14, 22, 38, 70]
]

to_do = []

# for mode,base_layer in enumerate(base_layers):
#     for width,scale in enumerate([1,2,3]):
#         for depth,end in enumerate([4,5,6]):
#             temp = base.copy()
#             current_layer = [i * scale for i in base_layer[:end]]
#             temp["layers"] = current_layer
#             temp["use_amplitude"] = False
#             name = f'model_{mode}_{width}_{depth}_without_amplitude'
#             temp["name"] = name
#             to_do.append(name+'.yml')
#             with open(root + 'configurations/' + name + '.yml','w') as f:
#                 yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

# for mode,base_layer in enumerate(base_layers):
#     for width,scale in enumerate([1,2,3]):
#         for depth,end in enumerate([4,5,6]):
#             temp = base.copy()
#             current_layer = [i * scale for i in base_layer[:end]]
#             temp["layers"] = current_layer
#             temp["use_amplitude"] = True
#             name = f'model_{mode}_{width}_{depth}_with_amplitude'
#             temp["name"] = name
#             to_do.append(name+'.yml')
#             with open(root + 'configurations/' + name + '.yml','w') as f:
#                 yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)




# #really final candidacy
temp = base.copy()
layers = [16, 32, 64, 128, 256, 512, 1024, 2048]
# for candidacy multi output Autoencoder
# for i,prefix in enumerate(['large','medium','small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 1
#     temp["skip_connection"] = False
#     name = f'model_final_autoencoder_{prefix}'
#     temp["name"] = name
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
#
# # for candidacy multi output Unet
# for i,prefix in enumerate(['large','medium','small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 1
#     temp["skip_connection"] = True
#     name = f'model_final_Unet_{prefix}'
#     temp["name"] = name
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
#
# # for candidacy multi output ResUnet
# for i,prefix in enumerate(['large','medium','small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 2
#     temp["skip_connection"] = True
#     name = f'model_final_ResUnet_{prefix}'
#     temp["name"] = name
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
#
# # for candidacy single output ResUnet small model
# layers = [16, 32, 64, 128, 256, 512]
# for i,prefix in enumerate(['15min','30min','45min','60min']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers
#     temp["use_amplitude"] = False
#     temp["model"] = 2
#     temp["multi"] = False
#     temp["output_index"] = i
#     temp["skip_connection"] = True
#     name = f'model_final_ResUnet_small_{prefix}'
#     temp["name"] = name
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
#
# for candidacy Persistence
# temp = base.copy()
# temp["width"] = 256
# temp["height"] = 256
# temp["patch_size"] = 256
# temp["numPatches"] = 32
# temp["layers"] = layers
# temp["use_amplitude"] = False
# temp["model"] = 0
# temp["multi"] = True
# temp["output_index"] = None
# temp["skip_connection"] = True
# name = f'model_final_Persistence'
# temp["name"] = name
# to_do.append(name+'.yml')
# with open(root + 'configurations/' + name + '.yml','w') as f:
#     yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)


#######extra- really long training epoch = 200
# need to modify job_batch.run for 12 hour instead of 4
# for i,prefix in enumerate(['large']):
#     temp = base.copy()
#     temp['epochs'] = 200
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)]
#     temp["use_amplitude"] = True
#     temp["model"] = 2
#     temp["skip_connection"] = True
#     name = f'model_ResUnet_{prefix}_long'
#     temp["name"] = name
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

## without batchnorm and with transposed conv
# for i,prefix in enumerate(['large']):
#     temp = base.copy()
#     temp['epochs'] = 30
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)]
#     temp["use_amplitude"] = True
#     temp["model"] = 3
#     temp["skip_connection"] = True
#     name = f'model_ResUnet_{prefix}_nobatchnorm_transconv'
#     temp["name"] = name
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

# # multi output ResUnet new dataset distinct train-val-test
# for i,prefix in enumerate(['large','medium','small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 2
#     temp["skip_connection"] = True
#     name = f'model_final_ResUnet_{prefix}'
#     temp["name"] = name
#     temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_distinct_train_val_test/"
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)


####################################################################################################################################################################################

#                                                                       12.10.2022
# for after NEURIPS paper, for NHESS paper
#for candidacy multi output Autoencoder
# for i,prefix in enumerate(['small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 1
#     temp["skip_connection"] = False
#     name = f'model_final_autoencoder_{prefix}'
#     temp["name"] = name
#     temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
#
# # for candidacy multi output Unet
# for i,prefix in enumerate(['small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 1
#     temp["skip_connection"] = True
#     name = f'model_final_Unet_{prefix}'
#     temp["name"] = name
#     temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
#
# # for candidacy multi output ResUnet
# for i,prefix in enumerate(['small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 2
#     temp["skip_connection"] = True
#     name = f'model_final_ResUnet_{prefix}'
#     temp["name"] = name
#     temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
#
# # for candidacy Persistence
# temp = base.copy()
# temp["width"] = 256
# temp["height"] = 256
# temp["patch_size"] = 256
# temp["numPatches"] = 32
# temp["layers"] = layers
# temp["use_amplitude"] = False
# temp["model"] = 0
# temp["multi"] = True
# temp["output_index"] = None
# temp["skip_connection"] = True
# name = f'model_final_Persistence'
# temp["name"] = name
# temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
# to_do.append(name+'.yml')
# with open(root + 'configurations/' + name + '.yml','w') as f:
#     yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
####################################################################################################################################################################################
# for paper multi output ResUnet changing DataGenerator parameters (flashThresh, p_ratio)


# for p_ratio in [0.5, 0.2, 0.1]:# do this for 0.0
#     for flashThresh in [1, 10, 50]:
#         prefix = 'small'
#         i = 2 # for small configuration
#         temp = base.copy()
#         temp["width"] = 256
#         temp["height"] = 256
#         temp["patch_size"] = 256
#         temp["numPatches"] = 32
#         temp["layers"] = layers[:len(layers)-i]
#         temp["use_amplitude"] = False
#         temp["model"] = 2
#         temp["skip_connection"] = True
#         name = f'model_final_ResUnet_{prefix}_{p_ratio}_{flashThresh}'
#         temp["name"] = name
#         temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#         temp["p_ratio"] = p_ratio
#         temp["flashThresh"] = flashThresh
#         to_do.append(name+'.yml')
#         with open(root + 'configurations/' + name + '.yml','w') as f:
#             yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
####################################################################################################################################################################################
#                                           5.12.2022
# testing no constraint for data generator
# for p_ratio in [1]:# do this for 1.0(no constraints)
#     for flashThresh in [1]:
#         prefix = 'small'
#         i = 2 # for small configuration
#         temp = base.copy()
#         temp["width"] = 256
#         temp["height"] = 256
#         temp["patch_size"] = 256
#         temp["numPatches"] = 32
#         temp["layers"] = layers[:len(layers)-i]
#         temp["use_amplitude"] = False
#         temp["model"] = 2
#         temp["skip_connection"] = True
#         name = f'model_final_ResUnet_{prefix}_{p_ratio}_{flashThresh}'
#         temp["name"] = name
#         temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#         temp["p_ratio"] = p_ratio
#         temp["flashThresh"] = flashThresh
#         to_do.append(name+'.yml')
#         with open(root + 'configurations/' + name + '.yml','w') as f:
#             yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)


# layers = [16, 32, 64, 128, 256, 512, 1024, 2048]
#for paper multi output Autoencoder
# for i,prefix in enumerate(['large','medium','small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 1
#     temp["skip_connection"] = False
#     name = f'model_final_autoencoder_{prefix}'
#     temp["name"] = name
#     temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

# # for paper multi output Unet
# for i,prefix in enumerate(['large','medium','small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 1
#     temp["skip_connection"] = True
#     name = f'model_final_Unet_{prefix}'
#     temp["name"] = name
#     temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

# # for paper multi output ResUnet
# for i,prefix in enumerate(['large','medium','small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 2
#     temp["skip_connection"] = True
#     name = f'model_final_ResUnet_{prefix}'
#     temp["name"] = name
#     temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
# for paper Persistence
# temp = base.copy()
# temp["width"] = 256
# temp["height"] = 256
# temp["patch_size"] = 256
# temp["numPatches"] = 32
# temp["layers"] = layers
# temp["use_amplitude"] = False
# temp["model"] = 0
# temp["multi"] = True
# temp["output_index"] = None
# temp["skip_connection"] = True
# name = f'model_final_Persistence'
# temp["name"] = name
# temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
# to_do.append(name+'.yml')
# with open(root + 'configurations/' + name + '.yml','w') as f:
#     yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

####################################################################################################################################################################################
#                                           14.12.2022
# no constraint for data generator paper NHESS
# layers = [16, 32, 64, 128, 256, 512, 1024, 2048]
# #for paper multi output Autoencoder
# for i,prefix in enumerate(['large','medium','small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 1
#     temp["skip_connection"] = False
#     name = f'model_final_autoencoder_{prefix}'
#     temp["name"] = name
#     temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#     temp["p_ratio"] = 1
#     temp["flashThresh"] = 0
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

# # # for paper multi output Unet
# for i,prefix in enumerate(['large','medium','small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 1
#     temp["skip_connection"] = True
#     name = f'model_final_Unet_{prefix}'
#     temp["name"] = name
#     temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#     temp["p_ratio"] = 1
#     temp["flashThresh"] = 0
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

# # # for paper multi output ResUnet
# for i,prefix in enumerate(['large','medium','small']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers[:len(layers)-i]
#     temp["use_amplitude"] = False
#     temp["model"] = 2
#     temp["skip_connection"] = True
#     name = f'model_final_ResUnet_{prefix}'
#     temp["name"] = name
#     temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#     temp["p_ratio"] = 1
#     temp["flashThresh"] = 0
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
# # for paper Persistence
# temp = base.copy()
# temp["width"] = 256
# temp["height"] = 256
# temp["patch_size"] = 256
# temp["numPatches"] = 32
# temp["layers"] = layers
# temp["use_amplitude"] = False
# temp["model"] = 0
# temp["multi"] = True
# temp["output_index"] = None
# temp["skip_connection"] = True
# name = f'model_final_Persistence'
# temp["name"] = name
# temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
# temp["p_ratio"] = 1
# temp["flashThresh"] = 0
# to_do.append(name+'.yml')
# with open(root + 'configurations/' + name + '.yml','w') as f:
#     yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
####################################################################################################################################################################################
#                                           15.12.2022
# testing inverted constraint for data generator
# prefix = 'inverted'
# i = 2 # for small configuration
# temp = base.copy()
# temp["width"] = 256
# temp["height"] = 256
# temp["patch_size"] = 256
# temp["numPatches"] = 32
# temp["layers"] = layers[:len(layers)-i]
# temp["use_amplitude"] = False
# temp["model"] = 2
# temp["skip_connection"] = True
# name = f'model_final_ResUnet_{prefix}'
# temp["name"] = name
# temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
# #temp["p_ratio"] = p_ratio
# #temp["flashThresh"] = flashThresh
# to_do.append(name+'.yml')
# with open(root + 'configurations/' + name + '.yml','w') as f:
#     yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

####################################################################################################################################################################################
#                                           20.12.2022

# # for NHESS single output ResUnet small model
# layers = [16, 32, 64, 128, 256, 512]
# for i,prefix in enumerate(['15min','30min','45min','60min']):
#     temp = base.copy()
#     temp["width"] = 256
#     temp["height"] = 256
#     temp["patch_size"] = 256
#     temp["numPatches"] = 32
#     temp["layers"] = layers
#     temp["use_amplitude"] = False
#     temp["model"] = 2
#     temp["multi"] = False
#     temp["output_index"] = i
#     temp["skip_connection"] = True
#     name = f'model_final_ResUnet_small_{prefix}'
#     temp["name"] = name
#     temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
#     temp["p_ratio"] = 1
#     temp["flashThresh"] = 0
#     to_do.append(name+'.yml')
#     with open(root + 'configurations/' + name + '.yml','w') as f:
#         yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
# 
####################################################################################################################################################################################
#                                           16.01.2023
# no constraint for data generator paper NHESS
# CSI value
layers = [16, 32, 64, 128, 256, 512, 1024, 2048]
#for paper multi output Autoencoder
for i,prefix in enumerate(['large','medium','small']):
    temp = base.copy()
    temp["width"] = 256
    temp["height"] = 256
    temp["patch_size"] = 256
    temp["numPatches"] = 32
    temp["layers"] = layers[:len(layers)-i]
    temp["use_amplitude"] = False
    temp["model"] = 1
    temp["skip_connection"] = False
    name = f'model_final_autoencoder_{prefix}'
    temp["name"] = name
    temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
    temp["p_ratio"] = 1
    temp["flashThresh"] = 0
    temp["tags"] = ["for paper"]
    to_do.append(name+'.yml')
    with open(root + 'configurations/' + name + '.yml','w') as f:
        yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

# # for paper multi output Unet
for i,prefix in enumerate(['large','medium','small']):
    temp = base.copy()
    temp["width"] = 256
    temp["height"] = 256
    temp["patch_size"] = 256
    temp["numPatches"] = 32
    temp["layers"] = layers[:len(layers)-i]
    temp["use_amplitude"] = False
    temp["model"] = 1
    temp["skip_connection"] = True
    name = f'model_final_Unet_{prefix}'
    temp["name"] = name
    temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
    temp["p_ratio"] = 1
    temp["flashThresh"] = 0
    temp["tags"] = ["for paper"]
    to_do.append(name+'.yml')
    with open(root + 'configurations/' + name + '.yml','w') as f:
        yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

# # for paper multi output ResUnet
for i,prefix in enumerate(['large','medium','small']):
    temp = base.copy()
    temp["width"] = 256
    temp["height"] = 256
    temp["patch_size"] = 256
    temp["numPatches"] = 32
    temp["layers"] = layers[:len(layers)-i]
    temp["use_amplitude"] = False
    temp["model"] = 2
    temp["skip_connection"] = True
    name = f'model_final_ResUnet_{prefix}'
    temp["name"] = name
    temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
    temp["p_ratio"] = 1
    temp["flashThresh"] = 0
    temp["tags"] = ["for paper"]
    to_do.append(name+'.yml')
    with open(root + 'configurations/' + name + '.yml','w') as f:
        yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
# for paper Persistence
temp = base.copy()
temp["width"] = 256
temp["height"] = 256
temp["patch_size"] = 256
temp["numPatches"] = 32
temp["layers"] = layers
temp["use_amplitude"] = False
temp["model"] = 0
temp["multi"] = True
temp["output_index"] = None
temp["skip_connection"] = True
name = f'model_final_Persistence'
temp["name"] = name
temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
temp["p_ratio"] = 1
temp["flashThresh"] = 0
temp["tags"] = ["for paper"]
to_do.append(name+'.yml')
with open(root + 'configurations/' + name + '.yml','w') as f:
    yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
####################################################################################################################################################################################
#                                           16.01.2023
# no constraint for data generator paper NHESS
# CSI value
# loss 3 continuous DICE ChangeLoss=False
layers = [16, 32, 64, 128, 256, 512, 1024, 2048]
#for paper multi output Autoencoder
for i,prefix in enumerate(['large','medium','small']):
    temp = base.copy()
    temp["width"] = 256
    temp["height"] = 256
    temp["patch_size"] = 256
    temp["numPatches"] = 32
    temp["layers"] = layers[:len(layers)-i]
    temp["use_amplitude"] = False
    temp["model"] = 1
    temp["skip_connection"] = False
    name = f'model_final_autoencoder_{prefix}_no_change_loss'
    temp["name"] = name
    temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
    temp["p_ratio"] = 1
    temp["flashThresh"] = 0
    temp["tags"] = ["no change loss"]
    temp["loss"] = 3
    temp["change_loss"] = False
    to_do.append(name+'.yml')
    with open(root + 'configurations/' + name + '.yml','w') as f:
        yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

# # for paper multi output Unet
for i,prefix in enumerate(['large','medium','small']):
    temp = base.copy()
    temp["width"] = 256
    temp["height"] = 256
    temp["patch_size"] = 256
    temp["numPatches"] = 32
    temp["layers"] = layers[:len(layers)-i]
    temp["use_amplitude"] = False
    temp["model"] = 1
    temp["skip_connection"] = True
    name = f'model_final_Unet_{prefix}_no_change_loss'
    temp["name"] = name
    temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
    temp["p_ratio"] = 1
    temp["flashThresh"] = 0
    temp["tags"] = ["no change loss"]
    temp["loss"] = 3
    temp["change_loss"] = False
    to_do.append(name+'.yml')
    with open(root + 'configurations/' + name + '.yml','w') as f:
        yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)

# # for paper multi output ResUnet
for i,prefix in enumerate(['large','medium','small']):
    temp = base.copy()
    temp["width"] = 256
    temp["height"] = 256
    temp["patch_size"] = 256
    temp["numPatches"] = 32
    temp["layers"] = layers[:len(layers)-i]
    temp["use_amplitude"] = False
    temp["model"] = 2
    temp["skip_connection"] = True
    name = f'model_final_ResUnet_{prefix}_no_change_loss'
    temp["name"] = name
    temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
    temp["p_ratio"] = 1
    temp["flashThresh"] = 0
    temp["tags"] = ["no change loss"]
    temp["loss"] = 3
    temp["change_loss"] = False
    to_do.append(name+'.yml')
    with open(root + 'configurations/' + name + '.yml','w') as f:
        yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
# for paper Persistence
temp = base.copy()
temp["width"] = 256
temp["height"] = 256
temp["patch_size"] = 256
temp["numPatches"] = 32
temp["layers"] = layers
temp["use_amplitude"] = False
temp["model"] = 0
temp["multi"] = True
temp["output_index"] = None
temp["skip_connection"] = True
name = f'model_final_Persistence_no_change_loss'
temp["name"] = name
temp["path"] = "/work/sci-sti-fr/semansou/dataSequenceCNN_2019_multi_output_periodic_and_distinct_train_val_test/"
temp["p_ratio"] = 1
temp["flashThresh"] = 0
temp["tags"] = ["no change loss"]
temp["loss"] = 3
temp["change_loss"] = False
to_do.append(name+'.yml')
with open(root + 'configurations/' + name + '.yml','w') as f:
    yaml.dump(temp,f,default_flow_style=False,Dumper=yaml.SafeDumper)
####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
# do not change this part
with open(root + 'to_do.pickle','wb') as f:
    pickle.dump(to_do,f)
