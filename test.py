# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:56:26 2023

@author: Zixia Zhou
"""

import tensorflow as tf
import time
import numpy as np
import random
from PIL import *
import os
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Lambda
from tensorflow.keras.layers import BatchNormalization,Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from blocks import *
from tensorflow.keras.layers import Add,Multiply
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD,RMSprop, Adam
from tensorflow.keras.layers import LeakyReLU
from PIL import Image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
from augment_utils import *
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from natsort import natsorted
from itertools import product

matplotlib.use('Agg') 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ[ 'CUDA_VISIBLE_DEVICES' ] = ''
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
my_seed = 0
os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)


#### predict
def get_train_data(data_dir):
    simples = {}
    Dir_file = os.listdir(data_dir)
    Dir_file = natsorted(Dir_file)
 #   Dir_file = Dir_file[0:2000]
    for file_name in Dir_file:
        captcha = file_name.split('.')[0]
        simples[data_dir + '/' + file_name] = captcha
    return simples

def gen_entire(res_test):
    row1 =  np.hstack((np.squeeze(res_test[0:8])))  
    row2 =  np.hstack((np.squeeze(res_test[8:16]))) 
    row3 =  np.hstack((np.squeeze(res_test[16:24]))) 
    row4 =  np.hstack((np.squeeze(res_test[24:32]))) 
    row5 =  np.hstack((np.squeeze(res_test[32:40]))) 
    row6 =  np.hstack((np.squeeze(res_test[40:48]))) 
    row7 =  np.hstack((np.squeeze(res_test[48:56]))) 
    row8 =  np.hstack((np.squeeze(res_test[56:64]))) 
    
    res_out_entire = np.vstack((row1,row2,row3,row4,row5,row6,row7,row8)) 
    res_out_entire=res_out_entire*255
    res_out_entire[res_out_entire < 0] = 0
    res_out_entire[res_out_entire > 255] = 255
    return res_out_entire
        

filepath_dapi='/dataWS_task1/PPDDAPI'
filepath_auto='/dataWS_task1/Autofluorescence'

simples_dapi = get_train_data(filepath_dapi)
file_simples_dapi = list(simples_dapi.keys())

simples_auto = get_train_data(filepath_auto)
file_simples_auto = list(simples_auto.keys())


generator=load_model('/models/model_task1.h5')
#generator=load_model('/models/model_task2.h5')

newsize=(2048,2048)
d=256
test_num=1
for img_num in range (0,test_num):
    file_name_dapi = file_simples_dapi[img_num]
    img_dapi=Image.open(file_name_dapi)
    img_dapi = img_dapi.resize(newsize)
    file_name_auto = file_simples_auto[img_num]
    img_auto=Image.open(file_name_auto)
    img_auto = img_auto.resize(newsize)
    w, h = img_dapi.size
    grid1 = product(range(0, h-h%d, d), range(0, w-w%d, d))
    grid2 = product(range(0, h-h%d, d), range(0, w-w%d, d))
    patch_all1=[]
    patch_all2=[]
    res_test1=[]
    res_test2=[]
    res_test3=[]
    res_test4=[]
    res_test5=[]
    res_test6=[]
    res_test7=[]
    res_test8=[]    
    
    for i, j in grid1:
        box = (j, i, j+d, i+d)
        patch_temp=img_dapi.crop(box)
        patch_temp=np.array(patch_temp)
        patch_all2.append(patch_temp)

    for i, j in grid2:
        box = (j, i, j+d, i+d)
        patch_temp=img_auto.crop(box)
        patch_temp=np.array(patch_temp)
        patch_all1.append(patch_temp)
        
    patch_all1=np.array(patch_all1)    
    patch_all2=np.array(patch_all2)    
        
    num_patch=patch_all1.shape[0]       
    for i in range (num_patch):
        print(i)
        test_img1=patch_all1[i]
        test_img2=patch_all2[i]
        test_img1=test_img1[np.newaxis,:, :]
        test_img1=test_img1/255
        test_img2=test_img2[np.newaxis,:, :]
        test_img2=test_img2/255        
        test_out=generator.predict([test_img1,test_img2])
        test_out=test_out*255
        res_test1.append(test_out[0])
        res_test2.append(test_out[1])
        res_test3.append(test_out[2])
        res_test4.append(test_out[3])
        res_test5.append(test_out[4])
        res_test6.append(test_out[5])
        res_test7.append(test_out[6])
        res_test8.append(test_out[7])
        
    res_out_entire1=gen_entire(res_test1)
    res_out_entire2=gen_entire(res_test2)
    res_out_entire3=gen_entire(res_test3)
    res_out_entire4=gen_entire(res_test4)
    res_out_entire5=gen_entire(res_test5)
    res_out_entire6=gen_entire(res_test6)
    res_out_entire7=gen_entire(res_test7)
    res_out_entire8=gen_entire(res_test8)

    
    img_dapi_base=Image.open(file_name_dapi).convert('L')
    img_dapi_base = img_dapi_base.resize(newsize)
    img_dapi_base=np.array(img_dapi_base) 

    img_dapi_base = (255-img_dapi_base)
    img_dapi_base[img_dapi_base < 3] = 0
    img_dapi_base=(img_dapi_base/(np.max(img_dapi_base)))*255
        
    res_out_entire1=res_out_entire1.astype(np.uint8)
    resout_name1='/out_WS/res_wh/'+str(img_num)+'_cd3.png'
    Image.fromarray(res_out_entire1).save(resout_name1)

    res_out_entire2=res_out_entire2.astype(np.uint8)
    resout_name2='/out_WS/res_wh/'+str(img_num)+'_cd20.png'
    Image.fromarray(res_out_entire2).save(resout_name2)    

    res_out_entire3=res_out_entire3.astype(np.uint8)
    resout_name3='/out_WS/res_wh/'+str(img_num)+'_foxp3.png'
    Image.fromarray(res_out_entire3).save(resout_name3)
    
    res_out_entire4=res_out_entire4.astype(np.uint8)
    resout_name4='/out_WS/res_wh/'+str(img_num)+'_pd1.png'
    Image.fromarray(res_out_entire4).save(resout_name4)    
    
    res_out_entire5=res_out_entire5.astype(np.uint8)
    resout_name5='/out_WS/res_bl/'+str(img_num)+'_cd3.png'
    Image.fromarray(res_out_entire5).save(resout_name5)    
    
    res_out_entire6=res_out_entire6.astype(np.uint8)
    resout_name6='/out_WS/res_bl/'+str(img_num)+'_cd20.png'
    Image.fromarray(res_out_entire6).save(resout_name6)    
    
    res_out_entire7=res_out_entire7.astype(np.uint8)
    resout_name7='/out_WS/res_bl/'+str(img_num)+'_foxp3.png'
    Image.fromarray(res_out_entire7).save(resout_name7)    
    
    res_out_entire8=res_out_entire8.astype(np.uint8)
    resout_name8='/out_WS/res_bl/'+str(img_num)+'_pd1.png'
    Image.fromarray(res_out_entire8).save(resout_name8)    
    
    w,h=res_out_entire5.shape
    out_final=np.zeros((w,h,3), dtype=float)  
    res_out_entire5=(res_out_entire5/(np.max(res_out_entire5)))*255
    res_out_entire5[res_out_entire5 < 30] = 0
    out_final[:,:,1]=res_out_entire5*1
    out_final[:,:,2]=img_dapi_base
    out_final=out_final.astype(np.uint8)
    resout_name1='/out_WS/out_entire_colored/'+str(img_num)+'_cd3.png'
    Image.fromarray(out_final).save(resout_name1)    
    
    #w,h=res_out_entire5.shape
    out_final=np.zeros((w,h,3), dtype=float)  
    res_out_entire6=(res_out_entire6/(np.max(res_out_entire6)))*255
    res_out_entire6[res_out_entire6 < 3] = 0
    out_final[:,:,0]=res_out_entire6*1.5 
    out_final[:,:,2]=img_dapi_base+res_out_entire6*1.5 
    out_final=out_final.astype(np.uint8)
    resout_name1='/out_WS/out_entire_colored/'+str(img_num)+'_cd20.png'
    Image.fromarray(out_final).save(resout_name1)   

    out_final=np.zeros((w,h,3), dtype=float)  
    res_out_entire7=(res_out_entire7/(np.max(res_out_entire7)))*255
    res_out_entire7[res_out_entire7 < 5] = 0
    out_final[:,:,0]=res_out_entire7*1.8    
    out_final[:,:,2]=img_dapi_base    
    out_final=out_final.astype(np.uint8)
    resout_name1='/out_WS/out_entire_colored/'+str(img_num)+'_foxp3.png'
    Image.fromarray(out_final).save(resout_name1)   

    out_final=np.zeros((w,h,3), dtype=float)   
    res_out_entire8=(res_out_entire8/(np.max(res_out_entire8)))*255
    res_out_entire8[res_out_entire8 < 8] = 0
    out_final[:,:,0]=res_out_entire8*1.1     
    out_final[:,:,2]=img_dapi_base
    out_final[:,:,1]=res_out_entire8*1.1 
    out_final=out_final.astype(np.uint8)
    resout_name1='/out_WS/out_entire_colored/'+str(img_num)+'_pd1.png'
    Image.fromarray(out_final).save(resout_name1)     

    out_final=np.zeros((w,h,3), dtype=float)     
    out_final[:,:,0]=res_out_entire6*1.2+res_out_entire7*1.8+res_out_entire8*1.1          
    out_final[:,:,1]=res_out_entire5*1.2+res_out_entire8*1.1 
    out_final[:,:,2]=img_dapi_base+res_out_entire6*1.2
    out_final=(out_final/(np.max(out_final)))*255
   
    
    out_final=out_final.astype(np.uint8)
    resout_name5='/out_WS/out_entire_colored/'+str(img_num)+'.png'
    Image.fromarray(out_final).save(resout_name5)       