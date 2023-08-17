# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 17:16:55 2022

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
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout,concatenate,Conv2DTranspose
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

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
matplotlib.use('Agg') 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
my_seed = 0
os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)

# this function generate a random mask for each input patch during each epoch
def adjustData(img1,gt1):     
        a1=random.randint(0,255)
        a2=random.randint(0,255)
        ps=random.randint(10,25)
        img1[a1:a1+ps, a2:a2+ps]=255
        img1 = img1 / 255 
        gt1 = gt1 / 255  
        return (img1,gt1)
    

shape1=(256,256,3)
shape2=(256,256,3)
n_channels1=3
n_output=3
n_channels2=3


input1 = Input(shape=shape1)


# abstract feature from two images
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input1)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='latent_feature_auto')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (drop5)
merge6 = concatenate([drop4,up6], axis = 3)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (conv6)
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv7)
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv8)
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
output_final = Conv2D(n_output, (1, 1), padding='same')(conv9)



pre_generator = Model(inputs=input1, outputs=output_final)
opt = SGD(lr=0.0001, momentum=0.9, clipvalue=5.0)
pre_generator.compile(loss=['mse'], optimizer=opt)

        

def trainGenerator(batch_size,path1, path2, aug_dict,target_size = (256,256),seed = 1):

    image1_datagen = ImageDataGenerator(**aug_dict)

    mask1_datagen = ImageDataGenerator(**aug_dict)

    
    image1_generator = image1_datagen.flow_from_directory(
        path1,
        class_mode=None,
        batch_size = batch_size,
        seed=seed)

    mask1_generator = mask1_datagen.flow_from_directory(
        path2,
        class_mode=None,
        batch_size = batch_size,
        seed=seed)
      
    train_generator = zip(image1_generator,mask1_generator)
    for (img1,mask1) in train_generator:
        img1,mask1 = adjustData(img1,mask1)
        yield (img1,mask1)
        
data_gen_args = dict(
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    shear_range=0.15,
                    zoom_range=0.15,                 
                    fill_mode='nearest')
#data_gen_args = dict()
path_auto='/pathology/data_task_AE/auto/'
#path_dapi='/pathology/data_task_AE/dapi/'
                                  
myGene = trainGenerator(4,path_auto,path_auto,data_gen_args)        
     

EarlyStop=EarlyStopping(monitor='loss',
                        patience=30,verbose=1, mode='min')

best_model_path='/pretrained/pretrained_MAE_AF.h5'
#best_model_path='/pretrained/pretrained_MAE_DAPI.h5'
   
mc = ModelCheckpoint(
    best_model_path,
    monitor="loss",
    mode="min",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)

