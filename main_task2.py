
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:26:54 2023

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
from tensorflow.keras.layers import Dense,concatenate
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout,Conv2DTranspose
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
matplotlib.use('Agg') 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
my_seed = 0
os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)


def gatingsignal(input, out_size, batchnorm=False):
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x
    
def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same')(x) 
    shape_theta_x = K.int_shape(theta_x)
    phi_g = layers.Conv2D(inter_shape, (1, 1), kernel_initializer='he_normal', padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3), strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2]//shape_g[2]), kernel_initializer='he_normal', padding='same')(phi_g)
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg) 
    upsample_psi = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': shape_x[3]})(upsample_psi)                          
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv2D(shape_x[3], (1, 1), kernel_initializer='he_normal', padding='same')(y)
    attenblock = layers.BatchNormalization()(result)
    return attenblock    
    
 
def squared_differences(pair_of_tensors):
    x, y = pair_of_tensors
    return K.abs(x - y)      

shape1=(256,256,3)
shape2=(256,256,3)
n_channels1=3
n_output=3
n_channels2=3
batchnorm=True

input1 = Input(shape=shape1)
input2=Input(shape=shape2)
input_high=Input(shape=shape2)   

# abstract feature from two images

# upload pretrained MAE-AF
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
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='latent_feature_A')(conv5)
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

pre_generator_auto = Model(inputs=input1, outputs=output_final)
opt = SGD(lr=0.0001, momentum=0.9, clipvalue=5.0)
pre_generator_auto.compile(loss=['mse'], optimizer=opt)
#pre_generator_auto.load_weights('/pretrained/pretrained_MAE_AF.h5')
extraction_auto = Model(inputs=pre_generator_auto.input,outputs=pre_generator_auto.get_layer('latent_feature_AF').output)

# upload pretrained MAE-DAPI
conv1d1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input2)
conv1d1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1d1)
pool1d = MaxPooling2D(pool_size=(2, 2))(conv1d1)
conv2d1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1d)
conv2d1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2d1)
pool2d = MaxPooling2D(pool_size=(2, 2))(conv2d1)
conv3d1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2d)
conv3d1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3d1)
pool3d = MaxPooling2D(pool_size=(2, 2))(conv3d1)
conv4d = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3d)
conv4d = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4d)
drop4d = Dropout(0.5)(conv4d)
pool4d = MaxPooling2D(pool_size=(2, 2))(drop4d)
conv5d = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4d)
conv5d = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='latent_feature_dapi')(conv5d)
drop5d = Dropout(0.5)(conv5d)
up6d = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (drop5d)
merge6d = concatenate([drop4d,up6d], axis = 3)
conv6d = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6d)
conv6d = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6d)
up7d = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (conv6d)
merge7d = concatenate([conv3d1,up7d], axis = 3)
conv7d = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7d)
conv7d = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7d)
up8d = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv7d)
merge8d = concatenate([conv2d1,up8d], axis = 3)
conv8d = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8d)
conv8d = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8d)
up9d = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv8d)
merge9d = concatenate([conv1d1,up9d], axis = 3)
conv9d = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9d)
conv9d = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9d)
conv9d = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9d)
output_finald = Conv2D(n_output, (1, 1), padding='same')(conv9d)

pre_generator_dapi = Model(inputs=input2, outputs=output_finald)
opt = SGD(lr=0.0001, momentum=0.9, clipvalue=5.0)
pre_generator_dapi.compile(loss=['mse'], optimizer=opt)
#pre_generator_dapi.load_weights('/pretrained/pretrained_MAE_DAPI.h5')
extraction_dapi = Model(inputs=pre_generator_dapi.input,outputs=pre_generator_dapi.get_layer('latent_feature_dapi').output)


# abstract feature from two images 
inter1 = extraction_auto.output
inter2 = extraction_dapi.output
for layer in extraction_auto.layers:
 #   layer.trainable=False
    layer._name = layer.name + str('_auto')
for layer in extraction_dapi.layers:
 #   layer.trainable=False    
    layer._name = layer.name + str('_dapi')

# combine dual-model latent features
inter = concatenate([inter1, inter2])
x = Conv2D(512, (5, 5), padding='same')(inter)
x = residual_block(x, input_channels=512, output_channels=256)
x = attention_block1(x, encoder_depth=1)
x = residual_block(x, input_channels=256, output_channels=128)  # 4x4
x = attention_block1(x, encoder_depth=1)
x = Conv2D(1024, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=0.05)(x)
x = concatenate([x, inter])
x = Conv2D(1024, (3, 3), padding='same', name='transfer_part')(x)


# generate imgs through multi-channel
## output1
gating_61 = gatingsignal(x, 128, batchnorm)
att_61a = attention_block(conv4, gating_61, 128)
att_61d = attention_block(conv4d, gating_61, 128)
u6_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (x)
u6_1 = concatenate([u6_1, att_61a,att_61d])
c6_1 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6_1)
c6_1 = Dropout(0.5) (c6_1)
c6_1 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6_1)

gating_71 = gatingsignal(c6_1, 64, batchnorm)
att_71a = attention_block(conv3, gating_71, 64)
att_71d = attention_block(conv3d1, gating_71, 64)
u7_1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6_1)
u7_1 = concatenate([u7_1, att_71a,att_71d])
c7_1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7_1)
c7_1 = Dropout(0.5) (c7_1)
c7_1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7_1)

gating_81 = gatingsignal(c7_1, 32, batchnorm)
att_81a = attention_block(conv2, gating_81, 32)
att_81d = attention_block(conv2d1, gating_81, 32)
u8_1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7_1)
u8_1 = concatenate([u8_1, att_81a,att_81d])
c8_1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8_1)
c8_1 = Dropout(0.5) (c8_1)
c8_1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8_1)

gating_91 = gatingsignal(c8_1, 16, batchnorm)
att_91a = attention_block(conv1, gating_91, 16)
att_91d = attention_block(conv1d1, gating_91, 16)
u9_1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8_1)
u9_1 = concatenate([u9_1, att_91a,att_91d], axis=3)
c9_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9_1)
#c9_1 = Dropout(0.5) (c9_1)
c9_1 = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9_1)

outputs_1 = Conv2D(3, (1, 1)) (c9_1)

## output2
gating_62 = gatingsignal(x, 128, batchnorm)
att_62a = attention_block(conv4, gating_62, 128)
att_62d = attention_block(conv4d, gating_62, 128)
u6_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (x)
u6_2 = concatenate([u6_2, att_62a, att_62d])
c6_2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6_2)
c6_2 = Dropout(0.5) (c6_2)
c6_2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6_2)

gating_72 = gatingsignal(c6_2, 64, batchnorm)
att_72a = attention_block(conv3, gating_72, 64)
att_72d = attention_block(conv3d1, gating_72, 64)
u7_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6_2)
u7_2 = concatenate([u7_2, att_72a,att_72d])
c7_2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7_2)
c7_2 = Dropout(0.5) (c7_2)
c7_2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7_2)

gating_82 = gatingsignal(c7_2, 32, batchnorm)
att_82a = attention_block(conv2, gating_82, 32)
att_82d = attention_block(conv2d1, gating_82, 32)
u8_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7_2)
u8_2 = concatenate([u8_2, att_82a,att_82d])
c8_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8_2)
c8_2 = Dropout(0.5) (c8_2)
c8_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8_2)

gating_92 = gatingsignal(c8_2, 16, batchnorm)
att_92a = attention_block(conv1, gating_92, 16)
att_92d = attention_block(conv1d1, gating_92, 16)
u9_2 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8_2)
u9_2 = concatenate([u9_2, att_92a,att_92d], axis=3)
c9_2 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9_2)
#c9_2 = Dropout(0.1) (c9_2)
c9_2 = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9_2)

outputs_2 = Conv2D(3, (1, 1)) (c9_2)


## output3
gating_63 = gatingsignal(x, 128, batchnorm)
att_63a = attention_block(conv4, gating_63, 128)
att_63d = attention_block(conv4d, gating_63, 128)
u6_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (x)
u6_3 = concatenate([u6_3, att_63a,att_63d])
c6_3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6_3)
c6_3 = Dropout(0.5) (c6_3)
c6_3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6_3)

gating_73 = gatingsignal(c6_3, 64, batchnorm)
att_73a = attention_block(conv3, gating_73, 64)
att_73d = attention_block(conv3d1, gating_73, 64)
u7_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6_3)
u7_3 = concatenate([u7_3, att_73a,att_73d])
c7_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7_3)
c7_3 = Dropout(0.5) (c7_3)
c7_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7_3)

gating_83 = gatingsignal(c7_3, 32, batchnorm)
att_83a = attention_block(conv2, gating_83, 32)
att_83d = attention_block(conv2d1, gating_83, 32)
u8_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7_3)
u8_3 = concatenate([u8_3, att_83a,att_83d])
c8_3 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8_3)
c8_3 = Dropout(0.5) (c8_3)
c8_3 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8_3)

gating_93 = gatingsignal(c8_3, 16, batchnorm)
att_93a = attention_block(conv1, gating_93, 16)
att_93d = attention_block(conv1d1, gating_93, 16)
u9_3 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8_3)
u9_3 = concatenate([u9_3, att_93a,att_93d], axis=3)
c9_3 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9_3)
#c9_3 = Dropout(0.1) (c9_3)
c9_3 = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9_3)

outputs_3 = Conv2D(3, (1, 1)) (c9_3)

## output4
gating_64 = gatingsignal(x, 128, batchnorm)
att_64a = attention_block(conv4, gating_64, 128)
att_64d = attention_block(conv4d, gating_64, 128)
u6_4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (x)
u6_4 = concatenate([u6_4, att_64a,att_64d])
c6_4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6_4)
c6_4 = Dropout(0.5) (c6_4)
c6_4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6_4)

gating_74 = gatingsignal(c6_4, 64, batchnorm)
att_74a = attention_block(conv3, gating_74, 64)
att_74d = attention_block(conv3d1, gating_74, 64)
u7_4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6_4)
u7_4 = concatenate([u7_4, att_74a,att_74d])
c7_4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7_4)
c7_4 = Dropout(0.5) (c7_4)
c7_4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7_4)

gating_84 = gatingsignal(c7_4, 32, batchnorm)
att_84a = attention_block(conv2, gating_84, 32)
att_84d = attention_block(conv2d1, gating_84, 32)
u8_4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7_4)
u8_4 = concatenate([u8_4, att_84a,att_84d])
c8_4 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8_4)
c8_4 = Dropout(0.5) (c8_4)
c8_4 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8_4)

gating_94 = gatingsignal(c8_4, 16, batchnorm)
att_94a = attention_block(conv1, gating_94, 16)
att_94d = attention_block(conv1d1, gating_94, 16)
u9_4 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8_4)
u9_4 = concatenate([u9_4, att_94a,att_94d], axis=3)
c9_4 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9_4)
#c9_4 = Dropout(0.1) (c9_4)
c9_4 = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9_4)

outputs_4 = Conv2D(3, (1, 1)) (c9_4)

#cal_diff (residual fine-tune sec)
diff1 = Lambda(squared_differences,name = "diff1")([outputs_1, input2])
diff1 = Conv2D(16, (3, 3), padding='same')(diff1)
diff1 = LeakyReLU(alpha=0.05)(diff1)
diff1 = Conv2D(3, (3, 3), padding='same')(diff1)
diff1 = LeakyReLU(alpha=0.05)(diff1)
out_residual1 = Conv2D(1, (1, 1), padding='same')(diff1)

diff2 = Lambda(squared_differences,name = "diff2")([outputs_2, input2])
diff2 = Conv2D(16, (3, 3), padding='same')(diff2)
diff2 = LeakyReLU(alpha=0.05)(diff2)
diff2 = Conv2D(3, (3, 3), padding='same')(diff2)
diff2 = LeakyReLU(alpha=0.05)(diff2)
out_residual2 = Conv2D(1, (1, 1), padding='same')(diff2)

diff3 = Lambda(squared_differences,name = "diff3")([outputs_3, input2])
diff3 = Conv2D(16, (3, 3), padding='same')(diff3)
diff3 = LeakyReLU(alpha=0.05)(diff3)
diff3 = Conv2D(3, (3, 3), padding='same')(diff3)
diff3 = LeakyReLU(alpha=0.05)(diff3)
out_residual3 = Conv2D(1, (1, 1), padding='same')(diff3)

diff4 = Lambda(squared_differences,name = "diff4")([outputs_4, input2])
diff4 = Conv2D(16, (3, 3), padding='same')(diff4)
diff4 = LeakyReLU(alpha=0.05)(diff4)
diff4 = Conv2D(3, (3, 3), padding='same')(diff4)
diff4 = LeakyReLU(alpha=0.05)(diff4)
out_residual4 = Conv2D(1, (1, 1), padding='same')(diff4)


#build model (output 1-4 are stained images with dapi background; output 5-8 are stained images only)
generator_task1 = Model(inputs=[input1,input2], outputs=[outputs_1,outputs_2,outputs_3,outputs_4,out_residual1,out_residual2,out_residual3,out_residual4])
opt = Adam(lr=0.00005)

#loss function (loss sec 1-4 are used to optimize the stained image with background; loss sec 5-8 are used to optimize the stained image only)
generator_task1.compile(loss=['mse','mse','mse','mse','mse','mse','mse','mse'], optimizer=opt)
generator_task1.load_weights('/models/model_task1.h5')
generator_pre= Model(inputs=generator_task1.input,outputs=generator_task1.get_layer('transfer_part').output)


# get pretrained combined feature from task1 model
x = generator_pre.output


# generate imgs through multi-channel
## output1
gating_61 = gatingsignal(x, 128, batchnorm)
att_61a = attention_block(conv4, gating_61, 128)
att_61d = attention_block(conv4d, gating_61, 128)
u6_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (x)
u6_1 = concatenate([u6_1, att_61a,att_61d])
c6_1 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6_1)
c6_1 = Dropout(0.5) (c6_1)
c6_1 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6_1)

gating_71 = gatingsignal(c6_1, 64, batchnorm)
att_71a = attention_block(conv3, gating_71, 64)
att_71d = attention_block(conv3d1, gating_71, 64)
u7_1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6_1)
u7_1 = concatenate([u7_1, att_71a,att_71d])
c7_1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7_1)
c7_1 = Dropout(0.5) (c7_1)
c7_1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7_1)

gating_81 = gatingsignal(c7_1, 32, batchnorm)
att_81a = attention_block(conv2, gating_81, 32)
att_81d = attention_block(conv2d1, gating_81, 32)
u8_1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7_1)
u8_1 = concatenate([u8_1, att_81a,att_81d])
c8_1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8_1)
c8_1 = Dropout(0.5) (c8_1)
c8_1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8_1)

gating_91 = gatingsignal(c8_1, 16, batchnorm)
att_91a = attention_block(conv1, gating_91, 16)
att_91d = attention_block(conv1d1, gating_91, 16)
u9_1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8_1)
u9_1 = concatenate([u9_1, att_91a,att_91d], axis=3)
c9_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9_1)
#c9_1 = Dropout(0.5) (c9_1)
c9_1 = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9_1)

outputs_1 = Conv2D(3, (1, 1)) (c9_1)

## output2
gating_62 = gatingsignal(x, 128, batchnorm)
att_62a = attention_block(conv4, gating_62, 128)
att_62d = attention_block(conv4d, gating_62, 128)
u6_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (x)
u6_2 = concatenate([u6_2, att_62a, att_62d])
c6_2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6_2)
c6_2 = Dropout(0.5) (c6_2)
c6_2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6_2)

gating_72 = gatingsignal(c6_2, 64, batchnorm)
att_72a = attention_block(conv3, gating_72, 64)
att_72d = attention_block(conv3d1, gating_72, 64)
u7_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6_2)
u7_2 = concatenate([u7_2, att_72a,att_72d])
c7_2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7_2)
c7_2 = Dropout(0.5) (c7_2)
c7_2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7_2)

gating_82 = gatingsignal(c7_2, 32, batchnorm)
att_82a = attention_block(conv2, gating_82, 32)
att_82d = attention_block(conv2d1, gating_82, 32)
u8_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7_2)
u8_2 = concatenate([u8_2, att_82a,att_82d])
c8_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8_2)
c8_2 = Dropout(0.5) (c8_2)
c8_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8_2)

gating_92 = gatingsignal(c8_2, 16, batchnorm)
att_92a = attention_block(conv1, gating_92, 16)
att_92d = attention_block(conv1d1, gating_92, 16)
u9_2 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8_2)
u9_2 = concatenate([u9_2, att_92a,att_92d], axis=3)
c9_2 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9_2)
#c9_2 = Dropout(0.1) (c9_2)
c9_2 = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9_2)

outputs_2 = Conv2D(3, (1, 1)) (c9_2)


## output3
gating_63 = gatingsignal(x, 128, batchnorm)
att_63a = attention_block(conv4, gating_63, 128)
att_63d = attention_block(conv4d, gating_63, 128)
u6_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (x)
u6_3 = concatenate([u6_3, att_63a,att_63d])
c6_3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6_3)
c6_3 = Dropout(0.5) (c6_3)
c6_3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6_3)

gating_73 = gatingsignal(c6_3, 64, batchnorm)
att_73a = attention_block(conv3, gating_73, 64)
att_73d = attention_block(conv3d1, gating_73, 64)
u7_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6_3)
u7_3 = concatenate([u7_3, att_73a,att_73d])
c7_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7_3)
c7_3 = Dropout(0.5) (c7_3)
c7_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7_3)

gating_83 = gatingsignal(c7_3, 32, batchnorm)
att_83a = attention_block(conv2, gating_83, 32)
att_83d = attention_block(conv2d1, gating_83, 32)
u8_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7_3)
u8_3 = concatenate([u8_3, att_83a,att_83d])
c8_3 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8_3)
c8_3 = Dropout(0.5) (c8_3)
c8_3 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8_3)

gating_93 = gatingsignal(c8_3, 16, batchnorm)
att_93a = attention_block(conv1, gating_93, 16)
att_93d = attention_block(conv1d1, gating_93, 16)
u9_3 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8_3)
u9_3 = concatenate([u9_3, att_93a,att_93d], axis=3)
c9_3 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9_3)
#c9_3 = Dropout(0.1) (c9_3)
c9_3 = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9_3)

outputs_3 = Conv2D(3, (1, 1)) (c9_3)


#cal_diff (residual fine-tune sec)
diff1 = Lambda(squared_differences,name = "diff1")([outputs_1, input2])
diff1 = Conv2D(16, (3, 3), padding='same')(diff1)
diff1 = LeakyReLU(alpha=0.05)(diff1)
diff1 = Conv2D(3, (3, 3), padding='same')(diff1)
diff1 = LeakyReLU(alpha=0.05)(diff1)
out_residual1 = Conv2D(1, (1, 1), padding='same')(diff1)

diff2 = Lambda(squared_differences,name = "diff2")([outputs_2, input2])
diff2 = Conv2D(16, (3, 3), padding='same')(diff2)
diff2 = LeakyReLU(alpha=0.05)(diff2)
diff2 = Conv2D(3, (3, 3), padding='same')(diff2)
diff2 = LeakyReLU(alpha=0.05)(diff2)
out_residual2 = Conv2D(1, (1, 1), padding='same')(diff2)

diff3 = Lambda(squared_differences,name = "diff3")([outputs_3, input2])
diff3 = Conv2D(16, (3, 3), padding='same')(diff3)
diff3 = LeakyReLU(alpha=0.05)(diff3)
diff3 = Conv2D(3, (3, 3), padding='same')(diff3)
diff3 = LeakyReLU(alpha=0.05)(diff3)
out_residual3 = Conv2D(1, (1, 1), padding='same')(diff3)

generator = Model(inputs=generator_pre.input, outputs=[outputs_1,outputs_2,outputs_3,out_residual1,out_residual2,out_residual3])
opt = Adam(lr=0.00005)

generator.compile(loss=['mse','mse','mse','mse','mse','mse'], optimizer=opt)





def adjustData(img1,img2,mask1,mask2,mask3,mask4,mask5,mask6):
        img1 = img1 / 255  
        img2 = img2 / 255  
        mask1 = mask1 / 255  
        mask2 = mask2 / 255 
        mask3 = mask3 / 255 
        mask4 = mask4 / 255 
        mask5 = mask5 / 255 
        mask6 = mask6 / 255 
        return (img1,img2,mask1,mask2,mask3,mask4,mask5,mask6)
        

def trainGenerator(batch_size,path1,path2,path3,path4,path5,path6,path7,path8, aug_dict,target_size = (256,256),seed = 1):

    image1_datagen = ImageDataGenerator(**aug_dict)
    image2_datagen = ImageDataGenerator(**aug_dict)
    mask1_datagen = ImageDataGenerator(**aug_dict)
    mask2_datagen = ImageDataGenerator(**aug_dict)
    mask3_datagen = ImageDataGenerator(**aug_dict)
    mask4_datagen = ImageDataGenerator(**aug_dict)
    mask5_datagen = ImageDataGenerator(**aug_dict)
    mask6_datagen = ImageDataGenerator(**aug_dict) 
    image1_generator = image1_datagen.flow_from_directory(
        path1,
        class_mode=None,
        batch_size = batch_size,
     #   target_size=(128, 128),
        seed=seed)
    image2_generator = image2_datagen.flow_from_directory(
        path2,
        class_mode=None,
        batch_size = batch_size,
    #    target_size=(128, 128),
        seed=seed)        
        
    mask1_generator = mask1_datagen.flow_from_directory(
        path3,
        class_mode=None,
        batch_size = batch_size,
     #   target_size=(128, 128),
        seed=seed)
    
    mask2_generator = mask2_datagen.flow_from_directory(
        path4,
        class_mode=None,
        batch_size = batch_size,
     #   target_size=(128, 128),
        seed=seed)

    mask3_generator = mask3_datagen.flow_from_directory(
        path5,
        class_mode=None,
        batch_size = batch_size,
      #  target_size=(128, 128),
        seed=seed)

    mask4_generator = mask4_datagen.flow_from_directory(
         path6,
         class_mode=None,
         batch_size = batch_size,
        # target_size=(128, 128),
         seed=seed)

    mask5_generator = mask5_datagen.flow_from_directory(
         path7,
         class_mode=None,
         batch_size = batch_size,
        # target_size=(128, 128),
         seed=seed)    
         
    mask6_generator = mask6_datagen.flow_from_directory(
         path8,
         class_mode=None,
         batch_size = batch_size,
        # target_size=(128, 128),
         seed=seed)          
    
    
    train_generator = zip(image1_generator,image2_generator, mask1_generator,mask2_generator,mask3_generator, mask4_generator,mask5_generator,mask6_generator)
    for (img1,img2,mask1,mask2,mask3,mask4,mask5,mask6) in train_generator:
        img1,img2,mask1,mask2,mask3,mask4,mask5,mask6 = adjustData(img1,img2,mask1,mask2,mask3,mask4,mask5,mask6)
        yield ([img1,img2],[mask1,mask2,mask3,mask4,mask5,mask6])
        
data_gen_args = dict(
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    shear_range=0.1,
                    zoom_range=0.15,                 
                    fill_mode='nearest')
#data_gen_args = dict()
path_auto='/data_task2/Training/input_auto/'
path_dapi='/data_task2/Training/input_dapi/'
path_cd8a='/data_task2/Training/output_cd8a/' 
path_cd163='/data_task2/Training/output_cd163/' 
path_pdl1='/data_task2/Training/output_pdl1/'  
path_residual_cd8a='/data_task2/Training/out_residual_cd8a/' 
path_residual_cd163='/data_task2/Training/out_residual_cd163/' 
path_residual_pdl1='/data_task2/Training/out_residual_pdl1/'                                    
myGene = trainGenerator(1,path_auto,path_dapi,path_cd8a,path_cd163,path_pdl1,path_residual_cd8a,path_residual_cd163,path_residual_pdl1,data_gen_args)             

data1_gen_args = dict()


pathv_auto='/data_task2/Testing/input_auto/'
pathv_dapi='/data_task2/Testing/input_dapi/'
pathv_cd8='/data_task2/Testing/output_cd8a/' 
pathv_cd163='/data_task2/Testing/output_cd163/'
pathv_pdl1='/data_task2/Testing/output_pdl1/'   
pathv_residual_cd8='/data_task2/Testing/out_residual_cd8a/'
pathv_residual_cd163='/data_task2/Testing/out_residual_cd163/'  
pathv_residual_pdl1='/data_task2/Testing/out_residual_pdl1/'            
myGene_val = trainGenerator(1,pathv_auto,pathv_dapi,pathv_cd8,pathv_cd163,pathv_pdl1,pathv_residual_cd8,pathv_residual_cd163,pathv_residual_pdl1,data1_gen_args)  

EarlyStop=EarlyStopping(monitor='val_loss',
                        patience=50,verbose=1, mode='min')
Reduce=ReduceLROnPlateau(monitor='val_loss',
                         factor=0.8,
                         patience=10,
                         verbose=1,
                         mode='min',
                         epsilon=0.00001,
                         cooldown=0,
                         min_lr=0)
best_model_path='/models/model_task2.h5'
mc = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, mode='min')     
generator.fit(myGene,steps_per_epoch=10240//4,validation_data = myGene_val,validation_steps = 1280//4, epochs=200,verbose=1,callbacks=[EarlyStop,Reduce,mc])
