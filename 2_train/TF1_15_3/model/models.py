#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Activation, Dense, Dropout, MaxPooling3D, Conv3D, Convolution2D, MaxPooling2D, SpatialDropout2D, Lambda, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import AlphaDropout
from keras.regularizers import l2
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras import activations, constraints, initializers, regularizers
from keras import optimizers
from keras.losses import binary_crossentropy, mean_squared_error
from keras.layers.advanced_activations import ELU
from keras.layers.wrappers import TimeDistributed
from keras.backend import l2_normalize
from keras.backend.common import set_floatx
from keras import backend as K
from keras.callbacks import TensorBoard, Callback, History
set_floatx('float16')
from keras import metrics
from model_utils.functions import mse_loss
import tensorflow as tf
#################LRCN with Sensor Input#################

def build_lrcn_sensor(w, h, d, s):
    image_inputs = Input(shape=(s, h, w, d), name = 'image_input')
    sensor_inputs = Input(shape=(1,), name= 'sensor_input')
    x = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(image_inputs)
    x = TimeDistributed(Convolution2D(filters=32, kernel_size=(3, 3), 
        strides=(2,2),data_format='channels_last', input_shape=(h, w, d), activation='relu', name = 'conv1'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Convolution2D(filters=32, kernel_size=(3, 3),
        strides=(2,2),data_format='channels_last', input_shape=(h, w, d), activation='relu', name = 'conv2'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Convolution2D(filters=64, kernel_size=(3, 3),
        strides=(2,2),data_format='channels_last', input_shape=(h, w, d), activation='relu', name = 'conv3'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Convolution2D(filters=64, kernel_size=(3, 3),
        strides=(1,1),data_format='channels_last', input_shape=(h, w, d), activation='relu', name = 'conv4'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Convolution2D(filters=64, kernel_size=(3, 3),
        strides=(1,1),data_format='channels_last', input_shape=(h, w, d), activation='relu', name = 'conv5'))(x)
    x = BatchNormalization()(x)
    # Fully connected layer
    x = TimeDistributed(Flatten())(x)
    x1 = LSTM(512, return_sequences=False, stateful=False)(x)
    a = Dense(256, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense1")(x1)
    a = BatchNormalization()(a)
    a = Dense(128, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense2")(a)
    a = BatchNormalization()(a)
    a = Dense(64, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense3")(a)
    a = BatchNormalization()(a)
    x2 = x1 #Concatenate()([x1, sensor_inputs])
    b = Dense(256, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense1")(x2)
    b = BatchNormalization()(b)
    b = Dense(128, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense2")(b)
    b = BatchNormalization()(b)
    b = Dense(64, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense3")(b)
    b = BatchNormalization()(b)
    angle_out = Dense(1, activation='linear', name= 'angle_out')(a)
    throttle_out = Dense(1, activation='linear', name= 'throttle_out')(b)
    model = Model(inputs=[image_inputs, sensor_inputs], outputs=[angle_out, throttle_out])
    optimizer = optimizers.adam(lr = 0.0001)	
    model.compile(loss= {'angle_out': 'mean_squared_error', 
    			 'throttle_out': 'mean_squared_error'},
                  optimizer=optimizer,
                   metrics={'angle_out': ['mse'], 'throttle_out': ['mse']}, loss_weights=[1, 1])
    model.summary()
    return model

def build_3d_sensor(w, h, d, s):
    image_inputs = Input(shape=(s, h, w, d), name = 'image_input')
    sensor_inputs = Input(shape=(1,), name= 'sensor_input')
    x = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(image_inputs)
    x = Conv3D(
        filters=32, kernel_size=(3,3,3), strides=(1,2,2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name = 'conv1')(x)
    x = BatchNormalization()(x)
    x = Conv3D(
        filters=32, kernel_size=(3,3,3), strides=(1,2,2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name = 'conv2')(x)
    x = BatchNormalization()(x)
    x = Conv3D(
        filters=64, kernel_size=(3,3,3), strides=(1,2,2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name = 'conv3')(x)
    x = BatchNormalization()(x)
    x = Conv3D(
        filters=64, kernel_size=(3,3,3), strides=(1,2,2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name = 'conv4')(x)
    x = BatchNormalization()(x)
    x = Conv3D(
        filters=64, kernel_size=(3,3,3), strides=(1,2,2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name = 'conv5')(x)
    x = BatchNormalization()(x)
    # Fully connected layer
    x = TimeDistributed(Flatten())(x)
    x1 = LSTM(512, return_sequences=False, stateful=False)(x)
    a = Dense(256, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense1")(x1)
    a = BatchNormalization()(a)
    a = Dense(128, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense2")(a)
    a = BatchNormalization()(a)
    a = Dense(64, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense3")(a)
    a = BatchNormalization()(a)
    x2 = x1 #Concatenate()([x1, sensor_inputs])
    b = Dense(256, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense1")(x2)
    b = BatchNormalization()(b)
    b = Dense(128, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense2")(b)
    b = BatchNormalization()(b)
    b = Dense(64, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense3")(b)
    b = BatchNormalization()(b)
    angle_out = Dense(1, activation='linear', name= 'angle_out')(a)
    throttle_out = Dense(1, activation='linear', name= 'throttle_out')(b)
    model = Model(inputs=[image_inputs, sensor_inputs], outputs=[angle_out, throttle_out])
    optimizer = optimizers.adam(lr = 0.00001)	
    model.compile(loss= {'angle_out': 'mean_squared_error', 
    			 'throttle_out': 'mean_squared_error'},
                  optimizer=optimizer,
                   metrics={'angle_out': ['mse'], 'throttle_out': ['mse']}, loss_weights=[1, 1])
    model.summary()
    return model

