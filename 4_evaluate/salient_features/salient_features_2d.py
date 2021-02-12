#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import keras
from keras.layers import Input, Dense, merge
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, Conv3D, MaxPooling3D, Reshape, BatchNormalization, Lambda
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import ELU
from keras import regularizers
from keras import optimizers
from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os, os.path
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML
from glob import iglob
from model.models import build_2d_cnn_baseline
import matplotlib
import matplotlib.animation as animation
from keras.backend.tensorflow_backend import set_session

global n_stacked
n_stacked = 1

model = build_2d_cnn_baseline(w=157, h=157, d=1)
model.load_weights('2d_keras_model.hdf5')
model.summary()
img_in = Input(shape=(157, 157, 1))
h = 74
w = 154
d = 1
x = Convolution2D(24, (5,5), strides=(2,2), activation='relu', name='conv1', input_shape=(h, w, d))(img_in)
x = Convolution2D(32, (5,5), strides=(2,2), activation='relu', name='conv2', input_shape=(h, w, d))(x)
x = Convolution2D(64, (5,5), strides=(2,2), activation='relu', name='conv3', input_shape=(h, w, d))(x)
x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name='conv4', input_shape=(h, w, d))(x)
conv_5 = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name='conv5', input_shape=(h, w, d))(x)
convolution_part = Model(inputs=img_in, outputs=conv_5)
for layer_num in ('1', '2', '3', '4', '5'):
    convolution_part.get_layer('conv' + layer_num).set_weights(model.get_layer('conv' + layer_num).get_weights())

inp = convolution_part.input                                           # input placeholder
outputs = [layer.output for layer in convolution_part.layers]          # all layer outputs
functor = K.function([inp], outputs)

kernel_3x3 = tf.constant(np.array([
        [[[1]], [[1]], [[1]]], 
        [[[1]], [[1]], [[1]]], 
        [[[1]], [[1]], [[1]]]
]), tf.float32)

kernel_5x5 = tf.constant(np.array([
        [[[1]], [[1]], [[1]], [[1]], [[1]]], 
        [[[1]], [[1]], [[1]], [[1]], [[1]]], 
        [[[1]], [[1]], [[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]], [[1]], [[1]]]
]), tf.float32)

layers_kernels = {5: kernel_3x3, 4: kernel_3x3, 3: kernel_5x5, 2: kernel_5x5, 1: kernel_5x5}

layers_strides = {5: [1, 1, 1, 1], 4: [1, 1, 1, 1], 3: [1, 2, 2, 1], 2: [1, 2, 2, 1], 1: [1, 2, 2, 1]}

def compute_visualisation_mask(img):
    activations = functor([np.array([img])])	
    upscaled_activation = np.ones((13,13))
    for layer in [5, 4, 3, 2, 1]:
        the_layers = np.mean(activations[layer], axis=3).squeeze(axis=0)
        averaged_activation = the_layers * upscaled_activation
        outputs_shape = (activations[layer - 1].shape[1], activations[layer - 1].shape[2])
        x = np.reshape(averaged_activation, (1, averaged_activation.shape[0],averaged_activation.shape[1],1))    
        modeltwo = Sequential()
        if layer == 5:
                modeltwo.add(Conv2DTranspose(filters=1, kernel_size=(3,3), strides=(1,1),
                input_shape=(13, 13, 1), data_format='channels_last',
                padding='valid'))
        if layer == 4:
                modeltwo.add(Conv2DTranspose(filters=1, kernel_size=(3,3), strides=(1,1),
                input_shape=(15, 15, 1), data_format='channels_last',
                padding='valid'))
        if layer == 3:
                modeltwo.add(Conv2DTranspose(filters=1, kernel_size=(5,5), strides=(2,2),
                input_shape=(17, 17, 1), data_format='channels_last',
                padding='valid'))
        if layer == 2:
                modeltwo.add(Conv2DTranspose(filters=1, kernel_size=(5,5), strides=(2,2),
                input_shape=(37, 37, 1), data_format='channels_last',
                padding='valid'))
        if layer == 1:
                modeltwo.add(Conv2DTranspose(filters=1, kernel_size=(5,5), strides=(2,2),
                input_shape=(77, 77, 1), data_format='channels_last',
                padding='valid'))
        result = modeltwo.predict(x)
        #result = result.squeeze(axis=0)
        upscaled_activation = np.reshape(result, outputs_shape)
    final_visualisation_mask = upscaled_activation
    return (final_visualisation_mask - np.min(final_visualisation_mask))/(np.max(final_visualisation_mask) - np.min(final_visualisation_mask))

def plot_movie_mp4(image_array):
    dpi = 100.0
    xpixels, ypixels = image_array[0].shape[0], image_array[0].shape[1]
    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
    im = plt.figimage(image_array[0])
    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array))
    #matplotlib.rcParams['animation.writer'] = 'avconv'
    display(HTML(anim.to_html5_video()))
    anim.save('/home/jesse/Desktop/DNRacing/deep_learning/animation.mp4', writer='imagemagick', fps=40)

csv_path = '/media/jesse/c334c9bc-7ae4-4ea7-84fb-6b8f5595aea2/v1/finaldataset.csv'
img_path = '/media/jesse/c334c9bc-7ae4-4ea7-84fb-6b8f5595aea2/v1/images/'

df = pd.read_csv(csv_path, encoding='utf-8')   
image_location = []
for i, row in tqdm(df.iterrows()):
        fname = row['image']
        image_location.append(fname[:-1])
        
imgs = []
alpha = 0.001
beta = 1.0 - alpha
counter = 0
img_stack = []
z = []
number = 8000
target_image = []	
for a in range(300):	
        display_img_stack = []
        img = cv2.imread('/media/jesse/c334c9bc-7ae4-4ea7-84fb-6b8f5595aea2/v1/images/'+ str(image_location[number]))	
        img = img[25:, :]
        img = cv2.resize(img, (157,157))
        #blurred = cv2.GaussianBlur(img, (1, 1), 0)
	#img = cv2.Laplacian(img, cv2.CV_64F)
        img = cv2.Canny(img, 1, 255, 1, L2gradient=True)
        img_orig =img
        #img = (img/127.5) - 1
        img = img.reshape(img.shape[0],img.shape[1],1)
        salient_mask = compute_visualisation_mask(img)
        salient_mask = cv2.add(salient_mask, salient_mask)
        blend = cv2.addWeighted(img_orig.astype('float32'), alpha, salient_mask, beta, 0.0)	
        imgs.append(blend)
        counter += 1
        number = number + 1
        print(counter)
        if counter >= 70800:
                break



plot_movie_mp4(imgs)
