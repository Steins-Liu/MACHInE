#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import keras
from keras.layers import Input, Dense, merge
from keras.layers.convolutional import Conv3DTranspose
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution3D, Conv3D, MaxPooling3D, Reshape, BatchNormalization, Lambda
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
from model.models import build_3d_cnn_baseline
import matplotlib
import matplotlib.animation as animation
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True, device_count = {'CPU' : 1, 'GPU' : 1})
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

global n_stacked
n_stacked = 2

model = build_3d_cnn_baseline(w=160, h=160, d=3, s=n_stacked)
model.load_weights('3d_keras_model.hdf5')

img_in = Input(shape=(n_stacked, 160, 160, 3), name='img_in')
h = 160
w = 160
d = 3
s = n_stacked
x = img_in
x = Conv3D(24, (5,5,5), strides=(1,2,2), activation='relu', name='conv1', border_mode='same', data_format='channels_last', input_shape=(s, h, w, d))(x)
x = Conv3D(32, (5,5,5), strides=(1,2,2), activation='relu', name='conv2', border_mode='same', data_format='channels_last', input_shape=(s, h, w, d))(x)
x = Conv3D(64, (5,5,5), strides=(1,2,2), activation='relu', name='conv3', border_mode='same', data_format='channels_last', input_shape=(s, h, w, d))(x)
x = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu', name='conv4', border_mode='same', data_format='channels_last', input_shape=(s, h, w, d))(x)
conv_5 = Conv3D(64, (3,3,3), strides=(1,1,1), activation='elu', name='conv5', border_mode='same', data_format='channels_last', input_shape=(s, h, w, d))(x)
convolution_part = Model(inputs=[img_in], outputs=[conv_5])
for layer_num in ('1', '2', '3', '4', '5'):
    convolution_part.get_layer('conv' + layer_num).set_weights(model.get_layer('conv' + layer_num).get_weights())

inp = convolution_part.input                                           # input placeholder
outputs = [layer.output for layer in convolution_part.layers]          # all layer outputs
functor = K.function([inp], outputs)

kernel_3x3x3 = tf.constant(np.array(
[
        [
                [[[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]]]
        ],
        [
                [[[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]]]
        ],
        [
                [[[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]]]
        ]
]
), tf.float32)

kernel_5x5x5 = tf.constant(np.array(
[
        [
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]]
        ],
        [
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]]
        ],
        [
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]]
        ],
        [
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]]
        ],
        [
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]]
        ]
]
), tf.float32)

layers_kernels = {5: kernel_3x3x3, 4: kernel_3x3x3, 3: kernel_5x5x5, 2: kernel_5x5x5, 1: kernel_5x5x5}

layers_strides = {5: [1, 1, 1, 1, 1], 4: [1, 1, 1, 1, 1], 3: [1, 1, 2, 2, 1], 2: [1, 1, 2, 2, 1], 1: [1, 1, 2, 2, 1]}

def compute_visualisation_mask(img):
    activations = functor([np.array([img])])
    upscaled_activation = np.ones((2,20,20))
    for layer in [5, 4, 3, 2, 1]:
        the_layers = np.mean(activations[layer], axis=4).squeeze(axis=0)
        averaged_activation = the_layers * upscaled_activation
        outputs_shape = (activations[layer - 1].shape[1], activations[layer - 1].shape[2], activations[layer - 1].shape[3])
        x = np.reshape(averaged_activation, (1, averaged_activation.shape[0],averaged_activation.shape[1],averaged_activation.shape[2],1))
        modeltwo = Sequential()
        if layer == 5:
            modeltwo.add(Conv3DTranspose(filters=1, kernel_size=(3,3,3), strides=(1,1,1),
            input_shape=(2, 20, 20, 1), data_format='channels_last',
            padding='same'))
        if layer == 4:
            modeltwo.add(Conv3DTranspose(filters=1, kernel_size=(3,3,3), strides=(1,1,1),
            input_shape=(2, 20, 20, 1), data_format='channels_last',
            padding='same'))
        if layer == 3:
            modeltwo.add(Conv3DTranspose(filters=1, kernel_size=(5,5,5), strides=(1,2,2),
            input_shape=(2, 20, 20, 1), data_format='channels_last',
            padding='same'))
        if layer == 2:
            modeltwo.add(Conv3DTranspose(filters=1, kernel_size=(5,5,5), strides=(1,2,2),
            input_shape=(2, 40, 40, 1), data_format='channels_last',
            padding='same'))
        if layer == 1:
            modeltwo.add(Conv3DTranspose(filters=1, kernel_size=(5,5,5), strides=(1,2,2),
            input_shape=(2, 80, 80, 1), data_format='channels_last',
            padding='same'))
        result = modeltwo.predict(x)
        result = result.squeeze(axis=0)
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
    anim.save('/home/jesse/Desktop/DNRacing/deep_learning/keras2tf/animation.mp4', writer='imagemagick', fps=40)

csv_path = '/media/jesse/c334c9bc-7ae4-4ea7-84fb-6b8f5595aea2/v1/finaldataset.csv'
img_path = '/media/jesse/c334c9bc-7ae4-4ea7-84fb-6b8f5595aea2/v1/images/'

df = pd.read_csv(csv_path, encoding='utf-8')
image_location = []
for i, row in tqdm(df.iterrows()):
    fname = row['image']
    image_location.append(fname[:-1])

imgs = []
alpha = 0.02
beta = 1.0 - alpha
counter = 0
img_stack = []
z = []
number = 800
numbertwo = 800
target_image = []
for a in range(100):
    display_img_stack = []
    for b in range(n_stacked):
        img = cv2.imread('/media/jesse/c334c9bc-7ae4-4ea7-84fb-6b8f5595aea2/v1/images/'+ str(image_location[number]))
        img = img[250:, :]
        img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
        img_stack.append(img.astype(np.float32))
        display_img_stack.append(img.astype(np.float32))
        number = number + 1
        counter += 1
    if counter == 1:
        cv2.imshow(str(img.shape), img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    if len(img_stack) > 1:
        img_stack = img_stack[-n_stacked:]
    z.append(np.stack(img_stack))
    img =  np.stack(img_stack)
    salient_mask = compute_visualisation_mask(img)
    for a in range(n_stacked):
        temp_img = cv2.imread('/media/jesse/c334c9bc-7ae4-4ea7-84fb-6b8f5595aea2/v1/images/'+ str(image_location[number]))
        temp_img = temp_img[250:, :]
        numbertwo = numbertwo + 1
        temp_img = cv2.resize(temp_img, (160, 160), interpolation=cv2.INTER_AREA)
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        salient_masked_one = salient_mask[0,:,:]
        salient_masked_two = salient_mask[1,:,:]
        salient_mask_stacked = np.dstack((salient_masked_one,salient_masked_two))
        salient_mask_stacked = np.dstack((salient_mask_stacked,salient_masked_two))
        blend = cv2.addWeighted(temp_img.astype('float32'), alpha, salient_mask_stacked, beta, 0.0)
        imgs.append(blend)
    print(counter/2)
    if counter >= 70800:
        break



plot_movie_mp4(imgs)
