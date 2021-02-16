#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import argparse
import pickle
import keras
import time
import cv2
import os
from model_test_utils.metrics import mean_absolute_relative_error
from model_test_utils.metrics import coefficient_of_determination
from model_utils.functions import BatchNormalizationF16
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.externals import joblib
from data_utils.data_processor import load_3d_motor_dataset
from keras.models import Model
from keras.layers import Input, Flatten, Activation, Dense, Dropout, MaxPooling3D, Conv3D, Convolution2D, Lambda, Concatenate, concatenate
from keras import optimizers, regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import AlphaDropout
from keras.losses import mse
from keras import callbacks
from keras.callbacks import TensorBoard, Callback, History
from keras.layers.advanced_activations import ELU
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.backend.common import set_floatx
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

dtype = 'float32'
K.set_floatx(dtype)

config = tf.ConfigProto()
config = tf.ConfigProto(
    allow_soft_placement=True,
    device_count={
        'CPU': 1,
        'GPU': 2})
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def multi_gpu_model(model, gpus):
    if isinstance(gpus, (list, tuple)):
        num_gpus = len(gpus)
        target_gpu_ids = gpus
    else:
        num_gpus = gpus
        target_gpu_ids = range(num_gpus)

    def get_slice(data, i, parts):
        shape = tf.shape(data)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // parts
        if i == num_gpus - 1:
            size = batch_size - step * i
        else:
            size = step
        size = tf.concat([size, input_shape], axis=0)
        stride = tf.concat([step, input_shape * 0], axis=0)
        start = stride * i
        return tf.slice(data, start, size)

    all_outputs = []
    for i in range(len(model.outputs)):
        all_outputs.append([])

    # Place a copy of the model on each GPU,
    # each getting a slice of the inputs.
    for i, gpu_id in enumerate(target_gpu_ids):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('replica_%d' % gpu_id):
                inputs = []
                # Retrieve a slice of the input.
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_i = Lambda(get_slice,
                                     output_shape=input_shape,
                                     arguments={'i': i,
                                                'parts': num_gpus})(x)
                    inputs.append(slice_i)

                # Apply model on slice
                # (creating a model replica on the target device).
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save the outputs for merging back together later.
                for o in range(len(outputs)):
                    all_outputs[o].append(outputs[o])

    # Merge outputs on CPU.
    with tf.device('/cpu:0'):
        merged = []
        for name, outputs in zip(model.output_names, all_outputs):
            merged.append(concatenate(outputs,
                                      axis=0, name=name))
        return Model(model.inputs, merged)


class PlotSteeringLoss(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.kl_losses = []
        self.mse_losses = []
        self.val_kl_losses = []
        self.val_mse_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.kl_losses.append(logs.get('steering_out_kl_loss'))
        self.mse_losses.append(logs.get('steering_out_reconstruction_loss'))
        self.val_kl_losses.append(logs.get('val_steering_out_kl_loss'))
        self.val_mse_losses.append(
            logs.get('val_steering_out_reconstruction_loss'))
        self.i += 1
        plt.plot(self.x, self.kl_losses, 'g')
        plt.plot(self.x, self.mse_losses, 'r')
        plt.plot(self.x, self.val_kl_losses, 'b')
        plt.plot(self.x, self.val_mse_losses, 'y')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['kl', 'mse', 'val_kl', 'val_mse'], loc='upper left')
        plt.savefig("3dcvae_v3_loss.png")


class PlotThrottleLoss(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.kl_losses = []
        self.mse_losses = []
        self.val_kl_losses = []
        self.val_mse_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.kl_losses.append(logs.get('throttle_out_kl_loss'))
        self.mse_losses.append(logs.get('throttle_out_reconstruction_loss'))
        self.val_kl_losses.append(logs.get('val_throttle_out_kl_loss'))
        self.val_mse_losses.append(
            logs.get('val_throttle_out_reconstruction_loss'))
        self.i += 1
        plt.plot(self.x, self.kl_losses, 'g')
        plt.plot(self.x, self.mse_losses, 'r')
        plt.plot(self.x, self.val_kl_losses, 'b')
        plt.plot(self.x, self.val_mse_losses, 'y')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['kl', 'mse', 'val_kl', 'val_mse'], loc='upper left')
        plt.savefig("3dcvae_v3_loss.png")

# Dataset Generator


class Generator(keras.utils.Sequence):
    def __init__(self, image_filenames, sensor_filenames,
                 steering, throttle, batch_size):
        self.image_filenames = image_filenames
        self.sensor_filenames = sensor_filenames
        self.steering = steering
        self.throttle = throttle
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) /
                        float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_image = self.image_filenames[idx *
                                           self.batch_size: (idx + 1) * self.batch_size]
        batch_sensor = self.sensor_filenames[idx *
                                             self.batch_size: (idx + 1) * self.batch_size]
        batch_steering = self.steering[idx *
                                       self.batch_size: (idx + 1) * self.batch_size]
        batch_throttle = self.throttle[idx *
                                       self.batch_size: (idx + 1) * self.batch_size]
        batch_img = []
        for a in range(len(batch_image)):
            batch_img.append((np.stack([(cv2.imread(batch_image[a][0])[28:, :]).astype(
                np.float32), (cv2.imread(batch_image[a][1])[28:, :]).astype(np.float32)])))
        return [np.stack(batch_img), batch_sensor], [
            batch_steering, batch_throttle]


def Predict_Image_Generator(image_filenames):
    img = []
    for a in range(len(image_filenames)):
        img.append(np.stack((np.stack([(cv2.imread(image_filenames[a][0])[28:, :]).astype(
            np.float32), (cv2.imread(image_filenames[a][1])[28:, :]).astype(np.float32)]))))
    return np.stack(img)


# Callback for Beta Weight
counter = 0


class changeBeta(Callback):
    global counter

    def __init__(self, beta):
        self.beta = beta

    def on_epoch_end(self, epoch, logs={}):
        global counter
        K.set_value(self.beta, K.get_value(K.constant((counter % 30) / 30.0)))
        counter = counter + 1
        print("Setting beta weight to =", str(K.eval(beta)))


beta = K.variable(1.)
instance_beta = changeBeta(beta)

# Define loss


def kl_final_steering_loss(sigma, mu):
    def kl_reconstruction_loss(y_true, y_pred):
        # Reconstruction loss
        reconstruction_loss = mse(y_true, y_pred)
        # KL divergence loss
        kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Total loss
        return K.mean(reconstruction_loss + (instance_beta.beta) * kl_loss)
    return kl_reconstruction_loss


def kl_final_throttle_loss(sigma, mu):
    def kl_reconstruction_loss(y_true, y_pred):
        # Reconstruction loss
        reconstruction_loss = mse(y_true, y_pred)
        # KL divergence loss
        kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Total loss
        return K.mean(reconstruction_loss + (instance_beta.beta) * kl_loss)
    return kl_reconstruction_loss


def kl_loss(sigma, mu):
    def kl_loss(y_true, y_pred):
        # KL divergence loss
        k_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
        k_loss = K.sum(k_loss, axis=-1)
        k_loss *= -0.5
        # Total loss
        return K.mean(k_loss)
    return kl_loss


def reconstruction_loss(y_true, y_pred):
    reconstruction_loss = mse(y_true, y_pred)
    return reconstruction_loss

# Print Out Callbacks


class EvaluateEndCallbackSteering(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print(
            'KL_loss: {}, Reconstruction_loss: {}'.format(
                logs.get('steering_out_kl_loss'),
                logs.get('steering_out_reconstruction_loss')))
        print(
            'Val_KL_loss: {}, Reconstruction_loss: {}'.format(
                logs.get('val_steering_out_kl_loss'),
                logs.get('val_steering_out_reconstruction_loss')))


class EvaluateEndCallbackThrottle(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print(
            'KL_loss: {}, Reconstruction_loss: {}'.format(
                logs.get('throttle_out_kl_loss'),
                logs.get('throttle_out_reconstruction_loss')))
        print(
            'Val_KL_loss: {}, Reconstruction_loss: {}'.format(
                logs.get('val_throttle_out_kl_loss'),
                logs.get('val_throttle_out_reconstruction_loss')))

# Define sampling with reparameterization trick


def sample_z(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim))
    return mu + K.exp(sigma / 2) * eps

# Define Steering Encoder


def build_lrcn_steering_cvae_encoder(w, h, d, s, latent_dim, sensor_dim):
    image_label = Input(shape=(s, h, w, d), name='image_label')
    sensor_state = Input(shape=(sensor_dim,), name='sensor_state')
    x = TimeDistributed(
        Lambda(
            lambda x: x / 127.5 - 1.0,
            name='lambda_0'),
        name="time_distributed_1")(image_label)
    cx = Conv3D(
        filters=32, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv1')(x)
    cx = BatchNormalization(name='bn_1')(cx)
    cx = Conv3D(
        filters=32, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv2')(cx)
    cx = BatchNormalization(name='bn_2')(cx)
    cx = Conv3D(
        filters=64, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv3')(cx)
    cx = BatchNormalization(name='bn_3')(cx)
    cx = Conv3D(
        filters=64, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv4')(cx)
    cx = BatchNormalization(name='bn_4')(cx)
    cx = Conv3D(
        filters=64, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv5')(cx)
    cx = BatchNormalization(name='bn_5')(cx)
    # Flatten and split into z_steering, z_throttle
    image_state = TimeDistributed(Flatten(), name="time_distributed_7")(cx)
    image_state = LSTM(
        512,
        return_sequences=False,
        stateful=False,
        name='lstm_1')(image_state)
    x1 = image_state
    a1 = Dense(
        256,
        activation='elu',
        kernel_regularizer=regularizers.l1_l2(
            l1=0.001,
            l2=0.0),
        name='dense_1')(x1)
    a2 = BatchNormalization(name='bn_6')(a1)
    mu_steering = Dense(latent_dim, name='latent_mu_steering')(a2)
    sigma_steering = Dense(latent_dim, name='latent_sigma_steering')(a2)
    x2 = x1  # Concatenate()([x1, sensor_state])
    b1 = Dense(
        256,
        activation='elu',
        kernel_regularizer=regularizers.l1_l2(
            l1=0.001,
            l2=0.0),
        name='dense_2')(x2)
    b2 = BatchNormalization(name='bn_7')(b1)
    mu_throttle = Dense(latent_dim, name='latent_mu_throttle')(b2)
    sigma_throttle = Dense(latent_dim, name='latent_sigma_throttle')(b2)
    # Use reparameterization trick
    z_steering = Lambda(sample_z, output_shape=(latent_dim, ),
                        name='lambda_1')([mu_steering, sigma_steering])
    z_throttle = Lambda(sample_z, output_shape=(latent_dim, ),
                        name='lambda_2')([mu_throttle, sigma_throttle])
    model = Model(
        inputs=[
            image_label,
            sensor_state],
        outputs=[
            image_state,
            mu_steering,
            sigma_steering,
            z_steering,
            mu_throttle,
            sigma_throttle,
            z_throttle],
        name='model_1')
    return model

# Define Deploy


def build_lrcn_cvae_deploy(w, h, d, s, latent_dim, sensor_dim):
    image_label = Input(shape=(s, h, w, d), name='image_label')
    sensor_state = Input(shape=(sensor_dim,), name='sensor_state')
    z_steering_inputs = Input(shape=(latent_dim, ), name='z_steering_inputs')
    z_throttle_inputs = Input(shape=(latent_dim, ), name='z_throttle_inputs')
    x = TimeDistributed(Lambda(lambda x: x / 127.5 - 1.0),
                        name="time_distributed_1")(image_label)
    cx = Conv3D(
        filters=32, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv1')(x)
    cx = BatchNormalization(name='bn_1')(cx)
    cx = Conv3D(
        filters=32, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv2')(cx)
    cx = BatchNormalization(name='bn_2')(cx)
    cx = Conv3D(
        filters=64, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv3')(cx)
    cx = BatchNormalization(name='bn_3')(cx)
    cx = Conv3D(
        filters=64, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv4')(cx)
    cx = BatchNormalization(name='bn_4')(cx)
    cx = Conv3D(
        filters=64, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv5')(cx)
    cx = BatchNormalization(name='bn_5')(cx)
    # Flatten and split into z_steering, z_throttle
    image_state = TimeDistributed(
        Flatten(
            name='flatten'),
        name="time_distributed_7")(cx)
    image_state = LSTM(
        512,
        return_sequences=False,
        stateful=False,
        name='lstm_1')(image_state)
    a = Concatenate()([z_steering_inputs, image_state])
    a1 = Dense(
        256,
        activation='elu',
        kernel_regularizer=regularizers.l1_l2(
            l1=0.001,
            l2=0.0),
        name='dense_steering')(a)
    a2 = BatchNormalization(name='bn_6')(a1)
    steering = Dense(1, activation='linear', name='steering_out')(a2)
    b = Concatenate()([z_throttle_inputs, image_state])
    b1 = Dense(
        256,
        activation='elu',
        kernel_regularizer=regularizers.l1_l2(
            l1=0.001,
            l2=0.0000),
        name='dense_throttle')(b)
    b2 = BatchNormalization(name='bn_7')(b1)
    throttle = Dense(1, activation='linear', name='throttle_out')(b2)
    model = Model(
        inputs=[
            image_label,
            sensor_state,
            z_steering_inputs,
            z_throttle_inputs],
        outputs=[
            steering,
            throttle])
    return model


def build_lrcn_cvae(w, h, d, s, latent_dim, sensor_dim):
    image_inputs = Input(shape=(s, h, w, d), name='image_inputs')
    sensor_state = Input(shape=(sensor_dim,), name='sensor_state')
    x = TimeDistributed(
        Lambda(
            lambda x: x /
            127.5 -
            1.0,
            name='lambda_0'))(image_inputs)
    cx = Conv3D(
        filters=32, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv1')(x)
    cx = BatchNormalization(name='bn_1')(cx)
    cx = Conv3D(
        filters=32, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv2')(cx)
    cx = BatchNormalization(name='bn_2')(cx)
    cx = Conv3D(
        filters=64, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv3')(cx)
    cx = BatchNormalization(name='bn_3')(cx)
    cx = Conv3D(
        filters=64, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv4')(cx)
    cx = BatchNormalization(name='bn_4')(cx)
    cx = Conv3D(
        filters=64, kernel_size=(3, 3, 3), strides=(1, 2, 2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name='conv5')(cx)
    cx = BatchNormalization(name='bn_5')(cx)
    # Flatten and split into z_steering, z_throttle
    image_state = TimeDistributed(Flatten())(cx)
    image_state = LSTM(512, return_sequences=False,
                       stateful=False)(image_state)
    x1 = image_state
    a = Dense(
        256,
        activation='elu',
        kernel_regularizer=regularizers.l1_l2(
            l1=0.001,
            l2=0.0000),
        name='dense_1')(x1)
    a = BatchNormalization(name='bn_6')(a)
    mu_steering = Dense(latent_dim, name='latent_mu_steering')(a)
    sigma_steering = Dense(latent_dim, name='latent_sigma_steering')(a)
    x2 = x1  # Concatenate()([x1, sensor_state])
    b1 = Dense(
        256,
        activation='elu',
        kernel_regularizer=regularizers.l1_l2(
            l1=0.001,
            l2=0.0000),
        name='dense_2')(x2)
    b2 = BatchNormalization(name='bn_7')(b1)
    mu_throttle = Dense(latent_dim, name='latent_mu_throttle')(b2)
    sigma_throttle = Dense(latent_dim, name='latent_sigma_throttle')(b2)
    # Use reparameterization trick
    z_s = Lambda(sample_z, output_shape=(latent_dim, ),
                 name='lambda_1')([mu_steering, sigma_steering])
    z_t = Lambda(sample_z, output_shape=(latent_dim, ),
                 name='lambda_2')([mu_throttle, sigma_throttle])

    z_steering = Concatenate()([z_s, image_state])
    z_throttle = Concatenate()([z_t, image_state])

    a1 = Dense(
        256,
        activation='elu',
        kernel_regularizer=regularizers.l1_l2(
            l1=0.001,
            l2=0.0000),
        name='dense_steering')(z_steering)
    a2 = BatchNormalization(name='bn_8')(a1)
    steering = Dense(1, activation='linear', name='steering_out')(a2)

    b1 = Dense(
        256,
        activation='elu',
        kernel_regularizer=regularizers.l1_l2(
            l1=0.001,
            l2=0.0000),
        name='dense_throttle')(z_throttle)
    b2 = BatchNormalization(name='bn_9')(b1)
    throttle = Dense(1, activation='linear', name='throttle_out')(b2)
    model = Model(
        inputs=[
            image_inputs, sensor_state], outputs=[
            steering, throttle])
    #vae                = multi_gpu_model(model, gpus=2)
    # vae.layers[-3].set_weights(model.get_weights())
    optimizer = optimizers.adam(lr=0.00001)
    model.compile(loss={'steering_out': kl_final_steering_loss(sigma_steering, mu_steering),
                        'throttle_out': kl_final_throttle_loss(sigma_throttle, mu_throttle)},
                  optimizer=optimizer, loss_weights=[1, 1],
                  metrics=[kl_final_steering_loss(sigma_steering, mu_steering), kl_final_throttle_loss(sigma_throttle, mu_throttle), kl_loss(sigma_steering, mu_steering), kl_loss(sigma_throttle, mu_throttle), reconstruction_loss])
    return model


def main(*args, **kwargs):
    global counter
    # Data & model configuration
    plot_steering_losses = PlotSteeringLoss()
    plot_throttle_losses = PlotThrottleLoss()
    img_width, img_height = kwargs['width'], kwargs['height']
    batch_size = kwargs['batch_size']
    if kwargs['n_jump'] == 0:
        kwargs['n_jump'] = kwargs['n_stacked']
    saved_weights_name = './3dcvae_v3.h5'
    saved_file_name = './3dcvae_v3.hdf5'
    saved_file_name_two = './3dcvae_v3_temp.hdf5'
    data_path = os.path.join(
        os.path.dirname(
            os.path.abspath(
                os.path.dirname(__file__))),
        'dataset')
    img_path = os.path.join(kwargs['img_path'])
    out_path = os.path.join(kwargs['out_path'])
    n_stacked = kwargs['n_stacked']
    train_images, train_sensors, val_images, val_sensors, test_images, test_sensors, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle = load_3d_motor_dataset(
        n_stacked, img_path, out_path, h=kwargs['height'], w=kwargs['width'], d=kwargs['depth'], concatenate=kwargs['concatenate'], prediction_mode=kwargs['prediction_mode'], val_size=0.2, test_size=0.1, n_jump=kwargs['n_jump'])
    print("number of steering train output sets:", train_steering.shape)
    print("number of steering validation output sets:", val_steering.shape)
    print("number of steering test output sets:", test_steering.shape)
    print("number of throttle train output sets:", train_throttle.shape)
    print("number of throttle validation output sets:", val_throttle.shape)
    print("number of throttle test output sets:", test_throttle.shape)

    # Initialize Generators
    training_batch_generator = Generator(
        train_images,
        train_sensors,
        train_steering,
        train_throttle,
        kwargs['batch_size'])
    validation_batch_generator = Generator(
        val_images,
        val_sensors,
        val_steering,
        val_throttle,
        kwargs['batch_size'])

    tfboard = TensorBoard(log_dir='./logs', histogram_freq=0)
    stop_callbacks = callbacks.EarlyStopping(
        monitor='val_throttle_out_kl_reconstruction_loss', patience=15,
        verbose=0, mode='min', min_delta=0
    )

    calls = [
        stop_callbacks,
        EvaluateEndCallbackSteering(),
        EvaluateEndCallbackThrottle(),
        changeBeta(beta),
        plot_steering_losses,
        tfboard]
    calls.append(callbacks.ModelCheckpoint(saved_weights_name, monitor='val_throttle_out_kl_reconstruction_loss',
                                           verbose=1, save_best_only=True, save_weights_only=True, mode='min'))

    ####Settings####
    latent_dim = 64
    sensor_dim = 1
    test_set_length = len(test_steering)
    val_set_length = len(val_steering)
    # Call the steering model
    model = build_lrcn_cvae(
        kwargs['width'], kwargs['height'],
        kwargs['depth'], kwargs['n_stacked'],
        latent_dim, sensor_dim
    )

    with tf.device('/device:GPU:1'):
        # Load Weights
        #model.load_weights('3dcvae_v3.h5')
        '''
        history = model.fit_generator(workers=8,
                                      generator=training_batch_generator,
                                      epochs=50000,
                                      verbose=0,
                                      callbacks=calls,
                                      validation_data=validation_batch_generator
                                      )
        '''
        # Load Test Data
        test_imgs = Predict_Image_Generator(test_images)
        val_imgs = Predict_Image_Generator(val_images)

        # Load the Encoder for visualization
        model_encoder = build_lrcn_steering_cvae_encoder(
            kwargs['width'], kwargs['height'],
            kwargs['depth'], kwargs['n_stacked'],
            latent_dim, sensor_dim
        )

        # Check the trainable status of the individual layers
        for layer in model.layers:
            print(layer.name, layer.trainable)

        model.load_weights(saved_weights_name)

        # Load the deploy and assign weights
        model_deploy = build_lrcn_cvae_deploy(
            kwargs['width'], kwargs['height'],
            kwargs['depth'], kwargs['n_stacked'],
            latent_dim, sensor_dim
        )

        model_deploy.get_layer('time_distributed_1').set_weights(
            model.get_layer('time_distributed_1').get_weights())
        model_deploy.get_layer('conv1').set_weights(
            model.get_layer('conv1').get_weights())
        model_deploy.get_layer('conv2').set_weights(
            model.get_layer('conv2').get_weights())
        model_deploy.get_layer('conv3').set_weights(
            model.get_layer('conv3').get_weights())
        model_deploy.get_layer('conv4').set_weights(
            model.get_layer('conv4').get_weights())
        model_deploy.get_layer('conv5').set_weights(
            model.get_layer('conv5').get_weights())
        model_deploy.get_layer('time_distributed_7').set_weights(
            model.get_layer('time_distributed_2').get_weights())
        model_deploy.get_layer('lstm_1').set_weights(
            model.get_layer('lstm_1').get_weights())
        model_deploy.get_layer('bn_1').set_weights(
            model.get_layer('bn_1').get_weights())
        model_deploy.get_layer('bn_2').set_weights(
            model.get_layer('bn_2').get_weights())
        model_deploy.get_layer('bn_3').set_weights(
            model.get_layer('bn_3').get_weights())
        model_deploy.get_layer('bn_4').set_weights(
            model.get_layer('bn_4').get_weights())
        model_deploy.get_layer('bn_5').set_weights(
            model.get_layer('bn_5').get_weights())
        model_deploy.get_layer('dense_steering').set_weights(
            model.get_layer('dense_steering').get_weights())
        model_deploy.get_layer('dense_throttle').set_weights(
            model.get_layer('dense_throttle').get_weights())
        # Note that bn_6 corresponds to bn_8!
        model_deploy.get_layer('bn_6').set_weights(
            model.get_layer('bn_8').get_weights())
        model_deploy.get_layer('bn_7').set_weights(
            model.get_layer('bn_9').get_weights())
        model_deploy.get_layer('steering_out').set_weights(
            model.get_layer('steering_out').get_weights())
        model_deploy.get_layer('throttle_out').set_weights(
            model.get_layer('throttle_out').get_weights())

        z_throttle = z_steering = np.zeros((test_set_length, latent_dim))
        [model_steering_test, model_throttle_test] = model_deploy.predict(
            [test_imgs, test_sensors, z_steering, z_throttle], batch_size=None, verbose=0)
        z_throttle = z_steering = np.zeros((val_set_length, latent_dim))
        [model_steering_val, model_throttle_val] = model_deploy.predict(
            [val_imgs, val_sensors, z_steering, z_throttle], batch_size=None, verbose=0)
        
        # val result
        print("val result...")
        mae = sqrt(mean_absolute_error(val_steering[:], model_steering_val[:]))
        print('steering mae: ' + str(mae))
        mae = sqrt(mean_absolute_error(val_throttle[:], model_throttle_val[:]))
        print('throttle mae: ' + str(mae))
        rmse = sqrt(mean_squared_error(val_steering[:], model_steering_val[:]))
        print('steering rmse: ' + str(rmse))
        rmse = sqrt(mean_squared_error(val_throttle[:], model_throttle_val[:]))
        print('throttle rmse: ' + str(rmse))
        R2_val = r2_score(val_steering[:], model_steering_val[:])
        print('Steering R^2: ' + str(R2_val))
        R2_val = r2_score(val_throttle[:], model_throttle_val[:])
        print('Motor R^2: ' + str(R2_val))
        
        # test result
        print("test result...")
        mae = sqrt(
            mean_absolute_error(
                test_steering[:],
                model_steering_test[:]))
        print('steering mae: ' + str(mae))
        mae = sqrt(
            mean_absolute_error(
                test_throttle[:],
                model_throttle_test[:]))
        print('throttle mae: ' + str(mae))
        rmse = sqrt(
            mean_squared_error(
                test_steering[:],
                model_steering_test[:]))
        print('steering rmse: ' + str(rmse))
        rmse = sqrt(
            mean_squared_error(
                test_throttle[:],
                model_throttle_test[:]))
        print('throttle rmse: ' + str(rmse))
        R2_test = r2_score(test_steering[:], model_steering_test[:])
        print('Steering R^2: ' + str(R2_test))
        R2_test = r2_score(test_throttle[:], model_throttle_test[:])
        print('Motor R^2: ' + str(R2_test))
        
        val_steering_dist = '3dcvae_motor_Val_steering_dist.png'
        test_steering_dist = '3dcvae_motor_Test_steering_dist.png'
        val_throttle_dist = '3dcvae_motor_Val_throttle_dist.png'
        test_throttle_dist = '3dcvae_motor_Test_throttle_dist.png'
          
        # Plot Validation Throttle Distribution
        plt.hist([val_throttle.flatten(), model_throttle_val.flatten()], color=[
                 'b', 'r'], bins=100, label=['val_target', 'val_predict'])
        plt.legend()
        plt.savefig(val_throttle_dist)
        plt.close()

        # Plot Test throttle Distribution
        plt.hist([test_throttle.flatten(), model_throttle_test.flatten()], color=[
                 'b', 'r'], bins=100, label=['test_target', 'test_predict'])
        plt.legend()
        plt.savefig(test_throttle_dist)
        plt.close()

        # Plot Validation Steering Distribution
        plt.hist([val_steering.flatten(), model_steering_val.flatten()], color=[
                 'b', 'r'], bins=100, label=['val_target', 'val_predict'])
        plt.legend()
        plt.savefig(val_steering_dist)
        plt.close()

        # Plot Test Steering Distribution
        plt.hist([test_steering.flatten(), model_steering_test.flatten()], color=[
                 'b', 'r'], bins=100, label=['test_target', 'test_predict'])
        plt.legend()
        plt.savefig(test_steering_dist)
        plt.close()
        # Load Test Data
        test_imgs = Predict_Image_Generator(test_images)
        val_imgs = Predict_Image_Generator(val_images)

        # Load the Encoder for visualization
        model_encoder = build_lrcn_steering_cvae_encoder(
            kwargs['width'], kwargs['height'],
            kwargs['depth'], kwargs['n_stacked'],
            latent_dim, sensor_dim
        )
        print("Model encoder arch")
        for layer in model_encoder.layers:
            print(layer.name, layer.trainable)
        model_encoder.get_layer('time_distributed_1').set_weights(
            model.get_layer('time_distributed_1').get_weights())
        model_encoder.get_layer('conv1').set_weights(
            model.get_layer('conv1').get_weights())
        model_encoder.get_layer('conv2').set_weights(
            model.get_layer('conv2').get_weights())
        model_encoder.get_layer('conv3').set_weights(
            model.get_layer('conv3').get_weights())
        model_encoder.get_layer('conv4').set_weights(
            model.get_layer('conv4').get_weights())
        model_encoder.get_layer('conv5').set_weights(
            model.get_layer('conv5').get_weights())
        model_encoder.get_layer('time_distributed_7').set_weights(
            model.get_layer('time_distributed_2').get_weights())
        model_encoder.get_layer('lstm_1').set_weights(
            model.get_layer('lstm_1').get_weights())
        model_encoder.get_layer('bn_1').set_weights(
            model.get_layer('bn_1').get_weights())
        model_encoder.get_layer('bn_2').set_weights(
            model.get_layer('bn_2').get_weights())
        model_encoder.get_layer('bn_3').set_weights(
            model.get_layer('bn_3').get_weights())
        model_encoder.get_layer('bn_4').set_weights(
            model.get_layer('bn_4').get_weights())
        model_encoder.get_layer('bn_5').set_weights(
            model.get_layer('bn_5').get_weights())
        model_encoder.get_layer('bn_6').set_weights(
            model.get_layer('bn_6').get_weights())
        model_encoder.get_layer('bn_7').set_weights(
            model.get_layer('bn_7').get_weights())
        model_encoder.get_layer('dense_1').set_weights(
            model.get_layer('dense_1').get_weights())
        model_encoder.get_layer('dense_2').set_weights(
            model.get_layer('dense_2').get_weights())
        model_encoder.get_layer('latent_mu_steering').set_weights(
            model.get_layer('latent_mu_steering').get_weights())
        model_encoder.get_layer('latent_sigma_steering').set_weights(
            model.get_layer('latent_sigma_steering').get_weights())
        model_encoder.get_layer('latent_mu_throttle').set_weights(
            model.get_layer('latent_mu_throttle').get_weights())
        model_encoder.get_layer('latent_sigma_throttle').set_weights(
            model.get_layer('latent_sigma_throttle').get_weights())
        model_encoder.get_layer('lambda_1').set_weights(
            model.get_layer('lambda_1').get_weights())
        model_encoder.get_layer('lambda_2').set_weights(
            model.get_layer('lambda_2').get_weights())

        model_deploy.save('3dcvae_v3.hdf5')

        z_throttle = z_steering = np.zeros((test_set_length, latent_dim))
        [model_steering, model_throttle] = model_deploy.predict(
            [test_imgs, test_sensors, z_steering, z_throttle], batch_size=None, verbose=0)
        z_throttle = z_steering = np.zeros((val_set_length, latent_dim))
        [model_steering_val, model_throttle_val] = model_deploy.predict(
            [val_imgs, val_sensors, z_steering, z_throttle], batch_size=None, verbose=0)

        # Plot Validation Steering Distribution
        plt.hist([val_steering.flatten(), model_steering_val.flatten()], color=[
                 'b', 'r'], bins=100, label=['val_target', 'val_predict'])
        plt.legend()
        plt.show()
        plt.close()

        # Plot Test Steering Distribution
        plt.hist([test_steering.flatten(), model_steering.flatten()], color=[
                 'b', 'r'], bins=100, label=['test_target', 'test_predict'])
        plt.legend()
        plt.show()
        plt.close()
        
    # Results visualization
    # Credits for original visualization code:
    # https://keras.io/examples/variational_autoencoder_deconv/
    def viz_latent_space(encoder, data):
        input_data, target_data = data
        image_state, mu_steering, _, _, mu_throttle, _, _ = encoder.predict(
            input_data)
        start_steering = target_data[0][0][0]
        end_steering = target_data[0][0][0]
        start_throttle = target_data[1][0][0]
        end_throttle = target_data[1][0][0]
        steering = []
        throttle = []
        for a in range(len(target_data[0])):
            steering.append(target_data[0][a][0])
            if steering[a] > end_steering:
                end_steering = steering[a]
            if steering[a] < start_steering:
                start_steering = steering[a]
            throttle.append(target_data[1][a][0])
            if throttle[a] > end_throttle:
                end_throttle = throttle[a]
            if throttle[a] < start_throttle:
                start_throttle = throttle[a]
        cmap = plt.cm.rainbow
        norm_steering = plt.Normalize(vmin=start_steering, vmax=end_steering)
        norm_throttle = plt.Normalize(vmin=start_throttle, vmax=end_throttle)
        plt.subplot(211)
        plt.xlim(min(mu_steering[:, 0]), max(mu_steering[:, 0]))
        plt.ylim(min(mu_steering[:, 1]), max(mu_steering[:, 1]))
        plt.scatter(mu_steering[:, 0], mu_steering[:, 1],
                    alpha=0.5, marker='.', c=cmap(norm_steering(steering)))
        plt.xlabel('z dim 1')
        plt.ylabel('z dim 2')
        norm = matplotlib.colors.Normalize(
            vmin=start_steering, vmax=end_steering)
        plt.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=norm,
                cmap=cmap),
            label='Steering')
        plt.subplot(212)
        plt.xlim(min(mu_throttle[:, 0]), max(mu_throttle[:, 0]))
        plt.ylim(min(mu_throttle[:, 1]), max(mu_throttle[:, 1]))
        plt.scatter(mu_throttle[:, 0], mu_throttle[:, 1],
                    alpha=0.5, marker='.', c=cmap(norm_throttle(throttle)))
        plt.xlabel('z dim 1')
        plt.ylabel('z dim 2')
        norm = matplotlib.colors.Normalize(
            vmin=start_throttle, vmax=end_throttle)
        plt.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=norm,
                cmap=cmap),
            label='Throttle')
        plt.show()

    # Plot results
    data = ([test_imgs, test_sensors], [test_steering, test_throttle])
    viz_latent_space(model_encoder, data)

    # Delete Val/Test Images
    test_imgs = []
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_stacked", help="# of stacked frame for time axis",
        type=int, default=2
    )
    parser.add_argument(
        "--n_jump", help="time interval to get input, 0 for n_jump=n_stacked",
        type=int, default=1
    )
    parser.add_argument(
        "--width", help="width of input images",
        type=int, default=164
    )
    parser.add_argument(
        "--height", help="height of input images",
        type=int, default=64
    )
    parser.add_argument(
        "--depth", help="the number of channels of input images",
        type=int, default=3
    )
    parser.add_argument(
        "--img_path", help="image directory",
        type=str, default=''
    )
    parser.add_argument(
        "--out_path", help="target csv filename",
        type=str, default='/mnt/c334c9bc-7ae4-4ea7-84fb-6b8f5595aea2/homecircuit/final/final.csv'
    )
    parser.add_argument(
        "--epochs", help="total number of training epochs",
        type=int, default=50000
    )
    parser.add_argument(
        "--batch_size", help="batch_size",
        type=int, default=32
    )
    parser.add_argument(
        "--concatenate", help="length of image",
        type=int, default=1
    )
    parser.add_argument(
        "--prediction_mode", help="prediction mode",
        type=str, default='linear'
    )
    args = parser.parse_args()
    main(**vars(args))
