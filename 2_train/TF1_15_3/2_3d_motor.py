#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import cv2
import pickle
import os
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.externals import joblib
from keras.backend.tensorflow_backend import set_session
from keras.backend.common import set_floatx
set_floatx('float16')
config = tf.ConfigProto(allow_soft_placement=True, device_count = {'CPU' : 1, 'GPU' : 2})
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.models import Sequential
from keras.layers import Flatten, Activation, Dense, Dropout, MaxPooling3D, Conv3D
from keras import optimizers
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import AlphaDropout
from keras import callbacks
from keras.callbacks import TensorBoard, Callback
from keras.layers.advanced_activations import ELU
from data_utils.data_processor import load_3d_motor_dataset
from model.models import build_3d_sensor
from model_test_utils.metrics import mean_absolute_relative_error
from model_test_utils.metrics import coefficient_of_determination
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from multiprocessing import Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

class EvaluateEndCallback(Callback):
    def on_evaluate_end(self, epoch, logs=None):
        print('epoch: {}, logs: {}'.format(epoch, logs))

class PlotSteeringLoss(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('angle_out_loss'))
        self.val_losses.append(logs.get('val_angle_out_loss'))
        self.i += 1
        
        plt.plot(self.x, self.losses, 'g')
        plt.plot(self.x, self.val_losses, 'r')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("3d_v2_steering_loss_history.png")
      
class PlotThrottleLoss(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('throttle_out_loss'))
        self.val_losses.append(logs.get('val_throttle_out_loss'))
        self.i += 1
        
        plt.plot(self.x, self.losses, 'g')
        plt.plot(self.x, self.val_losses, 'r')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("3d_v2_throttle_loss_history.png")

plot_steering_losses = PlotSteeringLoss()
plot_throttle_losses = PlotThrottleLoss()

class Generator(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, sensor_filenames, steering, throttle, batch_size) :
    self.image_filenames = image_filenames
    self.sensor_filenames = sensor_filenames
    self.steering = steering
    self.throttle = throttle
    self.batch_size = batch_size
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  def __getitem__(self, idx) :
    batch_image = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_sensor = self.sensor_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_steering = self.steering[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_throttle = self.throttle[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_img = []
    for a in range(len(batch_image)):
         batch_img.append((np.stack([(cv2.imread(batch_image[a][0])[38:, :]).astype(np.float16), (cv2.imread(batch_image[a][1])[38:, :]).astype(np.float16)])))
    return [np.stack(batch_img), batch_sensor], [batch_steering, batch_throttle]

def Predict_Image_Generator(image_filenames):
    img = []
    for a in range(len(image_filenames)):
         img.append(np.stack((np.stack([(cv2.imread(image_filenames[a][0])[38:, :]).astype(np.float16),(cv2.imread(image_filenames[a][1])[38:, :]).astype(np.float16)]))))
    return np.stack(img)

def main(*args, **kwargs):
    if kwargs['n_jump'] == 0:
        kwargs['n_jump'] = kwargs['n_stacked']
    
    saved_weight_name = './3d_v2_1.h5'
    saved_file_name = './3d_v2_1.hdf5'.format(
            kwargs['n_stacked'], kwargs['n_jump'], kwargs['depth'])

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(os.path.dirname(__file__))),
        'dataset'
    )
    img_path = os.path.join(kwargs['img_path'])
    out_path = os.path.join(kwargs['out_path'])
    n_stacked = kwargs['n_stacked']

    train_images, train_sensors, val_images, val_sensors, test_images, test_sensors, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle = load_3d_motor_dataset(
        n_stacked, img_path, out_path,
        h=kwargs['height'], w=kwargs['width'], d=kwargs['depth'], concatenate=kwargs['concatenate'], prediction_mode=kwargs['prediction_mode'], val_size=0.2, test_size=0.1, n_jump=kwargs['n_jump'])

    print("number of train images:", train_images.shape)
    print("number of validation images:", val_images.shape)
    print("number of test images:", test_images.shape)
    print("number of train sensors:", train_sensors.shape)
    print("number of validation sensors:", val_sensors.shape)
    print("number of test sensors:", test_sensors.shape)
    print("number of steering train output sets:", train_steering.shape)
    print("number of steering validation output sets:", val_steering.shape)
    print("number of steering test output sets:", test_steering.shape)
    print("number of throttle train output sets:", train_throttle.shape)
    print("number of throttle validation output sets:", val_throttle.shape)
    print("number of throttle test output sets:", test_throttle.shape)

    # Initialize Generators
    training_batch_generator = Generator(train_images, train_sensors, train_steering, train_throttle, kwargs['batch_size'])
    validation_batch_generator = Generator(val_images, val_sensors, val_steering, val_throttle, kwargs['batch_size'])

        
    with tf.device('/device:GPU:0'):
        tf.logging.set_verbosity(tf.logging.ERROR)
        tfboard = TensorBoard(log_dir='./logs', histogram_freq=0)
	# Call the model
        model = build_3d_sensor(
            kwargs['width'], kwargs['height'],
            kwargs['depth'], kwargs['n_stacked']
        )
        # input()
        stop_callbacks = callbacks.EarlyStopping(
                monitor='val_loss', patience=15, verbose=0, mode='min', min_delta=0
            )
        checkpoint = callbacks.ModelCheckpoint(
                saved_weight_name, monitor='val_loss',
                verbose=1, save_best_only=True, save_weights_only=True, mode='min'
            )
        
        #model.load_weights(saved_weight_name)
        
        history = model.fit_generator(workers=8,
                generator=training_batch_generator, 
                epochs= 20000, verbose=0,
                callbacks=[stop_callbacks,checkpoint, tfboard, EvaluateEndCallback(), plot_steering_losses],
                validation_data=validation_batch_generator
            )
        
        # save
        model.save('3d_v2_1.hdf5')
        
        # test
        print("Start test....")
        model.load_weights(saved_weight_name)

        # Load Val/Test Images
        val_imgs = Predict_Image_Generator(val_images) 
        test_imgs = Predict_Image_Generator(test_images)

        # Predict
        [model_steering_val, model_throttle_val] = model.predict([val_imgs, val_sensors], batch_size=None, verbose=0)
        [model_steering_test, model_throttle_test] = model.predict([test_imgs, test_sensors], batch_size=None, verbose=0)

        # Delete Val/Test Images  
        val_imgs = []
        test_imgs = []

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
    print('motorspeed R^2: ' + str(R2_val))
    R2_val = r2_score(val_throttle[:], model_throttle_val[:])
    print('Throttle R^2: ' + str(R2_val))

    # test result
    print("test result...")
    mae = sqrt(mean_absolute_error(test_steering[:], model_steering_test[:]))
    print('steering mae: ' + str(mae))
    mae = sqrt(mean_absolute_error(test_throttle[:], model_throttle_test[:]))
    print('throttle mae: ' + str(mae))
    rmse = sqrt(mean_squared_error(test_steering[:], model_steering_test[:]))
    print('steering rmse: ' + str(rmse))
    rmse = sqrt(mean_squared_error(test_throttle[:], model_throttle_test[:]))
    print('throttle rmse: ' + str(rmse))
    R2_test = r2_score(test_steering[:], model_steering_test[:])
    print('Steering R^2: ' + str(R2_test))
    R2_test = r2_score(test_throttle[:], model_throttle_test[:])
    print('Motor R^2: ' + str(R2_test))
 
    val_steering_dist      = '3d_motor_Val_steering_dist.png'
    test_steering_dist     = '3d_motor_Test_steering_dist.png'
    val_throttle_dist      = '3d_motor_Val_throttle_dist.png'
    test_throttle_dist     = '3d_motor_Test_throttle_dist.png'

    # Plot Validation Throttle Distribution
    plt.hist([val_throttle.flatten(), model_throttle_val.flatten()], color=['b', 'r'], bins=100, label=['val_target', 'val_predict'])
    plt.legend()
    plt.savefig(val_throttle_dist)
    plt.close()

    # Plot Test throttle Distribution 
    plt.hist([test_throttle.flatten(), model_throttle_test.flatten()], color=['b', 'r'], bins=100, label=['test_target', 'test_predict'])
    plt.legend()
    plt.savefig(test_throttle_dist)
    plt.close()

    # Plot Validation Steering Distribution
    plt.hist([val_steering.flatten(), model_steering_val.flatten()], color=['b', 'r'], bins=100, label=['val_target', 'val_predict'])
    plt.legend()
    plt.savefig(val_steering_dist)
    plt.close()

    # Plot Test Steering Distribution 
    plt.hist([test_steering.flatten(), model_steering_test.flatten()], color=['b', 'r'], bins=100, label=['test_target', 'test_predict'])
    plt.legend()
    plt.savefig(test_steering_dist)
    plt.close()

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
        type=int, default=54
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
        "--concatenate", help="target csv filename",
        type=int, default=2
    )
    parser.add_argument(
        "--prediction_mode", help="sets steering predictions from categorical cross entropy or from linear regression",
        type=str, default='linear'
    )

    args = parser.parse_args()
    main(**vars(args))

