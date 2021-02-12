from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
import time
import os
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from itertools import chain
import random
import bisect

def load_3d_motor_dataset(n_stacked, img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
    df = pd.read_csv(csv_path, encoding='utf-8')    
    x = []
    z = []
    steering = []
    throttle = []
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        thv = row['throttle']
        shaft_speed = row['shaftspeed']
        shaft_time = row['shafttime']
        batt = row['battery']                  
        fr_rpm = row['fr_rpm']
        fl_rpm = row['fl_rpm']
        rr_rpm = row['rr_rpm']
        rl_rpm = row['rl_rpm']
        roll = row['roll']
        pitch = row['pitch']
        yaw = row['yaw']
        acc_x = row['acc_x']
        acc_y = row['acc_y']
        acc_z = row['acc_z']
        ang_vel_x = row['ang_vel_x']
        ang_vel_y = row['ang_vel_y']
        ang_vel_z = row['ang_vel_z']
        img = os.path.join(img_path, fname[:-1])
        img_stack.append(img)
        if mini_counter == 1:
                    stv_prev = stv
                    thv_prev = shaft_speed
        if i+1 >= n_stacked and (i+1 - n_stacked) % n_jump == 0 and mini_counter != 1:
                    x.append(img_stack)
                    img_stack = img_stack[n_jump:]
                    steering.append(np.stack([stv]))
                    try: 
                       shaft_speed = np.float32(shaft_speed)
                    except:
                       shaft_speed = np.float32(shaft_speed[:-3])  
                    throttle.append(np.stack([shaft_speed/((30.497-1.0534)/2)]))   
                    sen_stack = [batt]
                    z.append(np.stack(sen_stack))
                    thv_prev = shaft_speed
                    stv_prev = stv
        mini_counter += 1
    x = np.stack(x)
    z = np.stack(z)
    steering = np.stack(steering) 
    throttle = np.stack(throttle)
    val_images, val_sensors, val_steering, val_throttle = test_images, test_sensors, test_steering, test_throttle = empty_images, empty_sensors, empty_steering, empty_throttle = train_images, train_sensors, train_steering, train_throttle = None, None, None, None      
    if val_size is not None:
       train_images, val_images, train_sensors, val_sensors, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                x, z, steering, throttle, test_size=test_size,
                random_state=123, shuffle=True
                )
    x = []
    z = []
    steering = []
    throttle = []	
    if test_size is not None:
       train_images, test_images, train_sensors, test_sensors, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                train_images, train_sensors, train_steering, train_throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    shuffle(train_images, train_sensors, train_steering, train_throttle)
    return train_images, train_sensors, val_images, val_sensors, test_images, test_sensors, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle


