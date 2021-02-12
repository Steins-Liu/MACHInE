#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import RPi.GPIO as GPIO
import serial
import string
import io
import pigpio
import argparse
import numpy as np
import math
import time
import sys
import os
from multiprocessing import Process
from threading import Thread
from tensorflow.python.platform import gfile
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
from model_utils import read_RPM
import tensorflow as tf

pi = pigpio.pi()

#-------------------------------------------------------------------------------
# A modified implementation of a PID controller obtained from example source code for the book "Real-World Instrumentation with Python"
# by J. M. Hughes, published by O'Reilly Media, December 2010,
# ISBN 978-0-596-80956-0.
#-------------------------------------------------------------------------------

class PID:
    # Simple PID control.
    def __init__(self):
        # initialze gains
        self.Kp = 20
        self.Kd = 0.5
        self.Ki = 50
        self.Initialize()

    def SetKp(self, invar):
        # Set proportional gain.
        self.Kp = invar

    def SetKi(self, invar):
        # Set integral gain.
        self.Ki = invar

    def SetKd(self, invar):
        # Set derivative gain.
        self.Kd = invar

    def SetPrevErr(self, preverr):
        # Set previous error value.
        self.prev_err = preverr

    def Initialize(self):
        # initialize delta t variables
        self.currtm = time.time()
        self.prevtm = self.currtm
        self.prev_err = 0
        # term result variables
        self.Cp = 0
        self.Ci = 0
        self.Cd = 0

    def Output(self, error):
        # Performs a PID computation and returns a control value based on
        #    the elapsed time (dt) and the error signal from a summing junction
        #    (the error parameter).
        self.currtm = time.time()               # get t
        dt = self.currtm - self.prevtm          # get delta t
        de = error - self.prev_err              # get delta error
        self.Cp = self.Kp * error               # proportional term
        self.Ci += error * dt                   # integral term
        self.Cd = 0
        if dt > 0:                              # no div by zero
            self.Cd = de/dt                     # derivative term
        self.prevtm = self.currtm               # save t for next pass
        self.prev_err = error                   # save t-1 error
        result = self.Cp + (self.Ki * self.Ci) + (self.Kd * self.Cd)
        if result > 50000:
            result = 50000
        if result < -50000:
            result = -50000
        return result

class PiVideoStream:
    def __init__(self, resolution=(162, 92), framerate=40):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.sensor_mode = 5
        self.camera.rotation = 180
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                                                     format="bgr", use_video_port=True)
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def main():
    # Load tflite Model
    saved_file_name = './tflite_model.tflite'
    interpreter = tf.lite.Interpreter(model_path=saved_file_name, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Arm Electronic Speed Controller
    for a in range(3):
        pi.hardware_PWM(13, 75, 960000)
    # Start Camera Thread
    threaded_capture = PiVideoStream().start()
    time.sleep(2.0)
    # Initialize Shaft Speed
    HALL_ONE = 26
    HALL_TWO = 20
    HALL_THREE = 21
    p1 = read_RPM.reader(pi, HALL_ONE, pulses_per_rev=6, min_RPM=60.0)
    p2 = read_RPM.reader(pi, HALL_TWO, pulses_per_rev=6, min_RPM=60.0)
    p3 = read_RPM.reader(pi, HALL_THREE, pulses_per_rev=6, min_RPM=60.0)
    # Initialize PID
    pid = PID()
    pid.Initialize()
    # Run Loop
    with tf.device('/cpu:0'):
        while True:
            image_1 = threaded_capture.read()
            speed = [p1.RPM(), p2.RPM(), p3.RPM()]
            try:
                speed.remove(max(p1.RPM(), p2.RPM(), p3.RPM()))
            except ValueError:
                pass
            # Do the prediction
            input_data = np.array([np.stack([image_1[22:, :]])], dtype=np.float32)
            interpreter.set_tensor(input_details[1]['index'], input_data)
            interpreter.set_tensor(input_details[0]['index'], np.array([np.stack([0.0])], dtype=np.float32))
            interpreter.invoke()
            [[steering]] = interpreter.get_tensor(output_details[0]['index'])
            [[motor_sp]] = interpreter.get_tensor(output_details[1]['index'])
            # Start the PID loop
            value = pid.Output(int(motor_sp*((30.497-1.0534)/2))-max(speed)/60)
            # Deploy Steering and Throttle
            steeringangle = (0.114)*1e6 + (steering)/32*1e6
            pi.hardware_PWM(13, 75, 514000 + int(value))
            pi.hardware_PWM(18, 75, int(steeringangle))
            

if __name__ == '__main__':
    main()
