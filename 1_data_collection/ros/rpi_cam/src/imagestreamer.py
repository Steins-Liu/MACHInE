#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from picamera.array import PiRGBArray
from picamera import PiCamera, Color
from threading import Thread
import time
length_of_stacked_images = 2
width_of_downsize = 160
height_of_downsize =160
	

class PiVideoStream:
	def __init__(self, resolution=(160, 160), framerate=40):
		# initialize the camera and stream
		self.camera = PiCamera()
		self.camera.sensor_mode = 5
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		#self.camera.shutter_speed = 400000
		#self.camera.zoom = (0.0, 0.2, 1.0, 1.0)
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

def cameraone():
    global img
    global cam
    global bridge
    bridge = CvBridge()
    increment_val = 0
    #start data acquisition
    print('Starting data acquisition...')
    threaded_capture = PiVideoStream().start()
    time.sleep(3.0)
    rospy.init_node('cameraone', anonymous=True)
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
	cv2_img = threaded_capture.read()
        cam_data = bridge.cv2_to_imgmsg(cv2_img, "bgr8")
        pub = rospy.Publisher("image_topic", Image, queue_size=1) 
        pub.publish(cam_data)
        rate.sleep()

if __name__ == '__main__':
	try: 
        	cameraone()
	except KeyboardInterrupt:
		print("Shutting down")
