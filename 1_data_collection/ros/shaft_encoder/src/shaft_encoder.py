#!/usr/bin/env python
import rospy
import math
import time
import numpy
import std_msgs
from std_msgs.msg import UInt16
from sensor_msgs.msg import Joy
from time import sleep
import pigpio
import read_RPM

# Author: Jesse Cha
# This ROS Node listens to one of the sensored DC motor 3 phase
# digital outputs and finds the wheel rotation count of the RC car. 

# Define Constants
WHEEL_DIAMETER = 0.0245 #Units in Meters
DRIVE_RATIO = 53./9.  # 53t spur, 9t pinion
MOTOR_POLES = 2  # brushless sensor counts per motor revolution
M_PI = 3.141592
V_SCALE = WHEEL_DIAMETER*M_PI / DRIVE_RATIO / MOTOR_POLES
HALL_ONE = 26
HALL_TWO = 20
HALL_THREE = 21

# Initialize GPIO
pi = pigpio.pi()

if __name__ == '__main__':
        p1 = read_RPM.reader(pi, HALL_ONE, pulses_per_rev =6, min_RPM=60.0)
        p2 = read_RPM.reader(pi, HALL_TWO, pulses_per_rev = 6, min_RPM=60.0)
        p3 = read_RPM.reader(pi, HALL_THREE, pulses_per_rev = 6, min_RPM=60.0)
        rospy.init_node('shaft_encoder')
        pub = rospy.Publisher("shaft_speed", std_msgs.msg.String, queue_size =1) # changed shaft_speed to shaft_encoder
        pubtwo = rospy.Publisher("shaft_time", std_msgs.msg.String, queue_size =1) # changed shaft_time to shaft_encoder
        while not rospy.is_shutdown():
                start = time.time()
                time.sleep(0.015)
                array = [p1.RPM(), p2.RPM(), p3.RPM()]
                array.remove(max(p1.RPM(), p2.RPM(), p3.RPM()))
                pub.publish(str(max(array)/60)) # Rotations per second
                end = time.time()
                pubtwo.publish(str(start-end))

   
