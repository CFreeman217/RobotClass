#!/usr/bin/env python

# Turtlebot code to follow a pre-defined global trajectory #
# v1 - Prop control of angular velocity to return to trajectory #

import time
import rospy
from math import sqrt,cos,sin,atan2
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu	
from nav_msgs.msg import Odometry
import sys, select, termios, tty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import random
import numpy as np


#### SET THIS MODE TO FALSE FOR CONSTANT SPEEP/DIRECTION ####
#### SET THIS MODE TO TRUE FOR RANDOM WAYPOINT EVASION   ####
random_evader = False
#############################################################

time_prev = time.time()
imu_data = Imu()
yaw = 0.0
twist = Twist()
odom = Odometry()

### Declaring GLOBAL Variables for Heading and Speed Control ###
yaw_error_prev = 0.0
yaw_error_sum = 0.0
spd_error_prev = 0.0
spd_error_sum = 0.0

dt = 0.1

def dist(p1, p2):
	return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

# This function takes the waypoints in nodes and steers to them
# one at a time to within way_toler 
def traj_pointer(tar_node, cur_loc):
	des_direction = atan2(tar_node[1]-cur_loc[1],tar_node[0]-cur_loc[0])
	tar_distance = dist(cur_loc, tar_node)
	return tar_distance, des_direction



def heading_control(des_direction,cur_head):
	Kp = 2.5 #0.02
	Ki = 0.00
	Kd = 0.06 #0.1
	ang_vel_limit = 1.5

	global yaw_error_prev
	global yaw_error_sum

	yaw_error = (des_direction - cur_head)

	# Small wrapping help for yaw angles that are actually close to each other
	if yaw_error > 6.2832:
		yaw_error = yaw_error - 6.2832
	if yaw_error < -6.2832:
		yaw_error = yaw_error + 6.2832

	yaw_error_dot = (yaw_error - yaw_error_prev)/dt
	yaw_error_sum = yaw_error_sum + yaw_error*dt
	yaw_error_prev = yaw_error

	ang_vel_cmd = Kp*yaw_error + Ki*yaw_error_sum + Kd*yaw_error_dot
	if ang_vel_cmd > ang_vel_limit:
		ang_vel_cmd = ang_vel_limit
	if ang_vel_cmd < -ang_vel_limit:
		ang_vel_cmd = -ang_vel_limit
	return ang_vel_cmd




def odom_callback(data):
	global odom, imu_data, yaw
	odom = data
	imu_data = data.pose.pose
	quat_list = [imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w]
	(roll, pitch, yaw) = euler_from_quaternion(quat_list)

def evasion_waypoints(random_evader):
	wayp = np.array([3.0, 3.0]) # Starting at 3,3 (TB1 start position from Launch file)
	if random_evader == True:
		for i in range(0,4): # Making 4 waypoints
			xy_rand = random.randint(-1.0, 2.0), random.randint(-1.0, 2.0) # biased to keep going in approximately same direction
			wayp = np.vstack((wayp,xy_rand)) # This keeps bot from attempting to circle back and forth (too much)
	else:
		wayp = np.vstack((wayp,[-5.0,5.0])) # Fixed target/trajectory. 

	return wayp # Return list of waypoints for trajectory/path following


if __name__=="__main__":
	
	settings = termios.tcgetattr(sys.stdin)

	rospy.init_node('turtlebot3_evader')

	pub = rospy.Publisher('/tb1/cmd_vel', Twist, queue_size=10)
	rospy.Subscriber("/tb1/odom",Odometry,odom_callback)

	turtlebot3_model = "burger"

	
	twist.angular.z = 0
	twist.linear.x = 0.0	
	twist.linear.y = 0
	twist.linear.z = 0
	twist.angular.x = 0
	twist.angular.y = 0
	
	pub.publish(twist) # Making sure vehicle is starting from rest.

	rospy.sleep(1.0) # let system initialize before taking off

	#trajectory = np.array(0) # Fix this, need to stick in random waypoints

	trajectory = evasion_waypoints(random_evader) # Call function to generate waypoints
	print "Trajectory of evader is ", trajectory

	cur_pose = odom.pose.pose.position
	cur_loc = cur_pose.x, cur_pose.y
	cur_wpt = 0 # Initialize which waypoint we are going to
	last_wpt = trajectory.shape[0] # Number of points in trajectory

	vel_cmd = 0.1 # Constant forward speed - half speed
	global yaw, cur_loc

	while(cur_wpt < last_wpt): # Stop when the last wpt has been reached
		last_wpt = trajectory.shape[0] # Updates as the algorithms are recalled
		cur_pose = odom.pose.pose.position
		cur_loc = cur_pose.x, cur_pose.y

		if (dist(cur_loc,trajectory[cur_wpt]) < 0.1): # move to next wpt when close
			cur_wpt = cur_wpt + 1
			if (cur_wpt == last_wpt):
				break

		dist_to_wpt, des_dir = traj_pointer(trajectory[cur_wpt], cur_loc) # returns distance and angle to waypoint
		ang_vel_cmd = heading_control(des_dir,yaw)

		twist = Twist()
		twist.linear.x = vel_cmd
		twist.angular.z = ang_vel_cmd
		
		pub.publish(twist) 
		
		#print twist.linear.x
		#print "Position",cur_pose.x, cur_pose.y, trajectory[cur_wpt]
		#print "Angles",perp_angle, wpt_angle, des_dir, yaw
		#print "Wpt Info", dist_to_traj, dist_to_wpt, fake_dist, dist_ratio
		#print "Cmds", ang_vel_cmd, vel_cmd

		rospy.sleep(dt)

	# Last waypoint reached, stop the turtlebot before terminating
	twist.linear.x = 0
	twist.angular.z = 0
	pub.publish(twist)
   

	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
