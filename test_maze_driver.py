#!/usr/bin/env python

import rospy, time, csv, os, datetime, math
from beginner_tutorials.msg import Position
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import sys, select
if os.name == 'nt':
  import msvcrt
else:
  import tty, termios
import dubins
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
from sensor_msgs.msg import Imu	
from test_maze_pathgen import djikstra

imu_data = Imu()
yaw = 0.0
twist = Twist()
twist2 = Twist()
odom = Odometry()

position = Position()

lin_max_vel = 0.2 # Maxmimum linear velocity

turning_radius = 0.1 # Dubins curve minimum turning radius
step_size = 1.0 # Dubins curve sampling step size
# Set desired waypoints to travel to

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = .75

WAFFLE_MAX_LIN_VEL = 0.26
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1

msg = """
Command and control. 

Turtlebot commands given: 

Follow Waypoints to arrive at the destination.
Uses PID to control heading and velocity
Maximum Linear velocity of 0.2 m/s
Maximum Angular velocity is set by simulation limits

CTRL-C to quit
"""
coordinates = djikstra()

# print coordinates

# Acceptable error is 5cm
err_allow = 0.05

gen_pts = []

dubin_pts = []

waypoints = []

distance = lambda x, y: math.sqrt(x**2 + y**2)
rads = lambda x : math.pi * x / 180.0
degs = lambda x : 180.0 * x / math.pi

def angle(coords1, coords2, coords3):
    dxi = coords2[0] - coords1[0]
    dyi = coords2[1] - coords1[1]
    dxo = coords3[0] - coords2[0]
    dyo = coords3[1] - coords2[1]
    incoming = math.atan2(dyi, dxi)
    outgoing = math.atan2(dyo, dxo)
    return np.mean([incoming,outgoing])*(180.0/math.pi)

# callback function for 3DOF linear position
def callback_pos(data):

	# the following line allows this function to access the variable
	# called position which exists in the global namespace, without
	# this statement using the global keyword, we will get an error
	# that the local variable "position" has been used prior to
	# being declared
	global position

	position.linear.x = data.pose.pose.position.x
	position.linear.y = data.pose.pose.position.y
	position.linear.z = data.pose.pose.position.z

# callback function for 3DOF rotation position
def callback_ori(data):

	global position

	position.angular.roll  = data.angular.roll
	position.angular.pitch = data.angular.pitch
	position.angular.yaw   = data.angular.yaw

def getKey():
    if os.name == 'nt':
      return msvcrt.getch()

    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def pid(command, meas, prev_err, freq, k_p=1, k_i=0, k_d=0):
    '''
    PID Controller Function:

    Inputs:
    command = Desired target value
    meas = Measured value
    prev_err = Error from previous iteration
    freq = Sampling frequency
    k_p = Proportional gain
    k_i = Integrator gain
    k_d = Derivative gain

    Outputs:
    lam_com = Output command signal
    err_prop = Measured error on this iteration
    '''
    # Numeric differentiation by backward finite divided difference method
    ddif = lambda h, f_0, f_1 : (f_0 - f_1) / h
    # Numeric integration by trapezoidal method
    trap = lambda h, f_0, f_1 : h * (f_0 - f_1) / 2
    # Current error
    err_prop = command - meas
    # Integrator error
    err_int = trap(freq, err_prop, prev_err)
    # Derivative error
    err_der = ddif(freq, err_prop, prev_err)
    # Combined lambda signal
    lam_com = (err_prop * k_p) + (err_int * k_i) + (err_der * k_d)
    return lam_com, err_prop

def get_filename(prefix, suffix, base_path):
    '''
    Gets a unique file name in the base path.
    
    Appends date and time information to file name and adds a number
    if the file name is stil not unique.

    prefix = Homework assignment name

    suffix = Extension

    base_path = Location of log file
    '''
    # Set base filename for compare
    fileNameBase = base_path + prefix + "_" + datetime.datetime.now().strftime("%b_%d_%H_%M")
    # Set base for numbering system if filename exists
    num = 1
    # Generate complete filename to check existence
    fileName = fileNameBase + suffix
    # Find a unique filename
    while os.path.isfile(fileName):
        # if the filename is not unique, add a number to the end of it
        fileName = fileNameBase + "_" + str(num) + suffix
        # increments the number in case the filename is still not unique
        num = num + 1
    return fileName

def gen_pathway(begin_pt, end_pt, min_rad, stp_size):
    '''
    Creates a dubins pathway between two points

    Inputs:
    begin_pt = (x_coordinate, y_coordinate, radian_angle)
    end_pt = (x_coordinate, y_coordinate, radian_angle)
    min_rad = Minimum turning radius
    stp_size = Step size for path samples

    Outputs:
    list of tuples [(x_coordinate, y_coordinate)]
    '''
    tunnel = dubins.shortest_path(begin_pt, end_pt, min_rad)
    samples, _ = tunnel.sample_many(stp_size)
    return [(samples[i][0], samples[i][1]) for i in range(len(samples))]

def absolute_angle(in_angle):
    ''' 
    Gets absolute angle
    Converts when input angle is more than 360 degrees
    '''
    count = int(in_angle) / 360
    
    out_angle = in_angle - count*360.0
    return out_angle

def angular_dist(angle_a, angle_b):
    '''
    Determines the difference between two angles
    '''
    remainder = (angle_a - angle_b) % 360.0
    if remainder >= 180.0:
        remainder -= 360.0
    return remainder

for index, value in enumerate(coordinates):
    last_co = len(coordinates)
    if index == 0:
        desang = 0
        # desang = math.atan2(value[1], value[0])
    elif index == last_co - 1:
        desang = 270
    else:
        desang = angle(coordinates[index-1], value, coordinates[index+1])
    waypoints.append((int(value[0]),int(value[1]), desang))

# print waypoints
way_rads = [(waypoints[i][0], waypoints[i][1], rads(waypoints[i][2])) for i in range(len(waypoints))]
for i in range(len(way_rads)-1):
    segment = gen_pathway(way_rads[i], way_rads[i+1], turning_radius, step_size)
    gen_pts.append(segment)

for segment in range(len(gen_pts)):
    x_vals = []
    y_vals = []
    for point in range(len(gen_pts[segment])):
        x_vals.append(gen_pts[segment][point][0])
        y_vals.append(gen_pts[segment][point][1])
        lab = 'Dubin Segment ' + str(segment)

        dubin_pts.append((gen_pts[segment][point][0],gen_pts[segment][point][1]))
    plt.scatter(x_vals, y_vals, label=lab, color='k')

if __name__=="__main__":
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)
    
    rospy.init_node('quat_2_eul')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('/eul', Position, callback_ori)
    rospy.Subscriber('/odom', Odometry, callback_pos)
    turtlebot3_model = rospy.get_param("model", "burger")

    # Start from zero
    status = 0
    target_linear_vel   = 0.0
    target_angular_vel  = 0.0
    pursuer_vel = 0
    # pub.publish(twist)
    rospy.sleep(1.0)
    print msg
    # the following code creates a base filename containing the data and time
    fileName = get_filename('test_maze','.csv',"./log_files/")
    # this will constitute the header for the columns in the csv file, this is simply because
    # it is the first line which will be written
    myData = ["pos_x, pos_y, pos_z, roll, pitch, yaw, tgt_lin, tgt_ang, des_heading,"]
    # Write header row to file name
    with open(fileName, 'a') as myFile:
        writer = csv.writer(myFile)
        writer.writerow(myData)
    # Stopping for 1 second helps with the next commands
    # time.sleep(1)
    # Get parameters from the simulation environment
    rate = rospy.Rate(100) # 100Hz update rate
    ang_err = 100 # Initializing error term for PID control of heading
    vel_err = 100 # Initializing error term for PID control of velocity
    wpt = 0 # Set waypoint number
    while (1):
        key = getKey()
        des_x = dubin_pts[wpt][0] # Desired x coordinate
        des_y = dubin_pts[wpt][1] # Desired y coordinate
        cur_x = position.linear.x # Current x position
        cur_y = position.linear.y # Current y position
        err_x = des_x - cur_x # Current x error
        err_y = des_y - cur_y # Current y error
        ### Get desired heading from error terms'''
        desired_heading = degs(math.atan2(err_y, err_x))
        curr_heading = position.angular.yaw
        # print "CurHead {} DesHead {} Curr(x,y) ({},{}), Des(x,y) ({},{})".format(curr_heading, desired_heading, cur_x, cur_y, des_x,des_y)
        ### Get distance to waypoint'''
        dist_err = math.sqrt(err_x**2 + err_y**2)
        ### If the current position is within the error limit, the robot has reached the waypoint'''
        if abs(dist_err) < err_allow:
            ### Cycle to the next waypoint'''
            print 'waypoint {} reached'.format(wpt)
            wpt += 1
            if wpt == len(dubin_pts):
                break
            ### When we reach the end of the waypoint list, stop and end the program '''
        else:
            
            if angular_dist(desired_heading, curr_heading) > 5.0:
                target_linear_vel = 0.05
            else:
                target_linear_vel = 0.1
            target_angular_vel, ang_err = pid(rads(desired_heading), rads(curr_heading), ang_err, 1, k_p=0.8, k_i=0.0, k_d=0.2)
            if abs(target_angular_vel) > BURGER_MAX_ANG_VEL:
                if target_angular_vel < 0:
                    target_angular_vel = -BURGER_MAX_ANG_VEL
                else:
                    target_angular_vel = BURGER_MAX_ANG_VEL

        # print status message
        # rate.sleep()

        # Sets linear velocity and pushes the value to the twist parameter incorporating some slop
        twist.linear.x = target_linear_vel; twist.linear.y = 0.0; twist.linear.z = 0.0
        # print "Linear Vel = {}".format(control_linear_vel)
        # Sets angular velocity and pushes the value to the twist parameter incorporating some slop
        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = target_angular_vel
        # Sends evader commands
        pub.publish(twist)

        myData = [cur_x,cur_y,position.linear.z,
                    position.angular.roll,position.angular.pitch,curr_heading,
                    target_linear_vel, target_angular_vel, desired_heading,]

        with open(fileName, 'a') as myFile:
            writer = csv.writer(myFile)
            writer.writerow(myData)

        if (key == '\x03'):
            break

    twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
    twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
    pub.publish(twist)


    if os.name != 'nt':
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    pos_x, pos_y, bot_yaw = np.loadtxt(fileName,
                            delimiter = ',',
                            skiprows = 1,
                            usecols=(0,1,5),
                            unpack=True)
    plt.plot(pos_x, pos_y, label="Recorded Position")
    plt.title("Maze Djikstra Position Data")
    plt.xlabel("X-Position (m)")
    plt.ylabel("Y-Position (m)")
    plt.grid()
    # plt.legend(loc='lower right')
    plt.savefig("./log_files/" + "test1_maze_paths.png", bbox_inches='tight')
    plt.show()