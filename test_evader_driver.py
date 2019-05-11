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
from test_evader_pathgen import astar

imu_data = Imu()
yaw = 0.0
twist = Twist()
twist2 = Twist()
odom = Odometry()

position = Position()

lin_max_vel = 0.2 # Maxmimum linear velocity

turning_radius = 0.3 # Dubins curve minimum turning radius
step_size = 0.25 # Dubins curve sampling step size
# Set desired waypoints to travel to

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

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
coordinates = astar()

# print coordinates

# Acceptable error is 5cm
err_allow = 0.05

gen_pts = []

dubin_pts = []

waypoints = []

distance = lambda x, y: math.sqrt(x**2 + y**2)
rads = lambda x : math.pi * x / 180.0
degs = lambda x : 180.0 * x / math.pi

def odom_callback(data):
	global odom, imu_data, yaw
	odom = data
	imu_data = data.pose.pose
	quat_list = [imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w]
	(roll, pitch, yaw) = euler_from_quaternion(quat_list)

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

def intercept(chaser_x0, chaser_y0, chaser_max_vel, evader_x0, evader_y0, evader_heading, evader_max_vel):
    distance = lambda x, y: math.sqrt(x**2 + y**2)
    rads = lambda x : math.pi * x / 180.0
    degs = lambda x : 180.0 * x / math.pi

    d_x0 = abs(evader_x0 - chaser_x0) # x distance between evader and chaser
    d_y0 = abs(evader_y0 - chaser_y0) # y distance between evader and chaser

    d_0 = distance(d_x0, d_y0) # total distance between evader and chaser
    e_dx = evader_max_vel*math.cos(rads(evader_heading)) # x-velocity of evader
    e_dy = evader_max_vel*math.sin(rads(evader_heading)) # y-velocity of evader
    dot_num = d_x0*e_dx + d_y0*e_dy
    if evader_max_vel == 0:
        evader_max_vel = 1
        print 'evader exception'
        angle = 0
        chaser_desired_vel = 0.1
        int_time = 1
    angle = math.acos(dot_num/(d_0*evader_max_vel))

    # Set up quadratic function
    q_a = (chaser_max_vel**2 - evader_max_vel**2)
    q_b = 2*(d_0*evader_max_vel)
    q_c = (d_0**2)*(-1)

    d_t_d = q_b**2 - 4*q_a*q_c
    # if d_t_d < 0:
        # print "INTERCEPT FUNCTION ERROR: NO INTERCEPT FOUND"
    
    d_tp = (-q_b + math.sqrt(d_t_d))/(2*q_a)
    d_tn = (-q_b - math.sqrt(d_t_d))/(2*q_a)
    int_time = 0
    
    if d_tn < 0 or d_tp < d_tn:
        # print "INTERCEPT FUNCTION REJECT NEGATIVE TIME"
        int_time = d_tp
    else:
        int_time = d_tn
    # if int_time < 0:
        # print "INTERCEPT FUNCTION ERROR: NO POSITIVE TIME FOUND"


    pi_x = evader_x0 + e_dx*int_time
    pi_y = evader_y0 + e_dy*int_time
    pi_dx = abs(chaser_x0 - pi_x)
    pi_dy = abs(chaser_y0 - pi_y)
    dp_int = distance(pi_dx, pi_dy)
    chaser_desired_vel = dp_int/int_time

    return angle, chaser_desired_vel, int_time

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
    pub = rospy.Publisher('/tb1/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('/tb1/eul', Position, callback_ori)
    rospy.Subscriber('/tb1/odom', Odometry, callback_pos)
    turtlebot3_model = rospy.get_param("model", "burger")
    
    pub2 = rospy.Publisher('/tb2/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber("/tb2/odom",Odometry,odom_callback)

    # Start from zero
    status = 0
    target_linear_vel   = 0.0
    target_angular_vel  = 0.0
    pursuer_vel = 0
    # pub.publish(twist)
    rospy.sleep(1.0)
    print msg
    # the following code creates a base filename containing the data and time
    fileName = get_filename('test','.csv',"./log_files/")
    # this will constitute the header for the columns in the csv file, this is simply because
    # it is the first line which will be written
    myData = ["evader_pos_x, evader_pos_y, evader_pos_z, evader_roll, evader_pitch, evader_yaw, evader_tgt_lin, evader_tgt_ang, evader_des_heading, pursuer_pos_x, pursuer_pos_y, pursuer_yaw, pursuer_ang_vel, pursuer_lin_vel, pursuer_desired_heading"]
    # Write header row to file name
    with open(fileName, 'a') as myFile:
        writer = csv.writer(myFile)
        writer.writerow(myData)
    # Stopping for 1 second helps with the next commands
    # time.sleep(1)
    # Get parameters from the simulation environment
    prev_x = 0
    prev_y = 0
    oldtime = time.time()
    rate = rospy.Rate(100) # 100Hz update rate
    ang_err = 100 # Initializing error term for PID control of heading
    vel_err = 100 # Initializing error term for PID control of velocity
    pur_ang_err = 100
    wpt = 0 # Set waypoint number
    while (1):
        cur_pose = odom.pose.pose.position
        cur_loc_x = cur_pose.x
        cur_loc_y = cur_pose.y
        pur_head = yaw
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
            
            if angular_dist(desired_heading, curr_heading) > 15.0:
                target_linear_vel = 0.075
            else:
                target_linear_vel = 0.15
                # target_linear_vel, vel_err = pid(0.0, distance, vel_err, 1/100.0, k_p=.98, k_i=0, k_d=.025)
                # target_linear_vel = abs(target_linear_vel)
            # if target_linear_vel > 0.15:
            #     target_linear_vel = 0.15
            target_angular_vel, ang_err = pid(rads(desired_heading), rads(curr_heading), ang_err, 1/100.0, k_p=0.8, k_i=0.0, k_d=0.2)
            if abs(target_angular_vel) > BURGER_MAX_ANG_VEL:
                if target_angular_vel < 0:
                    target_angular_vel = -BURGER_MAX_ANG_VEL
                else:
                    target_angular_vel = BURGER_MAX_ANG_VEL

 


        # print status message
        rate.sleep()

        # Sets linear velocity and pushes the value to the twist parameter incorporating some slop
        twist.linear.x = target_linear_vel; twist.linear.y = 0.0; twist.linear.z = 0.0
        # print "Linear Vel = {}".format(control_linear_vel)
        # Sets angular velocity and pushes the value to the twist parameter incorporating some slop
        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = target_angular_vel
        # Sends evader commands
        pub.publish(twist)


        int_heading, req_vel, _ = intercept(cur_loc_x, cur_loc_y, pursuer_vel, cur_x, cur_y, rads(curr_heading), target_linear_vel)
        
        pursuer_angular_vel, pur_ang_err = pid(int_heading, pur_head, pur_ang_err, 1/100.0, k_p=0.8, k_i=0.0, k_d=0.2)
        if abs(pursuer_angular_vel) > BURGER_MAX_ANG_VEL:
            if pursuer_angular_vel < 0:
                pursuer_angular_vel = -BURGER_MAX_ANG_VEL
            else:
                pursuer_angular_vel = BURGER_MAX_ANG_VEL

        # print "Pursuer Heading: {}".format(pur_head)

        # tar_vel = target_linear_vel
        # tar_heading = curr_heading
        
        if angular_dist(degs(int_heading), degs(pur_head)) > 15.0:
            pursuer_vel = 0.1
        else:
            pursuer_vel = 0.2
        twist2.linear.x = pursuer_vel; twist2.linear.y = 0.0; twist2.linear.z = 0.0
        twist2.angular.x = 0.0; twist2.angular.y = 0.0; twist2.angular.z = pursuer_angular_vel
        pub2.publish(twist2)
        

        myData = [cur_x,cur_y,position.linear.z,
                    position.angular.roll,position.angular.pitch,curr_heading,
                    target_linear_vel, target_angular_vel, desired_heading,
                    cur_loc_x, cur_loc_y, pur_head, pursuer_angular_vel, pursuer_vel, int_heading]

        with open(fileName, 'a') as myFile:
            writer = csv.writer(myFile)
            writer.writerow(myData)

        if (key == '\x03'):
            break
        
        catchx = abs(cur_x - cur_loc_x)
        catchy = abs(cur_y - cur_loc_y)
        caught = distance(catchx, catchy)
        print 'Distance to Target : {}'.format(caught)
        if caught < .25:
            twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
            twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
            twist2.linear.x = 0.0; twist2.linear.y = 0.0; twist2.linear.z = 0.0
            twist2.angular.x = 0.0; twist2.angular.y = 0.0; twist2.angular.z = 0.0
            pub.publish(twist)
            pub2.publish(twist2)
            break
    


    twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
    twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
    pub.publish(twist)
    # pub2.publish(twist)

    if os.name != 'nt':
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    ev_x, ev_y, ev_yaw, pur_x, pur_y, pur_yaw = np.loadtxt(fileName,
                            delimiter = ',',
                            skiprows = 1,
                            usecols=(0,1,5,9,10,11),
                            unpack=True)
    plt.plot(ev_x, ev_y, label="Evader Position")
    plt.plot(pur_x, pur_y, label="Pursuer Position")
    plt.title("Pursuer - Evader Position Data")
    plt.xlabel("X-Position (m)")
    plt.ylabel("Y-Position (m)")
    plt.grid()
    # plt.legend(loc='lower right')
    plt.savefig("./log_files/" + "test1_evader_paths_next.png", bbox_inches='tight')
    plt.show()