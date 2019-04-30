#!/usr/bin/env python
## module djikstra
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


imu_data = Imu()
yaw = 0.0
twist = Twist()
odom = Odometry()

position = Position()

lin_max_vel = 0.2 # Maxmimum linear velocity

turning_radius = 0.3 # Dubins curve minimum turning radius
step_size = 0.25 # Dubins curve sampling step size
# Set desired waypoints to travel to


map_size = [(0,0),(10,10)]

start_pos = (2,1)

goal_pos = (7,2)

obstacles = [(5,0),
             (5,1),
             (5,2),
             (5,3),
             (5,4),
             (0,5),
             (1,4),
             (2,3),
             (3,2),
             (3,3)]

class DJNode:
    '''
    Djikstra\'s Pathfinding Algorithm (and A* method):
    Finds shortest pathway between grid of points.
    Needs map size, start, end, and obstacle points list.
    '''
    num_nodes = 0
    map_nodes = {}
    all_coordinates = []
    pathway = []
    best_path = []
    start_node = 0
    end_goal = 0

    def __init__(self, x_location, y_location):
        ''' 
        Instantiates an instance of the Djikstra Node Class.
        Only needs an x and y location to create a member.
        Attaches unique number and important properties associated with the node.
        Generated as part of the 'set_map' classmethod.
        '''
        DJNode.num_nodes += 1
        self.xloc = x_location
        self.yloc = y_location
        self.cost = 0.0
        self.parent = None
        self.obstacle = False
        self.start_node = False
        self.goal_node = False
        self.visited = False       
        self.node_index = DJNode.num_nodes

    def distance(self, other):
        '''
        Calculates the distance between nodes. Returns float.
        Usage:
        distance = node_1.distance(node_2)
        '''
        a = self.xloc
        b = self.yloc
        c = other.xloc
        d = other.yloc
        return((abs(c - a)**2 + abs(d - b)**2)**0.5)
    
    @property
    def valid(self):
        '''
        Returns whether the given node is a valid coordinate
        '''
        # index = DJNode.coords_to_index(coords)
        if self.obstacle == False:
            return True
        return False
    
    def get_neighbors(self):
        cur_x = self.xloc
        cur_y = self.yloc
        above = cur_y + 1
        below = cur_y - 1
        left = cur_x - 1
        right = cur_x + 1
        neighbor_list = []
        positions = []
        positions.append(DJNode.coords_to_index((left, above)))
        positions.append(DJNode.coords_to_index((cur_x, above)))
        positions.append(DJNode.coords_to_index((right, above)))
        positions.append(DJNode.coords_to_index((left, cur_y)))
        positions.append(DJNode.coords_to_index((right, cur_y)))
        positions.append(DJNode.coords_to_index((left, below)))
        positions.append(DJNode.coords_to_index((cur_x, below)))
        positions.append(DJNode.coords_to_index((right, below)))
        for spot in positions:
            if spot != False:
                if DJNode.map_nodes[spot].valid == True:
                    if DJNode.map_nodes[spot].visited == False:
                        neighbor_list.append(spot)
        return neighbor_list
        
    def get_costs(self, neighbors, a_star=False):
        prices = []
        for node in neighbors:
            adj_node = DJNode.map_nodes[node]
            current_price = self.distance(adj_node) + self.cost
            if a_star:
                final_node = DJNode.map_nodes[DJNode.end_goal]
                current_price += adj_node.distance(final_node)
            if adj_node.cost == 0.0:
                if adj_node.start_node == False:
                    adj_node.cost = 999.0
            if current_price < adj_node.cost:
                adj_node.cost = current_price
                adj_node.parent = self.node_index
            prices.append((node, current_price))
        return prices
    
    def set_cost(self, parent_node, price):
        self.parent = parent_node.node_index
        self.cost += price

    @property
    def set_obstacle(self):
        ''' 
        Sets obstacles within the map of available points
        '''
        self.obstacle = True
    
    @property
    def set_goal(self):
        '''
        Sets the goal node
        '''
        self.goal_node = True
        DJNode.end_goal = self.node_index

    
    @property
    def set_start(self):
        '''
        Sets the starting node
        '''
        self.start_node = True
        self.visited = True
        DJNode.start_node = self.node_index
    
    @property
    def get_node_info(self):
        '''
        Prints important information about the node
        '''
        print 'NODE # {}, POS: {}, COST: {}, PARENT: {}, OBS = {}, VISIT = {}, GOAL = {}'.format(self.node_index, self.coordinates, self.cost, self.parent,self.obstacle, self.visited, self.goal_node)

    @property
    def coordinates(self):
        '''
        Gets a tuple of the current node coordinates
        '''
        return (self.xloc, self.yloc)
    
    @classmethod
    def set_map(cls, coords):
        '''
        Sets up the map of points and generates an instance of the node class
        for each point in the map.
        Takes a list of two tuples for lower left and upper right coordinate
        Usage:
        DJNode.set_map([(lowLeftX, lowLeftY), (upRightX, upRightY)])
        '''
        cls.map_limits = coords
        cls.x_min = coords[0][0]
        cls.x_max = coords[1][0]
        cls.y_min = coords[0][1]
        cls.y_max = coords[1][1]
        cls.x_coords = range(cls.x_min, cls.x_max + 1)
        cls.y_coords = range(cls.y_min, cls.y_max + 1)
        for j in cls.y_coords:
            for i in cls.x_coords:
                cls.map_nodes[DJNode.num_nodes] = cls(i,j)
                # Needed to speed up the coords_to_index function
                cls.all_coordinates.append((i,j))
                
    @classmethod
    def load_obstacles(cls, obstacles):
        '''
        Takes a list of obstacles and sets the node property on each class instance
        appropriately.
        Takes a list of x,y tuple coordinate pairs for obstacles.
        Usage:
        DJNode.load_obstacles([(obs1x, obs1y), (obs2x, obs2y),...,(obsNx, obsNy)])
        '''
        for obs in obstacles:
            node = cls.coords_to_index(obs)
            cls.map_nodes[node].set_obstacle

    @classmethod
    def set_goal_node(cls, coords):
        '''
        Sets the goal node property from a coordinate pair.
        Usage:
        DJNode.set_goal_node((goal_pos_x, goal_pos_y))
        '''
        node = cls.coords_to_index(coords)
        cls.map_nodes[node].set_goal

    @classmethod
    def set_starting_node(cls, coords):
        '''
        Sets the starting node property from a coordinate pair.
        Usage:
        DJNode.set_starting_node((start_pos_x, start_pos_y))
        '''
        node = cls.coords_to_index(coords)
        cls.map_nodes[node].set_start
    
    @classmethod
    def coords_to_index(cls, coords):
        '''
        Gets the index number of a node from given coordinates
        Usage:
        DJNode.coords_to_index((x_coordinate, y_coordinate))
        '''
        if coords in cls.all_coordinates:
            for node in cls.map_nodes:
                if cls.map_nodes[node].coordinates == coords:
                    return cls.map_nodes[node].node_index
        else:
            # The input coordinates were not created in the map initialization
            return False
    
    @classmethod
    def plot_current(cls, animate=False):

        box_color = '#FD9501'
        ln_width = 3

        plt.plot([cls.x_min, cls.x_max],[cls.y_min, cls.y_min], color=box_color, linewidth=ln_width)
        plt.plot([cls.x_min, cls.x_max],[cls.y_max, cls.y_max], color=box_color, linewidth=ln_width)
        plt.plot([cls.x_min, cls.x_min],[cls.y_min, cls.y_max], color=box_color, linewidth=ln_width)
        plt.plot([cls.x_max, cls.x_max],[cls.y_min, cls.y_max], color=box_color, linewidth=ln_width)
        path_nodes = [i[0] for i in cls.pathway]
        for node in cls.map_nodes.values():
            if node.visited == True:
                plt.annotate('{:1.1f}'.format(node.cost),(node.xloc, node.yloc), color = 'k')
            elif node.node_index in path_nodes:
                plt.annotate('{:1.1f}'.format(node.cost),(node.xloc, node.yloc), color = 'g')
            elif node.obstacle == True:
                plt.scatter(node.xloc, node.yloc, color='r', marker='s')
            else:
                plt.scatter(node.xloc, node.yloc, color='b')
        if animate:
            plt.pause(0.05)
            i_num += 1.0

        
        



    @classmethod
    def run_djikstra(cls):
        c_node_index = cls.start_node
        c_node = cls.map_nodes[c_node_index]
        while c_node.goal_node == False:
            adj_nodes = c_node.get_neighbors()
            adj_costs = c_node.get_costs(adj_nodes)
            cls.pathway = cls.pathway + adj_costs
            for option in cls.pathway:
                for index, option in enumerate(cls.pathway):
                    check_node = cls.map_nodes[option[0]]
                    if check_node.visited == True:
                        cls.pathway.pop(index)
            best_choice = min(cls.pathway, key = lambda x: x[1])
            # cls.plot_current()

            c_node.visited = True
            # print '{} : {} , Cost = {:1.2f}'.format(c_node.node_index, c_node.coordinates, c_node.cost)
            c_node = cls.map_nodes[best_choice[0]]
        
        while c_node.start_node == False:
            cls.best_path.append((c_node.xloc, c_node.yloc))
            c_node = cls.map_nodes[c_node.parent]
        
        cls.best_path.append((c_node.xloc, c_node.yloc))
        cls.plot_current()
        linexs = [i[0] for i in cls.best_path]
        lineys = [i[1] for i in cls.best_path]
        plt.plot(linexs, lineys, '--', color='b', linewidth=2)
        plt.title('Djikstra\'s Algorithm Pathfinding Method \n Distance = {:1.2f}'.format(cls.map_nodes[cls.coords_to_index((cls.best_path[0][0],cls.best_path[0][1]))].cost))

        plt.savefig('./log_files/HW5_prob4.png', bbox_inches='tight')
        plt.show()

    @classmethod
    def run_astar(cls):
        c_node_index = cls.start_node
        c_node = cls.map_nodes[c_node_index]
        while c_node.goal_node == False:
            adj_nodes = c_node.get_neighbors()
            adj_costs = c_node.get_costs(adj_nodes, a_star=True)
            cls.pathway = cls.pathway + adj_costs
            for option in cls.pathway:
                for index, option in enumerate(cls.pathway):
                    check_node = cls.map_nodes[option[0]]
                    if check_node.visited == True:
                        cls.pathway.pop(index)

            best_choice = min(cls.pathway, key = lambda x: x[1])
            # cls.plot_current(animate=True)

            c_node.visited = True
            # print '{} : {} , Cost = {:1.2f}'.format(c_node.node_index, c_node.coordinates, c_node.cost)
            c_node = cls.map_nodes[best_choice[0]]
        
        while c_node.start_node == False:
            cls.best_path.append((c_node.xloc, c_node.yloc))
            c_node = cls.map_nodes[c_node.parent]
        
        cls.best_path.append((c_node.xloc, c_node.yloc))
        cls.plot_current()
        linexs = [i[0] for i in cls.best_path]
        lineys = [i[1] for i in cls.best_path]
        plt.plot(linexs, lineys, '--', color='b', linewidth=2)
        plt.title('A-Star Pathfinding Method \n Distance = {:1.2f}'.format(cls.map_nodes[cls.coords_to_index((cls.best_path[0][0],cls.best_path[0][1]))].cost))

        plt.savefig('./log_files/test_pathfinding_evader.png', bbox_inches='tight')
        plt.show()




DJNode.set_map(map_size)
DJNode.load_obstacles(obstacles)
DJNode.set_goal_node(goal_pos)
DJNode.set_starting_node(start_pos)
# DJNode.run_djikstra()
DJNode.run_astar()
# a = DJNode.coords_to_index((3,3))
# print a
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


def odom_callback(data):
	global odom, imu_data, yaw
	odom = data
	imu_data = data.pose.pose
	quat_list = [imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w]
	(roll, pitch, yaw) = euler_from_quaternion(quat_list)
coordinates = DJNode.best_path[::-1]

def angle(coords1, coords2, coords3):
    dxi = coords2[0] - coords1[0]
    dyi = coords2[1] - coords1[1]
    dxo = coords3[0] - coords2[0]
    dyo = coords3[1] - coords2[1]
    incoming = math.atan2(dyi, dxi)
    outgoing = math.atan2(dyo, dxo)
    return np.mean([incoming,outgoing])*(180.0/math.pi)

waypoints = []

for index, value in enumerate(coordinates):
    last_co = len(coordinates)
    if index == 0:
        desang = 90
        # desang = math.atan2(value[1], value[0])
    elif index == last_co - 1:
        desang = 180
    else:
        desang = angle(coordinates[index-1], value, coordinates[index+1])
    waypoints.append((int(value[0]),int(value[1]), desang))




# Acceptable error is 5cm
err_allow = 0.05

gen_pts = []
dubin_pts = []

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

def makeSimpleProfile(output, input, slop):
    if input > output:
        output = min( input, output + slop )
    elif input < output:
        output = max( input, output - slop )
    else:
        output = input

    return output

def constrain(input, low, high):
    if input < low:
      input = low
    elif input > high:
      input = high
    else:
      input = input

    return input

def checkLinearLimitVelocity(vel):
    if turtlebot3_model == "burger":
      vel = constrain(vel, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
    elif turtlebot3_model == "waffle" or turtlebot3_model == "waffle_pi":
      vel = constrain(vel, -WAFFLE_MAX_LIN_VEL, WAFFLE_MAX_LIN_VEL)
    else:
      vel = constrain(vel, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)

    return vel

def checkAngularLimitVelocity(vel):
    if turtlebot3_model == "burger":

      vel = constrain(vel, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
    elif turtlebot3_model == "waffle" or turtlebot3_model == "waffle_pi":
      vel = constrain(vel, -WAFFLE_MAX_ANG_VEL, WAFFLE_MAX_ANG_VEL)
    else:
      vel = constrain(vel, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)

    return vel

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

distance = lambda x, y: math.sqrt(x**2 + y**2)
rads = lambda x : math.pi * x / 180.0
degs = lambda x : 180.0 * x / math.pi

# colors = cm.rainbow(np.linspace(0,1,10))

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
# print(dubin_pts)
# plt.show()
if __name__=="__main__":
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)
    
    rospy.init_node('turtlebot3_evader')
    pub = rospy.Publisher('/tb1/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber("/tb1/odom",Odometry,odom_callback)
    turtlebot3_model = rospy.get_param("model", "burger")
    # Start from zero
    status = 0
    target_linear_vel   = 0.0
    target_angular_vel  = 0.0
    control_linear_vel  = 0.0
    control_angular_vel = 0.0
    print msg
    # the following code creates a base filename containing the data and time
    fileName = get_filename('test','.csv',"./log_files/")
    # this will constitute the header for the columns in the csv file, this is simply because
    # it is the first line which will be written
    myData = ["pos_x, pos_y, pos_z, roll, pitch, yaw, tgt_lin, ctrl_lin, tgt_ang, ctrl_ang, des_heading"]
    # Write header row to file name
    with open(fileName, 'a') as myFile:
        writer = csv.writer(myFile)
        writer.writerow(myData)
    # Stopping for 1 second helps with the next commands
    time.sleep(1)
    # Get parameters from the simulation environment
    rospy.Subscriber('/tb1/eul', Position, callback_ori)
    rospy.Subscriber('/tb1/odom', Odometry, callback_pos)
    rate = rospy.Rate(100) # 100Hz update rate
    time.sleep(1) # pause for 1 second
    ang_err = 100 # Initializing error term for PID control of heading
    vel_err = 100 # Initializing error term for PID control of velocity
    wpt = 0 # Set waypoint number
    while(1):
        key = getKey()
        des_x = dubin_pts[wpt][0] # Desired x coordinate
        des_y = dubin_pts[wpt][1] # Desired y coordinate
        cur_x = position.linear.x # Current x position
        cur_y = position.linear.y # Current y position
        err_x = des_x - cur_x # Current x error
        err_y = des_y - cur_y # Current y error
        ### Get desired heading from error terms'''
        desired_heading = math.atan2(err_y, err_x)*(180.0/math.pi)
        ### Get distance to waypoint'''
        distance = math.sqrt(err_x**2 + err_y**2)
        ### If the current position is within the error limit, the robot has reached the waypoint'''
        if abs(distance) < err_allow:
            ### Cycle to the next waypoint'''
            print 'waypoint reached'
            wpt += 1
            ### When we reach the end of the waypoint list, stop and end the program '''
            if wpt == len(dubin_pts):
                target_linear_vel   = 0.0
                control_linear_vel  = 0.0
                target_angular_vel  = 0.0
                control_angular_vel = 0.0
                break
        else:
            if angular_dist(desired_heading, position.angular.yaw) > 5:
                if wpt == 0:
                    target_linear_vel = 0
            else:
                target_linear_vel, vel_err = pid(0.0, distance, vel_err, 1/100.0, k_p=.9, k_i=0, k_d=.05)
                target_linear_vel = abs(target_linear_vel)
            if target_linear_vel > lin_max_vel:
                target_linear_vel = lin_max_vel
            target_angular_vel, ang_err = pid(desired_heading, position.angular.yaw, ang_err, 1/100.0, k_p=.8, k_i=0.0, k_d=0.2)
            if target_angular_vel > BURGER_MAX_ANG_VEL:
                target_angular_vel = BURGER_MAX_ANG_VEL
        myData = [position.linear.x,position.linear.y,position.linear.z,
                    position.angular.roll,position.angular.pitch,position.angular.yaw,
                    target_linear_vel, control_linear_vel, target_angular_vel, control_angular_vel,
                    desired_heading]

        with open(fileName, 'a') as myFile:
            writer = csv.writer(myFile)
            writer.writerow(myData)

        # print status message
        # print "write complete, waiting"
        rate.sleep()
        twist = Twist()

        control_linear_vel = makeSimpleProfile(control_linear_vel, target_linear_vel, (LIN_VEL_STEP_SIZE/2.0))
        twist.linear.x = control_linear_vel; twist.linear.y = 0.0; twist.linear.z = 0.0

        control_angular_vel = makeSimpleProfile(control_angular_vel, target_angular_vel, (ANG_VEL_STEP_SIZE/2.0))
        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = control_angular_vel

        pub.publish(twist)
        if (key == '\x03'):
            break





    twist = Twist()
    twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
    twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
    pub.publish(twist)

    if os.name != 'nt':
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    pos_x, pos_y, yaw, tgt_lin, ctr_lin, tgt_ang, ctr_ang = np.loadtxt(fileName,
                            delimiter = ',',
                            skiprows = 1,
                            usecols=(0,1,5,6,7,8,9),
                            unpack=True)
    plt.plot(pos_x, pos_y, label="Actual Position")
    plt.title("Position Data")
    plt.xlabel("X-Position (m)")
    plt.ylabel("Y-Position (m)")
    plt.grid()
    # plt.legend(loc='lower right')
    plt.savefig("./log_files" + "/" + "test_dubins_evader.png", bbox_inches='tight')
    plt.show()
