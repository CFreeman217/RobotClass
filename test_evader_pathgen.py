#!/usr/bin/env python
## module test_evader_pathgen
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

        # plt.savefig('./log_files/test_pathfinding_evader_djikstra.png', bbox_inches='tight')
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

        # plt.savefig('./log_files/test_pathfinding_evader_astar.png', bbox_inches='tight')
        plt.show()




DJNode.set_map(map_size)
DJNode.load_obstacles(obstacles)
DJNode.set_goal_node(goal_pos)
DJNode.set_starting_node(start_pos)
def djikstra():
    DJNode.run_djikstra()
    path = DJNode.best_path[::-1]
    # print path
    return path

def astar():
    DJNode.run_astar()
    path = DJNode.best_path[::-1]
    # print path
    return path

# astar()

